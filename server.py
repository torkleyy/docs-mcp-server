"""
MCP server exposing local Markdown documentation via LlamaIndex.

This server uses the LlamaIndex library to build a persistent vector store
from a directory of Markdown files.  It exposes two tools to the Model
Context Protocol (MCP) client:

  1. ``search_docs(query: str, k: int = 6) -> list[{path, heading, score, snippet}]``
     Perform a similarity search over the indexed Markdown chunks and return
     up to ``k`` results.  Each result contains the relative path to the
     source file, the best available heading context, the similarity score,
     and a short excerpt from the matched section.

  2. ``ask_docs(question: str) -> {answer: str, citations: list[{path, heading, snippet}]}``
     Generate an answer to a natural‑language question using a retrieval‑augmented
     generation approach.  The answer is grounded in the indexed content.
     Citations are returned for each supporting chunk.

The server reads configuration from environment variables.  In particular,
``DOCS_DIR`` points to the directory containing your Markdown files and
defaults to ``./docs``.  If the directory does not exist or contains no
Markdown files, the server will exit with a non‑zero status at start up.

OpenAI credentials are loaded from the environment as well; see
``README.md`` for details on which variables must be defined.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from mcp.server.fastmcp import FastMCP

try:
    # Core LlamaIndex imports.  These are imported lazily to avoid
    # incurring import overhead or hard failures before we verify the docs
    # directory exists.  If a required submodule is missing, an ImportError
    # will be raised here rather than at runtime in tool functions.
    from llama_index.core import Document, PromptTemplate, Settings, StorageContext, VectorStoreIndex, load_index_from_storage
    from llama_index.core.node_parser import MarkdownNodeParser
except ImportError as exc:
    # Provide a clear error message if dependencies are missing.  This makes
    # it obvious during server startup which package is not installed.
    missing = str(exc).split(" '" )[-1].split("'")[0]
    raise SystemExit(
        f"Missing dependency: {missing}. Please install required packages as described in the README."
    )

# Provider-specific imports are deferred until we know which provider to use.

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Model provider configuration.  Set MODEL_PROVIDER to "ollama" to use local
# models via Ollama, or "openai" (default) to use OpenAI's API.
MODEL_PROVIDER: str = os.environ.get("MODEL_PROVIDER", "openai").lower()

# Names of the OpenAI models to use.  Update these constants to swap
# in different models without touching the retrieval logic.
OPENAI_CHAT_MODEL: str = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL: str = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Names of the Ollama models to use when MODEL_PROVIDER is "ollama".
OLLAMA_CHAT_MODEL: str = os.environ.get("OLLAMA_CHAT_MODEL", "llama3.2")
OLLAMA_EMBEDDING_MODEL: str = os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_HOST: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# The directory in which to persist the vector index.  Changing this will
# rebuild the index from scratch if the directory does not already exist.
PERSIST_DIR_NAME: str = ".vector_store"

# File that tracks modification times for incremental index updates.
METADATA_FILE_NAME: str = ".file_metadata.json"

# The maximum length of a snippet returned to the client.
SNIPPET_CHAR_LIMIT: int = 500

# Number of chunks to retrieve for RAG queries
RAG_TOP_K: int = 8

# System prompt for the RAG query engine
RAG_SYSTEM_PROMPT: str = """You are a helpful documentation assistant. Answer questions based solely on the provided context.

Rules:
- Be specific and detailed. List all relevant items when asked "what X are supported/available".
- Include concrete examples, syntax, or values from the documentation when relevant.
- If the context contains tables, lists, or enumerations, reproduce the key information.
- If the answer is not in the context, say "I don't have enough information to answer this."
- Do not make up information that isn't in the provided context.
"""

# Configure logging early.  Logs will be emitted to stdout.
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _get_docs_dir() -> Path:
    """Resolve the documentation directory from the environment.

    Returns:
        Path: The absolute path to the documentation directory.

    Raises:
        SystemExit: If the directory does not exist or contains no Markdown files.
    """
    docs_dir = os.environ.get("DOCS_DIR", "./docs")
    docs_path = Path(docs_dir).resolve()
    if not docs_path.exists() or not docs_path.is_dir():
        logger.error("DOCS_DIR '%s' does not exist or is not a directory", docs_path)
        raise SystemExit(1)

    # Look for at least one Markdown file (.md or .mdx)
    md_files = list(docs_path.rglob("*.md")) + list(docs_path.rglob("*.mdx"))
    if not md_files:
        logger.error("DOCS_DIR '%s' contains no Markdown (.md/.mdx) files", docs_path)
        raise SystemExit(1)

    logger.info("Using documentation directory: %s", docs_path)
    return docs_path


def _get_current_file_metadata(docs_path: Path) -> Dict[str, float]:
    """Get modification times for all markdown files in the docs directory.

    Args:
        docs_path: The documentation directory.

    Returns:
        Dict mapping relative file paths to their modification times.
    """
    metadata: Dict[str, float] = {}
    for file_path in docs_path.rglob("*"):
        if file_path.suffix.lower() in {".md", ".mdx"}:
            relative_path = str(file_path.relative_to(docs_path))
            metadata[relative_path] = file_path.stat().st_mtime
    return metadata


def _load_stored_metadata(persist_dir: Path) -> Dict[str, float]:
    """Load stored file metadata from disk.

    Args:
        persist_dir: The persistence directory.

    Returns:
        Dict mapping relative file paths to their stored modification times.
    """
    metadata_file = persist_dir / METADATA_FILE_NAME
    if metadata_file.exists():
        try:
            return json.loads(metadata_file.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_metadata(persist_dir: Path, metadata: Dict[str, float]) -> None:
    """Save file metadata to disk.

    Args:
        persist_dir: The persistence directory.
        metadata: Dict mapping relative file paths to modification times.
    """
    metadata_file = persist_dir / METADATA_FILE_NAME
    metadata_file.write_text(json.dumps(metadata, indent=2))


def _detect_changes(
    current: Dict[str, float], stored: Dict[str, float]
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Detect new, modified, and deleted files by comparing metadata.

    Args:
        current: Current file metadata (path -> mtime).
        stored: Previously stored file metadata.

    Returns:
        Tuple of (new_files, modified_files, deleted_files) as sets of paths.
    """
    current_files = set(current.keys())
    stored_files = set(stored.keys())

    new_files = current_files - stored_files
    deleted_files = stored_files - current_files
    modified_files = {
        f for f in current_files & stored_files
        if current[f] != stored[f]
    }

    return new_files, modified_files, deleted_files


def _parse_files_to_nodes(
    docs_path: Path, file_paths: Set[str], parser: "MarkdownNodeParser"
) -> List[Any]:
    """Parse a set of markdown files into nodes.

    Args:
        docs_path: The documentation directory.
        file_paths: Set of relative file paths to parse.
        parser: The MarkdownNodeParser instance.

    Returns:
        List of parsed nodes.
    """
    documents: List[Document] = []
    for relative_path in file_paths:
        file_path = docs_path / relative_path
        try:
            text = file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("Skipping %s due to read error: %s", file_path, e)
            continue
        # Use file_path as the doc_id so we can delete by reference later
        doc = Document(
            text=text,
            metadata={"file_path": relative_path},
            doc_id=relative_path,
        )
        documents.append(doc)

    if not documents:
        return []

    return parser.get_nodes_from_documents(documents)


def _build_or_load_index(docs_path: Path) -> VectorStoreIndex:
    """Build a vector index from the given docs or load it from disk.

    This function uses a persistent storage directory to avoid recomputation
    on subsequent runs.  If the persistence directory exists and contains
    data, the index is loaded.  Otherwise, a new index is built and saved.

    Args:
        docs_path (Path): The directory containing Markdown files.

    Returns:
        VectorStoreIndex: A LlamaIndex vector store built from the docs.
    """
    persist_dir = docs_path / PERSIST_DIR_NAME
    provider_file = persist_dir / ".provider"
    storage_context: Optional[StorageContext] = None

    # Check if provider has changed since last index build
    if persist_dir.exists() and provider_file.exists():
        stored_provider = provider_file.read_text().strip()
        if stored_provider != MODEL_PROVIDER:
            logger.warning(
                "Provider changed from '%s' to '%s'. Deleting incompatible index...",
                stored_provider,
                MODEL_PROVIDER,
            )
            shutil.rmtree(persist_dir)

    # Configure embedding model and LLM globally via Settings based on provider
    if MODEL_PROVIDER == "ollama":
        try:
            from llama_index.embeddings.ollama import OllamaEmbedding
            from llama_index.llms.ollama import Ollama
        except ImportError:
            raise SystemExit(
                "Ollama dependencies not installed. Run: uv sync --extra ollama"
            )
        logger.info("Using Ollama provider (host=%s)", OLLAMA_HOST)
        embed_model = OllamaEmbedding(model_name=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_HOST)
        llm = Ollama(model=OLLAMA_CHAT_MODEL, base_url=OLLAMA_HOST)
    else:
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.llms.openai import OpenAI
        logger.info("Using OpenAI provider")
        embed_model = OpenAIEmbedding(model=OPENAI_EMBEDDING_MODEL)
        llm = OpenAI(model=OPENAI_CHAT_MODEL)

    Settings.embed_model = embed_model
    Settings.llm = llm

    parser = MarkdownNodeParser.from_defaults()
    current_metadata = _get_current_file_metadata(docs_path)

    if persist_dir.exists() and any(persist_dir.iterdir()):
        logger.info("Loading existing index from %s", persist_dir)
        storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
        index = load_index_from_storage(storage_context)

        # Check for file changes and perform incremental update
        stored_metadata = _load_stored_metadata(persist_dir)
        new_files, modified_files, deleted_files = _detect_changes(
            current_metadata, stored_metadata
        )

        if new_files or modified_files or deleted_files:
            logger.info(
                "Detected changes: %d new, %d modified, %d deleted files",
                len(new_files),
                len(modified_files),
                len(deleted_files),
            )

            # Delete nodes for modified and deleted files
            files_to_remove = modified_files | deleted_files
            for file_path in files_to_remove:
                try:
                    index.delete_ref_doc(file_path, delete_from_docstore=True)
                    logger.debug("Removed nodes for: %s", file_path)
                except Exception as e:
                    logger.warning("Failed to remove nodes for %s: %s", file_path, e)

            # Parse and insert nodes for new and modified files
            files_to_add = new_files | modified_files
            if files_to_add:
                new_nodes = _parse_files_to_nodes(docs_path, files_to_add, parser)
                if new_nodes:
                    index.insert_nodes(new_nodes)
                    logger.info("Added %d nodes from %d files", len(new_nodes), len(files_to_add))

            # Persist updated index and metadata
            index.storage_context.persist(persist_dir=str(persist_dir))
            _save_metadata(persist_dir, current_metadata)
            logger.info("Index updated and persisted")
        else:
            logger.info("No file changes detected")

        return index

    logger.info("Building new index; this may take a while...")

    # Collect documents from the docs directory.  Each Document carries
    # metadata about its file path relative to docs_path.  Only files with
    # extensions .md or .mdx are considered.
    documents: List[Document] = []
    for file_path in docs_path.rglob("*"):
        if file_path.suffix.lower() not in {".md", ".mdx"}:
            continue
        try:
            text = file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("Skipping %s due to read error: %s", file_path, e)
            continue
        relative_path = str(file_path.relative_to(docs_path))
        # Use file_path as doc_id for incremental updates
        documents.append(Document(
            text=text,
            metadata={"file_path": relative_path},
            doc_id=relative_path,
        ))

    if not documents:
        logger.error("No readable Markdown files found under %s", docs_path)
        raise SystemExit(1)

    # Parse documents into nodes using the MarkdownNodeParser.  This splits
    # documents by heading and attaches a 'header_path' to the node metadata.
    nodes = parser.get_nodes_from_documents(documents)
    logger.info("Parsed %d nodes from %d documents", len(nodes), len(documents))

    # Build the index in memory first, then persist to disk.
    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=str(persist_dir))
    # Record provider and file metadata for change detection on next run
    provider_file.write_text(MODEL_PROVIDER)
    _save_metadata(persist_dir, current_metadata)
    logger.info("Index built and persisted at %s", persist_dir)
    return index


class DocsSearcher:
    """Encapsulates the vector index and provides search and QA methods."""

    def __init__(self, index: VectorStoreIndex) -> None:
        self._index = index

    def _format_result(self, node: Any, score: float) -> Dict[str, Union[str, float]]:
        """Convert a NodeWithScore into a result dictionary.

        Args:
            node: A TextNode or similar object containing content and metadata.
            score: The similarity score assigned by the retriever.

        Returns:
            Dict[str, Union[str, float]]: The formatted result.
        """
        metadata = node.metadata or {}
        file_path: str = metadata.get("file_path", "")
        header_path: str = metadata.get("header_path", "/")
        # Derive the most specific heading from the header_path.  The
        # header_path is formatted like "/Parent/Child/"; split and take
        # the last non‑empty segment.  If none exists, fall back to the file
        # name.
        heading: str = ""
        parts = [p for p in header_path.split("/") if p]
        if parts:
            heading = parts[-1]
        elif file_path:
            heading = Path(file_path).stem
        # Generate a snippet from the content (truncate at limit, preserve words).
        content: str = node.get_content()
        if len(content) <= SNIPPET_CHAR_LIMIT:
            snippet = content.strip()
        else:
            # Truncate at word boundary
            truncated = content[:SNIPPET_CHAR_LIMIT]
            last_space = truncated.rfind(" ")
            snippet = (truncated[:last_space] if last_space > SNIPPET_CHAR_LIMIT // 2 else truncated).strip() + "..."
        return {
            "path": file_path,
            "heading": heading,
            "score": score,
            "snippet": snippet,
        }

    async def search(self, query: str, k: int = 6) -> List[Dict[str, Union[str, float]]]:
        """Perform a similarity search over the vector store.

        Args:
            query (str): The search query.
            k (int, optional): The maximum number of results to return. Defaults to 6.

        Returns:
            List[Dict[str, Union[str, float]]]: A list of search results.
        """
        logger.info("search_docs called with query='%s', k=%d", query, k)
        retriever = self._index.as_retriever(similarity_top_k=k)
        results = retriever.retrieve(query)
        formatted = [self._format_result(r.node, r.score) for r in results]
        return formatted

    async def ask(self, question: str) -> Dict[str, Any]:
        """Answer a natural‑language question using retrieval‑augmented generation.

        Args:
            question (str): The question to answer.

        Returns:
            Dict[str, Any]: An answer and citations.
        """
        logger.info("ask_docs called with question='%s'", question)

        # Build a custom QA prompt with system instructions
        qa_prompt = PromptTemplate(
            RAG_SYSTEM_PROMPT + """
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the question.
Question: {query_str}
Answer: """
        )

        query_engine = self._index.as_query_engine(
            similarity_top_k=RAG_TOP_K,
            text_qa_template=qa_prompt,
        )
        response = query_engine.query(question)
        answer: str = str(response)
        citations: List[Dict[str, str]] = []
        for source_node in response.source_nodes:
            node = source_node.node
            metadata = node.metadata or {}
            file_path: str = metadata.get("file_path", "")
            header_path: str = metadata.get("header_path", "/")
            parts = [p for p in header_path.split("/") if p]
            heading: str = parts[-1] if parts else (Path(file_path).stem if file_path else "")
            content = node.get_content()
            if len(content) <= SNIPPET_CHAR_LIMIT:
                snippet = content.strip()
            else:
                truncated = content[:SNIPPET_CHAR_LIMIT]
                last_space = truncated.rfind(" ")
                snippet = (truncated[:last_space] if last_space > SNIPPET_CHAR_LIMIT // 2 else truncated).strip() + "..."
            citations.append({"path": file_path, "heading": heading, "snippet": snippet})
        return {"answer": answer, "citations": citations}


def build_app() -> FastMCP:
    """Construct and return the FastMCP application.

    Returns:
        FastMCP: The configured MCP application exposing our tools.
    """
    docs_path = _get_docs_dir()
    index = _build_or_load_index(docs_path)
    searcher = DocsSearcher(index)

    app = FastMCP()

    @app.tool(
        name="search_docs",
        description="Search the local documentation with a text query. Returns a list of matching sections with path, heading, score, and snippet.",
    )
    async def search_docs(query: str, k: int = 6) -> List[Dict[str, Union[str, float]]]:
        """Search the indexed documentation.

        Args:
            query (str): The search query.
            k (int, optional): The number of results to return. Defaults to 6.

        Returns:
            List[dict]: A list of results with ``path``, ``heading``, ``score`` and ``snippet``.
        """
        return await searcher.search(query, k)

    @app.tool(
        name="ask_docs",
        description="Ask a question over the local documentation. Returns an answer with citations.",
    )
    async def ask_docs(question: str) -> Dict[str, Any]:
        """Answer a question using the indexed documentation.

        Args:
            question (str): The question to answer.

        Returns:
            dict: A dictionary with ``answer`` and ``citations``.
        """
        return await searcher.ask(question)

    return app


def main() -> None:
    """Entry point for running the MCP server.

    This function simply constructs the application and hands control to the
    FastMCP runner.  When run with the MCP CLI (e.g., ``uv run mcp dev server.py``),
    MCP will invoke this function automatically.  When run directly via
    ``python server.py``, this function starts the server as well.
    """
    app = build_app()
    # The FastMCP runner automatically reads the ASGI app and exposes the
    # tools defined above.  When this module is run as a script, call run()
    # directly.  When executed under ``mcp dev``, the CLI will call run()
    # implicitly.
    app.run()


if __name__ == "__main__":
    main()