# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MCP server that indexes local Markdown documentation using LlamaIndex and OpenAI. Single-file Python application (`server.py`) exposing semantic search and RAG Q&A tools via FastMCP.

**See README.md for setup instructions, environment variables, and usage.**

## Development

```bash
# Install dependencies
uv sync

# Run server (set OPENAI_API_KEY first)
uv run python server.py
```

## Architecture

### Core Flow

1. **Startup**: Validates `DOCS_DIR`, builds or loads vector index from persistent storage
2. **Indexing**: Recursively reads `.md`/`.mdx` files, parses with MarkdownNodeParser (splits by heading), builds VectorStoreIndex
3. **Persistence**: Index cached in `{DOCS_DIR}/.vector_store/` - delete to force rebuild
4. **Tools**: `search_docs` (similarity search) and `ask_docs` (RAG with citations)

### Key Components

- **`_build_or_load_index()`** - Index lifecycle management (build or load from disk)
- **`DocsSearcher`** - Wraps index with search/ask methods, handles result formatting
- **`build_app()`** - FastMCP app factory that registers tools

### Metadata

Documents carry `file_path` (relative to DOCS_DIR). MarkdownNodeParser adds `header_path` to nodes (e.g., "/Section/Subsection/"). Both are used when formatting search results and citations.

## Notes

- Server exits if DOCS_DIR is invalid or contains no markdown files
- Global LlamaIndex config via `Settings.embed_model` and `Settings.llm`
- Changes to markdown files require deleting `.vector_store/` to take effect
- Keep README.md updated with environment variables and usage changes
