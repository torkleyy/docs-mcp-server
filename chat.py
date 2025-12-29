#!/usr/bin/env python3
"""Interactive CLI to chat with a Markdown documentation folder."""

import argparse
import asyncio
import os
import sys
from pathlib import Path


def main():
    # Suppress noisy logs by default
    import logging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("llama_index").setLevel(logging.WARNING)
    logging.getLogger("__main__").setLevel(logging.WARNING)
    logging.getLogger("server").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        description="Chat with your Markdown documentation using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/docs                    # Chat using OpenAI
  %(prog)s /path/to/docs --ollama           # Chat using local Ollama
  %(prog)s /path/to/docs -q "How do I..."   # Ask a single question
""",
    )
    parser.add_argument("docs_dir", type=Path, help="Path to Markdown documentation folder")
    parser.add_argument("--ollama", action="store_true", help="Use Ollama instead of OpenAI")
    parser.add_argument("--ollama-host", default="http://localhost:11434", help="Ollama server URL")
    parser.add_argument("-q", "--question", help="Ask a single question and exit")
    parser.add_argument("-s", "--search", help="Search docs and exit (no AI answer)")
    parser.add_argument("-k", "--results", type=int, default=6, help="Number of search results")
    parser.add_argument("--reindex", action="store_true", help="Force rebuild of vector index")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show debug logs")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("server").setLevel(logging.INFO)

    # Validate docs directory
    docs_path = args.docs_dir.resolve()
    if not docs_path.exists() or not docs_path.is_dir():
        print(f"Error: '{docs_path}' is not a valid directory", file=sys.stderr)
        sys.exit(1)

    md_files = list(docs_path.rglob("*.md")) + list(docs_path.rglob("*.mdx"))
    if not md_files:
        print(f"Error: No Markdown files found in '{docs_path}'", file=sys.stderr)
        sys.exit(1)

    # Set environment for server module
    os.environ["DOCS_DIR"] = str(docs_path)
    if args.ollama:
        os.environ["MODEL_PROVIDER"] = "ollama"
        os.environ["OLLAMA_HOST"] = args.ollama_host
    elif "OPENAI_API_KEY" not in os.environ:
        print("Error: OPENAI_API_KEY not set. Use --ollama for local models.", file=sys.stderr)
        sys.exit(1)

    # Delete index if reindex requested
    if args.reindex:
        import shutil
        vector_store = docs_path / ".vector_store"
        if vector_store.exists():
            shutil.rmtree(vector_store)
            print("Cleared existing index.")

    # Import after setting env vars
    from server import DocsSearcher, _build_or_load_index

    print(f"Loading docs from: {docs_path}")
    print(f"Found {len(md_files)} Markdown file(s)")
    provider = "Ollama" if args.ollama else "OpenAI"
    print(f"Using: {provider}\n")

    index = _build_or_load_index(docs_path)
    searcher = DocsSearcher(index)

    # Single search mode
    if args.search:
        results = asyncio.run(searcher.search(args.search, args.results))
        print_search_results(results)
        return

    # Single question mode
    if args.question:
        response = asyncio.run(searcher.ask(args.question))
        print_answer(response)
        return

    # Interactive mode
    print("Ready! Type your questions (or 'quit' to exit, '/search <query>' to search)\n")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Search command
        if user_input.startswith("/search "):
            query = user_input[8:].strip()
            if query:
                results = asyncio.run(searcher.search(query, args.results))
                print_search_results(results)
            continue

        # Ask question
        print("\nThinking...", end="", flush=True)
        response = asyncio.run(searcher.ask(user_input))
        print("\r" + " " * 12 + "\r", end="")  # Clear "Thinking..."
        print_answer(response)


def print_search_results(results):
    """Display search results."""
    if not results:
        print("No results found.")
        return
    print(f"\nFound {len(results)} result(s):\n")
    for i, r in enumerate(results, 1):
        score = r.get("score", 0)
        print(f"{i}. [{r['path']}] {r['heading']} (score: {score:.3f})")
        snippet = r["snippet"].replace("\n", " ")[:200]
        print(f"   {snippet}...")
        print()


def print_answer(response):
    """Display an answer with citations."""
    print(f"\n{response['answer']}\n")
    if response.get("citations"):
        print("-" * 40)
        print("Sources:")
        # Group headings by file path
        by_file: dict[str, list[str]] = {}
        for c in response["citations"]:
            path = c["path"]
            heading = c["heading"]
            if path not in by_file:
                by_file[path] = []
            if heading and heading not in by_file[path]:
                by_file[path].append(heading)
        for path, headings in by_file.items():
            if headings:
                print(f"  • {path} ({', '.join(headings)})")
            else:
                print(f"  • {path}")


if __name__ == "__main__":
    main()
