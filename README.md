# Local Documentation MCP Server

This repository contains a small Python application that exposes a local
collection of Markdown files as a Model Context Protocol (MCP) server using
[LlamaIndex](https://github.com/run-llama/llama_index) and
[FastMCP](https://github.com/jlowin/fastmcp).  The server indexes your
documentation directory, stores it in a persistent vector store, and
provides two AI‑callable tools:

* **`search_docs(query: str, k: int = 6)`** – perform a similarity search over
  your Markdown documentation.  Returns a list of the top‑`k` matching
  sections along with their relative file path, heading, similarity score
  and a short snippet.
* **`ask_docs(question: str)`** – answer a natural‑language question by
  retrieving relevant chunks and generating a response.  Returns both
  the answer and a list of citation objects with file paths, headings and
  snippets.

The project is intended as a template for hosting your own docs as an MCP
server.  It is fully deterministic, fails fast when misconfigured, and uses
a disk‑backed index to avoid re‑processing your files on every run.

## Quick start

These instructions assume you have the
[Astral **uv**](https://docs.astral.sh/uv/) package manager installed on
your system.  If you do not, follow the installation instructions in the
uv documentation.

1. **Clone this repository** (or copy the files into your own project).

2. **Create a virtual environment** using uv and activate it:

   ```sh
   uv venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**.  The `pyproject.toml` defines all required
   packages.  Install them into your venv with uv:

   ```sh
   uv pip install -e .
   ```

4. **Set environment variables**.  Copy `.env.example` to `.env` (or set
   variables directly in your shell) and fill in your OpenAI API key and
   optionally the docs directory.  For example:

   ```sh
   export OPENAI_API_KEY="sk-..."            # required – your OpenAI key
   export OPENAI_API_BASE="https://api.openai.com/v1"  # optional – override the base URL
   export DOCS_DIR="/absolute/path/to/your/docs"       # optional – defaults to ./docs
   ```

   The server will read `DOCS_DIR` and fail fast if the directory does
   not exist or contains no `.md`/`.mdx` files.

5. **Run the server**.  The easiest way to launch the server in development
   mode is via the MCP CLI.  Use uv’s `run` command to invoke it:

   ```sh
   uv run mcp dev server.py
   ```

   This command starts an ASGI application on `http://127.0.0.1:8000/mcp`.
   You should see log messages indicating whether the index was loaded or
   built.  On subsequent runs the existing persistent index will be
   reused.

## Testing the tools

You can exercise the MCP server using the official MCP client.  The
client can connect over HTTP or spawn a Python process locally.  The
example below uses the in‑process transport to call both tools from your
own code:

```python
import asyncio
from mcp.client import Client

async def main() -> None:
    # Launch the server in a subprocess via the MCP protocol.  The
    # server must be in your current working directory.
    client = Client("python", args=["server.py"])  # or http://127.0.0.1:8000/mcp

    # Search for a term in your docs
    search_results = await client.call_tool(
        "search_docs", {"query": "installation instructions", "k": 3}
    )
    print("Search results:", search_results)

    # Ask a question and receive an answer with citations
    answer = await client.call_tool(
        "ask_docs", {"question": "How do I install the project?"}
    )
    print("Answer:", answer)

asyncio.run(main())
```

The client will return JSON‑serializable data structures for both tools.

## Environment variables

The server reads the following environment variables:

| Variable            | Required | Description                                                         |
|---------------------|---------:|---------------------------------------------------------------------|
| `OPENAI_API_KEY`    |    Yes   | Your OpenAI API key.  Required to use the OpenAI chat and embedding models. |
| `OPENAI_API_BASE`   |     No   | Override the OpenAI API base URL (default: `https://api.openai.com/v1`). |
| `DOCS_DIR`          |     No   | Path to the Markdown documentation folder (default: `./docs`).      |
| `OPENAI_CHAT_MODEL` |     No   | Name of the OpenAI chat model (default: `gpt-3.5-turbo`).           |
| `OPENAI_EMBEDDING_MODEL` |  No | Name of the OpenAI embedding model (default: `text-embedding-3-small`). |

You can place these variables in an `.env` file or export them in your shell
before starting the server.  The `.env.example` file provides a template.

## Project structure

```
.
├── server.py          # MCP server implementation
├── README.md          # This file
├── pyproject.toml     # Dependency specification for uv
├── .env.example       # Example environment variables
└── docs/              # Place your Markdown files here (optional)
    └── ...
```

## Notes

* Only files ending in `.md` or `.mdx` are indexed.  Other files are ignored.
* The vector store is persisted under a `.vector_store` subdirectory within
  your docs folder.  Deleting this directory will force the index to be
  rebuilt on the next run.
* Both tools are asynchronous; FastMCP will handle running them under an
  event loop.  If you need synchronous versions, you can wrap the calls in
  `asyncio.run()` from your own code.

Happy hacking!