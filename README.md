# Local Documentation MCP Server

An MCP server that indexes local Markdown files and provides AI-powered search and Q&A tools using LlamaIndex and OpenAI.

## Tools

- **`search_docs(query, k=6)`** – Similarity search over your docs. Returns matching sections with path, heading, score, and snippet.
- **`ask_docs(question)`** – RAG-powered Q&A. Returns an answer with citations.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and an OpenAI API key.

```bash
# Clone and enter the directory
git clone https://github.com/torkleyy/docs-mcp-server
cd docs-mcp-server

# Install dependencies
uv sync
```

## Usage

### With Claude Code

Add to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "docs-mcp-server": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--directory", "/path/to/docs-mcp-server", "python", "server.py"],
      "env": {
        "DOCS_DIR": "/path/to/your/markdown/docs",
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

Or use the CLI:

```bash
claude mcp add --transport stdio docs-mcp-server \
  -e DOCS_DIR=/path/to/your/docs \
  -e OPENAI_API_KEY=sk-... \
  --scope project \
  -- uv run --directory /path/to/docs-mcp-server python server.py
```

### Standalone

```bash
export OPENAI_API_KEY="sk-..."
export DOCS_DIR="/path/to/your/markdown/docs"
uv run python server.py
```

## Environment Variables

| Variable | Required | Description |
|----------|:--------:|-------------|
| `OPENAI_API_KEY` | Yes | Your OpenAI API key |
| `DOCS_DIR` | No | Path to Markdown folder (default: `./docs`) |
| `OPENAI_API_BASE` | No | Override OpenAI API URL |
| `OPENAI_CHAT_MODEL` | No | Chat model (default: `gpt-4o-mini`) |
| `OPENAI_EMBEDDING_MODEL` | No | Embedding model (default: `text-embedding-3-small`) |

## Notes

- Only `.md` and `.mdx` files are indexed
- Vector store is cached in `.vector_store/` under your docs folder—delete it to rebuild
- First run takes longer due to embedding generation
