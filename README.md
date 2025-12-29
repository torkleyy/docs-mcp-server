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

### Chat CLI (Easiest)

Chat directly with your docs from the command line:

```bash
# With Ollama (local, no API key needed)
uv run docs-chat /path/to/your/docs --ollama

# With OpenAI
uv run docs-chat /path/to/your/docs

# Single question mode
uv run docs-chat /path/to/your/docs --ollama -q "How do I install?"

# Search mode (no AI, just find relevant sections)
uv run docs-chat /path/to/your/docs --ollama -s "configuration"
```

Interactive mode commands:
- Type questions to get AI answers with citations
- `/search <query>` - search without AI generation
- `quit` or `q` - exit

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
| `DOCS_DIR` | No | Path to Markdown folder (default: `./docs`) |
| `MODEL_PROVIDER` | No | `openai` (default) or `ollama` |
| `OPENAI_API_KEY` | Yes* | Your OpenAI API key (*not needed for Ollama) |
| `OPENAI_API_BASE` | No | Override OpenAI API URL |
| `OPENAI_CHAT_MODEL` | No | Chat model (default: `gpt-4o-mini`) |
| `OPENAI_EMBEDDING_MODEL` | No | Embedding model (default: `text-embedding-3-small`) |
| `OLLAMA_HOST` | No | Ollama server URL (default: `http://localhost:11434`) |
| `OLLAMA_CHAT_MODEL` | No | Ollama chat model (default: `llama3.2`) |
| `OLLAMA_EMBEDDING_MODEL` | No | Ollama embedding model (default: `nomic-embed-text`) |

## Using Local Models (Ollama)

For testing without OpenAI, you can use [Ollama](https://ollama.com) to run models locally.

### Setup

```bash
# Install Ollama (macOS)
brew install ollama

# Start Ollama server
ollama serve

# Pull required models (~2GB total)
ollama pull nomic-embed-text   # Embedding model (~274MB)
ollama pull llama3.2           # Chat model (~2GB, or use llama3.2:1b for ~1.3GB)
```

### Install Ollama dependencies

```bash
uv sync --extra ollama
```

### Run with Ollama

```bash
export MODEL_PROVIDER=ollama
export DOCS_DIR=/path/to/your/docs
uv run python server.py
```

## Notes

- Only `.md` and `.mdx` files are indexed
- Vector store is cached in `.vector_store/` under your docs folder—delete it to force full rebuild
- First run takes longer due to embedding generation
- **Incremental updates**: Changed/new/deleted markdown files are automatically detected on startup and only affected embeddings are rebuilt
- Switching providers (OpenAI ↔ Ollama) automatically rebuilds the index since embeddings are incompatible
