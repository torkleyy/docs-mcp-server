# Example Documentation

Welcome to the example documentation for the MCP server.  This file is
used solely for demonstration and testing purposes.  Feel free to
replace it with your own Markdown files.

## Installation

To install this project using uv:

1. Create a virtual environment:

   ```sh
   uv venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```sh
   uv pip install -e .
   ```

3. Set your environment variables and run the server:

   ```sh
   export OPENAI_API_KEY="sk-..."
   uv run mcp dev server.py
   ```

## Usage

Use the `search_docs` tool to find relevant sections by keyword and
`ask_docs` to ask natural‑language questions over the documentation.