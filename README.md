# slocate

Semantic locate. Language-agnostic semantic code search for your terminal.

```
$ slocate query "coordinate transforms"
[0.85] function_item `warp` — pixelflow-core/src/combinators/at.rs
[0.81] function_item `contramap` — pixelflow-core/src/domain.rs
[0.78] impl_item `At` — pixelflow-core/src/combinators/at.rs
```

slocate indexes your codebase using tree-sitter and BGE-base-en-v1.5 embeddings
(109M parameter BERT, 768-dim vectors), then provides sub-second semantic
search via an HNSW nearest-neighbor index stored in SQLite.

Designed as a RAG context provider for LLM coding assistants. A single hook
injects relevant code into every prompt automatically.

## Install

```bash
# From source
cargo install --path .
slocate install    # downloads model (~418MB, one-time), sets up daemon

# Or build a release binary
cargo build --profile dist
cp target/dist/slocate ~/.local/bin/
slocate install
```

## Quick start

```bash
slocate add-repo ~/src/myproject
slocate reindex
slocate query "error handling in the parser"
```

## How it works

1. **Parse** — tree-sitter splits source files into semantic chunks (functions,
   structs, impls, classes, etc.) across Rust, Python, Go, YAML, Starlark, and
   Markdown.

2. **Embed** — each chunk is embedded with BGE-base-en-v1.5 via
   [candle](https://github.com/huggingface/candle) (in-process, no Python).
   Batched inference, multi-threaded on CPU, optional Metal GPU.

3. **Index** — embeddings are stored in an HNSW graph inside a per-workspace
   SQLite database. Leiden community detection groups related chunks.

4. **Search** — queries are embedded and matched against the HNSW index.
   Results above a configurable similarity threshold are returned with source
   context.

5. **Incremental** — file mtime+size tracking means only changed files are
   re-embedded. No-op reindex: ~170ms. Full query: ~210ms.

## LLM integration

### Claude Code (hook)

Add to `.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [{
      "type": "command",
      "command": "slocate query",
      "timeout": 5000
    }]
  }
}
```

Every prompt automatically gets relevant code context injected.

### MCP server

Add to `.mcp.json`:

```json
{
  "mcpServers": {
    "slocate": {
      "command": "slocate",
      "args": ["serve"]
    }
  }
}
```

Exposes `search_code`, `index_workspace`, `note_to_self`, and `check_notes`
tools over JSON-RPC 2.0 / stdio.

### Git hooks

Auto-reindex on commit/merge:

```bash
# .githooks/post-commit and .githooks/post-merge
if command -v slocate >/dev/null 2>&1; then
    slocate reindex >/dev/null 2>&1 &
fi
```

## Configuration

`$XDG_CONFIG_HOME/slocate/config.toml` (default `~/.config/slocate/config.toml`):

```toml
[index]
workspaces = ["~/src/myproject", "~/src/other"]
reindex_interval_minutes = 10
extensions = ["rs", "py", "ts", "go", "c", "h", "md", "yaml", "yml", "bzl"]
embed_workers = 4
max_file_bytes = 1048576

[model]
dir = "~/.local/share/slocate/models/bge-base-en-v1.5"

[search]
top_k = 5
min_score = 0.72
```

## File layout

| Path | Purpose |
|------|---------|
| `$XDG_CONFIG_HOME/slocate/config.toml` | Configuration |
| `$XDG_DATA_HOME/slocate/models/` | Embedding model weights |
| `$XDG_DATA_HOME/slocate/registry/` | Workspace registry symlinks |
| `$XDG_STATE_HOME/slocate/daemon.log` | Daemon log |
| `<workspace>/.slocate/index.db` | Per-workspace index (co-located) |

## Performance

| Operation | Time |
|-----------|------|
| No-op reindex (441 files) | 170ms |
| Query (embed + HNSW search) | 210ms |
| Incremental reindex (2 files, 24 chunks) | 8s |
| Model load (mmap, lazy paging) | 72ms |

Model weights are memory-mapped. The OS pages data in on demand and reclaims
it when idle. No persistent memory cost.

## Environment variables

| Variable | Effect |
|----------|--------|
| `XDG_CONFIG_HOME` | Config directory (default `~/.config`) |
| `XDG_DATA_HOME` | Data directory (default `~/.local/share`) |
| `XDG_STATE_HOME` | State directory (default `~/.local/state`) |
| `SLOCATE_DEVICE` | `metal` for Metal GPU inference (default: CPU) |

## Requirements

- Rust 1.75+ (edition 2021)
- macOS: Accelerate framework (ships with Xcode)
- Linux: MKL or OpenBLAS
- ~418 MB disk for model weights (downloaded on first `slocate install`)

## License

Apache 2.0
