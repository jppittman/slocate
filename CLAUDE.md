# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
cargo build                    # Debug build
cargo build --release          # Release build
cargo build --profile dist     # Distribution build (LTO, strip, panic=abort)
cargo test                     # Run all tests (currently only spsc module has tests)
cargo test spsc                # Run spsc channel tests only
cargo clippy                   # Lint
cargo fmt --check              # Check formatting
```

Run the binary directly:
```bash
cargo run -- serve             # MCP server (JSON-RPC 2.0 over stdio)
cargo run -- reindex           # Re-embed changed files across all workspaces
cargo run -- query "search terms"
cargo run -- hook --backend claude "query"
cargo run -- install           # One-time setup: download model, configure daemon
cargo run -- add-repo ~/path   # Register workspace
cargo run -- gc                # Clean orphaned registry links + stale embed cache
```

## Architecture

**slocate** is a semantic code search tool that works as a RAG context provider for LLM coding assistants. It parses source files into semantic chunks, embeds them with BGE-small-en-v1.5 (384-dim BERT via candle, in-process, no Python), stores vectors in an HNSW index backed by SQLite, and returns ranked results at query time.

### Data Pipeline

```
Files → tree-sitter parse → chunks → BGE embedding → HNSW index (SQLite)
                                         ↑ cached via content hash (f16)
Query → "code: " prefix → embed → HNSW search → score filter → formatted output
```

### Key Modules

| Module | Role |
|--------|------|
| `main.rs` | CLI (clap derive), subcommand dispatch |
| `error.rs` | Unified `Error` enum, `Result<T>` alias — all errors must be explicit |
| `config.rs` | TOML config at `$XDG_CONFIG_HOME/slocate/config.toml`, tilde expansion |
| `parse.rs` | Tree-sitter chunking for Rust/Python/Go/YAML/Starlark/Markdown |
| `embed.rs` | BGE-small-en-v1.5 via candle. mmap'd weights, L2-normalized output |
| `store.rs` | SQLite persistence: chunks, HNSW graph, embed cache (f32→f16), file_meta |
| `vdb.rs` | HNSW implementation (M=16, ef_construction=200). Dot product similarity |
| `search.rs` | Query execution: embed → HNSW search → min_score filter → merge across workspaces |
| `reindex.rs` | 6-phase incremental pipeline: classify → delete → embed (parallel SPSC) → commit → HNSW update → Leiden |
| `download.rs` | HuggingFace model downloader with BERT config validation |
| `install.rs` | Full setup: model download, config write, binary copy, daemon install, first reindex |
| `registry.rs` | Workspace symlink registry at `$XDG_DATA_HOME/slocate/`. Index co-located at `<workspace>/.slocate/index.db` |
| `backends/` | `HookBackend` trait with Claude and Gemini formatters |
| `platform/` | Daemon setup: launchd plist (macOS) / systemd user timer (Linux) |
| `mcp_tools.rs` | MCP tool handlers: `search_code`, `index_workspace`, `note_to_self`, `check_notes` |
| `spsc.rs` | Wait-free single-producer/single-consumer channel (the only module with unit tests) |

### Platform-Conditional Compilation

candle backend differs per platform (Cargo.toml):
- **macOS:** `candle-core` with `metal` + `accelerate` features, `candle-nn` with `metal`
- **Linux:** `candle-core` with `mkl` feature

Device selection at runtime: `SLOCATE_DEVICE=metal` for GPU, CPU default.

### Concurrency Model

Reindexing uses N worker threads (1 for GPU, up to 4 for CPU). Each worker has a dedicated SPSC channel (`spsc.rs`) — the main thread distributes file batches and polls receivers for results. No shared locks beyond SQLite WAL mode.

### Vectors

All vectors are L2-normalized at embed time so cosine similarity = dot product. Stored in SQLite as f16 BLOBs (embed cache) or f32 BLOBs (HNSW nodes). Dimension mismatch between query and index triggers a loud error requiring reindex.

## Design Principles

- **No silent failures.** Every error path must produce a visible diagnostic. Use `error::Error` variants; never swallow errors.
- **Incremental by default.** File mtime+size tracking in `file_meta`, content-hash embed cache, HNSW fast-path (insert-only when no deletions).
- **XDG-compliant paths.** Config, data, and state directories follow XDG Base Directory spec with standard fallbacks.

## CI/CD

Release builds trigger on `v*` tags. Matrix: `{x86_64, aarch64} × {apple-darwin, linux-gnu}`. Uses `--profile dist` (LTO + strip + panic=abort). Outputs `.tar.gz` + `.sha256` to GitHub Releases.
