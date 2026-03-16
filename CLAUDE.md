# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
cargo build                    # Debug build
cargo build --release          # Release build
cargo build --profile dist     # Distribution build (LTO, strip, panic=abort)
cargo test                     # Run all tests
cargo clippy                   # Lint
cargo fmt --check              # Check formatting
cargo bench                    # Run benchmarks (criterion, embed throughput)
```

Run the binary directly:
```bash
cargo run -- serve             # MCP server (JSON-RPC 2.0 over stdio)
cargo run -- reindex           # Re-embed changed files across all workspaces
cargo run -- reindex --force   # Force full re-embedding (ignore mtime cache)
cargo run -- query "search terms"
cargo run -- query --json      # Accept JSON query on stdin
cargo run -- query --json-out  # Output results as JSON
cargo run -- hook --backend claude "query"
cargo run -- hook --backend gemini "query"
cargo run -- install           # One-time setup: download model, configure daemon
cargo run -- add-repo ~/path   # Register workspace
cargo run -- remove-repo ~/path  # Unregister workspace (deletes index)
cargo run -- repos             # List configured workspaces
cargo run -- gc                # Clean orphaned registry links + stale embed cache
cargo run -- claude-hook       # Claude Code UserPromptSubmit hook handler
cargo run -- gemini-hook       # Gemini CLI BeforeAgent hook handler
cargo run -- agent-hook        # Generic agent hook (default: Gemini backend)
```

## Architecture

**slocate** is a semantic code search tool that works as a RAG context provider for LLM coding assistants. It parses source files into semantic chunks, embeds them with BGE-small-en-v1.5 (384-dim BERT via candle, in-process, no Python), stores vectors in SQLite, and returns ranked results via hybrid BM25 + flat-scan semantic search at query time.

### Data Pipeline

```
Files → tree-sitter parse → chunks → BGE embedding → SQLite (chunks + FTS5 + embed cache)
                                         ↑ cached via content hash (f16)
Query → "code: " prefix → embed → flat dot-product scan + BM25 hybrid → MMR rerank → formatted output
```

### Key Modules

| Module | Role |
|--------|------|
| `main.rs` | CLI (clap derive), subcommand dispatch, MCP server loop |
| `lib.rs` | Public library surface for benchmarks/integration tests |
| `error.rs` | Unified `Error` enum, `Result<T>` alias — all errors must be explicit |
| `config.rs` | TOML config at `$XDG_CONFIG_HOME/slocate/config.toml`, tilde expansion |
| `parse.rs` | Tree-sitter chunking for Rust/Python/Go/C/YAML/Starlark/Markdown |
| `embed.rs` | BGE-small-en-v1.5 via candle. mmap'd weights, L2-normalized output |
| `store.rs` | SQLite persistence: chunks, embed cache (f32->f16), file_meta, FTS5 |
| `search.rs` | Query execution: embed -> flat dot-product scan + FTS5 BM25 hybrid -> MMR rerank -> merge across workspaces |
| `reindex.rs` | 4-phase incremental pipeline: classify -> delete -> embed (parallel SPSC) -> commit |
| `fastmath.rs` | RAII guard for FTZ/DAZ denormal flushing (x86_64 MXCSR / aarch64 FPCR) |
| `mcp.rs` | JSON-RPC 2.0 response/error builders for MCP protocol |
| `mcp_tools.rs` | MCP tool handlers: `search_code`, `index_workspace`, `note_to_self`, `check_notes` |
| `download.rs` | HuggingFace model downloader with BERT config validation |
| `install.rs` | Full setup: model download, config write, binary copy, daemon install, first reindex |
| `registry.rs` | Workspace symlink registry at `$XDG_DATA_HOME/slocate/`. Index co-located at `<workspace>/.slocate/index.db` |
| `backends/` | `HookBackend` trait with Claude (`claude.rs`) and Gemini (`gemini.rs`) formatters |
| `platform/` | Daemon setup: launchd plist (macOS) / systemd user timer (Linux) |
| `spsc.rs` | Wait-free single-producer/single-consumer channel |

### Platform-Conditional Compilation

candle backend differs per platform (Cargo.toml):
- **macOS:** `candle-core` with `metal` + `accelerate` features, `candle-nn` with `metal`
- **Linux:** `candle-core` with `cuda` feature, `candle-nn` with `cuda`

Device selection at runtime: `SLOCATE_DEVICE=metal` for GPU, CPU default.

Compiler flags (`.cargo/config.toml`): `target-cpu=native` + `--enable-unsafe-fp-math` for SIMD dot product throughput. The `fastmath.rs` module provides an RAII guard that sets FTZ/DAZ flags to prevent 100-1000x slowdowns from subnormal floats during embedding.

### Concurrency Model

Reindexing uses N worker threads (1 for GPU, up to 4 for CPU). Each worker has a dedicated SPSC channel (`spsc.rs`) -- the main thread distributes file batches and polls receivers for results. No shared locks beyond SQLite WAL mode.

### Vectors

All vectors are L2-normalized at embed time so cosine similarity = dot product. Stored in SQLite as f16 BLOBs (embed cache). Dimension mismatch between query and index triggers a loud error requiring reindex.

### Search

Hybrid retrieval combining flat-scan dot-product similarity (exact, no ANN approximation) with SQLite FTS5 BM25 keyword scoring. Results are merged with configurable `bm25_weight` and reranked via MMR (Maximal Marginal Relevance) with tunable `mmr_lambda`. At codebase scale (~500k chunks max), flat scan takes single-digit milliseconds, so approximate nearest neighbor structures add complexity without meaningful speedup.

## Tests

Tests are embedded in source modules (no separate `tests/` directory):

| Module | What's tested |
|--------|---------------|
| `spsc.rs` | 19 tests: channel semantics, blocking, capacity, drop safety |
| `reindex.rs` | 12 tests: content hashing, SPSC blocking send, CancelGuard, classify_files |
| `store.rs` | SQLite round-trip persistence, BM25 OR logic |

Run targeted tests: `cargo test spsc`, `cargo test reindex`, `cargo test store`.

## Design Principles

- **No silent failures.** Every error path must produce a visible diagnostic. Use `error::Error` variants; never swallow errors.
- **Incremental by default.** File mtime+size tracking in `file_meta`, content-hash embed cache.
- **Exact search.** Flat dot-product scan over all chunk vectors — no lossy ANN approximation.
- **XDG-compliant paths.** Config, data, and state directories follow XDG Base Directory spec with standard fallbacks.

## CI/CD

Release builds trigger on `v*` tags. Matrix: `{x86_64, aarch64} x {apple-darwin, linux-gnu}`. Uses `--profile dist` (LTO + strip + panic=abort). Outputs `.tar.gz` + `.sha256` to GitHub Releases.
