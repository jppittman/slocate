// slocate: language-agnostic semantic code search.
//
// Subcommands:
//   serve    — MCP server (JSON-RPC 2.0 over stdio) for Claude/Gemini/etc.
//   reindex  — walk workspaces, embed chunks, write index
//   query    — read {"prompt":"..."} from stdin, print top-k chunks to stdout
//   install  — lazy model download, daemon setup (launchd/systemd)
//   add-repo — register a workspace directory for indexing

mod backends;
mod config;
mod download;
mod embed;
mod error;
mod fastmath;
mod install;
mod leiden;
mod mcp;
mod mcp_tools;
mod parse;
mod platform;
mod registry;
mod reindex;
mod search;
mod spsc;
mod store;
mod vdb;

use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(name = "slocate", about = "Semantic code search")]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Start the MCP server (JSON-RPC 2.0 over stdio)
    Serve,
    /// Walk workspaces and rebuild the search index
    Reindex,
    /// Search the index. Reads query from stdin (plain text) or args.
    Query {
        /// Read {"prompt":"..."} JSON from stdin instead of plain text
        #[arg(long)]
        json: bool,
        /// Query string (if omitted, reads from stdin)
        query: Option<String>,
    },
    /// Search and format chunks using a specific LLM hook backend
    Hook {
        /// Backend to use: claude, gemini
        #[arg(long, default_value_t = BackendKind::Claude)]
        backend: BackendKind,
        /// The query/prompt to search for
        query: String,
    },
    /// Download model and set up daemon for periodic reindexing
    Install,
    /// Add a directory as an indexed workspace
    AddRepo {
        /// Path to the workspace directory
        path: String,
    },
    /// Remove a directory from indexed workspaces
    RemoveRepo {
        /// Path to remove
        path: String,
    },
    /// List configured workspaces
    Repos,
    /// Remove broken registry symlinks (workspaces that were deleted)
    Gc,
    /// Claude Code UserPromptSubmit hook: reads hook JSON from stdin, injects slocate context
    ClaudeHook,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum BackendKind {
    Claude,
    Gemini,
}

impl std::fmt::Display for BackendKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Claude => write!(f, "claude"),
            Self::Gemini => write!(f, "gemini"),
        }
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let cli = Cli::parse();
    let result = match cli.command {
        Cmd::Serve => cmd_serve(),
        Cmd::Reindex => {
            let config = config::Config::load().map_err(|e| {
                log::error!("Config error: {e}");
                e
            }).unwrap(); // Fail fast
            cmd_reindex(&config)
        }
        Cmd::Query { json, query } => {
            let config = config::Config::load().map_err(|e| {
                log::error!("Config error: {e}");
                e
            }).unwrap(); // Fail fast
            cmd_query(&config, json, query.as_deref())
        }
        Cmd::Hook { backend, query } => {
            let config = config::Config::load().map_err(|e| {
                log::error!("Config error: {e}");
                e
            }).unwrap(); // Fail fast
            let embedder = embed::Embedder::load(&config.model_dir()).map_err(|e| {
                log::error!("Embedder error: {e}");
                e
            }).unwrap(); // Fail fast
            let backend: Box<dyn backends::HookBackend> = match backend {
                BackendKind::Claude => Box::new(backends::claude::ClaudeBackend),
                BackendKind::Gemini => Box::new(backends::gemini::GeminiBackend),
            };
            cmd_hook(&embedder, &config, &*backend, &query)
        }
        Cmd::Install => {
            let config = config::Config::load().unwrap_or_else(|_| config::Config::default());
            install::install(&config)
        }
        Cmd::AddRepo { path } => cmd_add_repo(&path),
        Cmd::RemoveRepo { path } => cmd_remove_repo(&path),
        Cmd::Repos => cmd_repos(),
        Cmd::Gc => cmd_gc(),
        Cmd::ClaudeHook => cmd_claude_hook(),
    };
    if let Err(e) = result {
        log::error!("Fatal error: {e}");
        std::process::exit(1);
    }
}

fn cmd_serve() -> error::Result<()> {
    use std::io::{BufRead, Write};
    let config = config::Config::load().unwrap_or_default();

    // GPU embedder for bulk indexing workloads (auto-selects CUDA/Metal/CPU).
    let index_embedder = embed::Embedder::load(&config.model_dir())?;

    // CPU embedder for single-query search. Only allocate a second copy when
    // the index embedder is actually on GPU — otherwise reuse the same one.
    let query_embedder_owned: Option<embed::Embedder> = if index_embedder.is_gpu() {
        Some(embed::Embedder::load_on(&config.model_dir(), candle_core::Device::Cpu)?)
    } else {
        None
    };
    let embedders = mcp_tools::Embedders {
        index: &index_embedder,
        query: query_embedder_owned.as_ref().unwrap_or(&index_embedder),
    };

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut reader = std::io::BufReader::new(stdin.lock());
    let mut writer = std::io::BufWriter::new(stdout.lock());
    loop {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(e) => {
                eprintln!("stdin error: {e}");
                break;
            }
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let req: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("JSON error: {e}");
                continue;
            }
        };
        if let Some(resp) = mcp_tools::handle(&embedders, &config, &req) {
            let mut s = serde_json::to_string(&resp)
                .map_err(|e| error::Error::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("response serialization failed: {e}"),
                )))?;
            s.push('\n');
            writer.write_all(s.as_bytes())?;
            writer.flush()?;
        }
    }
    Ok(())
}

fn cmd_reindex(config: &config::Config) -> error::Result<()> {
    let embedder = embed::Embedder::load(&config.model_dir())?;
    let workspaces = config.expanded_workspaces();
    if workspaces.is_empty() {
        eprintln!("[reindex] No workspaces configured. Edit {}", config::config_file().display());
        return Ok(());
    }
    for ws in &workspaces {
        eprintln!("[reindex] Indexing {}", ws.display());
        reindex::reindex_workspace(&embedder, config, ws)?;
    }
    Ok(())
}

fn cmd_query(config: &config::Config, json: bool, inline: Option<&str>) -> error::Result<()> {
    let prompt = if let Some(q) = inline {
        q.to_string()
    } else {
        use std::io::Read;
        let mut input = String::new();
        std::io::stdin().read_to_string(&mut input)?;
        if json {
            let v: serde_json::Value = serde_json::from_str(input.trim())
                .map_err(|e| error::Error::Config(format!("invalid JSON on stdin: {e}")))?;
            v["prompt"]
                .as_str()
                .ok_or_else(|| error::Error::NotFound("missing 'prompt' field in stdin JSON".into()))?
                .to_string()
        } else {
            input.trim().to_string()
        }
    };
    if prompt.is_empty() {
        return Ok(());
    }
    let embedder = embed::Embedder::load(&config.model_dir())?;
    let backend = backends::claude::ClaudeBackend;
    let results = search::query_all_workspaces(&embedder, config, &prompt, &backend)?;
    if !results.is_empty() {
        println!("--- Relevant code context ---");
        println!("{results}");
        println!("--- End context ---");
    }
    Ok(())
}

fn cmd_hook(
    embedder: &embed::Embedder,
    config: &config::Config,
    backend: &dyn backends::HookBackend,
    query: &str,
) -> error::Result<()> {
    let scored = search::search_workspaces(embedder, config, query)?;
    let output = backend.format_results(&scored, config.search.top_k);
    if !output.is_empty() {
        println!("{output}");
    }
    Ok(())
}

fn cmd_add_repo(path: &str) -> error::Result<()> {
    let abs = std::fs::canonicalize(path)?;
    if !abs.is_dir() {
        return Err(error::Error::NotFound(format!("'{}' is not a directory", abs.display())));
    }

    let mut config = config::Config::load().unwrap_or_default();

    // Store with ~ prefix if under $HOME for portability.
    let home = std::env::var("HOME").unwrap_or_default();
    let store_path = if !home.is_empty() && abs.starts_with(&home) {
        format!("~/{}", abs.strip_prefix(&home).unwrap().display())
    } else {
        abs.display().to_string()
    };

    // Check for duplicates.
    if config.index.workspaces.iter().any(|w| {
        let expanded = config::expand_tilde_pub(w);
        expanded == abs
    }) {
        eprintln!("[slocate] Already registered: {store_path}");
        return Ok(());
    }

    config.index.workspaces.push(store_path.clone());
    config.save()?;
    eprintln!("[slocate] Added: {store_path}");
    eprintln!("[slocate] Run `slocate reindex` to build the index.");
    Ok(())
}

fn cmd_remove_repo(path: &str) -> error::Result<()> {
    let abs = std::fs::canonicalize(path)?;

    let mut config = config::Config::load().unwrap_or_default();
    let before = config.index.workspaces.len();

    config.index.workspaces.retain(|w| {
        config::expand_tilde_pub(w) != abs
    });

    if config.index.workspaces.len() == before {
        return Err(error::Error::NotFound(
            format!("'{}' is not in the workspace list", abs.display()),
        ));
    }

    config.save()?;

    // Delete the index data and registry symlink.
    registry::remove_index(&abs)?;
    eprintln!("[slocate] Removed: {} (index deleted)", abs.display());
    Ok(())
}

fn cmd_gc() -> error::Result<()> {
    let removed = registry::gc_registry()?;
    if removed > 0 {
        eprintln!("[slocate] Cleaned {removed} orphaned registry link(s).");
    } else {
        eprintln!("[slocate] No orphaned links found.");
    }

    // Purge stale embed cache entries (>30 days old) in each workspace.
    let config = config::Config::load().unwrap_or_default();
    for ws in config.expanded_workspaces() {
        let index_dir = match registry::index_dir(&ws) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let db = match store::Store::open(&index_dir) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let before = db.cache_count().unwrap_or(0);
        let purged = db.cache_gc(30).unwrap_or(0);
        if purged > 0 {
            eprintln!(
                "[slocate] {} embed cache: purged {purged}/{before} stale entries",
                ws.display()
            );
        }
    }
    Ok(())
}

/// Claude Code `UserPromptSubmit` hook handler.
///
/// Reads the hook event JSON from stdin, extracts the `prompt` field, runs a
/// slocate search, and writes the hook response JSON to stdout so Claude Code
/// injects the results as `additionalContext` before calling the model.
///
/// Input (from Claude Code):
///   {"session_id":"...","transcript_path":"...","cwd":"...","hook_event_name":"UserPromptSubmit","prompt":"..."}
///
/// Output:
///   {"continue":true,"hookSpecificOutput":{"additionalContext":"..."}}
///   or {"continue":true} when nothing was found.
fn cmd_claude_hook() -> error::Result<()> {
    use std::io::Read;
    let mut raw = String::new();
    std::io::stdin().read_to_string(&mut raw)?;

    let data: serde_json::Value = serde_json::from_str(raw.trim())
        .unwrap_or(serde_json::Value::Null);
    let prompt = data["prompt"].as_str().unwrap_or("").trim().to_string();

    if prompt.is_empty() {
        println!(r#"{{"continue":true}}"#);
        return Ok(());
    }

    let config = config::Config::load().unwrap_or_default();
    let embedder = embed::Embedder::load(&config.model_dir())?;
    let backend = backends::claude::ClaudeBackend;
    let context = search::query_all_workspaces(&embedder, &config, &prompt, &backend)?;

    let resp = if context.is_empty() {
        serde_json::json!({"continue": true})
    } else {
        serde_json::json!({
            "continue": true,
            "hookSpecificOutput": {
                "additionalContext": context
            }
        })
    };
    println!(
        "{}",
        serde_json::to_string(&resp)
            .map_err(|e| error::Error::Config(format!("hook JSON serialization failed: {e}")))?
    );
    Ok(())
}

fn cmd_repos() -> error::Result<()> {
    let config = config::Config::load().unwrap_or_default();
    if config.index.workspaces.is_empty() {
        eprintln!("No workspaces configured. Use `slocate add-repo <path>` to add one.");
        return Ok(());
    }
    for w in &config.index.workspaces {
        println!("{w}");
    }
    Ok(())
}
