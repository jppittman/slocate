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
mod install;
mod leiden;
mod mcp;
mod parse;
mod platform;
mod store;
mod tools;
mod vdb;

use clap::{Parser, Subcommand};

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
        #[arg(long, default_value = "claude")]
        backend: String,
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
}

fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        Cmd::Serve => cmd_serve(),
        Cmd::Reindex => {
            let config = config::Config::load().unwrap_or_else(|e| {
                eprintln!("Config error: {e}");
                std::process::exit(1);
            });
            cmd_reindex(&config)
        }
        Cmd::Query { json, query } => {
            let config = config::Config::load().unwrap_or_else(|e| {
                eprintln!("Config error: {e}");
                std::process::exit(1);
            });
            cmd_query(&config, json, query.as_deref())
        }
        Cmd::Hook { backend, query } => {
            let config = config::Config::load().unwrap_or_else(|e| {
                eprintln!("Config error: {e}");
                std::process::exit(1);
            });
            let embedder = embed::Embedder::load(&config.model_dir()).unwrap_or_else(|e| {
                eprintln!("Embedder error: {e}");
                std::process::exit(1);
            });
            let backend = backends::from_name(&backend).unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(1);
            });
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
    };
    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn cmd_serve() -> Result<(), String> {
    use std::io::{BufRead, Write};
    let config = config::Config::load().unwrap_or_default();
    let embedder = embed::Embedder::load(&config.model_dir())?;
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
        if let Some(resp) = tools::handle(&embedder, &config, &req) {
            let mut s = serde_json::to_string(&resp)
                .map_err(|e| format!("response serialization failed: {e}"))?;
            s.push('\n');
            writer
                .write_all(s.as_bytes())
                .map_err(|e| format!("stdout write failed: {e}"))?;
            writer
                .flush()
                .map_err(|e| format!("stdout flush failed: {e}"))?;
        }
    }
    Ok(())
}

pub fn cmd_reindex(config: &config::Config) -> Result<(), String> {
    let embedder = embed::Embedder::load(&config.model_dir())?;
    let workspaces = config.expanded_workspaces();
    if workspaces.is_empty() {
        eprintln!("[reindex] No workspaces configured. Edit {}", config::config_file().display());
        return Ok(());
    }
    for ws in &workspaces {
        eprintln!("[reindex] Indexing {}", ws.display());
        tools::reindex_workspace(&embedder, config, ws)?;
    }
    Ok(())
}

fn cmd_query(config: &config::Config, json: bool, inline: Option<&str>) -> Result<(), String> {
    let prompt = if let Some(q) = inline {
        q.to_string()
    } else {
        use std::io::Read;
        let mut input = String::new();
        std::io::stdin()
            .read_to_string(&mut input)
            .map_err(|e| format!("failed to read stdin: {e}"))?;
        if json {
            let v: serde_json::Value = serde_json::from_str(input.trim())
                .map_err(|e| format!("invalid JSON on stdin: {e}"))?;
            v["prompt"]
                .as_str()
                .ok_or("missing 'prompt' field in stdin JSON")?
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
    let results = tools::query_all_workspaces(&embedder, config, &prompt, &backend)?;
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
) -> Result<(), String> {
    let scored = tools::search_workspaces(embedder, config, query)?;
    let output = backend.format_results(&scored, config.search.top_k);
    if !output.is_empty() {
        println!("{output}");
    }
    Ok(())
}

fn cmd_add_repo(path: &str) -> Result<(), String> {
    let abs = std::fs::canonicalize(path)
        .map_err(|e| format!("cannot resolve '{}': {e}", path))?;
    if !abs.is_dir() {
        return Err(format!("'{}' is not a directory", abs.display()));
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

fn cmd_remove_repo(path: &str) -> Result<(), String> {
    let abs = std::fs::canonicalize(path)
        .map_err(|e| format!("cannot resolve '{}': {e}", path))?;

    let mut config = config::Config::load().unwrap_or_default();
    let before = config.index.workspaces.len();

    config.index.workspaces.retain(|w| {
        config::expand_tilde_pub(w) != abs
    });

    if config.index.workspaces.len() == before {
        return Err(format!("'{}' is not in the workspace list", abs.display()));
    }

    config.save()?;

    // Delete the index data and registry symlink.
    store::remove_index(&abs)?;
    eprintln!("[slocate] Removed: {} (index deleted)", abs.display());
    Ok(())
}

fn cmd_gc() -> Result<(), String> {
    let removed = store::gc_registry()?;
    if removed > 0 {
        eprintln!("[slocate] Cleaned {removed} orphaned registry link(s).");
    } else {
        eprintln!("[slocate] No orphaned links found.");
    }
    Ok(())
}

fn cmd_repos() -> Result<(), String> {
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
