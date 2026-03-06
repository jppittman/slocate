use crate::config::{self, Config};
use std::path::PathBuf;

const MCP_ENTRY: &str = r#"{"type":"stdio","command":"slocate","args":["serve"]}"#;

pub fn install(config: &Config) -> crate::error::Result<()> {
    // 1. Ensure model is downloaded.
    log::info!("Checking model...");
    crate::download::ensure_model(&config.model_dir())?;

    // 2. Write config if it doesn't exist.
    config.save()?;
    let config_file = config::config_file();
    log::info!("Config written to {}", config_file.display());

    // 3. Ensure state dir exists (for daemon log).
    let state = config::state_dir();
    std::fs::create_dir_all(&state)?;

    // 4. Copy binary to ~/.local/bin so hook commands and PATH users find it.
    let exe = std::env::current_exe()?;
    install_binary(&exe)?;

    // 5. Set up platform daemon (launchd on macOS, systemd on Linux).
    crate::platform::setup_daemon(&exe, config)?;

    // 6. Patch shell profile to add ~/.local/bin to PATH if needed.
    patch_shell_path()?;

    // 7. Configure MCP servers for Claude Code and Gemini CLI (best-effort).
    let home = std::env::var("HOME").unwrap_or_default();
    configure_claude_mcp(&home);
    configure_gemini_mcp(&home);

    // 8. Register UserPromptSubmit hook in Claude Code settings (best-effort).
    configure_claude_hook(&home);

    // 9. Run first reindex.
    log::info!("Running initial reindex (this may take a few minutes for large workspaces)...");
    crate::cmd_reindex(config)?;

    let log_file = config::log_file();
    eprintln!("\n[slocate] Install complete.");
    eprintln!("  Binary:  ~/.local/bin/slocate");
    eprintln!("  Config:  {}", config_file.display());
    eprintln!("  Daemon:  reindex every {} min", config.index.reindex_interval_minutes);
    eprintln!("  Log:     {}", log_file.display());

    Ok(())
}

fn install_binary(exe: &std::path::Path) -> crate::error::Result<()> {
    let home = std::env::var("HOME")
        .map_err(|_| crate::error::Error::Config("HOME not set".into()))?;
    let bin_dir = PathBuf::from(&home).join(".local/bin");
    std::fs::create_dir_all(&bin_dir)?;
    let dest = bin_dir.join("slocate");

    // Don't copy if we're already running from the install location.
    let exe_canon = std::fs::canonicalize(exe).unwrap_or_else(|_| exe.to_path_buf());
    let dest_canon = std::fs::canonicalize(&dest).unwrap_or_else(|_| dest.clone());
    if exe_canon == dest_canon {
        log::warn!("Binary already at {}", dest.display());
        return Ok(());
    }

    std::fs::copy(exe, &dest).map_err(|e| {
        log::error!("Failed to copy binary to {}: {e}", dest.display());
        e
    })?;

    // Make executable (should already be, but be explicit).
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o755);
        std::fs::set_permissions(&dest, perms)?;
    }

    log::info!("Binary installed to {}", dest.display());
    Ok(())
}

/// Append `export PATH="$HOME/.local/bin:$PATH"` to the user's shell profile
/// if ~/.local/bin isn't already on PATH or in the profile.
fn patch_shell_path() -> crate::error::Result<()> {
    let home = std::env::var("HOME")
        .map_err(|_| crate::error::Error::Config("HOME not set".into()))?;
    let bin_dir = PathBuf::from(&home).join(".local/bin");

    // Already on PATH? Nothing to do.
    if let Ok(path) = std::env::var("PATH") {
        if path.split(':').any(|p| {
            let p = p.replace("$HOME", &home).replace('~', &home);
            PathBuf::from(p) == bin_dir
        }) {
            log::debug!("~/.local/bin already on PATH");
            return Ok(());
        }
    }

    // Pick the right profile file.
    let shell = std::env::var("SHELL").unwrap_or_default();
    let profile = if shell.ends_with("zsh") {
        PathBuf::from(&home).join(".zshrc")
    } else {
        PathBuf::from(&home).join(".bashrc")
    };

    // Check if profile already has the line.
    let line = r#"export PATH="$HOME/.local/bin:$PATH""#;
    if let Ok(contents) = std::fs::read_to_string(&profile) {
        if contents.contains(".local/bin") {
            log::debug!("~/.local/bin already in {}", profile.display());
            return Ok(());
        }
    }

    // Append it.
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&profile)?;
    use std::io::Write;
    writeln!(f, "\n# Added by slocate install")?;
    writeln!(f, "{line}")?;

    log::info!("Added ~/.local/bin to PATH in {}", profile.display());
    eprintln!("[slocate] Restart your shell or: source {}", profile.display());

    Ok(())
}

/// Merge slocate MCP server entry into ~/.claude.json (Claude Code user-scope config).
/// Logs and returns on any error — not having Claude Code installed is not a bug.
fn configure_claude_mcp(home: &str) {
    let path = PathBuf::from(home).join(".claude.json");
    match patch_mcp_json(&path) {
        Ok(true) => eprintln!("[slocate] Registered MCP server in {}", path.display()),
        Ok(false) => log::info!("slocate MCP server already in {}", path.display()),
        Err(e) => log::warn!("Could not configure Claude Code MCP (skipping): {e}"),
    }
}

/// Merge slocate MCP server entry into ~/.gemini/settings.json (Gemini CLI config).
/// Logs and returns on any error — not having Gemini CLI installed is not a bug.
fn configure_gemini_mcp(home: &str) {
    let path = PathBuf::from(home).join(".gemini/settings.json");
    // Only attempt if the directory already exists (Gemini CLI was set up).
    if !path.parent().map(|p| p.exists()).unwrap_or(false) {
        log::debug!("~/.gemini/ not found, skipping Gemini CLI MCP config");
        return;
    }
    match patch_mcp_json(&path) {
        Ok(true) => eprintln!("[slocate] Registered MCP server in {}", path.display()),
        Ok(false) => log::info!("slocate MCP server already in {}", path.display()),
        Err(e) => log::warn!("Could not configure Gemini CLI MCP (skipping): {e}"),
    }
}

/// Register `slocate claude-hook` as a Claude Code `UserPromptSubmit` hook.
/// Patches `~/.claude/settings.json`. Logs and returns on any error.
fn configure_claude_hook(home: &str) {
    let path = PathBuf::from(home).join(".claude/settings.json");
    match patch_claude_hook_json(&path) {
        Ok(true) => eprintln!("[slocate] Registered UserPromptSubmit hook in {}", path.display()),
        Ok(false) => log::info!("slocate hook already in {}", path.display()),
        Err(e) => log::warn!("Could not configure Claude Code hook (skipping): {e}"),
    }
}

/// Inject the slocate `UserPromptSubmit` hook entry into `~/.claude/settings.json`.
/// Returns `true` if modified, `false` if already present.
fn patch_claude_hook_json(path: &std::path::Path) -> Result<bool, String> {
    let mut root: serde_json::Value = if path.exists() {
        let raw = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        serde_json::from_str(&raw).map_err(|e| format!("parse error: {e}"))?
    } else {
        serde_json::Value::Object(serde_json::Map::new())
    };

    let hooks = root
        .as_object_mut()
        .ok_or("settings.json root is not a JSON object")?
        .entry("hooks")
        .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()))
        .as_object_mut()
        .ok_or("hooks is not a JSON object")?
        .entry("UserPromptSubmit")
        .or_insert_with(|| serde_json::Value::Array(Vec::new()))
        .as_array_mut()
        .ok_or("UserPromptSubmit is not an array")?;

    // Check if the slocate hook is already present.
    let already = hooks.iter().any(|entry| {
        entry["hooks"]
            .as_array()
            .map(|arr| {
                arr.iter().any(|h| {
                    h["command"].as_str() == Some("slocate claude-hook")
                })
            })
            .unwrap_or(false)
    });
    if already {
        return Ok(false);
    }

    hooks.push(serde_json::json!({
        "hooks": [{"type": "command", "command": "slocate claude-hook"}]
    }));

    // Ensure the parent directory exists.
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let out = serde_json::to_string_pretty(&root).map_err(|e| e.to_string())?;
    std::fs::write(path, out).map_err(|e| e.to_string())?;
    Ok(true)
}

/// Read a JSON file (or start fresh), insert `mcpServers.slocate`, write back.
/// Returns `true` if the file was modified, `false` if slocate was already present.
fn patch_mcp_json(path: &std::path::Path) -> Result<bool, String> {
    let mut root: serde_json::Value = if path.exists() {
        let raw = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        serde_json::from_str(&raw).map_err(|e| format!("parse error: {e}"))?
    } else {
        serde_json::Value::Object(serde_json::Map::new())
    };

    let servers = root
        .as_object_mut()
        .ok_or("root is not a JSON object")?
        .entry("mcpServers")
        .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()))
        .as_object_mut()
        .ok_or("mcpServers is not a JSON object")?;

    if servers.contains_key("slocate") {
        return Ok(false);
    }

    let entry: serde_json::Value =
        serde_json::from_str(MCP_ENTRY).expect("MCP_ENTRY is valid JSON");
    servers.insert("slocate".to_string(), entry);

    let out = serde_json::to_string_pretty(&root).map_err(|e| e.to_string())?;
    std::fs::write(path, out).map_err(|e| e.to_string())?;
    Ok(true)
}
