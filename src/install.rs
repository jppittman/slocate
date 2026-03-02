use crate::config::{self, Config};
use std::path::PathBuf;

pub fn install(config: &Config) -> Result<(), String> {
    // 1. Ensure model is downloaded.
    eprintln!("[slocate] Checking model...");
    crate::download::ensure_model(&config.model_dir())?;

    // 2. Write config if it doesn't exist.
    config.save()?;
    let config_file = config::config_file();
    eprintln!("[slocate] Config written to {}", config_file.display());

    // 3. Ensure state dir exists (for daemon log).
    let state = config::state_dir();
    std::fs::create_dir_all(&state)
        .map_err(|e| format!("failed to create state dir {}: {e}", state.display()))?;

    // 4. Copy binary to ~/.local/bin so hook commands and PATH users find it.
    let exe = std::env::current_exe()
        .map_err(|e| format!("cannot determine binary path: {e}"))?;
    install_binary(&exe)?;

    // 5. Set up platform daemon (launchd on macOS, systemd on Linux).
    crate::platform::setup_daemon(&exe, config)?;

    // 6. Patch shell profile to add ~/.local/bin to PATH if needed.
    patch_shell_path()?;

    // 7. Run first reindex.
    eprintln!("[slocate] Running initial reindex...");
    crate::cmd_reindex(config)?;

    let log_file = config::log_file();
    eprintln!("\n[slocate] Install complete.");
    eprintln!("  Binary:  ~/.local/bin/slocate");
    eprintln!("  Config:  {}", config_file.display());
    eprintln!("  Daemon:  reindex every {} min", config.index.reindex_interval_minutes);
    eprintln!("  Log:     {}", log_file.display());
    eprintln!();
    eprintln!("  To use as a Claude Code RAG hook, add to .claude/settings.json:");
    eprintln!("    \"UserPromptSubmit\": [{{");
    eprintln!("      \"hooks\": [{{");
    eprintln!("        \"type\": \"command\",");
    eprintln!("        \"command\": \"slocate query\",");
    eprintln!("        \"timeout\": 5000");
    eprintln!("      }}]");
    eprintln!("    }}]");

    Ok(())
}

fn install_binary(exe: &std::path::Path) -> Result<(), String> {
    let home = std::env::var("HOME").map_err(|_| "HOME not set".to_string())?;
    let bin_dir = PathBuf::from(&home).join(".local/bin");
    std::fs::create_dir_all(&bin_dir)
        .map_err(|e| format!("failed to create {}: {e}", bin_dir.display()))?;
    let dest = bin_dir.join("slocate");

    // Don't copy if we're already running from the install location.
    let exe_canon = std::fs::canonicalize(exe).unwrap_or_else(|_| exe.to_path_buf());
    let dest_canon = std::fs::canonicalize(&dest).unwrap_or_else(|_| dest.clone());
    if exe_canon == dest_canon {
        eprintln!("[slocate] Binary already at {}", dest.display());
        return Ok(());
    }

    std::fs::copy(exe, &dest)
        .map_err(|e| format!("failed to copy binary to {}: {e}", dest.display()))?;

    // Make executable (should already be, but be explicit).
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o755);
        std::fs::set_permissions(&dest, perms)
            .map_err(|e| format!("chmod failed: {e}"))?;
    }

    eprintln!("[slocate] Binary installed to {}", dest.display());
    Ok(())
}

/// Append `export PATH="$HOME/.local/bin:$PATH"` to the user's shell profile
/// if ~/.local/bin isn't already on PATH or in the profile.
fn patch_shell_path() -> Result<(), String> {
    let home = std::env::var("HOME").map_err(|_| "HOME not set".to_string())?;
    let bin_dir = PathBuf::from(&home).join(".local/bin");

    // Already on PATH? Nothing to do.
    if let Ok(path) = std::env::var("PATH") {
        if path.split(':').any(|p| {
            let p = p.replace("$HOME", &home).replace('~', &home);
            PathBuf::from(p) == bin_dir
        }) {
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
            return Ok(());
        }
    }

    // Append it.
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&profile)
        .map_err(|e| format!("failed to open {}: {e}", profile.display()))?;
    use std::io::Write;
    writeln!(f, "\n# Added by slocate install")
        .map_err(|e| format!("write failed: {e}"))?;
    writeln!(f, "{line}")
        .map_err(|e| format!("write failed: {e}"))?;

    eprintln!(
        "[slocate] Added ~/.local/bin to PATH in {}",
        profile.display()
    );
    eprintln!("[slocate] Restart your shell or: source {}", profile.display());

    Ok(())
}
