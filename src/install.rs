use crate::config::{self, Config};
use std::path::PathBuf;

pub fn install(config: &Config) -> crate::error::Result<()> {
    // 1. Ensure model is downloaded.
    eprintln!("[slocate] Checking model...");
    crate::download::ensure_model(&config.model_dir())?;

    // 2. Write config if it doesn't exist.
    config.save()?;
    let config_file = config::config_file();
    eprintln!("[slocate] Config written to {}", config_file.display());

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
    eprintln!("  Claude Code — add to .claude/settings.json:");
    eprintln!("    \"hooks\": {{");
    eprintln!("      \"UserPromptSubmit\": [{{");
    eprintln!("        \"hooks\": [{{");
    eprintln!("          \"type\": \"command\",");
    eprintln!("          \"command\": \"slocate hook --backend claude \\\"$PROMPT\\\"\",");
    eprintln!("          \"timeout\": 5000");
    eprintln!("        }}]");
    eprintln!("      }}]");
    eprintln!("    }}");
    eprintln!();
    eprintln!("  Gemini CLI — add to .gemini/settings.json:");
    eprintln!("    \"hooks\": {{");
    eprintln!("      \"user_prompt_submit\": [{{");
    eprintln!("        \"hooks\": [{{");
    eprintln!("          \"type\": \"command\",");
    eprintln!("          \"command\": \"slocate hook --backend gemini \\\"$PROMPT\\\"\",");
    eprintln!("          \"timeout\": 5000");
    eprintln!("        }}]");
    eprintln!("      }}]");
    eprintln!("    }}");

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
        eprintln!("[slocate] Binary already at {}", dest.display());
        return Ok(());
    }

    std::fs::copy(exe, &dest)?;

    // Make executable (should already be, but be explicit).
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o755);
        std::fs::set_permissions(&dest, perms)?;
    }

    eprintln!("[slocate] Binary installed to {}", dest.display());
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
        .open(&profile)?;
    use std::io::Write;
    writeln!(f, "\n# Added by slocate install")?;
    writeln!(f, "{line}")?;

    eprintln!(
        "[slocate] Added ~/.local/bin to PATH in {}",
        profile.display()
    );
    eprintln!("[slocate] Restart your shell or: source {}", profile.display());

    Ok(())
}
