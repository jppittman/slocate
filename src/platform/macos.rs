use crate::config::{self, Config};
use std::path::PathBuf;
use std::process::Command;

pub fn setup_daemon(_exe: &std::path::Path, config: &Config) -> crate::error::Result<()> {
    let home = std::env::var("HOME")
        .map_err(|_| crate::error::Error::Config("HOME not set".into()))?;
    let plist_dir = PathBuf::from(&home).join("Library/LaunchAgents");
    std::fs::create_dir_all(&plist_dir)?;

    let plist_path = plist_dir.join("io.slocate.plist");

    // Always reference the installed binary so the daemon survives rebuilds.
    let bin_path = PathBuf::from(&home).join(".local/bin/slocate");
    let program = bin_path.display().to_string();

    let log_file = config::log_file();
    let log_path = log_file.display().to_string();

    // Ensure state dir exists for the log file.
    if let Some(parent) = log_file.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let interval = config.index.reindex_interval_minutes * 60;
    let plist = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>io.slocate</string>
    <key>ProgramArguments</key>
    <array>
        <string>{program}</string>
        <string>reindex</string>
    </array>
    <key>StartInterval</key><integer>{interval}</integer>
    <key>RunAtLoad</key><true/>
    <key>StandardErrorPath</key>
    <string>{log_path}</string>
    <key>StandardOutPath</key>
    <string>{log_path}</string>
</dict>
</plist>"#,
    );

    // Unload existing agent first (ignore errors if not loaded).
    if plist_path.exists() {
        let _ = Command::new("launchctl")
            .args(["unload", &plist_path.display().to_string()])
            .output();
    }

    std::fs::write(&plist_path, &plist)?;

    let status = Command::new("launchctl")
        .args(["load", &plist_path.display().to_string()])
        .status()?;
    if !status.success() {
        return Err(crate::error::Error::Config(format!(
            "launchctl load exited with {}",
            status.code().unwrap_or(-1)
        )));
    }

    eprintln!("[slocate] launchd agent installed: io.slocate");
    eprintln!(
        "[slocate] Reindex every {} min, log: {}",
        config.index.reindex_interval_minutes,
        log_path,
    );
    Ok(())
}
