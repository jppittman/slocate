use crate::config::{self, Config};
use std::path::PathBuf;
use std::process::Command;

pub fn setup_daemon(exe: &std::path::Path, config: &Config) -> Result<(), String> {
    let unit_dir = xdg_systemd_dir()?;
    std::fs::create_dir_all(&unit_dir)
        .map_err(|e| format!("failed to create systemd user dir: {e}"))?;

    let log_file = config::log_file();
    let log_path = log_file.display().to_string();

    // Ensure state dir exists for the log file.
    if let Some(parent) = log_file.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create state dir: {e}"))?;
    }

    let service = format!(
        "[Unit]\nDescription=slocate reindexer\n\n\
         [Service]\nType=oneshot\nExecStart={exe} reindex\n\
         StandardError=append:{log_path}\n",
        exe = exe.display(),
    );
    std::fs::write(unit_dir.join("slocate.service"), service)
        .map_err(|e| format!("failed to write service: {e}"))?;

    let interval_sec = config.index.reindex_interval_minutes * 60;
    let timer = format!(
        "[Unit]\nDescription=slocate reindex timer\n\n\
         [Timer]\nOnBootSec=60\nOnUnitActiveSec={interval_sec}\n\n\
         [Install]\nWantedBy=timers.target\n",
    );
    std::fs::write(unit_dir.join("slocate.timer"), timer)
        .map_err(|e| format!("failed to write timer: {e}"))?;

    for cmd in [
        vec!["daemon-reload"],
        vec!["enable", "--now", "slocate.timer"],
    ] {
        let status = Command::new("systemctl")
            .arg("--user")
            .args(&cmd)
            .status()
            .map_err(|e| format!("systemctl failed: {e}"))?;
        if !status.success() {
            return Err(format!(
                "systemctl --user {} exited {status}",
                cmd.join(" ")
            ));
        }
    }
    eprintln!("[slocate] systemd user timer installed: slocate.timer");
    Ok(())
}

fn xdg_systemd_dir() -> Result<PathBuf, String> {
    let home = std::env::var("HOME").map_err(|_| "HOME not set".to_string())?;
    Ok(PathBuf::from(home).join(".config/systemd/user"))
}
