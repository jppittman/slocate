use crate::config::{self, Config};
use std::path::PathBuf;
use std::process::Command;

pub fn setup_daemon(exe: &std::path::Path, config: &Config) -> crate::error::Result<()> {
    let unit_dir = xdg_systemd_dir()?;
    std::fs::create_dir_all(&unit_dir)?;

    let log_file = config::log_file();
    let log_path = log_file.display().to_string();

    // Ensure state dir exists for the log file.
    if let Some(parent) = log_file.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let service = format!(
        "[Unit]\nDescription=slocate reindexer\n\n\
         [Service]\nType=oneshot\nExecStart={exe} reindex\n\
         StandardError=append:{log_path}\n",
        exe = exe.display(),
    );
    std::fs::write(unit_dir.join("slocate.service"), service)?;

    let interval_sec = config.index.reindex_interval_minutes * 60;
    let timer = format!(
        "[Unit]\nDescription=slocate reindex timer\n\n\
         [Timer]\nOnBootSec=60\nOnUnitActiveSec={interval_sec}\n\n\
         [Install]\nWantedBy=timers.target\n",
    );
    std::fs::write(unit_dir.join("slocate.timer"), timer)?;

    for cmd in [
        vec!["daemon-reload"],
        vec!["enable", "--now", "slocate.timer"],
    ] {
        let status = Command::new("systemctl")
            .arg("--user")
            .args(&cmd)
            .status();
        match status {
            Ok(s) if s.success() => {}
            Ok(s) => {
                eprintln!(
                    "[slocate] warning: systemctl --user {} exited {s}",
                    cmd.join(" ")
                );
                eprintln!(
                    "[slocate] Daemon setup failed (no user session bus?).\n\
                     [slocate] Unit files written to {}/.\n\
                     [slocate] Run `systemctl --user enable --now slocate.timer` manually,\n\
                     [slocate] or run `slocate reindex` on a cron/schedule.",
                    unit_dir.display()
                );
                return Ok(());
            }
            Err(e) => {
                eprintln!("[slocate] warning: systemctl not available: {e}");
                eprintln!(
                    "[slocate] Unit files written to {}/.\n\
                     [slocate] Enable manually or run `slocate reindex` on a cron/schedule.",
                    unit_dir.display()
                );
                return Ok(());
            }
        }
    }
    eprintln!("[slocate] systemd user timer installed: slocate.timer");
    Ok(())
}

fn xdg_systemd_dir() -> crate::error::Result<PathBuf> {
    let home = std::env::var("HOME")
        .map_err(|_| crate::error::Error::Config("HOME not set".into()))?;
    Ok(PathBuf::from(home).join(".config/systemd/user"))
}
