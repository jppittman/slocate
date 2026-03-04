use crate::config::{self, Config};
use std::path::PathBuf;
use std::process::Command;

pub fn setup_daemon(exe: &std::path::Path, config: &Config) -> crate::error::Result<()> {
    // Pre-flight check: systemd --user requires a functional session bus.
    // systemctl will often hang for 30s then fail with "Transport endpoint is not connected"
    // if these are missing or invalid.
    if std::env::var("XDG_RUNTIME_DIR").is_err() && std::env::var("DBUS_SESSION_BUS_ADDRESS").is_err() {
        return Err(crate::error::Error::Config(
            "Neither XDG_RUNTIME_DIR nor DBUS_SESSION_BUS_ADDRESS is set. \
             systemd --user requires a functional user session. \
             If you are in an SSH session, ensure pam_systemd is working, \
             or try 'loginctl enable-linger $USER' and log in again.".into()
        ));
    }

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
         [Service]\nType=oneshot\nExecStart=\"{exe}\" reindex\n\
         StandardError=journal\n\
         TimeoutStopSec=5s\n",
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
        log::debug!("Running systemctl --user {}", cmd.join(" "));
        let status = Command::new("systemctl")
            .arg("--user")
            .args(&cmd)
            .status();
        match status {
            Ok(s) if s.success() => {
                log::debug!("systemctl --user {} succeeded", cmd.join(" "));
            }
            Ok(s) => {
                let err_msg = format!("systemctl --user {} failed with status {}", cmd.join(" "), s);
                log::error!("{}", err_msg);
                if s.code() == Some(1) {
                     log::error!("Note: 'Transport endpoint is not connected' often means the user session bus is not running. Try 'export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$(id -u)/bus'");
                }
                return Err(crate::error::Error::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    err_msg
                )));
            }
            Err(e) => {
                log::error!("systemctl not found or failed to execute: {e}");
                return Err(crate::error::Error::Io(e));
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
