use crate::config::{self, Config};
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

/// Timeout for systemctl commands. In environments without a user session
/// bus (containers, WSL1, CI), systemctl can hang waiting for D-Bus.
const SYSTEMCTL_TIMEOUT: Duration = Duration::from_secs(15);

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
        let status = run_with_timeout(
            Command::new("systemctl").arg("--user").args(&cmd),
            SYSTEMCTL_TIMEOUT,
        );
        match status {
            Ok(Some(s)) if s.success() => {}
            Ok(Some(s)) => {
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
            Ok(None) => {
                eprintln!(
                    "[slocate] warning: systemctl --user {} timed out after {}s \
                     (no user session bus?)",
                    cmd.join(" "),
                    SYSTEMCTL_TIMEOUT.as_secs(),
                );
                eprintln!(
                    "[slocate] Unit files written to {}/.\n\
                     [slocate] Enable manually or run `slocate reindex` on a cron/schedule.",
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

/// Run a command with a timeout. Returns `Ok(Some(status))` on normal exit,
/// `Ok(None)` if the timeout expires (child is killed), or `Err` if the
/// command could not be spawned.
fn run_with_timeout(
    cmd: &mut Command,
    timeout: Duration,
) -> std::io::Result<Option<std::process::ExitStatus>> {
    let mut child = cmd.spawn()?;
    let start = Instant::now();
    loop {
        match child.try_wait()? {
            Some(status) => return Ok(Some(status)),
            None if start.elapsed() >= timeout => {
                let _ = child.kill();
                let _ = child.wait(); // reap zombie
                return Ok(None);
            }
            None => std::thread::sleep(Duration::from_millis(100)),
        }
    }
}

fn xdg_systemd_dir() -> crate::error::Result<PathBuf> {
    let home = std::env::var("HOME")
        .map_err(|_| crate::error::Error::Config("HOME not set".into()))?;
    Ok(PathBuf::from(home).join(".config/systemd/user"))
}
