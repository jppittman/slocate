#[cfg(target_os = "macos")]
mod macos;

#[cfg(target_os = "linux")]
mod linux;

use crate::config::Config;

pub fn setup_daemon(exe: &std::path::Path, config: &Config) -> crate::error::Result<()> {
    #[cfg(target_os = "macos")]
    {
        macos::setup_daemon(exe, config)
    }
    #[cfg(target_os = "linux")]
    {
        linux::setup_daemon(exe, config)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        let _ = (exe, config);
        eprintln!(
            "[slocate] Daemon setup not supported on this platform. \
             Run `slocate reindex` manually."
        );
        Ok(())
    }
}
