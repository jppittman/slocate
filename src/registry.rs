use std::path::{Path, PathBuf};

/// Returns the index directory for a workspace.
///
/// Data lives at `<workspace>/.slocate/` -- delete the workspace and the
/// index goes with it. No leaked space.
///
/// A symlink from `~/.local/share/slocate/<hash>` points back to the
/// workspace's `.slocate/` dir so we can enumerate all indexed dirs
/// without scanning the filesystem. Broken symlinks = workspace was deleted,
/// `slocate gc` prunes them.
pub fn index_dir(workspace_root: &Path) -> crate::error::Result<PathBuf> {
    let dir = workspace_root.join(".slocate");
    if !dir.exists() {
        std::fs::create_dir_all(&dir)?;
    }

    // Maintain the reverse symlink registry (best-effort).
    let _ = ensure_registry_link(workspace_root, &dir);

    Ok(dir)
}

/// Remove the index for a workspace and its registry symlink.
pub fn remove_index(workspace_root: &Path) -> crate::error::Result<()> {
    let dir = workspace_root.join(".slocate");
    if dir.exists() {
        std::fs::remove_dir_all(&dir)?;
    }
    // Remove the registry symlink.
    if let Ok(link) = registry_link_path(workspace_root) {
        if link.symlink_metadata().is_ok() {
            let _ = std::fs::remove_file(&link);
        }
    }
    Ok(())
}

/// Remove registry symlinks whose targets no longer exist (workspace deleted).
pub fn gc_registry() -> crate::error::Result<usize> {
    let base = registry_base()?;
    if !base.exists() {
        return Ok(0);
    }
    let mut removed = 0;
    let entries = std::fs::read_dir(&base)?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        // Symlink whose target is gone -> orphan.
        if path.symlink_metadata().is_ok() && !path.exists() {
            let _ = std::fs::remove_file(&path);
            removed += 1;
        }
    }
    Ok(removed)
}

fn registry_base() -> crate::error::Result<PathBuf> {
    Ok(crate::config::data_dir().join("registry"))
}

fn registry_link_path(workspace_root: &Path) -> crate::error::Result<PathBuf> {
    let base = registry_base()?;
    Ok(base.join(dir_hash(workspace_root)))
}

fn ensure_registry_link(workspace_root: &Path, index_dir: &Path) -> crate::error::Result<()> {
    let base = registry_base()?;
    std::fs::create_dir_all(&base)?;
    let link = base.join(dir_hash(workspace_root));
    if !link.exists() {
        #[cfg(unix)]
        std::os::unix::fs::symlink(index_dir, &link)?;
    }
    Ok(())
}

/// Stable hash of a workspace path -> 16 hex chars.
fn dir_hash(path: &Path) -> String {
    let mut h: u64 = 5381;
    for b in path.display().to_string().bytes() {
        h = h.wrapping_mul(33).wrapping_add(b as u64);
    }
    format!("{:016x}", h)
}
