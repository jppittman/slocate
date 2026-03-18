use rusqlite::{Connection, params};
use std::collections::HashMap;
use std::path::Path;

/// Shared embedding cache backed by a standalone SQLite database.
///
/// Lives at `$XDG_DATA_HOME/slocate/embed_cache.db` so that identical chunks
/// across multiple workspaces are embedded once and shared. The per-workspace
/// `index.db` no longer stores embed cache entries.
pub struct EmbedCache {
    conn: Connection,
}

impl EmbedCache {
    /// Open (or create) the shared embed cache at the default XDG data path.
    pub fn open() -> crate::error::Result<Self> {
        let path = crate::config::data_dir().join("embed_cache.db");
        Self::open_at(&path)
    }

    /// Open at an explicit path (for tests or custom locations).
    pub fn open_at(path: &Path) -> crate::error::Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path)?;
        conn.pragma(None, "journal_mode", "WAL", |_| Ok(()))?;
        conn.pragma(None, "synchronous", "NORMAL", |_| Ok(()))?;
        conn.pragma(None, "busy_timeout", 5000, |_| Ok(()))?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS embed_cache (
                content_hash TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                created_at INTEGER NOT NULL DEFAULT (strftime('%s','now'))
            );",
        )?;
        Ok(Self { conn })
    }

    pub fn get_batch(&self, hashes: &[String]) -> crate::error::Result<HashMap<String, Vec<f32>>> {
        let mut out = HashMap::new();
        let mut s = self.conn.prepare_cached(
            "SELECT vector FROM embed_cache WHERE content_hash = ?1",
        )?;
        for h in hashes {
            if let Ok(b) = s.query_row(params![h], |row| row.get::<_, Vec<u8>>(0)) {
                out.insert(h.clone(), super::decode_vector(&b));
            }
        }
        Ok(out)
    }

    pub fn put(&self, entries: &[(String, Vec<f32>)]) -> crate::error::Result<()> {
        let mut s = self.conn.prepare_cached(
            "INSERT OR REPLACE INTO embed_cache (content_hash, vector) VALUES (?1, ?2)",
        )?;
        for (h, v) in entries {
            s.execute(params![h, super::encode_vector(v)])?;
        }
        Ok(())
    }

    pub fn gc(&self, max_age_days: u32) -> crate::error::Result<usize> {
        let cutoff = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64
            - (max_age_days as i64 * 86400);
        Ok(self.conn.execute(
            "DELETE FROM embed_cache WHERE created_at < ?1",
            params![cutoff],
        )?)
    }

    pub fn count(&self) -> crate::error::Result<usize> {
        Ok(self
            .conn
            .query_row("SELECT COUNT(*) FROM embed_cache", [], |row| row.get(0))?)
    }

    /// Migrate entries from a per-workspace index.db embed_cache table into this
    /// shared cache. Returns the number of entries migrated. Safe to call even if
    /// the old table does not exist.
    pub fn migrate_from(&self, index_db_path: &Path) -> crate::error::Result<usize> {
        let old = Connection::open_with_flags(
            index_db_path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;

        // Check if the old table exists.
        let has_table: bool = old
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='embed_cache'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .map(|c| c > 0)
            .unwrap_or(false);

        if !has_table {
            return Ok(0);
        }

        let mut read = old.prepare("SELECT content_hash, vector, created_at FROM embed_cache")?;
        let mut write = self.conn.prepare_cached(
            "INSERT OR IGNORE INTO embed_cache (content_hash, vector, created_at) VALUES (?1, ?2, ?3)",
        )?;

        let mut count = 0usize;
        let rows = read.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Vec<u8>>(1)?,
                row.get::<_, i64>(2)?,
            ))
        })?;
        for row in rows {
            let (hash, vec, ts) = row?;
            count += write.execute(params![hash, vec, ts])? as usize;
        }
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_cache() -> (std::path::PathBuf, EmbedCache) {
        let dir = std::env::temp_dir().join(format!(
            "slocate_test_cache_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .subsec_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("embed_cache.db");
        let cache = EmbedCache::open_at(&path).unwrap();
        (dir, cache)
    }

    #[test]
    fn round_trip_put_get() {
        let (_dir, cache) = temp_cache();
        let entries = vec![
            ("hash_a".to_string(), vec![1.0f32; 384]),
            ("hash_b".to_string(), vec![0.5f32; 384]),
        ];
        cache.put(&entries).unwrap();

        let got = cache
            .get_batch(&["hash_a".into(), "hash_b".into(), "hash_c".into()])
            .unwrap();
        assert!(got.contains_key("hash_a"));
        assert!(got.contains_key("hash_b"));
        assert!(!got.contains_key("hash_c"));
        assert_eq!(got["hash_a"].len(), 384);
    }

    #[test]
    fn gc_removes_old_entries() {
        let (_dir, cache) = temp_cache();
        // Insert with an ancient timestamp.
        cache
            .conn
            .execute(
                "INSERT INTO embed_cache (content_hash, vector, created_at) VALUES (?1, ?2, ?3)",
                params!["old_hash", super::super::encode_vector(&vec![0.1f32; 384]), 1000],
            )
            .unwrap();
        cache
            .put(&[("new_hash".into(), vec![0.2f32; 384])])
            .unwrap();

        assert_eq!(cache.count().unwrap(), 2);
        let purged = cache.gc(30).unwrap();
        assert_eq!(purged, 1);
        assert_eq!(cache.count().unwrap(), 1);
    }

    #[test]
    fn migrate_from_old_db() {
        let (_dir, cache) = temp_cache();

        // Create a fake old index.db with embed_cache table.
        let old_path = _dir.join("old_index.db");
        let old_conn = Connection::open(&old_path).unwrap();
        old_conn
            .execute_batch(
                "CREATE TABLE embed_cache (
                    content_hash TEXT PRIMARY KEY,
                    vector BLOB NOT NULL,
                    created_at INTEGER NOT NULL DEFAULT (strftime('%s','now'))
                );",
            )
            .unwrap();
        old_conn
            .execute(
                "INSERT INTO embed_cache (content_hash, vector) VALUES (?1, ?2)",
                params!["migrated_hash", super::super::encode_vector(&vec![0.3f32; 384])],
            )
            .unwrap();
        drop(old_conn);

        let migrated = cache.migrate_from(&old_path).unwrap();
        assert_eq!(migrated, 1);

        let got = cache.get_batch(&["migrated_hash".into()]).unwrap();
        assert!(got.contains_key("migrated_hash"));
    }

    #[test]
    fn migrate_from_no_table() {
        let (_dir, cache) = temp_cache();
        let old_path = _dir.join("empty_index.db");
        let old_conn = Connection::open(&old_path).unwrap();
        old_conn.execute_batch("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);").unwrap();
        drop(old_conn);

        let migrated = cache.migrate_from(&old_path).unwrap();
        assert_eq!(migrated, 0);
    }
}
