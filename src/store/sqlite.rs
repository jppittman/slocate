use rusqlite::{Connection, params, OptionalExtension};
use std::path::Path;
use crate::parse::ChunkKind;
use super::{Chunk, Db, FileMeta, Note};

pub struct SqliteDb { conn: Connection }

impl SqliteDb {
    pub fn open(index_dir: &Path) -> crate::error::Result<Self> {
        let db_path = index_dir.join("index.db");
        let conn = Connection::open(&db_path)?;
        conn.pragma(None, "journal_mode", "WAL", |_| Ok(()))?;
        conn.pragma(None, "synchronous", "NORMAL", |_| Ok(()))?;
        conn.execute_batch("
            CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, kind TEXT NOT NULL, name TEXT NOT NULL, source_path TEXT NOT NULL, source TEXT NOT NULL, vector BLOB);
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(name, source, content='chunks', content_rowid='rowid');
        ")?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS notes (id TEXT PRIMARY KEY, text TEXT NOT NULL, tags TEXT NOT NULL, timestamp TEXT NOT NULL, vector BLOB NOT NULL);"
        )?;
        Ok(Self { conn })
    }
}

impl Db for SqliteDb {
    fn begin(&self) -> crate::error::Result<()> { self.conn.execute_batch("BEGIN")?; Ok(()) }
    fn commit(&self) -> crate::error::Result<()> { self.conn.execute_batch("COMMIT")?; Ok(()) }
    fn rollback(&self) -> crate::error::Result<()> { self.conn.execute_batch("ROLLBACK")?; Ok(()) }

    fn save_meta(&self, key: &str, value: &str) -> crate::error::Result<()> {
        self.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)", params![key, value])?;
        Ok(())
    }
    fn get_meta(&self, key: &str) -> crate::error::Result<Option<String>> {
        let mut s = self.conn.prepare("SELECT value FROM meta WHERE key = ?1")?;
        Ok(s.query_row(params![key], |row| row.get(0)).optional()?)
    }

    fn insert_chunk(&self, c: &Chunk, v: &[f32]) -> crate::error::Result<()> {
        self.conn.execute("INSERT OR REPLACE INTO chunks (id, kind, name, source_path, source, vector) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![c.id, c.kind.as_str(), c.name, c.source_path, c.source, encode_vector(v)])?;
        let rowid = self.conn.last_insert_rowid();
        self.conn.execute("INSERT OR REPLACE INTO chunks_fts (rowid, name, source) VALUES (?1, ?2, ?3)",
            params![rowid, c.name, c.source])?;
        Ok(())
    }
    fn load_all_chunks_with_vectors(&self) -> crate::error::Result<Vec<(Chunk, Vec<f32>)>> {
        let mut s = self.conn.prepare("SELECT id, kind, name, source_path, source, vector FROM chunks WHERE vector IS NOT NULL")?;
        let r = s.query_map([], |row| Ok((Chunk {
            id: row.get(0)?, kind: ChunkKind::from_db_str(&row.get::<_, String>(1)?),
            name: row.get(2)?, source_path: row.get(3)?, source: row.get(4)?,
        }, decode_vector(&row.get::<_, Vec<u8>>(5)?))))?;
        r.collect::<rusqlite::Result<Vec<_>>>().map_err(|e| e.into())
    }
    fn get_chunks_by_ids(&self, ids: &[String]) -> crate::error::Result<Vec<Chunk>> {
        if ids.is_empty() { return Ok(Vec::new()); }
        let p: String = ids.iter().enumerate().map(|(i, _)| format!("?{}", i + 1)).collect::<Vec<_>>().join(",");
        let sql = format!("SELECT id, kind, name, source_path, source FROM chunks WHERE id IN ({})", p);
        let mut s = self.conn.prepare(&sql)?;
        let r = s.query_map(rusqlite::params_from_iter(ids), |row| Ok(Chunk {
            id: row.get(0)?, kind: ChunkKind::from_db_str(&row.get::<_, String>(1)?),
            name: row.get(2)?, source_path: row.get(3)?, source: row.get(4)?,
        }))?;
        r.collect::<rusqlite::Result<Vec<_>>>().map_err(|e| e.into())
    }
    fn get_chunks_with_vectors_by_ids(&self, ids: &[String]) -> crate::error::Result<Vec<(Chunk, Vec<f32>)>> {
        if ids.is_empty() { return Ok(Vec::new()); }
        let p: String = ids.iter().enumerate().map(|(i, _)| format!("?{}", i + 1)).collect::<Vec<_>>().join(",");
        let sql = format!("SELECT id, kind, name, source_path, source, vector FROM chunks WHERE id IN ({})", p);
        let mut s = self.conn.prepare(&sql)?;
        let r = s.query_map(rusqlite::params_from_iter(ids), |row| Ok((Chunk {
            id: row.get(0)?, kind: ChunkKind::from_db_str(&row.get::<_, String>(1)?),
            name: row.get(2)?, source_path: row.get(3)?, source: row.get(4)?,
        }, decode_vector(&row.get::<_, Vec<u8>>(5)?))))?;
        r.collect::<rusqlite::Result<Vec<_>>>().map_err(|e| e.into())
    }

    fn ensure_file_meta_table(&self) -> crate::error::Result<()> {
        self.conn.execute_batch("CREATE TABLE IF NOT EXISTS file_meta (rel_path TEXT PRIMARY KEY, mtime_ns INTEGER NOT NULL, size INTEGER NOT NULL);")?;
        Ok(())
    }
    fn load_file_meta(&self) -> crate::error::Result<std::collections::HashMap<String, FileMeta>> {
        let mut s = self.conn.prepare("SELECT rel_path, mtime_ns, size FROM file_meta")?;
        let r = s.query_map([], |row| Ok((row.get::<_, String>(0)?, FileMeta { mtime_ns: row.get(1)?, size: row.get(2)? })))?;
        r.collect::<rusqlite::Result<std::collections::HashMap<_, _>>>().map_err(|e| e.into())
    }
    fn update_file_meta(&self, entries: &[(String, FileMeta)]) -> crate::error::Result<()> {
        let mut s = self.conn.prepare_cached("INSERT OR REPLACE INTO file_meta (rel_path, mtime_ns, size) VALUES (?1, ?2, ?3)")?;
        for (p, m) in entries { s.execute(params![p, m.mtime_ns, m.size])?; }
        Ok(())
    }

    fn remove_files(&mut self, rel_paths: &[String]) -> crate::error::Result<()> {
        let tx = self.conn.transaction()?;
        for path in rel_paths {
            tx.execute("DELETE FROM file_meta WHERE rel_path = ?1", params![path])?;
            let mut s = tx.prepare("SELECT rowid, name, source FROM chunks WHERE source_path = ?1")?;
            let chunks: Vec<(i64, String, String)> = s.query_map(params![path], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?.filter_map(|r| r.ok()).collect();
            for (rid, name, source) in chunks {
                tx.execute("INSERT INTO chunks_fts(chunks_fts, rowid, name, source) VALUES('delete', ?1, ?2, ?3)", params![rid, name, source])?;
            }
            tx.execute("DELETE FROM chunks WHERE source_path = ?1", params![path])?;
        }
        tx.commit()?; Ok(())
    }

    fn remove_chunks_for_file(&self, rel_path: &str) -> crate::error::Result<()> {
        let mut s = self.conn.prepare("SELECT rowid, name, source FROM chunks WHERE source_path = ?1")?;
        let chunks: Vec<(i64, String, String)> = s.query_map(params![rel_path], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?.filter_map(|r| r.ok()).collect();
        for (rid, name, source) in chunks {
            self.conn.execute("INSERT INTO chunks_fts(chunks_fts, rowid, name, source) VALUES('delete', ?1, ?2, ?3)", params![rid, name, source])?;
        }
        self.conn.execute("DELETE FROM chunks WHERE source_path = ?1", params![rel_path])?;
        Ok(())
    }

    fn bm25_search(&self, query: &str, limit: usize) -> crate::error::Result<Vec<(String, f32)>> {
        let fts_query = sanitize_fts_query(query);
        if fts_query.is_empty() { return Ok(Vec::new()); }
        let mut s = self.conn.prepare_cached("SELECT c.id, bm25(chunks_fts) AS score FROM chunks_fts JOIN chunks c ON chunks_fts.rowid = c.rowid WHERE chunks_fts MATCH ?1 ORDER BY bm25(chunks_fts) LIMIT ?2")?;
        let r: Vec<(String, f64)> = s.query_map(params![fts_query, limit as i64], |row| Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?)))?.filter_map(|r| r.ok()).collect();
        if r.is_empty() { return Ok(Vec::new()); }
        let b = r[0].1;
        Ok(r.into_iter().map(|(id, s)| (id, if b == 0.0 { 0.0 } else { (s / b) as f32 })).collect())
    }

    fn upsert_note(&self, note: &Note) -> crate::error::Result<()> {
        self.conn.execute("INSERT OR REPLACE INTO notes (id, text, tags, timestamp, vector) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![note.id, note.text, serde_json::to_string(&note.tags).unwrap(), note.timestamp, encode_vector(&note.vector)])?;
        Ok(())
    }
    fn load_notes(&self) -> crate::error::Result<Vec<Note>> {
        let mut s = self.conn.prepare("SELECT id, text, tags, timestamp, vector FROM notes")?;
        let r = s.query_map([], |row| {
            let ts_str: String = row.get(3)?;
            Ok(Note { id: row.get(0)?, text: row.get(1)?, tags: serde_json::from_str(&row.get::<_, String>(2)?).unwrap_or_default(),
                timestamp: ts_str.parse().unwrap_or(0), vector: decode_vector(&row.get::<_, Vec<u8>>(4)?) })
        })?;
        r.collect::<rusqlite::Result<Vec<_>>>().map_err(|e| e.into())
    }
}

use super::{encode_vector, decode_vector};
fn sanitize_fts_query(q: &str) -> String {
    let stopwords = ["for", "the", "and", "but", "not", "with", "this", "that", "code", "implement", "implemented"];
    q.split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|t| !t.is_empty() && !stopwords.contains(&t.to_lowercase().as_str()))
        .map(|t| format!("\"{t}\""))
        .collect::<Vec<_>>()
        .join(" OR ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::ChunkKind;

    fn tempdir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "slocate_test_store_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .subsec_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_bm25_or_logic() -> crate::error::Result<()> {
        let dir = tempdir();
        let db = SqliteDb::open(&dir)?;

        let c1 = Chunk {
            id: "1".into(),
            kind: ChunkKind::Function,
            name: "log2".into(),
            source_path: "arm.rs".into(),
            source: "fn log2(self) { // NEON implementation }".into(),
        };
        let c2 = Chunk {
            id: "2".into(),
            kind: ChunkKind::Function,
            name: "test".into(),
            source_path: "test.rs".into(),
            source: "fn test() { // some other code }".into(),
        };

        db.insert_chunk(&c1, &[0.1; 384])?;
        db.insert_chunk(&c2, &[0.2; 384])?;

        let results = db.bm25_search("log2 missing_word", 10)?;
        assert_eq!(results.len(), 1, "Expected 1 result for 'log2 missing_word', got {:?}", results);
        assert_eq!(results[0].0, "1");

        let results = db.bm25_search("log2 neon", 10)?;
        assert_eq!(results.len(), 1, "Expected 1 result for 'log2 neon', got {:?}", results);
        assert_eq!(results[0].0, "1");

        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }
}
