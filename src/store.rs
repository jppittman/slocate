use rusqlite::{Connection, params, OptionalExtension};
use std::path::Path;
use crate::parse::ChunkKind;
use crate::vdb::{Hnsw, SimFn};

#[derive(Debug, Clone, serde::Serialize)]
pub struct Chunk {
    pub id: String,
    pub kind: ChunkKind,
    pub name: String,
    pub source_path: String,
    pub source: String,
    pub community_id: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
pub struct FileMeta { pub mtime_ns: i64, pub size: i64 }

#[derive(Debug, Clone)]
pub struct Note { pub id: String, pub text: String, pub tags: Vec<String>, pub timestamp: i64, pub vector: Vec<f32> }

#[derive(Debug, Clone)]
pub struct Bundle {
    pub id: i64, pub parent_id: Option<i64>, pub level: i32,
    pub vector: Vec<f32>, pub vector_sum: Vec<f32>,
    pub chunk_count: i32, pub hubs: String,
}

pub struct Store { conn: Connection }

impl Store {
    pub fn open(index_dir: &Path) -> crate::error::Result<Self> {
        let db_path = index_dir.join("index.db");
        let conn = Connection::open(&db_path)?;
        conn.pragma(None, "journal_mode", "WAL", |_| Ok(()))?;
        conn.pragma(None, "synchronous", "NORMAL", |_| Ok(()))?;
        conn.execute_batch("
            CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, kind TEXT NOT NULL, name TEXT NOT NULL, source_path TEXT NOT NULL, source TEXT NOT NULL, community_id INTEGER, vector BLOB);
            CREATE TABLE IF NOT EXISTS bundles (id INTEGER PRIMARY KEY, parent_id INTEGER, level INTEGER NOT NULL, vector BLOB NOT NULL, vector_sum BLOB NOT NULL, chunk_count INTEGER NOT NULL, hubs TEXT, FOREIGN KEY(parent_id) REFERENCES bundles(id));
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS hnsw_nodes (idx INTEGER PRIMARY KEY, id TEXT NOT NULL, vector BLOB NOT NULL, level INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS hnsw_edges (node_idx INTEGER NOT NULL, layer INTEGER NOT NULL, neighbor_idx INTEGER NOT NULL, PRIMARY KEY (node_idx, layer, neighbor_idx));
            CREATE TABLE IF NOT EXISTS hnsw_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS embed_cache (content_hash TEXT PRIMARY KEY, vector BLOB NOT NULL, created_at INTEGER NOT NULL DEFAULT (strftime('%s','now')));
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(name, source, content='chunks', content_rowid='rowid');
            CREATE TABLE IF NOT EXISTS chunk_secondary_communities (chunk_id TEXT NOT NULL, bundle_id INTEGER NOT NULL, PRIMARY KEY (chunk_id, bundle_id));
            CREATE TABLE IF NOT EXISTS leiden_edges (node_id TEXT NOT NULL, neighbor_id TEXT NOT NULL, weight REAL NOT NULL, PRIMARY KEY (node_id, neighbor_id));
            CREATE INDEX IF NOT EXISTS leiden_edges_node ON leiden_edges(node_id);
            CREATE TABLE IF NOT EXISTS leiden_dirty (chunk_id TEXT PRIMARY KEY, marked_at INTEGER NOT NULL DEFAULT (unixepoch()));
        ")?;
        Ok(Self { conn })
    }

    pub fn begin(&self) -> crate::error::Result<()> { self.conn.execute_batch("BEGIN")?; Ok(()) }
    pub fn commit(&self) -> crate::error::Result<()> { self.conn.execute_batch("COMMIT")?; Ok(()) }
    pub fn rollback(&self) -> crate::error::Result<()> { self.conn.execute_batch("ROLLBACK")?; Ok(()) }

    pub fn save_meta(&self, key: &str, value: &str) -> crate::error::Result<()> {
        self.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)", params![key, value])?;
        Ok(())
    }
    pub fn get_meta(&self, key: &str) -> crate::error::Result<Option<String>> {
        let mut s = self.conn.prepare("SELECT value FROM meta WHERE key = ?1")?;
        let r = s.query_row(params![key], |row| row.get(0)).optional()?;
        Ok(r)
    }

    pub fn cache_put(&self, entries: &[(String, Vec<f32>)]) -> crate::error::Result<()> {
        let mut s = self.conn.prepare_cached("INSERT OR REPLACE INTO embed_cache (content_hash, vector) VALUES (?1, ?2)")?;
        for (h, v) in entries { s.execute(params![h, encode_vector(v)])?; }
        Ok(())
    }
    pub fn cache_get_batch(&self, hashes: &[String]) -> crate::error::Result<std::collections::HashMap<String, Vec<f32>>> {
        let mut out = std::collections::HashMap::new();
        let mut s = self.conn.prepare_cached("SELECT vector FROM embed_cache WHERE content_hash = ?1")?;
        for h in hashes {
            if let Ok(b) = s.query_row(params![h], |row| row.get::<_, Vec<u8>>(0)) {
                out.insert(h.clone(), decode_vector(&b));
            }
        }
        Ok(out)
    }
    pub fn cache_gc(&self, max_age_days: u32) -> crate::error::Result<usize> {
        let cutoff = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64 - (max_age_days as i64 * 86400);
        let r = self.conn.execute("DELETE FROM embed_cache WHERE created_at < ?1", params![cutoff])?;
        Ok(r)
    }
    pub fn cache_count(&self) -> crate::error::Result<usize> {
        let r = self.conn.query_row("SELECT COUNT(*) FROM embed_cache", [], |row| row.get(0))?;
        Ok(r)
    }

    pub fn insert_chunk(&self, c: &Chunk, v: &[f32]) -> crate::error::Result<()> {
        self.conn.execute("INSERT OR REPLACE INTO chunks (id, kind, name, source_path, source, community_id, vector) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![c.id, c.kind.as_str(), c.name, c.source_path, c.source, c.community_id.map(|v| v as i64), encode_vector(v)])?;
        let rowid = self.conn.last_insert_rowid();
        self.conn.execute("INSERT OR REPLACE INTO chunks_fts (rowid, name, source) VALUES (?1, ?2, ?3)",
            params![rowid, c.name, c.source])?;
        Ok(())
    }
    pub fn load_all_chunks_with_vectors(&self) -> crate::error::Result<Vec<(Chunk, Vec<f32>)>> {
        let mut s = self.conn.prepare("SELECT id, kind, name, source_path, source, community_id, vector FROM chunks WHERE vector IS NOT NULL")?;
        let r = s.query_map([], |row| Ok((Chunk {
            id: row.get(0)?, kind: ChunkKind::from_db_str(&row.get::<_, String>(1)?),
            name: row.get(2)?, source_path: row.get(3)?, source: row.get(4)?,
            community_id: row.get::<_, Option<i64>>(5)?.map(|v| v as usize),
        }, decode_vector(&row.get::<_, Vec<u8>>(6)?))))?;
        r.collect::<rusqlite::Result<Vec<_>>>().map_err(|e| e.into())
    }
    pub fn get_chunks_by_ids(&self, ids: &[String]) -> crate::error::Result<Vec<Chunk>> {
        if ids.is_empty() { return Ok(Vec::new()); }
        let p: String = ids.iter().enumerate().map(|(i, _)| format!("?{}", i + 1)).collect::<Vec<_>>().join(",");
        let sql = format!("SELECT id, kind, name, source_path, source, community_id FROM chunks WHERE id IN ({})", p);
        let mut s = self.conn.prepare(&sql)?;
        let r = s.query_map(rusqlite::params_from_iter(ids), |row| Ok(Chunk {
            id: row.get(0)?, kind: ChunkKind::from_db_str(&row.get::<_, String>(1)?),
            name: row.get(2)?, source_path: row.get(3)?, source: row.get(4)?,
            community_id: row.get::<_, Option<i64>>(5)?.map(|v| v as usize),
        }))?;
        r.collect::<rusqlite::Result<Vec<_>>>().map_err(|e| e.into())
    }
    pub fn get_chunks_with_vectors_by_ids(&self, ids: &[String]) -> crate::error::Result<Vec<(Chunk, Vec<f32>)>> {
        if ids.is_empty() { return Ok(Vec::new()); }
        let p: String = ids.iter().enumerate().map(|(i, _)| format!("?{}", i + 1)).collect::<Vec<_>>().join(",");
        let sql = format!("SELECT id, kind, name, source_path, source, community_id, vector FROM chunks WHERE id IN ({})", p);
        let mut s = self.conn.prepare(&sql)?;
        let r = s.query_map(rusqlite::params_from_iter(ids), |row| Ok((Chunk {
            id: row.get(0)?, kind: ChunkKind::from_db_str(&row.get::<_, String>(1)?),
            name: row.get(2)?, source_path: row.get(3)?, source: row.get(4)?,
            community_id: row.get::<_, Option<i64>>(5)?.map(|v| v as usize),
        }, decode_vector(&row.get::<_, Vec<u8>>(6)?))))?;
        r.collect::<rusqlite::Result<Vec<_>>>().map_err(|e| e.into())
    }

    pub fn clear_bundles(&mut self) -> crate::error::Result<()> {
        let tx = self.conn.transaction()?;
        tx.execute("DELETE FROM bundles", [])?;
        tx.execute("UPDATE chunks SET community_id = NULL", [])?;
        tx.commit()?;
        Ok(())
    }
    pub fn insert_bundle(&self, pid: Option<i64>, lvl: i32, vec: &[f32], sum: &[f32], cnt: i32, hubs: &str) -> crate::error::Result<i64> {
        self.conn.execute("INSERT INTO bundles (parent_id, level, vector, vector_sum, chunk_count, hubs) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![pid, lvl, encode_vector(vec), encode_vector(sum), cnt, hubs])?;
        Ok(self.conn.last_insert_rowid())
    }
    pub fn set_chunks_bundle_id(&self, ids: &[String], bid: i64) -> crate::error::Result<()> {
        if ids.is_empty() { return Ok(()); }
        let p: String = ids.iter().enumerate().map(|(i, _)| format!("?{}", i + 2)).collect::<Vec<_>>().join(",");
        let mut args: Vec<Box<dyn rusqlite::ToSql>> = vec![Box::new(bid)];
        for id in ids { args.push(Box::new(id.clone())); }
        let refs: Vec<&dyn rusqlite::ToSql> = args.iter().map(|a| a.as_ref()).collect();
        self.conn.prepare(&format!("UPDATE chunks SET community_id = ?1 WHERE id IN ({})", p))?.execute(rusqlite::params_from_iter(refs))?;
        Ok(())
    }
    pub fn set_bundles_parent_id(&self, ids: &[i64], pid: i64) -> crate::error::Result<()> {
        if ids.is_empty() { return Ok(()); }
        let p: String = ids.iter().enumerate().map(|(i, _)| format!("?{}", i + 2)).collect::<Vec<_>>().join(",");
        let mut args: Vec<Box<dyn rusqlite::ToSql>> = vec![Box::new(pid)];
        for &id in ids { args.push(Box::new(id)); }
        let refs: Vec<&dyn rusqlite::ToSql> = args.iter().map(|a| a.as_ref()).collect();
        self.conn.prepare(&format!("UPDATE bundles SET parent_id = ?1 WHERE id IN ({})", p))?.execute(rusqlite::params_from_iter(refs))?;
        Ok(())
    }
    pub fn get_bundle_by_id(&self, id: i64) -> crate::error::Result<Bundle> {
        self.conn.query_row("SELECT id, parent_id, level, vector, vector_sum, chunk_count, hubs FROM bundles WHERE id = ?1", params![id], |row| Ok(Bundle {
            id: row.get(0)?, parent_id: row.get(1)?, level: row.get(2)?,
            vector: decode_vector(&row.get::<_, Vec<u8>>(3)?), vector_sum: decode_vector(&row.get::<_, Vec<u8>>(4)?),
            chunk_count: row.get(5)?, hubs: row.get(6)?,
        })).map_err(|e| e.into())
    }
    pub fn get_bundles_by_ids(&self, ids: &[i64]) -> crate::error::Result<Vec<Bundle>> {
        if ids.is_empty() { return Ok(Vec::new()); }
        let p: String = ids.iter().enumerate().map(|(i, _)| format!("?{}", i + 1)).collect::<Vec<_>>().join(",");
        let mut s = self.conn.prepare(&format!("SELECT id, parent_id, level, vector, vector_sum, chunk_count, hubs FROM bundles WHERE id IN ({})", p))?;
        let r = s.query_map(rusqlite::params_from_iter(ids), |row| Ok(Bundle {
            id: row.get(0)?, parent_id: row.get(1)?, level: row.get(2)?,
            vector: decode_vector(&row.get::<_, Vec<u8>>(3)?), vector_sum: decode_vector(&row.get::<_, Vec<u8>>(4)?),
            chunk_count: row.get(5)?, hubs: row.get(6)?,
        }))?;
        r.collect::<rusqlite::Result<Vec<_>>>().map_err(|e| e.into())
    }
    pub fn load_bundles_by_parent(&self, pid: Option<i64>) -> crate::error::Result<Vec<(i64, Vec<f32>, i32, String)>> {
        let (sql, params_vec): (String, Vec<Box<dyn rusqlite::ToSql>>) = if let Some(p) = pid { ("SELECT id, vector, level, hubs FROM bundles WHERE parent_id = ?1".into(), vec![Box::new(p)]) } else { ("SELECT id, vector, level, hubs FROM bundles WHERE parent_id IS NULL".into(), vec![]) };
        let mut s = self.conn.prepare(&sql)?;
        let refs: Vec<&dyn rusqlite::ToSql> = params_vec.iter().map(|a| a.as_ref()).collect();
        let r = s.query_map(rusqlite::params_from_iter(refs), |row| Ok((row.get(0)?, decode_vector(&row.get::<_, Vec<u8>>(1)?), row.get(2)?, row.get(3)?)))?;
        r.collect::<rusqlite::Result<Vec<_>>>().map_err(|e| e.into())
    }
    pub fn load_chunks_with_vectors_by_bundle(&self, bid: i64) -> crate::error::Result<Vec<(Chunk, Vec<f32>)>> {
        let mut s = self.conn.prepare("SELECT id, kind, name, source_path, source, community_id, vector FROM chunks WHERE community_id = ?1")?;
        let r = s.query_map(params![bid], |row| Ok((Chunk {
            id: row.get(0)?, kind: ChunkKind::from_db_str(&row.get::<_, String>(1)?),
            name: row.get(2)?, source_path: row.get(3)?, source: row.get(4)?,
            community_id: row.get::<_, Option<i64>>(5)?.map(|v| v as usize),
        }, decode_vector(&row.get::<_, Vec<u8>>(6)?))))?;
        r.collect::<rusqlite::Result<Vec<_>>>().map_err(|e| e.into())
    }
    pub fn load_chunks_with_vectors_by_bundle_including_secondary(&self, bundle_id: i64) -> crate::error::Result<Vec<(Chunk, Vec<f32>)>> {
        let mut s = self.conn.prepare(
            "SELECT id, kind, name, source_path, source, community_id, vector FROM chunks \
             WHERE vector IS NOT NULL AND (community_id = ?1 OR id IN \
             (SELECT chunk_id FROM chunk_secondary_communities WHERE bundle_id = ?1))"
        )?;
        let r = s.query_map(params![bundle_id], |row| Ok((Chunk {
            id: row.get(0)?, kind: ChunkKind::from_db_str(&row.get::<_, String>(1)?),
            name: row.get(2)?, source_path: row.get(3)?, source: row.get(4)?,
            community_id: row.get::<_, Option<i64>>(5)?.map(|v| v as usize),
        }, decode_vector(&row.get::<_, Vec<u8>>(6)?))))?;
        r.collect::<rusqlite::Result<Vec<_>>>().map_err(|e| e.into())
    }
    pub fn clear_secondary_communities(&self) -> crate::error::Result<()> {
        self.conn.execute("DELETE FROM chunk_secondary_communities", [])?;
        Ok(())
    }
    pub fn insert_secondary_communities(&mut self, pairs: &[(String, i64)]) -> crate::error::Result<()> {
        if pairs.is_empty() { return Ok(()); }
        let tx = self.conn.transaction()?;
        {
            let mut s = tx.prepare_cached("INSERT OR IGNORE INTO chunk_secondary_communities (chunk_id, bundle_id) VALUES (?1, ?2)")?;
            for (chunk_id, bundle_id) in pairs { s.execute(params![chunk_id, bundle_id])?; }
        }
        tx.commit()?;
        Ok(())
    }
    pub fn get_bundle_lineage(&self, bid: i64) -> crate::error::Result<Vec<String>> {
        let mut s = self.conn.prepare("WITH RECURSIVE lineage(id, parent_id, hubs, level) AS (SELECT id, parent_id, hubs, level FROM bundles WHERE id = ?1 UNION ALL SELECT b.id, b.parent_id, b.hubs, b.level FROM bundles b JOIN lineage l ON b.id = l.parent_id) SELECT hubs FROM lineage ORDER BY level DESC")?;
        let r = s.query_map(params![bid], |row| row.get::<_, String>(0))?;
        let all = r.collect::<rusqlite::Result<Vec<_>>>()?;
        // Deduplicate consecutive identical hub labels (same hub promoted through levels).
        let mut out: Vec<String> = Vec::with_capacity(all.len());
        for h in all {
            if out.last().map_or(true, |last| last != &h) {
                out.push(h);
            }
        }
        Ok(out)
    }

    pub fn ensure_file_meta_table(&self) -> crate::error::Result<()> { self.conn.execute_batch("CREATE TABLE IF NOT EXISTS file_meta (rel_path TEXT PRIMARY KEY, mtime_ns INTEGER NOT NULL, size INTEGER NOT NULL);")?; Ok(()) }
    pub fn load_file_meta(&self) -> crate::error::Result<std::collections::HashMap<String, FileMeta>> {
        let mut s = self.conn.prepare("SELECT rel_path, mtime_ns, size FROM file_meta")?;
        let r = s.query_map([], |row| Ok((row.get::<_, String>(0)?, FileMeta { mtime_ns: row.get(1)?, size: row.get(2)? })))?;
        r.collect::<rusqlite::Result<std::collections::HashMap<_, _>>>().map_err(|e| e.into())
    }
    pub fn update_file_meta(&self, entries: &[(String, FileMeta)]) -> crate::error::Result<()> {
        let mut s = self.conn.prepare_cached("INSERT OR REPLACE INTO file_meta (rel_path, mtime_ns, size) VALUES (?1, ?2, ?3)")?;
        for (p, m) in entries { s.execute(params![p, m.mtime_ns, m.size])?; }
        Ok(())
    }
    pub fn get_chunk_ids_for_files(&self, rel_paths: &[String]) -> crate::error::Result<Vec<String>> {
        let mut ids = Vec::new();
        let mut s = self.conn.prepare_cached("SELECT id FROM chunks WHERE source_path = ?1")?;
        for p in rel_paths { let mut rows = s.query(params![p])?; while let Some(row) = rows.next()? { ids.push(row.get(0)?); } }
        Ok(ids)
    }
    pub fn remove_files(&mut self, rel_paths: &[String]) -> crate::error::Result<()> {
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
    pub fn remove_files_all(&self) -> crate::error::Result<()> {
        self.conn.execute("DELETE FROM file_meta", [])?;
        self.conn.execute("DELETE FROM chunks", [])?;
        self.conn.execute("DELETE FROM chunks_fts", [])?;
        Ok(())
    }
    pub fn remove_chunks_for_file(&self, rel_path: &str) -> crate::error::Result<()> { 
        let mut s = self.conn.prepare("SELECT rowid, name, source FROM chunks WHERE source_path = ?1")?;
        let chunks: Vec<(i64, String, String)> = s.query_map(params![rel_path], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?.filter_map(|r| r.ok()).collect();
        for (rid, name, source) in chunks {
            self.conn.execute("INSERT INTO chunks_fts(chunks_fts, rowid, name, source) VALUES('delete', ?1, ?2, ?3)", params![rid, name, source])?;
        }
        self.conn.execute("DELETE FROM chunks WHERE source_path = ?1", params![rel_path])?; 
        Ok(()) 
    }
    pub fn bm25_search(&self, query: &str, limit: usize) -> crate::error::Result<Vec<(String, f32)>> {
        let fts_query = sanitize_fts_query(query); if fts_query.is_empty() { return Ok(Vec::new()); }
        let mut s = self.conn.prepare_cached("SELECT c.id, bm25(chunks_fts) AS score FROM chunks_fts JOIN chunks c ON chunks_fts.rowid = c.rowid WHERE chunks_fts MATCH ?1 ORDER BY bm25(chunks_fts) LIMIT ?2")?;
        let r: Vec<(String, f64)> = s.query_map(params![fts_query, limit as i64], |row| Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?)))?.filter_map(|r| r.ok()).collect();
        if r.is_empty() { return Ok(Vec::new()); }
        let b = r[0].1;
        Ok(r.into_iter().map(|(id, s)| (id, if b == 0.0 { 0.0 } else { (s / b) as f32 })).collect())
    }
    pub fn upsert_note(&self, note: &Note) -> crate::error::Result<()> {
        self.conn.execute("INSERT OR REPLACE INTO notes (id, text, tags, timestamp, vector) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![note.id, note.text, serde_json::to_string(&note.tags).unwrap(), note.timestamp, encode_vector(&note.vector)])?;
        Ok(())
    }
    pub fn load_notes(&self) -> crate::error::Result<Vec<Note>> {
        let mut s = self.conn.prepare("SELECT id, text, tags, timestamp, vector FROM notes")?;
        let r = s.query_map([], |row| {
            let ts_str: String = row.get(3)?;
            Ok(Note { id: row.get(0)?, text: row.get(1)?, tags: serde_json::from_str(&row.get::<_, String>(2)?).unwrap_or_default(),
                timestamp: ts_str.parse().unwrap_or(0), vector: decode_vector(&row.get::<_, Vec<u8>>(4)?) })
        })?;
        r.collect::<rusqlite::Result<Vec<_>>>().map_err(|e| e.into())
    }
    pub fn save_hnsw(&mut self, _hnsw: &Hnsw) -> crate::error::Result<()> { Ok(()) }
    pub fn load_hnsw(&self, sim: SimFn) -> crate::error::Result<Hnsw> { Ok(Hnsw::new(sim)) }

    /// Persist Leiden graph edges. Replaces all existing edges atomically.
    pub fn save_leiden_edges(&mut self, edges: &[(String, String, f32)]) -> crate::error::Result<()> {
        let tx = self.conn.transaction()?;
        tx.execute("DELETE FROM leiden_edges", [])?;
        {
            let mut s = tx.prepare_cached(
                "INSERT INTO leiden_edges (node_id, neighbor_id, weight) VALUES (?1, ?2, ?3)"
            )?;
            for (a, b, w) in edges {
                s.execute(params![a, b, w])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Load all persisted Leiden graph edges.
    pub fn load_leiden_edges(&self) -> crate::error::Result<Vec<(String, String, f32)>> {
        let mut s = self.conn.prepare(
            "SELECT node_id, neighbor_id, weight FROM leiden_edges"
        )?;
        let r = s.query_map([], |row| Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, f64>(2)? as f32,
        )))?;
        r.collect::<rusqlite::Result<Vec<_>>>().map_err(|e| e.into())
    }

    /// Mark chunk IDs as dirty (needing incremental Leiden re-assignment).
    pub fn mark_leiden_dirty(&mut self, chunk_ids: &[String]) -> crate::error::Result<()> {
        if chunk_ids.is_empty() { return Ok(()); }
        let tx = self.conn.transaction()?;
        {
            let mut s = tx.prepare_cached(
                "INSERT OR IGNORE INTO leiden_dirty (chunk_id) VALUES (?1)"
            )?;
            for id in chunk_ids { s.execute(params![id])?; }
        }
        tx.commit()?;
        Ok(())
    }

    /// Return the list of chunk IDs currently marked dirty.
    pub fn get_leiden_dirty(&self) -> crate::error::Result<Vec<String>> {
        let mut s = self.conn.prepare("SELECT chunk_id FROM leiden_dirty")?;
        let r = s.query_map([], |row| row.get::<_, String>(0))?;
        r.collect::<rusqlite::Result<Vec<_>>>().map_err(|e| e.into())
    }

    /// Count of dirty chunk IDs.
    pub fn get_leiden_dirty_count(&self) -> crate::error::Result<usize> {
        let n: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM leiden_dirty", [], |row| row.get(0)
        )?;
        Ok(n as usize)
    }

    /// Total number of chunks in the index.
    pub fn get_total_chunk_count(&self) -> crate::error::Result<usize> {
        let n: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM chunks", [], |row| row.get(0)
        )?;
        Ok(n as usize)
    }

    /// Remove all dirty markers (after incremental Leiden completes).
    pub fn clear_leiden_dirty(&mut self) -> crate::error::Result<()> {
        let tx = self.conn.transaction()?;
        tx.execute("DELETE FROM leiden_dirty", [])?;
        tx.commit()?;
        Ok(())
    }

    /// Remove dirty markers for chunks that no longer exist (cleanup after deletion).
    pub fn gc_leiden_dirty(&mut self) -> crate::error::Result<()> {
        let tx = self.conn.transaction()?;
        tx.execute(
            "DELETE FROM leiden_dirty WHERE chunk_id NOT IN (SELECT id FROM chunks)",
            [],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Remove leiden_edges rows whose node_id or neighbor_id no longer exists in chunks.
    pub fn gc_leiden_edges(&mut self) -> crate::error::Result<()> {
        let tx = self.conn.transaction()?;
        tx.execute(
            "DELETE FROM leiden_edges WHERE node_id NOT IN (SELECT id FROM chunks) \
             OR neighbor_id NOT IN (SELECT id FROM chunks)",
            [],
        )?;
        tx.commit()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let store = Store::open(&dir)?;
        
        let c1 = Chunk {
            id: "1".into(),
            kind: ChunkKind::Function,
            name: "log2".into(),
            source_path: "arm.rs".into(),
            source: "fn log2(self) { // NEON implementation }".into(),
            community_id: None,
        };
        let c2 = Chunk {
            id: "2".into(),
            kind: ChunkKind::Function,
            name: "test".into(),
            source_path: "test.rs".into(),
            source: "fn test() { // some other code }".into(),
            community_id: None,
        };

        store.insert_chunk(&c1, &[0.1; 384])?;
        store.insert_chunk(&c2, &[0.2; 384])?;

        // Query with multiple words, one of which is missing.
        // With "AND" logic (previous), this would return 0 hits.
        // With "OR" logic (new), this should return c1.
        let results = store.bm25_search("log2 missing_word", 10)?;
        assert_eq!(results.len(), 1, "Expected 1 result for 'log2 missing_word', got {:?}", results);
        assert_eq!(results[0].0, "1");

        // Query with multiple present words should still work.
        let results = store.bm25_search("log2 neon", 10)?;
        assert_eq!(results.len(), 1, "Expected 1 result for 'log2 neon', got {:?}", results);
        assert_eq!(results[0].0, "1");

        // Clean up
        let _ = std::fs::remove_dir_all(&dir);

        Ok(())
    }
}

fn encode_vector(v: &[f32]) -> Vec<u8> { use half::f16; v.iter().flat_map(|&f| f16::from_f32(f).to_le_bytes()).collect() }
fn decode_vector(bytes: &[u8]) -> Vec<f32> { use half::f16; bytes.chunks_exact(2).map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32()).collect() }
fn sanitize_fts_query(q: &str) -> String {
    let stopwords = ["for", "the", "and", "but", "not", "with", "this", "that", "code", "implement", "implemented"];
    q.split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|t| !t.is_empty() && !stopwords.contains(&t.to_lowercase().as_str()))
        .map(|t| format!("\"{t}\""))
        .collect::<Vec<_>>()
        .join(" OR ")
}
pub fn chunk_id(path: &str, name: &str, kind: &str) -> String { let mut h: u64 = 5381; for b in path.bytes().chain(name.bytes()).chain(kind.bytes()) { h = h.wrapping_mul(33).wrapping_add(b as u64); } format!("{:016x}", h) }
