use rusqlite::{Connection, params};
use std::path::Path;

use crate::parse::ChunkKind;
use crate::vdb::{Hnsw, HnswNode, SimFn};

// ─── Data types ───────────────────────────────────────────────────────────────

/// Chunk metadata. Vectors live in the HNSW index, not here.
#[derive(Debug, Clone)]
pub struct Chunk {
    pub id: String,
    pub kind: ChunkKind,
    pub name: String,
    pub source_path: String,
    pub source: String,
    /// Leiden community label. `None` until `set_community_ids` is called.
    pub community_id: Option<usize>,
}

/// File-level metadata for incremental reindex change detection.
#[derive(Debug, Clone, Copy)]
pub struct FileMeta {
    pub mtime_ns: i64,
    pub size: i64,
}

#[derive(Debug, Clone)]
pub struct Note {
    pub id: String,
    pub text: String,
    pub tags: Vec<String>,
    pub timestamp: i64,
    pub vector: Vec<f32>, // Notes are few enough for brute-force; no HNSW needed.
}

// ─── Store ────────────────────────────────────────────────────────────────────

pub struct Store {
    conn: Connection,
}

impl Store {
    pub fn open(index_dir: &Path) -> crate::error::Result<Self> {
        let db_path = index_dir.join("index.db");
        let conn = Connection::open(&db_path)?;
        conn.execute_batch("PRAGMA journal_mode=WAL;")?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS chunks (
                id           TEXT PRIMARY KEY,
                kind         TEXT NOT NULL,
                name         TEXT NOT NULL,
                source_path  TEXT NOT NULL,
                source       TEXT NOT NULL,
                community_id INTEGER
            );
            CREATE TABLE IF NOT EXISTS notes (
                id        TEXT PRIMARY KEY,
                text      TEXT NOT NULL,
                tags      TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                vector    BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS hnsw_nodes (
                idx    INTEGER PRIMARY KEY,
                id     TEXT NOT NULL,
                vector BLOB NOT NULL,
                level  INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS hnsw_edges (
                node_idx     INTEGER NOT NULL,
                layer        INTEGER NOT NULL,
                neighbor_idx INTEGER NOT NULL,
                PRIMARY KEY (node_idx, layer, neighbor_idx)
            );
            CREATE TABLE IF NOT EXISTS hnsw_meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS embed_cache (
                content_hash TEXT PRIMARY KEY,
                vector       BLOB NOT NULL,
                created_at   INTEGER NOT NULL DEFAULT (strftime('%s','now'))
            );",
        )?;
        // Migrate existing DBs that predate the community_id column.
        let _ = conn.execute_batch(
            "ALTER TABLE chunks ADD COLUMN community_id INTEGER;",
        );
        Ok(Self { conn })
    }

    // ─── Transactions ────────────────────────────────────────────────────────

    pub fn begin(&self) -> crate::error::Result<()> {
        self.conn.execute_batch("BEGIN")?;
        Ok(())
    }

    pub fn commit(&self) -> crate::error::Result<()> {
        self.conn.execute_batch("COMMIT")?;
        Ok(())
    }

    // ─── Embed cache ──────────────────────────────────────────────────────────

    /// Insert embedding vectors into the cache. Safe inside an explicit txn.
    /// Vectors are stored as f16 (halves storage, negligible loss for normalized vecs).
    pub fn cache_put(&self, entries: &[(String, Vec<f32>)]) -> crate::error::Result<()> {
        let mut stmt = self.conn.prepare_cached(
            "INSERT OR REPLACE INTO embed_cache (content_hash, vector) VALUES (?1, ?2)",
        )?;
        for (hash, vec) in entries {
            let blob = encode_vector(vec);
            stmt.execute(params![hash, blob])?;
        }
        Ok(())
    }

    /// Look up a batch of content hashes in the embed cache.
    /// Returns a map from hash → vector for hits. WAL mode allows concurrent
    /// readers, so this is safe to call from worker threads with their own
    /// read-only Store connection.
    pub fn cache_get_batch(&self, hashes: &[String]) -> crate::error::Result<std::collections::HashMap<String, Vec<f32>>> {
        if hashes.is_empty() {
            return Ok(std::collections::HashMap::new());
        }
        let mut out = std::collections::HashMap::with_capacity(hashes.len());
        let mut stmt = self.conn.prepare_cached(
            "SELECT vector FROM embed_cache WHERE content_hash = ?1",
        )?;
        for h in hashes {
            if let Ok(blob) = stmt.query_row(params![h], |row| {
                let raw: Vec<u8> = row.get(0)?;
                Ok(raw)
            }) {
                let vec = decode_vector(&blob);
                if !vec.is_empty() {
                    out.insert(h.clone(), vec);
                }
            }
        }
        Ok(out)
    }

    /// Delete cache entries older than `max_age_days`. Returns count removed.
    pub fn cache_gc(&self, max_age_days: u32) -> crate::error::Result<usize> {
        let cutoff = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| crate::error::Error::Embed(format!("system time error: {e}")))?
            .as_secs() as i64
            - (max_age_days as i64 * 86400);
        let removed = self.conn.execute(
            "DELETE FROM embed_cache WHERE created_at < ?1",
            params![cutoff],
        )?;
        Ok(removed)
    }

    /// Count of entries in the embed cache.
    pub fn cache_count(&self) -> crate::error::Result<usize> {
        let count = self.conn.query_row(
            "SELECT COUNT(*) FROM embed_cache", [], |row| row.get(0),
        )?;
        Ok(count)
    }

    // ─── Chunks ───────────────────────────────────────────────────────────────

    /// Insert a single chunk.
    pub fn insert_chunk(&self, c: &Chunk) -> crate::error::Result<()> {
        let cid = c.community_id.map(|v| v as i64);
        self.conn.execute(
            "INSERT OR REPLACE INTO chunks (id, kind, name, source_path, source, community_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![c.id, c.kind.as_str(), c.name, c.source_path, c.source, cid],
        )?;
        Ok(())
    }

    /// Fetch specific chunks by id. Order is not guaranteed.
    pub fn get_chunks_by_ids(&self, ids: &[String]) -> crate::error::Result<Vec<Chunk>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        // Build parameterised IN clause.
        let placeholders: String = ids
            .iter()
            .enumerate()
            .map(|(i, _)| format!("?{}", i + 1))
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "SELECT id, kind, name, source_path, source, community_id \
             FROM chunks WHERE id IN ({placeholders})"
        );
        let mut stmt = self.conn.prepare(&sql)?;

        let params: Vec<&dyn rusqlite::ToSql> =
            ids.iter().map(|id| id as &dyn rusqlite::ToSql).collect();

        let rows = stmt.query_map(params.as_slice(), |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, Option<i64>>(5)?,
            ))
        })?;

        rows.map(|r| {
            let (id, kind_str, name, source_path, source, community_id) = r?;
            Ok(Chunk {
                id,
                kind: ChunkKind::from_db_str(&kind_str),
                name,
                source_path,
                source,
                community_id: community_id.map(|v| v as usize),
            })
        })
        .collect()
    }

    /// Bulk-update the `community_id` column for a list of (chunk_id, community_id) pairs.
    pub fn set_community_ids(&mut self, assignments: &[(String, usize)]) -> crate::error::Result<()> {
        let tx = self.conn.transaction()?;
        {
            let mut stmt = tx.prepare("UPDATE chunks SET community_id = ?1 WHERE id = ?2")?;
            for (id, cid) in assignments {
                stmt.execute(params![*cid as i64, id])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    // ─── Notes ────────────────────────────────────────────────────────────────

    pub fn upsert_note(&self, note: &Note) -> crate::error::Result<()> {
        let tags_json = serde_json::to_string(&note.tags)
            .map_err(|e| crate::error::Error::Parse(format!("tags serialization failed: {e}")))?;
        self.conn.execute(
            "INSERT OR REPLACE INTO notes (id, text, tags, timestamp, vector)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                note.id,
                note.text,
                tags_json,
                note.timestamp,
                encode_vector(&note.vector),
            ],
        )?;
        Ok(())
    }

    pub fn load_notes(&self) -> crate::error::Result<Vec<Note>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, text, tags, timestamp, vector FROM notes",
        )?;
        let rows = stmt.query_map([], |row| {
            let ts_str: String = row.get(3)?;
            let ts = ts_str.parse::<i64>().unwrap_or(0);
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                ts,
                row.get::<_, Vec<u8>>(4)?,
            ))
        })?;
        rows.map(|r| {
            let (id, text, tags_json, timestamp, vec_bytes) = r?;
            let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();
            Ok(Note {
                id,
                text,
                tags,
                timestamp,
                vector: decode_vector(&vec_bytes),
            })
        })
        .collect()
    }

    // ─── HNSW persistence ─────────────────────────────────────────────────────

    /// Persist the entire HNSW graph atomically.
    pub fn save_hnsw(&mut self, hnsw: &Hnsw) -> crate::error::Result<()> {
        let tx = self.conn.transaction()?;

        tx.execute("DELETE FROM hnsw_nodes", [])?;
        tx.execute("DELETE FROM hnsw_edges", [])?;
        tx.execute("DELETE FROM hnsw_meta", [])?;

        {
            let mut node_stmt = tx.prepare(
                "INSERT INTO hnsw_nodes (idx, id, vector, level)
                 VALUES (?1, ?2, ?3, ?4)",
            )?;
            let mut edge_stmt = tx.prepare(
                "INSERT INTO hnsw_edges (node_idx, layer, neighbor_idx)
                 VALUES (?1, ?2, ?3)",
            )?;

            for (idx, node) in hnsw.nodes.iter().enumerate() {
                node_stmt.execute(params![
                    idx as i64,
                    node.id,
                    encode_vector(&node.vector),
                    node.level as i64,
                ])?;

                for (layer, neighbors) in node.neighbors.iter().enumerate() {
                    for &nb in neighbors {
                        edge_stmt.execute(params![idx as i64, layer as i64, nb as i64])?;
                    }
                }
            }
        }

        // Store scalar metadata.
        let ep_val = hnsw
            .entry_point
            .map(|i| i.to_string())
            .unwrap_or_else(|| "null".to_string());
        for (k, v) in [
            ("entry_point", ep_val.as_str()),
            ("max_level", &hnsw.max_level.to_string()),
            ("m", &hnsw.m.to_string()),
            ("m_max0", &hnsw.m_max0.to_string()),
            ("ef_construction", &hnsw.ef_construction.to_string()),
        ] {
            tx.execute(
                "INSERT INTO hnsw_meta (key, value) VALUES (?1, ?2)",
                params![k, v],
            )?;
        }

        tx.commit()?;
        Ok(())
    }

    /// Load the HNSW graph from SQLite. Returns an empty index if nothing is saved.
    pub fn load_hnsw(&self, sim: SimFn) -> crate::error::Result<Hnsw> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM hnsw_nodes", [], |r| r.get(0),
        )?;

        if count == 0 {
            return Ok(Hnsw::new(sim));
        }

        // Load scalar metadata.
        let meta = |key: &str| -> crate::error::Result<String> {
            Ok(self.conn.query_row(
                "SELECT value FROM hnsw_meta WHERE key = ?1",
                [key],
                |r| r.get::<_, String>(0),
            )?)
        };

        let entry_point: Option<usize> = match meta("entry_point")?.as_str() {
            "null" => None,
            s => Some(
                s.parse::<usize>()
                    .map_err(|e| crate::error::Error::Parse(format!("parse entry_point failed: {e}")))?,
            ),
        };
        let max_level = meta("max_level")?
            .parse::<usize>()
            .map_err(|e| crate::error::Error::Parse(format!("parse max_level failed: {e}")))?;
        let m = meta("m")?
            .parse::<usize>()
            .map_err(|e| crate::error::Error::Parse(format!("parse m failed: {e}")))?;
        let ef_construction = meta("ef_construction")?
            .parse::<usize>()
            .map_err(|e| crate::error::Error::Parse(format!("parse ef_construction failed: {e}")))?;

        // Load nodes ordered by idx.
        let mut stmt = self.conn.prepare(
            "SELECT idx, id, vector, level FROM hnsw_nodes ORDER BY idx",
        )?;
        let node_rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Vec<u8>>(2)?,
                row.get::<_, i64>(3)?,
            ))
        })?;

        let mut nodes: Vec<HnswNode> = Vec::with_capacity(count as usize);
        for r in node_rows {
            let (_idx, id, vec_bytes, level) = r?;
            let level = level as usize;
            nodes.push(HnswNode {
                id,
                vector: decode_vector(&vec_bytes),
                level,
                neighbors: vec![Vec::new(); level + 1],
            });
        }

        // Load edges and populate neighbor lists.
        let mut edge_stmt = self.conn.prepare(
            "SELECT node_idx, layer, neighbor_idx FROM hnsw_edges",
        )?;
        let edges = edge_stmt.query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
            ))
        })?;

        for r in edges {
            let (node_idx, layer, neighbor_idx) = r?;
            let ni = node_idx as usize;
            let li = layer as usize;
            if ni < nodes.len() && li < nodes[ni].neighbors.len() {
                nodes[ni].neighbors[li].push(neighbor_idx as usize);
            }
        }

        Ok(Hnsw::from_saved(nodes, entry_point, max_level, m, ef_construction, sim))
    }
}

// ─── File metadata for incremental reindex ───────────────────────────────────

impl Store {
    /// Ensure the file_meta table exists (migration for older DBs).
    pub fn ensure_file_meta_table(&self) -> crate::error::Result<()> {
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS file_meta (
                rel_path  TEXT PRIMARY KEY,
                mtime_ns  INTEGER NOT NULL,
                size      INTEGER NOT NULL
            );",
        )?;
        Ok(())
    }

    /// Load all file metadata into a map for fast lookup.
    pub fn load_file_meta(&self) -> crate::error::Result<std::collections::HashMap<String, FileMeta>> {
        let mut stmt = self.conn.prepare(
            "SELECT rel_path, mtime_ns, size FROM file_meta",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
            ))
        })?;
        let mut map = std::collections::HashMap::new();
        for r in rows {
            let (path, mtime_ns, size) = r?;
            map.insert(path, FileMeta { mtime_ns, size });
        }
        Ok(map)
    }

    /// Replace file metadata for a batch of files (within a transaction).
    /// Insert/replace file metadata entries.
    ///
    /// Safe to call inside an explicit `begin()`/`commit()` block — does NOT
    /// start its own transaction. If called outside a transaction, each row is
    /// auto-committed individually.
    pub fn update_file_meta(&self, entries: &[(String, FileMeta)]) -> crate::error::Result<()> {
        let mut stmt = self.conn.prepare_cached(
            "INSERT OR REPLACE INTO file_meta (rel_path, mtime_ns, size)
             VALUES (?1, ?2, ?3)",
        )?;
        for (path, meta) in entries {
            stmt.execute(params![path, meta.mtime_ns, meta.size])?;
        }
        Ok(())
    }

    /// Remove file metadata and chunks for deleted files.
    pub fn remove_files(&mut self, rel_paths: &[String]) -> crate::error::Result<()> {
        if rel_paths.is_empty() {
            return Ok(());
        }
        let tx = self.conn.transaction()?;
        for path in rel_paths {
            tx.execute("DELETE FROM file_meta WHERE rel_path = ?1", params![path])?;
            tx.execute("DELETE FROM chunks WHERE source_path = ?1", params![path])?;
        }
        tx.commit()?;
        Ok(())
    }

    /// Remove chunks for a given file (before re-inserting updated ones).
    pub fn remove_chunks_for_file(&self, rel_path: &str) -> crate::error::Result<()> {
        self.conn.execute(
            "DELETE FROM chunks WHERE source_path = ?1", params![rel_path],
        )?;
        Ok(())
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// djb2 hash over concatenated bytes, formatted as 16 hex chars.
pub fn chunk_id(path: &str, name: &str, kind: &str) -> String {
    let mut h: u64 = 5381;
    for b in path.bytes().chain(name.bytes()).chain(kind.bytes()) {
        h = h.wrapping_mul(33).wrapping_add(b as u64);
    }
    format!("{:016x}", h)
}

/// Encode f32 vectors as f16 on disk — halves storage with negligible
/// precision loss for L2-normalized vectors in [-1, 1].
fn encode_vector(v: &[f32]) -> Vec<u8> {
    use half::f16;
    v.iter()
        .flat_map(|&f| f16::from_f32(f).to_le_bytes())
        .collect()
}

fn decode_vector(bytes: &[u8]) -> Vec<f32> {
    use half::f16;
    bytes
        .chunks_exact(2)
        .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect()
}
