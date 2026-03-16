use std::collections::HashMap;
use crate::parse::ChunkKind;
use crate::error::Result;

#[derive(Debug, Clone, serde::Serialize)]
pub struct Chunk {
    pub id: String,
    pub kind: ChunkKind,
    pub name: String,
    pub source_path: String,
    pub source: String,
}

#[derive(Debug, Clone, Copy)]
pub struct FileMeta { pub mtime_ns: i64, pub size: i64 }

#[derive(Debug, Clone)]
pub struct Note { pub id: String, pub text: String, pub tags: Vec<String>, pub timestamp: i64, pub vector: Vec<f32> }

/// Storage backend. Implement this to swap SQLite for another store (BigQuery,
/// Postgres, etc.).
///
/// `begin`/`commit`/`rollback` default to no-ops for backends without
/// explicit transaction support.
pub trait Db {
    fn begin(&self) -> Result<()> { Ok(()) }
    fn commit(&self) -> Result<()> { Ok(()) }
    fn rollback(&self) -> Result<()> { Ok(()) }

    fn insert_chunk(&self, c: &Chunk, v: &[f32]) -> Result<()>;
    fn load_all_chunks_with_vectors(&self) -> Result<Vec<(Chunk, Vec<f32>)>>;
    fn get_chunks_by_ids(&self, ids: &[String]) -> Result<Vec<Chunk>>;
    fn get_chunks_with_vectors_by_ids(&self, ids: &[String]) -> Result<Vec<(Chunk, Vec<f32>)>>;
    fn remove_chunks_for_file(&self, rel_path: &str) -> Result<()>;
    fn remove_files(&mut self, rel_paths: &[String]) -> Result<()>;

    fn cache_put(&self, entries: &[(String, Vec<f32>)]) -> Result<()>;
    fn cache_get_batch(&self, hashes: &[String]) -> Result<HashMap<String, Vec<f32>>>;
    fn cache_gc(&self, max_age_days: u32) -> Result<usize>;
    fn cache_count(&self) -> Result<usize>;

    fn ensure_file_meta_table(&self) -> Result<()>;
    fn load_file_meta(&self) -> Result<HashMap<String, FileMeta>>;
    fn update_file_meta(&self, entries: &[(String, FileMeta)]) -> Result<()>;

    fn bm25_search(&self, query: &str, limit: usize) -> Result<Vec<(String, f32)>>;

    fn save_meta(&self, key: &str, value: &str) -> Result<()>;
    fn get_meta(&self, key: &str) -> Result<Option<String>>;

    fn upsert_note(&self, note: &Note) -> Result<()>;
    fn load_notes(&self) -> Result<Vec<Note>>;
}

pub mod sqlite;
pub use sqlite::SqliteDb;

/// Stable 64-bit FNV chunk identifier derived from file path + name + kind.
pub fn chunk_id(path: &str, name: &str, kind: &str) -> String {
    let mut h: u64 = 5381;
    for b in path.bytes().chain(name.bytes()).chain(kind.bytes()) {
        h = h.wrapping_mul(33).wrapping_add(b as u64);
    }
    format!("{:016x}", h)
}
