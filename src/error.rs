use std::fmt;

/// Unified error type for slocate.
///
/// Each variant maps to a conceptual domain. `Display` produces human-readable
/// messages suitable for stderr / MCP error responses.
#[derive(Debug)]
pub enum Error {
    /// Filesystem, process, or general I/O errors.
    Io(std::io::Error),
    /// SQLite / rusqlite errors.
    Db(rusqlite::Error),
    /// Embedding model errors (tokenization, forward pass, tensor ops).
    Embed(String),
    /// Configuration loading or serialization errors.
    Config(String),
    /// Model download errors (HTTP, write).
    Download(String),
    /// Source code parsing errors (tree-sitter).
    Parse(String),
    /// Something expected was not found (workspace, argument, field).
    NotFound(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::Db(e) => write!(f, "database error: {e}"),
            Error::Embed(msg) => write!(f, "embed error: {msg}"),
            Error::Config(msg) => write!(f, "config error: {msg}"),
            Error::Download(msg) => write!(f, "download error: {msg}"),
            Error::Parse(msg) => write!(f, "parse error: {msg}"),
            Error::NotFound(msg) => write!(f, "not found: {msg}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            Error::Db(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<rusqlite::Error> for Error {
    fn from(e: rusqlite::Error) -> Self {
        Error::Db(e)
    }
}
