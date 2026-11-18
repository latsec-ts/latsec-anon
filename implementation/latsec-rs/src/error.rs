//! Error type shared by the crate.

use std::fmt;

/// Errors from loading, validating or using public data.
///
/// Protocol-level outcomes (invocation errors and identifiable aborts) have their
/// own types in [`crate::latsec`].
#[derive(Debug)]
pub enum Error {
    /// An I/O failure, with the path involved.
    Io(String, std::io::Error),
    /// Malformed JSON or a JSON document of the wrong shape.
    Json(String),
    /// Data that parses but violates a stated requirement.
    Invalid(String),
    /// A malformed or non-canonical binary encoding.
    Decode(String),
}

impl Error {
    pub fn invalid(msg: impl Into<String>) -> Self {
        Error::Invalid(msg.into())
    }

    pub fn decode(msg: impl Into<String>) -> Self {
        Error::Decode(msg.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(path, err) => write!(f, "{path}: {err}"),
            Error::Json(msg) => write!(f, "JSON error: {msg}"),
            Error::Invalid(msg) => write!(f, "invalid data: {msg}"),
            Error::Decode(msg) => write!(f, "decoding error: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Error::Json(err.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;
