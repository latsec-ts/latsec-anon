//! Loading of the SEC matrices extracted by `tools/extract_fixtures.py`.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::intmat::IntMatrix;

/// Where an extracted matrix came from.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct MatrixSource {
    /// The notebook the matrix was read from: a path in the artifact, or a
    /// description if the notebook is not part of it.
    pub notebook: String,
    pub cells: BTreeMap<String, u32>,
    /// SHA-256 of the sources of those cells; identifies the matrix data.
    pub cell_sources_sha256: String,
    pub index_base: String,
}

/// An SEC matrix of the paper or of the SEC paper [46], with its planted supports.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct AuthorMatrix {
    pub format: String,
    pub name: String,
    pub source: MatrixSource,
    pub n: usize,
    pub m: usize,
    pub k: usize,
    pub c_exp: usize,
    /// The expander parameter `gamma`, as written in the notebook (`"5/40"`).
    pub gamma: String,
    /// The expander parameter `eps_exp`, as written in the notebook (`"0.48"`).
    pub eps_exp: String,
    pub k_p: usize,
    /// The matrix `G = [g | G_hat]`, one row per party.
    pub g: Vec<Vec<i64>>,
    /// Zero-based planted supports ([46, Algorithm 2]).
    pub planted_supports: Vec<Vec<usize>>,
    /// The witness of each planted support, in the order of its support.
    pub planted_witnesses: Vec<Vec<i64>>,
}

impl AuthorMatrix {
    pub const FORMAT: &'static str = "latsec-author-matrix-v1";

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|e| Error::Io(path.display().to_string(), e))?;
        Self::from_json(&text)
    }

    pub fn from_json(text: &str) -> Result<Self> {
        let matrix: AuthorMatrix = serde_json::from_str(text)?;
        if matrix.format != Self::FORMAT {
            return Err(Error::invalid(format!("unexpected fixture format {}", matrix.format)));
        }
        if matrix.k != matrix.m + 1 || matrix.g.len() != matrix.n || matrix.g.iter().any(|r| r.len() != matrix.k) {
            return Err(Error::invalid("matrix dimensions do not match n, m, k"));
        }
        if matrix.planted_supports.len() != matrix.planted_witnesses.len() {
            return Err(Error::invalid("supports and witnesses differ in number"));
        }
        Ok(matrix)
    }

    pub fn matrix(&self) -> IntMatrix {
        IntMatrix::from_i64(self.k, &self.g)
    }
}
