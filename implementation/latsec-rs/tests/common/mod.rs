//! Helpers shared by the integration tests.
#![allow(dead_code)]

use std::sync::Arc;

use latsec::latsec::SessionId;
use latsec::profile::Profile;
use latsec::sampling::SeededRandom;
use latsec::scheme::{ActiveSet, PartyId, Scheme};
use latsec::sim::Deployment;

pub fn repo_path(rel: &str) -> String {
    format!("{}/../{}", env!("CARGO_MANIFEST_DIR"), rel)
}

pub fn load_profile(id: &str) -> Profile {
    Profile::load(repo_path(&format!("profiles/{id}.json"))).unwrap()
}

pub fn scheme(id: &str) -> Arc<Scheme> {
    Arc::new(Scheme::new(load_profile(id)).unwrap())
}

/// A deployment with keys derived from `seed`, and the stream that continues it.
pub fn deployment(id: &str, seed: &str) -> (Deployment, SeededRandom) {
    let mut rng = SeededRandom::new(seed.as_bytes(), "test-deployment");
    let deployment = Deployment::keygen(scheme(id), &mut rng);
    (deployment, rng)
}

pub fn sid(tag: u64) -> SessionId {
    let mut bytes = [0u8; 32];
    bytes[..8].copy_from_slice(&tag.to_le_bytes());
    SessionId(bytes)
}

pub fn set(parties: &[PartyId]) -> ActiveSet {
    ActiveSet::new(parties).unwrap()
}

/// All subsets of `{0, .., n-1}` of the given size, in lexicographic order.
pub fn subsets(n: usize, size: usize) -> Vec<Vec<PartyId>> {
    fn extend(n: usize, size: usize, start: usize, current: &mut Vec<PartyId>, out: &mut Vec<Vec<PartyId>>) {
        if current.len() == size {
            out.push(current.clone());
            return;
        }
        for next in start..n {
            current.push(next as PartyId);
            extend(n, size, next + 1, current, out);
            current.pop();
        }
    }
    let mut out = Vec::new();
    extend(n, size, 0, &mut Vec::new(), &mut out);
    out
}
