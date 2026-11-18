//! Deterministic test vectors: one seeded key generation and signing session per
//! profile, recorded as digests of the canonical encodings (and, for small
//! profiles, the encodings themselves).
//!
//! The vectors pin down the whole pipeline: seeded randomness, the exact sampler's
//! use of it, matrix expansion, reconstruction, hashing, challenge expansion and
//! the codec.  Any change to one of them changes a digest.

use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::latsec::{verify, SessionId, SessionMessage};
use crate::sampling::{RandomSource, SeededRandom};
use crate::scheme::{ActiveSet, PartyId, Scheme};
use crate::sim::{Deployment, SessionOutcome};
use crate::stats::random_subset;
use crate::xof::Xof;

/// Domain of the digests in a test vector.
const KAT_DOMAIN: &str = "LaTSEC-v1/test-vector";
/// Encodings up to this many bytes are stored in full.
const INLINE_LIMIT: usize = 1024;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn digest(parts: &[Vec<u8>]) -> String {
    let mut xof = Xof::new(KAT_DOMAIN);
    for part in parts {
        xof.absorb(part);
    }
    hex(&xof.digest(32))
}

/// A recorded session.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TestVector {
    pub format: String,
    pub profile: String,
    pub profile_digest: String,
    /// Seed of the deterministic randomness (`SeededRandom` with label `test-vector`).
    pub seed: String,
    /// The signer set: the first admissible set of `t_sig` parties drawn from the stream.
    pub signers: Vec<PartyId>,
    pub session_id: String,
    /// The application payload, as text.
    pub payload: String,
    /// `lambda_act`, as carried in the session message.
    pub lambda: Vec<i64>,
    /// The encoded session message `msg` (header, `sid`, `act`, `lambda_act`, payload).
    pub msg: String,
    /// Sizes in bytes of the canonical encodings.
    pub bytes_msg: usize,
    pub bytes_vk: usize,
    pub bytes_abort_key: usize,
    pub bytes_share: usize,
    pub bytes_commitment: usize,
    pub bytes_opening: usize,
    pub bytes_partial_response: usize,
    pub bytes_signature: usize,
    /// Digests of the encodings (`SHAKE256`, domain `LaTSEC-v1/test-vector`, 32 bytes).
    pub digest_vk: String,
    pub digest_abort_keys: String,
    pub digest_shares: String,
    pub digest_commitments: String,
    pub digest_openings: String,
    pub digest_partial_responses: String,
    pub digest_signature: String,
    /// The encodings themselves when they are short.
    pub vk: Option<String>,
    pub signature: Option<String>,
}

impl TestVector {
    pub const FORMAT: &'static str = "latsec-test-vector-v2";

    /// Runs key generation and one signing session with randomness derived from `seed`.
    pub fn generate(scheme: Arc<Scheme>, seed: &str) -> Result<TestVector> {
        let mut rng = SeededRandom::new(seed.as_bytes(), "test-vector");
        let mut deployment = Deployment::keygen(Arc::clone(&scheme), &mut rng);
        let profile = scheme.profile();
        let mut act = None;
        for _ in 0..10_000 {
            let parties: Vec<PartyId> =
                random_subset(&mut rng, profile.sec.n, profile.sec.t_sig).into_iter().map(|p| p as PartyId).collect();
            let candidate = ActiveSet::new(&parties)?;
            if scheme.lambda(&candidate).is_ok() {
                act = Some(candidate);
                break;
            }
        }
        let act = act.ok_or_else(|| Error::invalid("no admissible signer set found"))?;
        let mut sid = [0u8; 32];
        rng.fill(&mut sid);
        let payload = format!("test vector for {}", profile.id);
        let shares: Vec<Vec<u8>> =
            deployment.signers.iter().map(|s| s.share().to_bytes(&scheme)).collect::<Result<_>>()?;
        let (outcome, _) = deployment
            .sign(SessionId(sid), &act, payload.as_bytes(), &mut rng)
            .map_err(|e| Error::invalid(format!("signing failed: {e}")))?;
        let SessionOutcome::Signed { signature, transcript } = outcome else {
            return Err(Error::invalid("honest session aborted"));
        };
        if !verify(&scheme, &deployment.vk, &transcript.msg, &signature) {
            return Err(Error::invalid("signature does not verify"));
        }
        let vk = deployment.vk.to_bytes(&scheme);
        let abort_keys: Vec<Vec<u8>> = deployment.abort_keys.iter().map(|ak| ak.to_bytes(&scheme)).collect();
        let commitments: Vec<Vec<u8>> = transcript.commitments.iter().map(|c| c.0.clone()).collect();
        let openings: Vec<Vec<u8>> = transcript.openings.iter().map(|o| o.to_bytes(&scheme)).collect();
        let responses: Vec<Vec<u8>> =
            transcript.responses.iter().map(|r| r.to_bytes(&scheme)).collect::<Result<_>>()?;
        let sig = signature.to_bytes(&scheme)?;
        let inline = |bytes: &[u8]| (bytes.len() <= INLINE_LIMIT).then(|| hex(bytes));
        Ok(TestVector {
            format: Self::FORMAT.to_string(),
            profile: profile.id.clone(),
            profile_digest: profile.digest.clone(),
            seed: seed.to_string(),
            signers: act.parties().to_vec(),
            session_id: hex(&sid),
            payload,
            lambda: SessionMessage::from_bytes(&scheme, &transcript.msg)?.lambda,
            msg: hex(&transcript.msg),
            bytes_msg: transcript.msg.len(),
            bytes_vk: vk.len(),
            bytes_abort_key: abort_keys[0].len(),
            bytes_share: shares[0].len(),
            bytes_commitment: commitments[0].len(),
            bytes_opening: openings[0].len(),
            bytes_partial_response: responses[0].len(),
            bytes_signature: sig.len(),
            digest_vk: digest(std::slice::from_ref(&vk)),
            digest_abort_keys: digest(&abort_keys),
            digest_shares: digest(&shares),
            digest_commitments: digest(&commitments),
            digest_openings: digest(&openings),
            digest_partial_responses: digest(&responses),
            digest_signature: digest(std::slice::from_ref(&sig)),
            vk: inline(&vk),
            signature: inline(&sig),
        })
    }

    pub fn load(path: impl AsRef<Path>) -> Result<TestVector> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|e| Error::Io(path.display().to_string(), e))?;
        let vector: TestVector = serde_json::from_str(&text)?;
        if vector.format != Self::FORMAT {
            return Err(Error::invalid("unexpected test-vector format"));
        }
        Ok(vector)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let mut text = serde_json::to_string_pretty(self)?;
        // Keep the integer lists on one line each.
        for field in ["signers", "lambda"] {
            if let Some(start) = text.find(&format!("\"{field}\": [")) {
                let end = start + text[start..].find(']').expect("closing bracket");
                let compact = text[start..end].split_whitespace().collect::<Vec<_>>().join(" ").replace("[ ", "[");
                text.replace_range(start..end, &compact);
            }
        }
        text.push('\n');
        std::fs::write(path, text).map_err(|e| Error::Io(path.display().to_string(), e))
    }
}
