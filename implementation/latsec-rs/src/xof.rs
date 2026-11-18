//! SHAKE256 with domain separation and unambiguous input framing.
//!
//! Every hash in the protocol is `SHAKE256(frame(domain) || frame(x_1) || ...)`,
//! where `frame(x)` is the byte length of `x` as a little-endian `u64` followed by
//! `x`.  Distinct domains therefore give independent functions, and no two distinct
//! input tuples collide as byte strings.

use sha3::digest::{ExtendableOutput, Update, XofReader};
use sha3::Shake256;

/// The domain strings used by this crate.
pub mod domain {
    /// Digest of a profile's canonical encoding.
    pub const PROFILE: &str = "LaTSEC-v1/profile";
    /// Expansion of the public matrix `A'` from its seed.
    pub const EXPAND_A: &str = "LaTSEC-v1/expand-A";
    /// Digest of a verification key.
    pub const VERIFICATION_KEY: &str = "LaTSEC-v1/vk";
    /// Digest of a reconstruction vector.
    pub const LAMBDA: &str = "LaTSEC-v1/lambda";
    /// Round-1 commitments `H_cmt`.
    pub const COMMITMENT: &str = "LaTSEC-v1/commit";
    /// The challenge hash `H_c`, whose stream feeds the sample-in-ball shuffle.
    pub const CHALLENGE: &str = "LaTSEC-v1/challenge";
    /// Deterministic randomness for fixtures and instance generation.
    pub const FIXTURE_RANDOMNESS: &str = "LaTSEC-v1/fixture-randomness";
}

/// An absorbing SHAKE256 instance with framed inputs.
#[derive(Clone)]
pub struct Xof {
    state: Shake256,
}

impl Xof {
    pub fn new(domain: &str) -> Self {
        let mut xof = Xof { state: Shake256::default() };
        xof.absorb(domain.as_bytes());
        xof
    }

    /// Absorbs one length-framed field.
    pub fn absorb(&mut self, data: &[u8]) -> &mut Self {
        self.state.update(&(data.len() as u64).to_le_bytes());
        self.state.update(data);
        self
    }

    /// Absorbs an integer as an eight-byte little-endian field.
    pub fn absorb_u64(&mut self, value: u64) -> &mut Self {
        self.absorb(&value.to_le_bytes())
    }

    pub fn finalize(self) -> XofOutput {
        XofOutput { reader: Box::new(self.state.finalize_xof()) }
    }

    /// Finalizes and returns the first `len` output bytes.
    pub fn digest(self, len: usize) -> Vec<u8> {
        let mut out = vec![0u8; len];
        self.finalize().read(&mut out);
        out
    }
}

/// The output stream of a finalized [`Xof`].
pub struct XofOutput {
    reader: Box<dyn XofReader + Send>,
}

impl XofOutput {
    pub fn read(&mut self, buf: &mut [u8]) {
        self.reader.read(buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn framing_separates_inputs() {
        let mut a = Xof::new("d");
        a.absorb(b"ab").absorb(b"c");
        let mut b = Xof::new("d");
        b.absorb(b"a").absorb(b"bc");
        assert_ne!(a.digest(32), b.digest(32));
        assert_ne!(Xof::new("d1").digest(32), Xof::new("d2").digest(32));
    }

    #[test]
    fn matches_plain_shake256() {
        // The framed input for domain "" and no fields is eight zero bytes.
        let framed = Xof::new("").digest(32);
        let mut plain = Shake256::default();
        plain.update(&[0u8; 8]);
        let mut expected = [0u8; 32];
        plain.finalize_xof().read(&mut expected);
        assert_eq!(framed, expected);
        // SHAKE256 of the empty string (FIPS 202 test vector).
        let mut empty = [0u8; 32];
        Shake256::default().finalize_xof().read(&mut empty);
        assert_eq!(hex(&empty), "46b9dd2b0ba88d13233b3feb743eeb243fcd52ea62b81b82b50c27646ed5762f");
    }

    #[test]
    fn output_is_a_stream() {
        let mut xof = Xof::new("stream");
        xof.absorb_u64(7);
        let whole = xof.clone().digest(64);
        let mut out = xof.finalize();
        let mut first = [0u8; 10];
        let mut second = [0u8; 54];
        out.read(&mut first);
        out.read(&mut second);
        assert_eq!(&whole[..10], &first);
        assert_eq!(&whole[10..], &second);
    }
}
