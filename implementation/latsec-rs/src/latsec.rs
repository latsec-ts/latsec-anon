//! The three-round LaTSEC threshold signature: Figure 1 of the paper
//! and its full specification (Appendix I of the full version).
//!
//! # Session input and the admissibility predicate
//!
//! The signed message `msg` is a [`SessionMessage`]: it canonically encodes the
//! session identifier, the signer set, the reconstruction vector `lambda_act` and
//! the application payload.  [`admissible`] is the predicate
//! `P(tspar, vk, sid, act, msg)` of Figure 1: `msg` parses, names `(sid, act)`,
//! `|act| >= t_sig`, `lambda_act^T G_act = e_1^T` and the coefficient bounds hold.
//! These are exact integer checks, so they do not depend on how `lambda_act` was
//! found; [`SessionMessage::new`] uses this crate's Algorithm 1.
//!
//! # Instantiation of the hashes
//!
//! * `H_cmt(sid, vk, act, i, w_i)` is SHAKE256 truncated to `2 kappa` bits.
//! * `H_c(sid, vk, act, msg, w)` maps into the challenge set `C` directly: the
//!   framed inputs seed a SHAKE256 stream, and the unbiased "sample in ball"
//!   shuffle of [`sample_challenge`] reads that stream.  Given a uniform stream the
//!   output is exactly uniform on `C`, so modelling SHAKE256 as a random oracle
//!   gives a random oracle into `C` (Sections 3.1 and 8.1).  A signature is
//!   `(c, z)` with `c` ranked in `b_c` bits ([`crate::codec`]).
//! * Hashes take a key digest instead of the key: `vkd = H(profile digest, seed, t)`.
//!   The profile digest covers the codec version and the reconstruction-algorithm
//!   identifier.
//!
//! # Outcomes
//!
//! Three kinds of failure are kept apart:
//!
//! * [`InvocationError`]: the call itself is malformed (wrong round, unknown
//!   session, inadmissible signer set, budget exhausted).  Nobody is blamed.
//! * [`Round3Output::Abort`] and [`CombineOutput::Abort`]: an identifiable abort
//!   naming the parties whose messages failed a public check.
//! * [`verify`] returning `false`.
//!
//! # Missing and malformed peer messages
//!
//! Figure 1: "a missing or unparsable peer message fails the check that uses it,
//! so its sender is blamed."  Here:
//!
//! * [`Signer::round2`] and [`Signer::round3`] take the messages that arrived.  A
//!   member of `act` without a message is a missing sender; a caller that cannot
//!   decode a message leaves it out.  A commitment is only used in Round 3, so a
//!   missing or wrong-length commitment is stored in Round 2 and its sender is
//!   blamed in Round 3, as is the sender of a missing or malformed opening (wrong
//!   shape or an unreduced coefficient).  The first such signer in the canonical
//!   order of `act` is named.
//! * A [`Transcript`] has one slot per signer; a missing or undecodable message is
//!   entered as [`Opening::missing`] or [`PartialResponse::missing`], which are
//!   malformed.  [`combine`] blames the sender of every malformed opening and of
//!   every response that fails `PartialVerify`.  [`partial_verify`] rejects a
//!   party whose own opening is malformed, or whose response is malformed or
//!   longer than `B1`, before it looks at anyone else's messages.  If an opening
//!   is malformed the aggregate `w`, and with it the challenge, is undefined:
//!   [`combine`] then blames those senders together with the senders of
//!   responses that fail the checks that need no challenge (shape and
//!   `|z_i|_2 <= B1`).
//! * Messages from outside `act` and duplicates are faults of the delivery layer
//!   and stay invocation errors, as does a signer's own message coming back
//!   altered: consistent delivery is assumed, and an honest signer must not
//!   blame itself.  (Appendix I of the full version has no such assertion in `ShareSign_3`.)
//!
//! # Model
//!
//! Key generation is by a trusted dealer.  Round messages are assumed to be
//! delivered consistently and authentically to all signers (see [`crate::sim`]).
//! The arithmetic and sampling backends are variable-time references.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use zeroize::Zeroize;

use crate::codec::{self, tag, BitReader, BitWriter};
use crate::error::{Error, Result};
use crate::profile::DIGEST_BYTES;
use crate::ring::{Challenge, PolyQ, PolyZ};
use crate::sampling::{random_bit, uniform_below, RandomSource};
use crate::scheme::{combine_z, norm_squared, ActiveSet, Inadmissible, PartyId, PublicMatrix, Scheme};
use crate::xof::{domain, Xof, XofOutput};

impl RandomSource for XofOutput {
    fn fill(&mut self, buf: &mut [u8]) {
        self.read(buf);
    }
}

/// A session identifier.  Each signer accepts a given identifier once per key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SessionId(pub [u8; 32]);

/// The signed message `msg` of a session: it canonically encodes `(sid, act,
/// lambda_act)` and the application payload (Section 3.1 of the paper).
///
/// Encoding: header, the 32-byte `sid`, the signer set (`u16` count and `u16`
/// indices, little-endian, strictly increasing), one entry of `lambda_act` per
/// signer stored as `x + B` in `bitlen(2B)` bits for the coefficient bound `B` of
/// the admissibility predicate, zero padding to a byte, then the payload, which
/// runs to the end.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionMessage {
    pub sid: SessionId,
    pub act: ActiveSet,
    pub lambda: Vec<i64>,
    pub payload: Vec<u8>,
}

impl SessionMessage {
    /// The session message with `lambda_act` from this crate's Algorithm 1, as the
    /// session initiator would build it.  Fails if the set is not admissible.
    pub fn new(
        scheme: &Scheme,
        sid: SessionId,
        act: &ActiveSet,
        payload: &[u8],
    ) -> std::result::Result<Self, Inadmissible> {
        let lambda = scheme.lambda(act)?;
        Ok(SessionMessage { sid, act: act.clone(), lambda: lambda.as_ref().clone(), payload: payload.to_vec() })
    }

    /// Fails if a coefficient of `lambda_act` exceeds the predicate's coefficient
    /// bound, or if `lambda_act` and `act` differ in length.
    pub fn to_bytes(&self, scheme: &Scheme) -> Result<Vec<u8>> {
        let bound = u128::from(scheme.profile().admissibility.linf_bound());
        if self.lambda.len() != self.act.len() {
            return Err(Error::invalid("lambda_act does not match the signer set"));
        }
        let mut w = BitWriter::with_header(tag::SESSION_MESSAGE);
        w.bytes(&self.sid.0);
        w.bytes(&self.act.encode());
        let lambda: Vec<i128> = self.lambda.iter().map(|&x| i128::from(x)).collect();
        codec::write_bounded(&mut w, &[lambda], bound)?;
        let mut out = w.finish();
        out.extend_from_slice(&self.payload);
        Ok(out)
    }

    /// Decodes a session message.  Everything before the payload has a length
    /// fixed by the profile and the number of signers
    /// ([`limits::session_head_len`]).  The payload runs to the end of `data` and
    /// has no length field, so its size is the application's to bound: see
    /// [`SessionMessage::from_bytes_with_limit`].
    pub fn from_bytes(scheme: &Scheme, data: &[u8]) -> Result<Self> {
        Self::from_bytes_with_limit(scheme, data, usize::MAX)
    }

    /// As [`SessionMessage::from_bytes`], rejecting a payload of more than
    /// `max_payload` bytes before anything else is read.
    pub fn from_bytes_with_limit(scheme: &Scheme, data: &[u8], max_payload: usize) -> Result<Self> {
        let mut r = BitReader::with_header(data, tag::SESSION_MESSAGE)?;
        let sid = SessionId(r.bytes(32)?.try_into().expect("32 bytes"));
        let count = usize::from(u16::from_le_bytes(r.bytes(2)?.try_into().expect("2 bytes")));
        if count == 0 || count > scheme.n() {
            return Err(Error::decode("signer set is empty or larger than the number of parties"));
        }
        let head = limits::session_head_len(scheme, count);
        if data.len() < head {
            return Err(Error::decode("encoding is too short"));
        }
        if data.len() - head > max_payload {
            return Err(Error::decode("payload exceeds the application's limit"));
        }
        let mut set = (count as u16).to_le_bytes().to_vec();
        set.extend_from_slice(r.bytes(2 * count)?);
        let act = ActiveSet::decode(&set)?;
        if act.parties().iter().any(|&p| usize::from(p) >= scheme.n()) {
            return Err(Error::decode("signer set names an unknown party"));
        }
        let bound = u128::from(scheme.profile().admissibility.linf_bound());
        let lambda = codec::read_bounded(&mut r, 1, count, bound)?.remove(0);
        let consumed = r.finish_prefix()?;
        Ok(SessionMessage {
            sid,
            act,
            lambda: lambda.into_iter().map(|x| x as i64).collect(),
            payload: data[consumed..].to_vec(),
        })
    }
}

/// The admissibility predicate `P(tspar, vk, sid, act, msg)` of Figure 1: `msg`
/// parses as a [`SessionMessage`] for `(sid, act)`, `|act| >= t_sig`,
/// `lambda_act^T G_act = e_1^T`, and `lambda_act` meets the coefficient bounds.
/// Returns the parsed message.
pub fn admissible(
    scheme: &Scheme,
    sid: &SessionId,
    act: &ActiveSet,
    msg: &[u8],
) -> std::result::Result<SessionMessage, Inadmissible> {
    let parsed = SessionMessage::from_bytes(scheme, msg).map_err(|_| Inadmissible::MalformedMessage)?;
    if &parsed.sid != sid || &parsed.act != act {
        return Err(Inadmissible::ContextMismatch);
    }
    scheme.check_supplied_lambda(act, &parsed.lambda)?;
    Ok(parsed)
}

/// The verification key `vk = (A, t)`, with `A` given by its seed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerificationKey {
    seed: [u8; 32],
    t: Vec<PolyQ>,
    matrix: PublicMatrix,
    digest: Vec<u8>,
}

impl VerificationKey {
    fn new(scheme: &Scheme, seed: [u8; 32], t: Vec<PolyQ>) -> Self {
        let profile = scheme.profile();
        let matrix = PublicMatrix::expand(scheme.ring(), profile.p, profile.ell, &seed);
        let mut xof = Xof::new(domain::VERIFICATION_KEY);
        xof.absorb(scheme.digest()).absorb(&seed).absorb(&codec::residues_to_bytes(&t, scheme.ring().q()));
        VerificationKey { seed, t, matrix, digest: xof.digest(DIGEST_BYTES) }
    }

    pub fn seed(&self) -> &[u8; 32] {
        &self.seed
    }

    /// `t = A s mod q`.
    pub fn t(&self) -> &[PolyQ] {
        &self.t
    }

    pub fn matrix(&self) -> &PublicMatrix {
        &self.matrix
    }

    /// The digest that stands for this key (and its profile) in every hash.
    pub fn digest(&self) -> &[u8] {
        &self.digest
    }

    /// Header, profile digest, seed, then `t` packed.
    pub fn to_bytes(&self, scheme: &Scheme) -> Vec<u8> {
        let mut w = BitWriter::with_header(tag::VERIFICATION_KEY);
        w.bytes(scheme.digest());
        w.bytes(&self.seed);
        codec::write_residues(&mut w, &self.t, scheme.ring().q());
        w.finish()
    }

    pub fn from_bytes(scheme: &Scheme, data: &[u8]) -> Result<Self> {
        limits::check_exact(data, limits::verification_key_len(scheme))?;
        let mut r = BitReader::with_header(data, tag::VERIFICATION_KEY)?;
        if r.bytes(DIGEST_BYTES)? != scheme.digest() {
            return Err(Error::decode("verification key belongs to another profile"));
        }
        let seed: [u8; 32] = r.bytes(32)?.try_into().expect("32 bytes");
        let t = codec::read_residues(&mut r, scheme.profile().p, scheme.ring().degree(), scheme.ring().q())?;
        r.finish()?;
        Ok(Self::new(scheme, seed, t))
    }
}

/// The abort key `ak_i = t_i = A s_i mod q` of one party.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AbortKey {
    pub index: PartyId,
    pub t_i: Vec<PolyQ>,
}

impl AbortKey {
    pub fn to_bytes(&self, scheme: &Scheme) -> Vec<u8> {
        let mut w = BitWriter::with_header(tag::ABORT_KEY);
        w.bytes(&self.index.to_le_bytes());
        codec::write_residues(&mut w, &self.t_i, scheme.ring().q());
        w.finish()
    }

    pub fn from_bytes(scheme: &Scheme, data: &[u8]) -> Result<Self> {
        limits::check_exact(data, limits::abort_key_len(scheme))?;
        let mut r = BitReader::with_header(data, tag::ABORT_KEY)?;
        let index = u16::from_le_bytes(r.bytes(2)?.try_into().expect("2 bytes"));
        if usize::from(index) >= scheme.n() {
            return Err(Error::decode("abort key for an unknown party"));
        }
        let t_i = codec::read_residues(&mut r, scheme.profile().p, scheme.ring().degree(), scheme.ring().q())?;
        r.finish()?;
        Ok(AbortKey { index, t_i })
    }
}

/// The signing share `sk_i = s_i` of one party.  Erased on drop.
#[derive(Clone, PartialEq, Eq)]
pub struct SecretShare {
    index: PartyId,
    s_i: Vec<PolyZ>,
}

impl fmt::Debug for SecretShare {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SecretShare(party {}, <redacted>)", self.index)
    }
}

impl Drop for SecretShare {
    fn drop(&mut self) {
        self.s_i.zeroize();
    }
}

impl SecretShare {
    pub fn index(&self) -> PartyId {
        self.index
    }

    /// The share as integer polynomials.  For tests and SEC-level reconstruction.
    pub fn expose(&self) -> &[PolyZ] {
        &self.s_i
    }

    /// Header, party index, then the coefficients Golomb-Rice coded with the
    /// party's parameter, bounded by `beta_sh`.  Fails if a coefficient exceeds
    /// that bound, which the tail analysis makes negligible for honestly dealt
    /// shares.
    pub fn to_bytes(&self, scheme: &Scheme) -> Result<Vec<u8>> {
        let mut w = BitWriter::with_header(tag::SECRET_SHARE);
        w.bytes(&self.index.to_le_bytes());
        codec::write_rice(&mut w, &self.s_i, share_rice_parameter(scheme, self.index), scheme.beta_sh())?;
        Ok(w.finish())
    }

    pub fn from_bytes(scheme: &Scheme, data: &[u8]) -> Result<Self> {
        let mut r = BitReader::with_header(data, tag::SECRET_SHARE)?;
        let index = u16::from_le_bytes(r.bytes(2)?.try_into().expect("2 bytes"));
        if usize::from(index) >= scheme.n() {
            return Err(Error::decode("share for an unknown party"));
        }
        limits::check_max(data, limits::max_share_len(scheme, index))?;
        let k = share_rice_parameter(scheme, index);
        let s_i = codec::read_rice(&mut r, scheme.profile().ell, scheme.ring().degree(), k, scheme.beta_sh())?;
        r.finish()?;
        Ok(SecretShare { index, s_i })
    }
}

/// The Golomb-Rice parameter of party `index`'s share, which depends on `|g_i|`.
fn share_rice_parameter(scheme: &Scheme, index: PartyId) -> u32 {
    let profile = scheme.profile();
    let abs_g = profile.sec.g[usize::from(index)][0].unsigned_abs();
    profile.encoding.rice_share[usize::from(abs_g)]
}

/// Everything the trusted dealer hands out.
#[derive(Debug)]
pub struct KeyMaterial {
    pub vk: VerificationKey,
    /// One abort key per party, in party order.
    pub abort_keys: Vec<AbortKey>,
    /// One share per party, in party order.
    pub shares: Vec<SecretShare>,
}

/// `LaTSEC.Keygen` with a trusted dealer (Algorithm 3).  The secret `s` and the
/// masking matrix are erased before returning.
pub fn keygen(scheme: &Scheme, rng: &mut dyn RandomSource) -> KeyMaterial {
    let profile = scheme.profile();
    let (d, ell) = (scheme.ring().degree(), profile.ell);
    let mut seed = [0u8; 32];
    rng.fill(&mut seed);
    let matrix = PublicMatrix::expand(scheme.ring(), profile.p, ell, &seed);
    let mut secret: Vec<PolyZ> = (0..ell)
        .map(|_| {
            let mut poly = vec![0i128; d];
            scheme.secret_sampler().sample_into(rng, &mut poly);
            poly
        })
        .collect();
    let dealt = scheme.deal(&secret, rng);
    let t = matrix.mul_z(scheme.ring(), &secret);
    secret.zeroize();
    let abort_keys = dealt
        .iter()
        .enumerate()
        .map(|(i, share)| AbortKey { index: i as PartyId, t_i: matrix.mul_z(scheme.ring(), share) })
        .collect();
    let shares = dealt.into_iter().enumerate().map(|(i, s_i)| SecretShare { index: i as PartyId, s_i }).collect();
    KeyMaterial { vk: VerificationKey::new(scheme, seed, t), abort_keys, shares }
}

/// A Round-1 message: the commitment `cmt_i`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Commitment(pub Vec<u8>);

/// A Round-2 message: the opening `w_i = A r_i mod q`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Opening {
    pub w: Vec<PolyQ>,
}

impl Opening {
    /// The transcript entry of a signer whose opening is missing or could not be
    /// decoded.  It is malformed, so its sender is blamed.
    pub fn missing() -> Self {
        Opening { w: Vec::new() }
    }
}

impl Opening {
    pub fn to_bytes(&self, scheme: &Scheme) -> Vec<u8> {
        let mut w = BitWriter::with_header(tag::OPENING);
        codec::write_residues(&mut w, &self.w, scheme.ring().q());
        w.finish()
    }

    pub fn from_bytes(scheme: &Scheme, data: &[u8]) -> Result<Self> {
        limits::check_exact(data, limits::opening_len(scheme))?;
        let mut r = BitReader::with_header(data, tag::OPENING)?;
        let w = codec::read_residues(&mut r, scheme.profile().p, scheme.ring().degree(), scheme.ring().q())?;
        r.finish()?;
        Ok(Opening { w })
    }
}

/// A Round-3 message: the partial response `z_i = c s_i + r_i` over the integers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartialResponse {
    pub z: Vec<PolyZ>,
}

impl PartialResponse {
    /// The transcript entry of a signer whose response is missing or could not be
    /// decoded.  It is malformed, so its sender is blamed.
    pub fn missing() -> Self {
        PartialResponse { z: Vec::new() }
    }
}

impl PartialResponse {
    /// Header, then the coefficients Golomb-Rice coded with the profile's response
    /// parameter.  Fails if a coefficient exceeds `B1` in absolute value; such a
    /// response would be rejected by partial verification anyway.
    pub fn to_bytes(&self, scheme: &Scheme) -> Result<Vec<u8>> {
        let mut w = BitWriter::with_header(tag::PARTIAL_RESPONSE);
        codec::write_rice(&mut w, &self.z, scheme.profile().encoding.rice_response, scheme.b1())?;
        Ok(w.finish())
    }

    /// Rejects an input longer than [`limits::max_response_len`] before reading it.
    pub fn from_bytes(scheme: &Scheme, data: &[u8]) -> Result<Self> {
        limits::check_max(data, limits::max_response_len(scheme))?;
        let mut r = BitReader::with_header(data, tag::PARTIAL_RESPONSE)?;
        let (count, d, k) = (scheme.profile().ell, scheme.ring().degree(), scheme.profile().encoding.rice_response);
        let z = codec::read_rice(&mut r, count, d, k, scheme.b1())?;
        r.finish()?;
        Ok(PartialResponse { z })
    }
}

/// A signature `(c, z)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Signature {
    pub challenge: Challenge,
    pub z: Vec<PolyZ>,
}

impl Signature {
    /// Header, the Golomb-Rice parameter `k` of `z` as one byte, the ranked
    /// challenge, then `z` coded with `k`.  The width of `z` depends on
    /// `lambda_act`, so `k` is chosen per signature: the one giving the shortest
    /// code, the smallest on ties.  Fails if the challenge is not in `C` or a
    /// coefficient of `z` exceeds `B2` in absolute value.
    pub fn to_bytes(&self, scheme: &Scheme) -> Result<Vec<u8>> {
        if !well_formed_challenge(scheme, &self.challenge) {
            return Err(Error::invalid("challenge is not in the challenge set"));
        }
        let k = codec::shortest_rice(&self.z);
        let mut w = BitWriter::with_header(tag::SIGNATURE);
        w.bytes(&[k as u8]);
        codec::write_challenge(&mut w, &self.challenge);
        codec::write_rice(&mut w, &self.z, k, scheme.b2())?;
        Ok(w.finish())
    }

    /// Rejects a parameter `k` other than the one [`Signature::to_bytes`] chooses.
    ///
    /// Two checks come before any coefficient is read: the input is no longer
    /// than [`limits::max_signature_len`], and it is long enough for its `k`
    /// (every codeword has at least `k + 1` bits).  Together they bound the work
    /// of the decoder, including the unary runs, by the profile; `k = 0`, the
    /// canonical parameter of a short enough response, stays valid.
    pub fn from_bytes(scheme: &Scheme, data: &[u8]) -> Result<Self> {
        limits::check_max(data, limits::max_signature_len(scheme))?;
        let mut r = BitReader::with_header(data, tag::SIGNATURE)?;
        let k = u32::from(r.bytes(1)?[0]);
        let body_bits = 8 * (data.len() as u128 - 3);
        if k >= 127 || limits::challenge_bits(scheme) + limits::dimension(scheme) * u128::from(k + 1) > body_bits {
            return Err(Error::decode("Rice parameter out of range for this encoding"));
        }
        let challenge =
            codec::read_challenge(&mut r, scheme.ring().degree(), scheme.profile().challenge.omega_c as usize)?;
        let z = codec::read_rice(&mut r, scheme.profile().ell, scheme.ring().degree(), k, scheme.b2())?;
        r.finish()?;
        if !codec::is_shortest_rice(&z, k) {
            return Err(Error::decode("Rice parameter is not the canonical one"));
        }
        Ok(Signature { challenge, z })
    }
}

/// Lengths of the encoded protocol objects, derived from the profile.  Decoders
/// reject an input of the wrong or of excessive length before reading it.
///
/// Keys and openings have one length.  For Golomb-Rice coded objects the limit is
/// an upper bound on the encodings of the values that can be valid:
///
/// * a share has coefficients of absolute value at most `beta_sh`;
/// * a partial response that passes `PartialVerify` has `|z_i|_2 <= B1`;
/// * a signature that verifies has `|z|_2 <= B2`, and its parameter is the one
///   giving the shortest code, so its length is at most the bound for any `k`.
///
/// A longer input encodes a value that would be rejected anyway.  The bounds
/// allow for the worst distribution of a given norm and are therefore above the
/// typical sizes (about 2% for a signature, more for the objects with a fixed
/// parameter).  The application payload of a session message is not covered: see
/// [`SessionMessage::from_bytes_with_limit`].
pub mod limits {
    use super::*;
    use num_bigint::BigUint;

    const HEADER: usize = 2;

    fn bytes_of_bits(bits: &BigUint) -> usize {
        usize::try_from((bits + 7u8) >> 3usize).unwrap_or(usize::MAX)
    }

    pub(super) fn check_exact(data: &[u8], len: usize) -> Result<()> {
        if data.len() == len {
            Ok(())
        } else {
            Err(Error::decode("encoding has the wrong length"))
        }
    }

    pub(super) fn check_max(data: &[u8], len: usize) -> Result<()> {
        if data.len() <= len {
            Ok(())
        } else {
            Err(Error::decode("encoding is longer than any valid one"))
        }
    }

    /// `D = d ell`, the number of coefficients of a share or response.
    pub(super) fn dimension(scheme: &Scheme) -> u128 {
        (scheme.ring().degree() * scheme.profile().ell) as u128
    }

    /// `b_c`: the rank of the challenge support and its signs.
    pub(super) fn challenge_bits(scheme: &Scheme) -> u128 {
        let weight = scheme.profile().challenge.omega_c as usize;
        u128::from(codec::challenge_rank_bits(scheme.ring().degree(), weight)) + weight as u128
    }

    fn residues_len(scheme: &Scheme) -> usize {
        codec::packed_len(scheme.profile().p * scheme.ring().degree(), codec::residue_width(scheme.ring().q()))
    }

    pub fn verification_key_len(scheme: &Scheme) -> usize {
        HEADER + DIGEST_BYTES + 32 + residues_len(scheme)
    }

    pub fn abort_key_len(scheme: &Scheme) -> usize {
        HEADER + 2 + residues_len(scheme)
    }

    pub fn opening_len(scheme: &Scheme) -> usize {
        HEADER + residues_len(scheme)
    }

    /// The length of a session message with `signers` signers and no payload.
    pub fn session_head_len(scheme: &Scheme, signers: usize) -> usize {
        let width = codec::signed_width(u128::from(scheme.profile().admissibility.linf_bound()));
        HEADER + 32 + 2 + 2 * signers + codec::packed_len(signers, width)
    }

    pub fn max_share_len(scheme: &Scheme, index: PartyId) -> usize {
        let k = share_rice_parameter(scheme, index);
        HEADER + 2 + bytes_of_bits(&codec::rice_max_bits_linf(dimension(scheme) as usize, k, scheme.beta_sh()))
    }

    /// Computed once per [`Scheme`].
    pub fn max_response_len(scheme: &Scheme) -> usize {
        *scheme.max_response_len.get_or_init(|| {
            let k = scheme.profile().encoding.rice_response;
            HEADER + bytes_of_bits(&codec::rice_max_bits_l2(dimension(scheme) as usize, k, scheme.b1_squared()))
        })
    }

    /// Computed once per [`Scheme`].
    pub fn max_signature_len(scheme: &Scheme) -> usize {
        *scheme.max_signature_len.get_or_init(|| {
            let count = dimension(scheme) as usize;
            let l1 = codec::l1_bound(count, scheme.b2_squared());
            let z_bits = (0..127).map(|k| codec::rice_max_bits_l1(count, k, &l1)).min().expect("nonempty");
            HEADER + 1 + bytes_of_bits(&(z_bits + challenge_bits(scheme)))
        })
    }
}

/// Whether `c` has the profile's degree and weight.
fn well_formed_challenge(scheme: &Scheme, c: &Challenge) -> bool {
    c.coeffs().len() == scheme.ring().degree() && c.weight() == scheme.profile().challenge.omega_c as usize
}

/// `H_cmt(sid, vk, act, i, w_i)`.
pub fn commitment_hash(
    scheme: &Scheme,
    sid: &SessionId,
    vk: &VerificationKey,
    act: &ActiveSet,
    party: PartyId,
    w: &[PolyQ],
) -> Commitment {
    let mut xof = Xof::new(domain::COMMITMENT);
    xof.absorb(&sid.0).absorb(vk.digest()).absorb(&act.encode());
    xof.absorb_u64(u64::from(party)).absorb(&codec::residues_to_bytes(w, scheme.ring().q()));
    Commitment(xof.digest(scheme.profile().challenge.commitment_bytes))
}

/// A uniform element of the challenge set `C = {c : |c|_inf = 1, |c|_1 = omega_c}`
/// from a stream of uniform bytes.
///
/// For `i = d - omega_c, ..., d - 1` a uniform `j` in `[0, i]` is drawn by rejection
/// sampling and a sign bit follows; then `c[i] = c[j]` and `c[j] = +-1`.  This is the
/// unbiased "sample in ball" shuffle: given a uniform stream, every element of `C`
/// has probability exactly `1 / |C|`.
pub fn sample_challenge(scheme: &Scheme, stream: &mut dyn RandomSource) -> Challenge {
    let d = scheme.ring().degree();
    let weight = scheme.profile().challenge.omega_c as usize;
    let mut coeffs = vec![0i8; d];
    for i in d - weight..d {
        let j = uniform_below(stream, i as u128 + 1) as usize;
        let negative = random_bit(stream);
        coeffs[i] = coeffs[j];
        coeffs[j] = if negative { -1 } else { 1 };
    }
    Challenge::from_coeffs(coeffs).expect("ternary by construction")
}

/// `c = H_c(sid, vk, act, msg, w)`: [`sample_challenge`] on the SHAKE256 stream
/// seeded with the framed inputs `(sid, vkd, act, msg, w, d, omega_c)`.
pub fn hash_to_challenge(
    scheme: &Scheme,
    sid: &SessionId,
    vk: &VerificationKey,
    act: &ActiveSet,
    msg: &[u8],
    w: &[PolyQ],
) -> Challenge {
    let mut xof = Xof::new(domain::CHALLENGE);
    xof.absorb(&sid.0).absorb(vk.digest()).absorb(&act.encode()).absorb(msg);
    xof.absorb(&codec::residues_to_bytes(w, scheme.ring().q()));
    xof.absorb_u64(scheme.ring().degree() as u64).absorb_u64(u64::from(scheme.profile().challenge.omega_c));
    sample_challenge(scheme, &mut xof.finalize())
}

/// `w = sum_j lambda_j w_j mod q` and the challenge `c = H_c(sid, vk, act, msg, w)`.
fn session_challenge(
    scheme: &Scheme,
    vk: &VerificationKey,
    sid: &SessionId,
    act: &ActiveSet,
    msg: &[u8],
    lambda: &[i64],
    openings: &[&Opening],
) -> Challenge {
    let items: Vec<&[PolyQ]> = openings.iter().map(|o| o.w.as_slice()).collect();
    let w = scheme.combine_q(lambda, &items);
    hash_to_challenge(scheme, sid, vk, act, msg, &w)
}

/// `A z - c t mod q`.
fn response_image(scheme: &Scheme, vk: &VerificationKey, z: &[PolyZ], c: &Challenge, t: &[PolyQ]) -> Vec<PolyQ> {
    let ring = scheme.ring();
    let az = vk.matrix.mul_z(ring, z);
    az.iter().zip(t).map(|(a, t_poly)| ring.sub(a, &c.mul_q(ring.modulus(), t_poly))).collect()
}

fn well_formed_z(scheme: &Scheme, z: &[PolyZ]) -> bool {
    z.len() == scheme.profile().ell && z.iter().all(|p| p.len() == scheme.ring().degree())
}

fn well_formed_q(scheme: &Scheme, w: &[PolyQ]) -> bool {
    let (d, q) = (scheme.ring().degree(), scheme.ring().q());
    w.len() == scheme.profile().p && w.iter().all(|p| p.len() == d && p.iter().all(|&x| x < q))
}

/// A malformed invocation.  Never attributed to another party.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InvocationError {
    /// This signer has already used the session identifier.
    SessionIdReused,
    /// The signer is not in the signer set.
    NotInActiveSet,
    /// The signer set fails the admissibility predicate.
    Inadmissible(Inadmissible),
    /// The per-key participation budget is spent.
    ParticipationCapReached,
    /// No running session with that identifier.
    UnknownSession,
    /// The session is in another round.
    WrongRound { expected: u8, found: u8 },
    /// The signer set differs from the one fixed in Round 1.
    ActiveSetMismatch,
    /// A message from outside the signer set, two messages from one party, or a
    /// transcript without one slot per signer.
    MessageSetMismatch,
    /// `PartialVerify` for one party cannot be evaluated because the opening of
    /// another, named here, is missing or malformed.
    MalformedMessage(PartyId),
    /// The signer's own commitment was not delivered intact.
    OwnCommitmentMismatch,
    /// The signer's own opening was not delivered intact.
    OwnOpeningMismatch,
    /// A key or share does not belong to this party or profile.
    KeyMismatch,
}

impl fmt::Display for InvocationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InvocationError::Inadmissible(reason) => write!(f, "signer set rejected: {reason}"),
            other => write!(f, "{other:?}"),
        }
    }
}

impl std::error::Error for InvocationError {}

impl From<Inadmissible> for InvocationError {
    fn from(reason: Inadmissible) -> Self {
        InvocationError::Inadmissible(reason)
    }
}

/// The result of Round 3 for one signer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Round3Output {
    /// The partial response `z_i`.
    Response(PartialResponse),
    /// An identifiable abort: the listed party's commitment or opening is missing
    /// or malformed, or its opening does not match its commitment.
    Abort { blamed: Vec<PartyId> },
}

/// The status of a session identifier at one signer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionStatus {
    /// Never used by this signer.
    Unused,
    /// Running, having completed the given round.
    Running { round: u8 },
    /// Completed or aborted; its ephemeral state has been erased.
    Finished,
}

/// Ephemeral state of one running session.  The mask is erased on drop.
struct Session {
    round: u8,
    act: ActiveSet,
    msg: Vec<u8>,
    lambda: Vec<i64>,
    mask: Vec<PolyZ>,
    opening: Opening,
    commitment: Commitment,
    /// The Round-1 messages in the order of `act`; `None` for a missing one.
    commitments: Vec<Option<Commitment>>,
}

impl Drop for Session {
    fn drop(&mut self) {
        self.mask.zeroize();
    }
}

/// One signer: its share, its running sessions, the identifiers it has used and
/// its participation counter.
///
/// The used-identifier set and the counter live in memory only.  Using one share
/// from several processes, or restoring an old copy of this state, would defeat
/// both; persistence is outside this prototype.
pub struct Signer {
    scheme: Arc<Scheme>,
    vk: Arc<VerificationKey>,
    share: SecretShare,
    sessions: HashMap<SessionId, Session>,
    used: HashSet<SessionId>,
    participations: u128,
}

impl Signer {
    pub fn new(scheme: Arc<Scheme>, vk: Arc<VerificationKey>, share: SecretShare) -> Result<Self> {
        if !scheme.supports_signing() {
            return Err(Error::invalid("SEC-only profiles do not support signing"));
        }
        if usize::from(share.index) >= scheme.n()
            || share.s_i.len() != scheme.profile().ell
            || share.s_i.iter().any(|p| p.len() != scheme.ring().degree())
        {
            return Err(Error::invalid("share does not match the profile"));
        }
        Ok(Signer { scheme, vk, share, sessions: HashMap::new(), used: HashSet::new(), participations: 0 })
    }

    pub fn index(&self) -> PartyId {
        self.share.index
    }

    /// The signer's share.
    pub fn share(&self) -> &SecretShare {
        &self.share
    }

    /// Sessions this signer has taken part in, including aborted ones.
    pub fn participations(&self) -> u128 {
        self.participations
    }

    pub fn session_status(&self, sid: &SessionId) -> SessionStatus {
        match self.sessions.get(sid) {
            Some(session) => SessionStatus::Running { round: session.round },
            None if self.used.contains(sid) => SessionStatus::Finished,
            None => SessionStatus::Unused,
        }
    }

    /// Erases the ephemeral state of a session.  The identifier stays used.
    pub fn abort_session(&mut self, sid: &SessionId) {
        self.sessions.remove(sid);
    }

    /// `ShareSign_1` (Figure 1): checks the identifier, membership and the
    /// predicate `P`, samples the mask `r_i` and returns `cmt_i`.  The
    /// reconstruction vector is the one `msg` carries.
    pub fn round1(
        &mut self,
        sid: SessionId,
        act: &ActiveSet,
        msg: &[u8],
        rng: &mut dyn RandomSource,
    ) -> std::result::Result<Commitment, InvocationError> {
        if self.used.contains(&sid) {
            return Err(InvocationError::SessionIdReused);
        }
        if !act.contains(self.index()) {
            return Err(InvocationError::NotInActiveSet);
        }
        let lambda = admissible(&self.scheme, &sid, act, msg)?.lambda;
        if self.participations >= self.scheme.profile().budgets.q_participate {
            return Err(InvocationError::ParticipationCapReached);
        }
        // Reserve the budget unit and the identifier before any mask exists.
        self.participations += 1;
        self.used.insert(sid);

        let (d, ell) = (self.scheme.ring().degree(), self.scheme.profile().ell);
        let mask: Vec<PolyZ> = (0..ell)
            .map(|_| {
                let mut poly = vec![0i128; d];
                self.scheme.mask_sampler().sample_into(rng, &mut poly);
                poly
            })
            .collect();
        let opening = Opening { w: self.vk.matrix.mul_z(self.scheme.ring(), &mask) };
        let commitment = commitment_hash(&self.scheme, &sid, &self.vk, act, self.index(), &opening.w);
        self.sessions.insert(
            sid,
            Session {
                round: 1,
                act: act.clone(),
                msg: msg.to_vec(),
                lambda,
                mask,
                opening,
                commitment: commitment.clone(),
                commitments: Vec::new(),
            },
        );
        Ok(commitment)
    }

    /// The running session `sid`, which must have completed `expected_round` with
    /// the signer set `act`.
    fn running<'a>(
        sessions: &'a mut HashMap<SessionId, Session>,
        sid: &SessionId,
        act: &ActiveSet,
        expected_round: u8,
    ) -> std::result::Result<&'a mut Session, InvocationError> {
        let session = sessions.get_mut(sid).ok_or(InvocationError::UnknownSession)?;
        if session.round != expected_round {
            return Err(InvocationError::WrongRound { expected: expected_round, found: session.round });
        }
        if &session.act != act {
            return Err(InvocationError::ActiveSetMismatch);
        }
        Ok(session)
    }

    /// `ShareSign_2` (Figure 1): takes the commitments that arrived from the
    /// signers in `act`, stores them and returns this signer's opening.  A
    /// missing or wrong-length commitment is not checked here: Round 3 uses it and
    /// blames its sender.
    pub fn round2(
        &mut self,
        sid: &SessionId,
        act: &ActiveSet,
        commitments: &[(PartyId, Commitment)],
    ) -> std::result::Result<Opening, InvocationError> {
        let index = self.index();
        let session = Self::running(&mut self.sessions, sid, act, 1)?;
        let ordered = in_set_order(act, commitments)?;
        let own = act.position(index).expect("checked in round 1");
        if ordered[own] != Some(&session.commitment) {
            return Err(InvocationError::OwnCommitmentMismatch);
        }
        session.commitments = ordered.into_iter().map(|commitment| commitment.cloned()).collect();
        session.round = 2;
        Ok(session.opening.clone())
    }

    /// `ShareSign_3` (Figure 1): takes the openings that arrived from the signers
    /// in `act`, checks them against the commitments in the canonical order of
    /// `act`, and returns the partial response or an identifiable abort naming the
    /// first signer whose commitment or opening is missing, malformed or
    /// inconsistent.  The session's ephemeral state is erased either way.
    pub fn round3(
        &mut self,
        sid: &SessionId,
        act: &ActiveSet,
        openings: &[(PartyId, Opening)],
    ) -> std::result::Result<Round3Output, InvocationError> {
        let index = self.index();
        let scheme = Arc::clone(&self.scheme);
        let vk = Arc::clone(&self.vk);
        let session = Self::running(&mut self.sessions, sid, act, 2)?;
        let ordered = in_set_order(act, openings)?;
        let own = act.position(index).expect("checked in round 1");
        if ordered[own] != Some(&session.opening) {
            return Err(InvocationError::OwnOpeningMismatch);
        }
        // The openings are borrowed from the caller: no copy is made of them.
        let mut checked: Vec<&Opening> = Vec::with_capacity(act.len());
        for ((&party, opening), commitment) in act.parties().iter().zip(ordered).zip(&session.commitments) {
            let consistent = match (&opening, commitment) {
                (Some(opening), Some(commitment)) => {
                    well_formed_q(&scheme, &opening.w)
                        && &commitment_hash(&scheme, sid, &vk, act, party, &opening.w) == commitment
                }
                _ => false,
            };
            match opening {
                Some(opening) if consistent => checked.push(opening),
                _ => {
                    self.sessions.remove(sid);
                    return Ok(Round3Output::Abort { blamed: vec![party] });
                }
            }
        }
        let challenge = session_challenge(&scheme, &vk, sid, act, &session.msg, &session.lambda, &checked);
        let z: Vec<PolyZ> = self
            .share
            .s_i
            .iter()
            .zip(&session.mask)
            .map(|(s, r)| {
                let mut product = challenge.mul_z(s);
                let sum = product.iter().zip(r).map(|(a, b)| a + b).collect();
                product.zeroize();
                sum
            })
            .collect();
        // The mask is never used again; dropping the session erases it.
        self.sessions.remove(sid);
        Ok(Round3Output::Response(PartialResponse { z }))
    }
}

/// Arranges per-party messages in the canonical order of `act`, with `None` for a
/// member that sent nothing.  A message from outside `act` or a second message
/// from one party is an error.  The messages are borrowed, not copied.
fn in_set_order<'a, T>(
    act: &ActiveSet,
    messages: &'a [(PartyId, T)],
) -> std::result::Result<Vec<Option<&'a T>>, InvocationError> {
    let mut slots: Vec<Option<&'a T>> = vec![None; act.len()];
    for (party, message) in messages {
        let position = act.position(*party).ok_or(InvocationError::MessageSetMismatch)?;
        if slots[position].replace(message).is_some() {
            return Err(InvocationError::MessageSetMismatch);
        }
    }
    Ok(slots)
}

/// The public transcript of a session, with per-party messages in the canonical
/// order of `act`.  A missing or undecodable message is entered as
/// [`Opening::missing`] or [`PartialResponse::missing`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Transcript {
    pub sid: SessionId,
    pub act: ActiveSet,
    /// The encoded [`SessionMessage`], which carries `lambda_act`.
    pub msg: Vec<u8>,
    pub commitments: Vec<Commitment>,
    pub openings: Vec<Opening>,
    pub responses: Vec<PartialResponse>,
}

/// Checks the shape of a transcript and returns the `lambda_act` its message
/// carries.  An inadmissible session input is a malformed invocation; malformed
/// peer messages are left to the checks that use them.
fn check_transcript(
    scheme: &Scheme,
    abort_keys: &[AbortKey],
    tr: &Transcript,
) -> std::result::Result<Vec<i64>, InvocationError> {
    let lambda = admissible(scheme, &tr.sid, &tr.act, &tr.msg)?.lambda;
    let count = tr.act.len();
    if tr.commitments.len() != count || tr.openings.len() != count || tr.responses.len() != count {
        return Err(InvocationError::MessageSetMismatch);
    }
    if abort_keys.len() != scheme.n() || abort_keys.iter().enumerate().any(|(i, ak)| usize::from(ak.index) != i) {
        return Err(InvocationError::KeyMismatch);
    }
    Ok(lambda)
}

/// The checks of `PartialVerify` that need no challenge: the shape of `z_i` and
/// `|z_i|_2 <= B1`.
fn response_in_bounds(scheme: &Scheme, response: &PartialResponse) -> bool {
    well_formed_z(scheme, &response.z) && &norm_squared(&response.z) <= scheme.b1_squared()
}

/// `PartialVerify` (Algorithm 7) for the signer at `position` in `tr.act`, given
/// the session challenge.
fn partial_verify_with(
    scheme: &Scheme,
    vk: &VerificationKey,
    abort_key: &AbortKey,
    challenge: &Challenge,
    opening: &Opening,
    response: &PartialResponse,
) -> bool {
    response_in_bounds(scheme, response) && relation_holds(scheme, vk, abort_key, challenge, opening, response)
}

/// `A z_i - c t_i = w_i (mod q)`, for a response of the right shape.
fn relation_holds(
    scheme: &Scheme,
    vk: &VerificationKey,
    abort_key: &AbortKey,
    challenge: &Challenge,
    opening: &Opening,
    response: &PartialResponse,
) -> bool {
    well_formed_q(scheme, &abort_key.t_i)
        && response_image(scheme, vk, &response.z, challenge, &abort_key.t_i) == opening.w
}

/// `PartialVerify` (Figure 1): whether party `party`'s response satisfies
/// `A z_i - c t_i = w_i (mod q)` and `|z_i|_2 <= B1`.
///
/// The checks on `party`'s own messages that need no challenge come first: a
/// missing or malformed opening, or a response that is missing, malformed or
/// longer than `B1`, fails whatever the other signers sent.  Only then, if the
/// opening of another signer is malformed, the challenge is undefined and the
/// call is an [`InvocationError::MalformedMessage`] naming the first such signer;
/// this is not a verdict on `party`.
pub fn partial_verify(
    scheme: &Scheme,
    vk: &VerificationKey,
    abort_keys: &[AbortKey],
    tr: &Transcript,
    party: PartyId,
) -> std::result::Result<bool, InvocationError> {
    let lambda = check_transcript(scheme, abort_keys, tr)?;
    let position = tr.act.position(party).ok_or(InvocationError::NotInActiveSet)?;
    if !well_formed_q(scheme, &tr.openings[position].w) || !response_in_bounds(scheme, &tr.responses[position]) {
        return Ok(false);
    }
    if let Some((&other, _)) =
        tr.act.parties().iter().zip(&tr.openings).find(|(_, opening)| !well_formed_q(scheme, &opening.w))
    {
        return Err(InvocationError::MalformedMessage(other));
    }
    let openings: Vec<&Opening> = tr.openings.iter().collect();
    let challenge = session_challenge(scheme, vk, &tr.sid, &tr.act, &tr.msg, &lambda, &openings);
    // The shape and the norm of the response were checked above.
    Ok(relation_holds(
        scheme,
        vk,
        &abort_keys[usize::from(party)],
        &challenge,
        &tr.openings[position],
        &tr.responses[position],
    ))
}

/// The result of [`combine`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CombineOutput {
    Signature(Signature),
    /// An identifiable abort naming every party whose opening is missing or
    /// malformed or whose partial response failed.
    Abort {
        blamed: Vec<PartyId>,
    },
}

/// `Combine` (Figure 1): partially verifies every response and, if all pass,
/// returns `sig = (c, sum_i lambda_i z_i)`.  With a missing or malformed opening
/// there is no challenge; the blame set is then the senders of such openings and
/// of responses that are malformed or longer than `B1`.
pub fn combine(
    scheme: &Scheme,
    vk: &VerificationKey,
    abort_keys: &[AbortKey],
    tr: &Transcript,
) -> std::result::Result<CombineOutput, InvocationError> {
    let lambda = check_transcript(scheme, abort_keys, tr)?;
    if tr.openings.iter().any(|opening| !well_formed_q(scheme, &opening.w)) {
        let blamed = tr
            .act
            .parties()
            .iter()
            .zip(tr.openings.iter().zip(&tr.responses))
            .filter(|(_, (opening, response))| {
                !well_formed_q(scheme, &opening.w) || !response_in_bounds(scheme, response)
            })
            .map(|(&party, _)| party)
            .collect();
        return Ok(CombineOutput::Abort { blamed });
    }
    let openings: Vec<&Opening> = tr.openings.iter().collect();
    let challenge = session_challenge(scheme, vk, &tr.sid, &tr.act, &tr.msg, &lambda, &openings);
    let blamed: Vec<PartyId> = tr
        .act
        .parties()
        .iter()
        .enumerate()
        .filter(|&(position, &party)| {
            !partial_verify_with(
                scheme,
                vk,
                &abort_keys[usize::from(party)],
                &challenge,
                &tr.openings[position],
                &tr.responses[position],
            )
        })
        .map(|(_, &party)| party)
        .collect();
    if !blamed.is_empty() {
        return Ok(CombineOutput::Abort { blamed });
    }
    let items: Vec<&[PolyZ]> = tr.responses.iter().map(|r| r.z.as_slice()).collect();
    Ok(CombineOutput::Signature(Signature { challenge, z: combine_z(&lambda, &items) }))
}

/// `Verify` (Figure 1): parses `(sid, act)` from `msg` and accepts iff
/// `c = H_c(sid, vk, act, msg, A z - c t)` and `|z|_2 <= B2`.
///
/// Like the paper's `Verify`, this does not evaluate the admissibility predicate;
/// `msg` only has to parse.  A deployment that wants verifiers to enforce a
/// signing policy on `act` must check it separately.
pub fn verify(scheme: &Scheme, vk: &VerificationKey, msg: &[u8], sig: &Signature) -> bool {
    let Ok(parsed) = SessionMessage::from_bytes(scheme, msg) else {
        return false;
    };
    if !well_formed_challenge(scheme, &sig.challenge) || !well_formed_z(scheme, &sig.z) {
        return false;
    }
    if &norm_squared(&sig.z) > scheme.b2_squared() {
        return false;
    }
    let w = response_image(scheme, vk, &sig.z, &sig.challenge, &vk.t);
    hash_to_challenge(scheme, &parsed.sid, vk, &parsed.act, msg, &w) == sig.challenge
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::{Admissibility, Category, Evidence, Profile, ProfileInputs};

    fn scheme(d: usize, omega_c: u32) -> Scheme {
        // Every row of G_hat (the last two columns) has exactly one nonzero entry.
        let g = vec![vec![1, 1, 0], vec![0, 0, 1], vec![1, 0, 1], vec![0, 1, 0]];
        let profile = Profile::derive(ProfileInputs {
            id: "challenge-test".into(),
            category: Category::Functional,
            description: String::new(),
            provenance: vec![],
            kappa: 128,
            d,
            q: crate::ring::largest_ntt_prime_below(40, d).unwrap(),
            p: 1,
            ell: 2,
            matrix_name: "toy".into(),
            matrix_source: None,
            g,
            t_uf: 1,
            t_sig: 2,
            k_c_det: None,
            q_h: 1 << 10,
            q_sign_global: 1 << 10,
            q_participate: 16,
            accounting: String::new(),
            sigma_s: 4.0,
            sigma_big_r: 4.0,
            sigma_r: 65536.0,
            omega_c,
            tail_rule: crate::params::TailRule::ClosedForm,
            admissibility: Admissibility::L1 { l1_cap: 4 },
            recon: crate::intmat::ReconParams::DEFAULT,
            evidence: Evidence::default(),
        })
        .unwrap();
        Scheme::new(profile).unwrap()
    }

    fn seeded_challenge(scheme: &Scheme, seed: &[u8]) -> Challenge {
        sample_challenge(scheme, &mut crate::sampling::SeededRandom::new(seed, "challenge-test"))
    }

    #[test]
    fn challenges_have_the_fixed_weight() {
        for (d, weight) in [(16usize, 4u32), (16, 16), (64, 1), (512, 30), (512, 60)] {
            let scheme = scheme(d, weight);
            for tag in 0u8..20 {
                let c = seeded_challenge(&scheme, &[tag; 32]);
                assert_eq!(c.coeffs().len(), d);
                assert_eq!(c.weight(), weight as usize);
                assert_eq!(c, seeded_challenge(&scheme, &[tag; 32]), "sampling is deterministic");
            }
            assert_ne!(seeded_challenge(&scheme, &[0; 32]), seeded_challenge(&scheme, &[1; 32]));
        }
    }

    /// Positions and signs are close to uniform: over many samples every position
    /// is hit about equally often and the signs balance.
    #[test]
    fn challenge_positions_and_signs_are_balanced() {
        let (d, weight, trials) = (16usize, 4u32, 8000u32);
        let scheme = scheme(d, weight);
        let mut hits = vec![0u32; d];
        let mut positive = 0u32;
        let mut stream = crate::sampling::SeededRandom::new(b"balance", "challenge-test");
        for _ in 0..trials {
            let c = sample_challenge(&scheme, &mut stream);
            for (i, &x) in c.coeffs().iter().enumerate() {
                if x != 0 {
                    hits[i] += 1;
                }
                if x == 1 {
                    positive += 1;
                }
            }
        }
        // Each position: Binomial(8000, 1/4), mean 2000, standard deviation 39.
        assert!(hits.iter().all(|&h| (1800..2200).contains(&h)), "{hits:?}");
        // Signs: Binomial(32000, 1/2), mean 16000, standard deviation 89.
        assert!((15500..16500).contains(&positive), "{positive}");
    }

    #[test]
    fn commitment_binds_every_input() {
        let scheme = scheme(16, 4);
        let mut rng = crate::sampling::SeededRandom::new(b"c", "commitment");
        let material = keygen(&scheme, &mut rng);
        let act = ActiveSet::new(&[0, 1, 2]).unwrap();
        let other_act = ActiveSet::new(&[0, 1, 3]).unwrap();
        let w = material.abort_keys[0].t_i.clone();
        let mut other_w = w.clone();
        other_w[0][0] = (other_w[0][0] + 1) % scheme.ring().q();
        let id = SessionId([1; 32]);
        let base = commitment_hash(&scheme, &id, &material.vk, &act, 0, &w);
        let variants = [
            commitment_hash(&scheme, &SessionId([2; 32]), &material.vk, &act, 0, &w),
            commitment_hash(&scheme, &id, &material.vk, &other_act, 0, &w),
            commitment_hash(&scheme, &id, &material.vk, &act, 1, &w),
            commitment_hash(&scheme, &id, &material.vk, &act, 0, &other_w),
        ];
        assert_eq!(base.0.len(), 32);
        assert!(variants.iter().all(|v| v != &base));
        let other_key = keygen(&scheme, &mut rng);
        assert_ne!(base, commitment_hash(&scheme, &id, &other_key.vk, &act, 0, &w));
    }
}
