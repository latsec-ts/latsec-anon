//! The runtime form of a validated profile, with the SEC layer on top of it:
//! signer sets, reconstruction vectors and admissibility, dealing, reconstruction,
//! and the public matrix `A = [I_p | A']`.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex, OnceLock};

use num_bigint::BigUint;
use num_traits::ToPrimitive;
use zeroize::Zeroize;

use crate::error::{Error, Result};
use crate::intmat::{recon_vector, IntMatrix};
use crate::profile::{Admissibility, Profile};
use crate::ring::{PolyQ, PolyZ, Ring};
use crate::sampling::{DiscreteGaussian, RandomSource};
use crate::xof::{domain, Xof};

/// A party index.  Zero-based here; party `i` of the paper is index `i - 1`.
pub type PartyId = u16;

/// A set of parties in canonical (strictly increasing) order.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ActiveSet(Vec<PartyId>);

impl ActiveSet {
    /// Canonicalizes a list of parties.  Duplicates and the empty set are rejected.
    pub fn new(parties: &[PartyId]) -> Result<Self> {
        let mut sorted = parties.to_vec();
        sorted.sort_unstable();
        if sorted.is_empty() || sorted.windows(2).any(|w| w[0] == w[1]) {
            return Err(Error::invalid("a signer set must be nonempty and without duplicates"));
        }
        Ok(ActiveSet(sorted))
    }

    pub fn parties(&self) -> &[PartyId] {
        &self.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn contains(&self, party: PartyId) -> bool {
        self.0.binary_search(&party).is_ok()
    }

    /// The position of `party` in canonical order.
    pub fn position(&self, party: PartyId) -> Option<usize> {
        self.0.binary_search(&party).ok()
    }

    /// Canonical encoding: the number of parties and each index, as little-endian
    /// 16-bit integers.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(2 + 2 * self.0.len());
        out.extend_from_slice(&(self.0.len() as u16).to_le_bytes());
        for party in &self.0 {
            out.extend_from_slice(&party.to_le_bytes());
        }
        out
    }

    /// Decodes [`ActiveSet::encode`], rejecting non-canonical input.
    pub fn decode(data: &[u8]) -> Result<Self> {
        if data.len() < 2 {
            return Err(Error::decode("signer set is too short"));
        }
        let count = usize::from(u16::from_le_bytes([data[0], data[1]]));
        if count == 0 || data.len() != 2 + 2 * count {
            return Err(Error::decode("signer set has the wrong length"));
        }
        let parties: Vec<PartyId> = data[2..].chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        if parties.windows(2).any(|w| w[0] >= w[1]) {
            return Err(Error::decode("signer set is not strictly increasing"));
        }
        Ok(ActiveSet(parties))
    }
}

/// Why a signer set (or a supplied reconstruction vector) may not be used.  This
/// is a session-selection outcome and never blames a signer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Inadmissible {
    /// A party index is not below `n`.
    UnknownParty(PartyId),
    /// Fewer than `t_sig` parties.
    TooFewSigners { have: usize, need: usize },
    /// `e_1^T` is not in the integer row span of `G_act`.
    NotAuthorized,
    /// `|lambda|_inf` exceeds `beta_lambda` (or reaches `q`).
    CoefficientTooLarge { linf: u64, bound: u64 },
    /// `wt(lambda)` exceeds `omega_lambda`.
    TooManyNonzero { weight: usize, bound: usize },
    /// `|lambda|_1` exceeds the cap `L`.
    L1TooLarge { l1: u64, bound: u64 },
    /// A supplied vector does not satisfy `lambda^T G_act = e_1^T`.
    NotAReconstructionVector,
    /// The session message does not parse.
    MalformedMessage,
    /// The session message names another session identifier or signer set.
    ContextMismatch,
}

impl fmt::Display for Inadmissible {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Inadmissible::UnknownParty(p) => write!(f, "party {p} is not part of the instance"),
            Inadmissible::TooFewSigners { have, need } => write!(f, "{have} signers, {need} required"),
            Inadmissible::NotAuthorized => write!(f, "the set cannot reconstruct"),
            Inadmissible::CoefficientTooLarge { linf, bound } => {
                write!(f, "reconstruction coefficient {linf} exceeds {bound}")
            }
            Inadmissible::TooManyNonzero { weight, bound } => {
                write!(f, "reconstruction vector has weight {weight}, limit {bound}")
            }
            Inadmissible::L1TooLarge { l1, bound } => {
                write!(f, "reconstruction vector has l1 norm {l1}, limit {bound}")
            }
            Inadmissible::NotAReconstructionVector => write!(f, "the supplied vector does not reconstruct"),
            Inadmissible::MalformedMessage => write!(f, "the session message does not parse"),
            Inadmissible::ContextMismatch => write!(f, "the session message names another session or signer set"),
        }
    }
}

/// The public matrix `A = [I_p | A']` with `A'` stored in the NTT domain.
///
/// `A'` is expanded from a 32-byte seed directly into NTT-domain coefficients (as
/// ML-KEM and ML-DSA do), so its definition depends on the root of unity recorded
/// in the profile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicMatrix {
    p: usize,
    ell: usize,
    /// `p x (ell - p)` polynomials, NTT domain, Montgomery form.
    a_hat: Vec<Vec<PolyQ>>,
}

impl PublicMatrix {
    /// Expands `A'` from `seed`: entry `(i, j)` is read from
    /// `SHAKE256(frame("LaTSEC-v1/expand-A") || frame(seed) || frame(i) || frame(j))`
    /// by rejection sampling of `ceil(lg q)`-bit little-endian integers.
    pub fn expand(ring: &Ring, p: usize, ell: usize, seed: &[u8; 32]) -> Self {
        let q = ring.q();
        let bits = 128 - (q - 1).leading_zeros();
        let nbytes = bits.div_ceil(8) as usize;
        let mask = if bits == 128 { u128::MAX } else { (1u128 << bits) - 1 };
        let mut a_hat = Vec::with_capacity(p);
        for i in 0..p {
            let mut row = Vec::with_capacity(ell - p);
            for j in 0..ell - p {
                let mut xof = Xof::new(domain::EXPAND_A);
                xof.absorb(seed).absorb_u64(i as u64).absorb_u64(j as u64);
                let mut stream = xof.finalize();
                let mut poly = Vec::with_capacity(ring.degree());
                while poly.len() < ring.degree() {
                    let mut bytes = [0u8; 16];
                    stream.read(&mut bytes[..nbytes]);
                    let x = u128::from_le_bytes(bytes) & mask;
                    if x < q {
                        poly.push(ring.modulus().to_mont(x));
                    }
                }
                row.push(poly);
            }
            a_hat.push(row);
        }
        PublicMatrix { p, ell, a_hat }
    }

    /// `A v mod q` for `v` in `R_q^ell`.  The transformed copy of `v` is erased,
    /// since `v` may be the image of a secret.
    pub fn mul_q(&self, ring: &Ring, v: &[PolyQ]) -> Vec<PolyQ> {
        assert_eq!(v.len(), self.ell);
        let m = ring.modulus();
        let mut tail: Vec<PolyQ> = v[self.p..]
            .iter()
            .map(|poly| {
                let mut f = poly.clone();
                ring.ntt(&mut f);
                f
            })
            .collect();
        let mut out = Vec::with_capacity(self.p);
        for (i, row) in self.a_hat.iter().enumerate() {
            let mut acc = ring.zero();
            for (a, x) in row.iter().zip(&tail) {
                for ((dst, &ac), &xc) in acc.iter_mut().zip(a).zip(x) {
                    *dst = m.add(*dst, m.mont_mul(ac, xc));
                }
            }
            ring.intt(&mut acc);
            out.push(ring.add(&acc, &v[i]));
        }
        tail.zeroize();
        out
    }

    /// `A v mod q` for an integer vector `v` in `R^ell`.
    pub fn mul_z(&self, ring: &Ring, v: &[PolyZ]) -> Vec<PolyQ> {
        let mut reduced: Vec<PolyQ> = v.iter().map(|poly| ring.reduce(poly)).collect();
        let out = self.mul_q(ring, &reduced);
        reduced.zeroize();
        out
    }
}

/// Bounded cache of reconstruction results, keyed by signer set.
struct ReconCache {
    capacity: usize,
    order: VecDeque<ActiveSet>,
    entries: HashMap<ActiveSet, std::result::Result<Arc<Vec<i64>>, Inadmissible>>,
    hits: u64,
    misses: u64,
}

/// A validated profile with everything precomputed for execution.
pub struct Scheme {
    profile: Profile,
    digest: Vec<u8>,
    ring: Ring,
    g: IntMatrix,
    g_rows: Vec<Vec<i64>>,
    gauss_s: DiscreteGaussian,
    gauss_big_r: DiscreteGaussian,
    gauss_r: DiscreteGaussian,
    beta_sh: u128,
    b1: u128,
    b2: u128,
    b1_squared: BigUint,
    b2_squared: BigUint,
    /// Length limits of the decoders, computed on first use ([`crate::latsec::limits`]).
    pub(crate) max_response_len: OnceLock<usize>,
    pub(crate) max_signature_len: OnceLock<usize>,
    cache: Mutex<ReconCache>,
}

impl fmt::Debug for Scheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Scheme({})", self.profile.id)
    }
}

impl Scheme {
    /// Default number of cached reconstruction vectors.
    pub const DEFAULT_CACHE_CAPACITY: usize = 4096;

    /// Validates the profile and prepares the runtime data.
    pub fn new(profile: Profile) -> Result<Self> {
        profile.validate()?;
        let ring = Ring::with_root(profile.ring.d, profile.ring.q, profile.ring.psi)?;
        let g_rows: Vec<Vec<i64>> = profile.sec.g.iter().map(|r| r.iter().map(|&x| i64::from(x)).collect()).collect();
        let g = IntMatrix::from_i64(profile.sec.k, &g_rows);
        let (sigma_s, sigma_big_r, sigma_r) = profile.sigmas()?;
        let to_u128 = |x: &BigUint| x.to_u128().ok_or_else(|| Error::invalid("bound exceeds 128 bits"));
        let (beta_sh, b1, b2) =
            (to_u128(&profile.bounds.beta_sh)?, to_u128(&profile.bounds.b1)?, to_u128(&profile.bounds.b2)?);
        Ok(Scheme {
            digest: profile.compute_digest(),
            ring,
            g,
            g_rows,
            gauss_s: DiscreteGaussian::new(sigma_s),
            gauss_big_r: DiscreteGaussian::new(sigma_big_r),
            gauss_r: DiscreteGaussian::new(sigma_r),
            beta_sh,
            b1,
            b2,
            b1_squared: &profile.bounds.b1 * &profile.bounds.b1,
            b2_squared: &profile.bounds.b2 * &profile.bounds.b2,
            max_response_len: OnceLock::new(),
            max_signature_len: OnceLock::new(),
            cache: Mutex::new(ReconCache {
                capacity: Self::DEFAULT_CACHE_CAPACITY,
                order: VecDeque::new(),
                entries: HashMap::new(),
                hits: 0,
                misses: 0,
            }),
            profile,
        })
    }

    pub fn profile(&self) -> &Profile {
        &self.profile
    }

    /// Whether the profile is meant for signing.  SEC-only profiles are not: their
    /// modulus is not sized for the signature's MSIS bound.
    pub fn supports_signing(&self) -> bool {
        self.profile.category != crate::profile::Category::SecOnly
    }

    /// The profile digest that every protocol hash is bound to.
    pub fn digest(&self) -> &[u8] {
        &self.digest
    }

    pub fn ring(&self) -> &Ring {
        &self.ring
    }

    pub fn n(&self) -> usize {
        self.profile.sec.n
    }

    pub fn k(&self) -> usize {
        self.profile.sec.k
    }

    /// The SEC matrix `G`.
    pub fn matrix(&self) -> &IntMatrix {
        &self.g
    }

    pub fn beta_sh(&self) -> u128 {
        self.beta_sh
    }

    pub fn b1(&self) -> u128 {
        self.b1
    }

    pub fn b2(&self) -> u128 {
        self.b2
    }

    pub fn b1_squared(&self) -> &BigUint {
        &self.b1_squared
    }

    pub fn b2_squared(&self) -> &BigUint {
        &self.b2_squared
    }

    /// The sampler of the signing masks, `D_{sigma_r}`.
    pub fn mask_sampler(&self) -> &DiscreteGaussian {
        &self.gauss_r
    }

    /// The sampler of the signing secret, `D_{sigma_s}`.
    pub fn secret_sampler(&self) -> &DiscreteGaussian {
        &self.gauss_s
    }

    /// Sets the number of cached reconstruction results and clears the cache.
    pub fn set_cache_capacity(&self, capacity: usize) {
        let mut cache = self.cache.lock().unwrap();
        cache.capacity = capacity;
        cache.entries.clear();
        cache.order.clear();
    }

    /// `(hits, misses)` of the reconstruction cache.
    pub fn cache_statistics(&self) -> (u64, u64) {
        let cache = self.cache.lock().unwrap();
        (cache.hits, cache.misses)
    }

    fn check_parties(&self, act: &ActiveSet) -> std::result::Result<(), Inadmissible> {
        match act.parties().iter().find(|&&p| usize::from(p) >= self.n()) {
            Some(&p) => Err(Inadmissible::UnknownParty(p)),
            None => Ok(()),
        }
    }

    /// The reconstruction vector of an authorized set, ignoring the signing
    /// threshold and the coefficient bounds.  This is the SEC-level algorithm.
    pub fn recon_vector(&self, set: &ActiveSet) -> std::result::Result<Vec<i64>, Inadmissible> {
        self.check_parties(set)?;
        let rows: Vec<usize> = set.parties().iter().map(|&p| usize::from(p)).collect();
        let lambda =
            recon_vector(&self.g.select_rows(&rows), self.profile.recon.params()).ok_or(Inadmissible::NotAuthorized)?;
        lambda
            .iter()
            .map(|x| x.to_i64().ok_or(Inadmissible::CoefficientTooLarge { linf: u64::MAX, bound: i64::MAX as u64 }))
            .collect()
    }

    /// The coefficient bounds of the admissibility predicate in force.
    pub fn check_bounds(&self, lambda: &[i64]) -> std::result::Result<(), Inadmissible> {
        let linf = lambda.iter().map(|x| x.unsigned_abs()).max().unwrap_or(0);
        let l1 = lambda.iter().fold(0u64, |acc, x| acc.saturating_add(x.unsigned_abs()));
        let weight = lambda.iter().filter(|&&x| x != 0).count();
        if u128::from(linf) >= self.ring.q() {
            return Err(Inadmissible::CoefficientTooLarge {
                linf,
                bound: (self.ring.q() - 1).min(u64::MAX as u128) as u64,
            });
        }
        match self.profile.admissibility {
            Admissibility::Paper { beta_lambda, omega_lambda } => {
                if linf > u64::from(beta_lambda) {
                    return Err(Inadmissible::CoefficientTooLarge { linf, bound: u64::from(beta_lambda) });
                }
                if weight > omega_lambda as usize {
                    return Err(Inadmissible::TooManyNonzero { weight, bound: omega_lambda as usize });
                }
            }
            Admissibility::L1 { l1_cap } => {
                if l1 > u64::from(l1_cap) {
                    return Err(Inadmissible::L1TooLarge { l1, bound: u64::from(l1_cap) });
                }
            }
        }
        Ok(())
    }

    /// The admissibility predicate `P(act)`: at least `t_sig` known parties, an
    /// authorized set, and a reconstruction vector within the profile's bounds.
    /// Returns the fixed vector `lambda_act`, in the canonical order of `act`.
    pub fn lambda(&self, act: &ActiveSet) -> std::result::Result<Arc<Vec<i64>>, Inadmissible> {
        {
            let mut cache = self.cache.lock().unwrap();
            if let Some(found) = cache.entries.get(act).cloned() {
                cache.hits += 1;
                return found;
            }
            cache.misses += 1;
        }
        let result = self.lambda_uncached(act).map(Arc::new);
        let mut cache = self.cache.lock().unwrap();
        if cache.capacity > 0 && !cache.entries.contains_key(act) {
            while cache.entries.len() >= cache.capacity {
                match cache.order.pop_front() {
                    Some(oldest) => cache.entries.remove(&oldest),
                    None => break,
                };
            }
            cache.order.push_back(act.clone());
            cache.entries.insert(act.clone(), result.clone());
        }
        result
    }

    fn lambda_uncached(&self, act: &ActiveSet) -> std::result::Result<Vec<i64>, Inadmissible> {
        self.check_parties(act)?;
        if act.len() < self.profile.sec.t_sig {
            return Err(Inadmissible::TooFewSigners { have: act.len(), need: self.profile.sec.t_sig });
        }
        let lambda = self.recon_vector(act)?;
        self.check_bounds(&lambda)?;
        Ok(lambda)
    }

    /// Checks a reconstruction vector supplied by a coordinator: the set must have
    /// at least `t_sig` known parties, `lambda^T G_act = e_1^T` must hold over the
    /// integers, and the profile's bounds must hold.  The vector need not be the
    /// one this crate's algorithm returns.
    pub fn check_supplied_lambda(&self, act: &ActiveSet, lambda: &[i64]) -> std::result::Result<(), Inadmissible> {
        self.check_parties(act)?;
        if act.len() < self.profile.sec.t_sig {
            return Err(Inadmissible::TooFewSigners { have: act.len(), need: self.profile.sec.t_sig });
        }
        if lambda.len() != act.len() {
            return Err(Inadmissible::NotAReconstructionVector);
        }
        let rows: Vec<&Vec<i64>> = act.parties().iter().map(|&party| &self.g_rows[usize::from(party)]).collect();
        for col in 0..self.k() {
            let acc: i128 =
                lambda.iter().zip(&rows).map(|(&coeff, row)| i128::from(coeff) * i128::from(row[col])).sum();
            if acc != i128::from(col == 0) {
                return Err(Inadmissible::NotAReconstructionVector);
            }
        }
        self.check_bounds(lambda)
    }

    /// `SEC.Deal`: shares `s_i = g_i s + sum_j G_hat[i][j] R_j` of a secret in
    /// `R^ell`, with a fresh Gaussian masking matrix `R` that is erased afterwards.
    pub fn deal(&self, secret: &[PolyZ], rng: &mut dyn RandomSource) -> Vec<Vec<PolyZ>> {
        let (d, ell) = (self.ring.degree(), self.profile.ell);
        assert!(secret.len() == ell && secret.iter().all(|p| p.len() == d));
        let masks = self.k() - 1;
        let mut r: Vec<Vec<PolyZ>> = (0..masks)
            .map(|_| {
                (0..ell)
                    .map(|_| {
                        let mut poly = vec![0i128; d];
                        self.gauss_big_r.sample_into(rng, &mut poly);
                        poly
                    })
                    .collect()
            })
            .collect();
        let shares = self
            .g_rows
            .iter()
            .map(|row| {
                let mut share = vec![vec![0i128; d]; ell];
                add_scaled_z(&mut share, row[0], secret);
                for (j, mask) in r.iter().enumerate() {
                    add_scaled_z(&mut share, row[j + 1], mask);
                }
                share
            })
            .collect();
        r.zeroize();
        shares
    }

    /// `SEC.Rec` over the integers: at least `k_c` parties and an authorized set.
    /// The coefficient bounds of the signing predicate are not required here.
    pub fn reconstruct(&self, set: &ActiveSet, shares: &[&[PolyZ]]) -> std::result::Result<Vec<PolyZ>, Inadmissible> {
        if set.len() < self.profile.sec.t_sig {
            return Err(Inadmissible::TooFewSigners { have: set.len(), need: self.profile.sec.t_sig });
        }
        assert_eq!(shares.len(), set.len());
        let lambda = self.recon_vector(set)?;
        Ok(combine_z(&lambda, shares))
    }

    /// The linear combination `sum_i lambda_i v_i mod q` of public images.
    pub fn combine_q(&self, lambda: &[i64], items: &[&[PolyQ]]) -> Vec<PolyQ> {
        assert_eq!(lambda.len(), items.len());
        let count = items.first().map_or(0, |v| v.len());
        let mut out = vec![self.ring.zero(); count];
        for (&coeff, item) in lambda.iter().zip(items) {
            for (acc, poly) in out.iter_mut().zip(item.iter()) {
                self.ring.add_scaled(acc, coeff, poly);
            }
        }
        out
    }
}

/// `acc += scalar * v` for integer polynomial vectors.
fn add_scaled_z(acc: &mut [PolyZ], scalar: i64, v: &[PolyZ]) {
    if scalar == 0 {
        return;
    }
    let s = i128::from(scalar);
    for (dst, src) in acc.iter_mut().zip(v) {
        for (x, &y) in dst.iter_mut().zip(src) {
            *x += s * y;
        }
    }
}

/// The integer linear combination `sum_i lambda_i v_i`.
pub fn combine_z(lambda: &[i64], items: &[&[PolyZ]]) -> Vec<PolyZ> {
    assert_eq!(lambda.len(), items.len());
    let Some(first) = items.first() else { return Vec::new() };
    let mut out: Vec<PolyZ> = first.iter().map(|p| vec![0i128; p.len()]).collect();
    for (&coeff, item) in lambda.iter().zip(items) {
        add_scaled_z(&mut out, coeff, item);
    }
    out
}

/// The squared Euclidean norm of an integer polynomial vector.
pub fn norm_squared(v: &[PolyZ]) -> BigUint {
    let mut total = BigUint::from(0u8);
    let mut acc: u128 = 0;
    for &x in v.iter().flatten() {
        let a = x.unsigned_abs();
        if a >> 63 == 0 {
            let square = a * a;
            match acc.checked_add(square) {
                Some(sum) => acc = sum,
                None => {
                    total += acc;
                    acc = square;
                }
            }
        } else {
            let big = BigUint::from(a);
            total += &big * &big;
        }
    }
    total + acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_squared_matches_big_integers() {
        let v: Vec<PolyZ> = vec![
            vec![0, 1, -1, 3, -(1 << 62), (1 << 63) - 1],
            vec![1 << 63, -(1 << 63), (1 << 100) + 7, -((1 << 119) - 1), i128::from(u64::MAX), 5],
        ];
        let expected: BigUint = v
            .iter()
            .flatten()
            .map(|x| {
                let a = BigUint::from(x.unsigned_abs());
                &a * &a
            })
            .sum();
        assert_eq!(norm_squared(&v), expected);
        assert_eq!(norm_squared(&[]), BigUint::from(0u8));
        // Many large squares overflow a 128-bit accumulator; the sum must not wrap.
        let many = vec![vec![(1i128 << 62) + 1; 64]];
        let one = BigUint::from((1u128 << 62) + 1);
        assert_eq!(norm_squared(&many), &one * &one * 64u32);
    }

    #[test]
    fn integer_combination() {
        let a: Vec<PolyZ> = vec![vec![1, 2], vec![3, 4]];
        let b: Vec<PolyZ> = vec![vec![10, 20], vec![30, 40]];
        let c: Vec<PolyZ> = vec![vec![-1, 0], vec![0, 1]];
        let out = combine_z(&[2, -1, 0], &[&a, &b, &c]);
        assert_eq!(out, vec![vec![-8, -16], vec![-24, -32]]);
        assert!(combine_z(&[], &[]).is_empty());
    }

    #[test]
    fn signer_set_encoding() {
        let set = ActiveSet::new(&[300, 2, 17]).unwrap();
        assert_eq!(set.parties(), &[2, 17, 300]);
        assert_eq!(set.encode(), vec![3, 0, 2, 0, 17, 0, 44, 1]);
        assert_eq!(ActiveSet::decode(&set.encode()).unwrap(), set);
        assert_eq!(set.position(17), Some(1));
        assert_eq!(set.position(18), None);
        assert!(set.contains(300) && !set.contains(0));
        assert_eq!(set.len(), 3);
    }

    /// `A v` computed through the NTT equals `v_top + A' v_bottom` computed with
    /// schoolbook products on the coefficient form of `A'`.
    #[test]
    fn public_matrix_product_matches_the_definition() {
        let mid = crate::ring::largest_ntt_prime_below(40, 64).unwrap();
        for (d, q) in [(16usize, 97u128), (64, mid), (512, (1 << 66) - 16383)] {
            let ring = Ring::new(d, q).unwrap();
            let (p, ell) = (2, 5);
            let seed = [9u8; 32];
            let matrix = PublicMatrix::expand(&ring, p, ell, &seed);
            assert_eq!(matrix, PublicMatrix::expand(&ring, p, ell, &seed), "expansion is deterministic");
            assert_ne!(matrix, PublicMatrix::expand(&ring, p, ell, &[8u8; 32]));
            // Coefficient form of A': leave Montgomery form, then invert the NTT.
            let a_prime: Vec<Vec<PolyQ>> = matrix
                .a_hat
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|poly| {
                            let mut plain: PolyQ = poly.iter().map(|&x| ring.modulus().mont_mul(x, 1)).collect();
                            assert!(plain.iter().all(|&x| x < q));
                            ring.intt(&mut plain);
                            plain
                        })
                        .collect()
                })
                .collect();
            let v: Vec<PolyZ> =
                (0..ell).map(|i| (0..d).map(|j| ((i * 1009 + j * 7919) as i128 % 2001) - 1000).collect()).collect();
            let got = matrix.mul_z(&ring, &v);
            for i in 0..p {
                let mut expected = ring.reduce(&v[i]);
                for j in 0..ell - p {
                    expected = ring.add(&expected, &ring.mul_schoolbook(&a_prime[i][j], &ring.reduce(&v[p + j])));
                }
                assert_eq!(got[i], expected, "d = {d}, q = {q}, row {i}");
            }
        }
    }
}
