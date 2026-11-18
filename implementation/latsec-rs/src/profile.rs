//! Machine-readable parameter profiles.
//!
//! A profile fixes everything the protocol needs: the ring, the SEC matrix and
//! thresholds, exact Gaussian widths, the challenge weight, tail factors, the
//! admissibility predicate, integer verification radii, and the identifiers of the
//! reconstruction algorithm, hash and codec.  It also carries evidence fields
//! (provenance, matrix constants, estimator results, claim status) that do not
//! affect protocol behavior.
//!
//! # Exactness
//!
//! Real-valued inputs (widths, tail factors) are stored as dyadic rationals
//! `num / 2^frac_bits`.  All integer bounds are derived from them in exact integer
//! arithmetic by [`Profile::derive_bounds`], and [`Profile::validate`] recomputes
//! them on load.  Logarithms printed anywhere are display values only.
//!
//! ```text
//! D            = d * ell
//! sqrtD_up     = ceil(sqrt(D) * 2^32) / 2^32
//! beta_s       = theta_s * sigma_s * sqrtD_up
//! beta_R       = theta_R * sigma_R
//! beta_sh      = beta_g * beta_s + c_exp * beta_R        (stored rounded up)
//! beta_r       = theta_r * sigma_r * sqrtD_up
//! B1           = ceil(omega_c * sqrtD_up * beta_sh + beta_r)
//! B2           = L * B1,   L = omega_lambda * beta_lambda  or the l1 cap
//! B_MSIS^2     = 4 * (B2^2 + omega_c)
//! ```
//!
//! # Digest
//!
//! [`Profile::compute_digest`] hashes every field that affects protocol behavior or
//! the bounds.  Descriptions, provenance and evidence are not covered.

use std::path::Path;

use num_bigint::BigUint;
use num_traits::{One, ToPrimitive, Zero};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::fixture::MatrixSource;
use crate::intmat::{LllDelta, ReconParams};
use crate::params::{self, ceil_lg, tail_exponent_bits, tail_theta, TailRule};
use crate::ring::Ring;
use crate::sampling::GaussianSigma;
use crate::xof::{domain, Xof};

/// The profile format implemented here.
pub const PROFILE_FORMAT: &str = "latsec-profile-v1";
/// The only Gaussian convention accepted: `Pr[x] ~ exp(-x^2 / (2 sigma^2))`.
pub const GAUSSIAN_CONVENTION: &str = "sigma-exp-half";
/// Identifier of the sampler in [`crate::sampling`].
pub const SAMPLER_ID: &str = "cks-exact-v1";
/// Identifier of the reconstruction algorithm in [`crate::intmat`].
pub const RECON_ALGORITHM_ID: &str = "xlatsec-recon-v1";
/// Identifier of the hash construction in [`crate::xof`].
pub const XOF_ID: &str = "shake256-framed-v1";
/// The codec version in [`crate::codec`].  Version 1 carried a `2 kappa`-bit
/// challenge digest in signatures and version 2 the challenge polynomial at two
/// bits per coefficient; both stored Gaussian components at fixed width.  Version
/// 3 ranks the challenge, Golomb-Rice codes the Gaussian components, and adds the
/// session message that carries `(sid, act, lambda_act)`.
pub const CODEC_VERSION: u32 = 3;
/// Length in bytes of profile and key digests.
pub const DIGEST_BYTES: usize = 64;
/// Fractional bits of `sqrtD_up` and of the tail factors.
const FIXED_FRAC_BITS: u32 = 32;

mod dec_big {
    use num_bigint::BigUint;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(value: &BigUint, ser: S) -> Result<S::Ok, S::Error> {
        ser.serialize_str(&value.to_string())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(de: D) -> Result<BigUint, D::Error> {
        let text = String::deserialize(de)?;
        if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) || (text.len() > 1 && text.starts_with('0')) {
            return Err(serde::de::Error::custom("expected a canonical decimal integer string"));
        }
        text.parse().map_err(serde::de::Error::custom)
    }
}

mod dec_u128 {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(value: &u128, ser: S) -> Result<S::Ok, S::Error> {
        ser.serialize_str(&value.to_string())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(de: D) -> Result<u128, D::Error> {
        let text = String::deserialize(de)?;
        if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) || (text.len() > 1 && text.starts_with('0')) {
            return Err(serde::de::Error::custom("expected a canonical decimal integer string"));
        }
        text.parse().map_err(serde::de::Error::custom)
    }
}

/// A nonnegative dyadic rational `num / 2^frac_bits`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dyadic {
    #[serde(with = "dec_big")]
    pub num: BigUint,
    pub frac_bits: u32,
}

impl Dyadic {
    pub fn from_integer(value: u128) -> Self {
        Dyadic { num: BigUint::from(value), frac_bits: 0 }
    }

    /// The smallest dyadic with `frac_bits` fractional bits that is at least
    /// `value * (1 + 2^-40)`.  The guard covers `f64` rounding in the caller.
    pub fn round_up_from_f64(value: f64, frac_bits: u32) -> Result<Self> {
        let sigma = GaussianSigma::round_up_from_f64(value, frac_bits)?;
        Ok(Dyadic { num: BigUint::from(sigma.num), frac_bits })
    }

    pub fn mul(&self, other: &Dyadic) -> Dyadic {
        Dyadic { num: &self.num * &other.num, frac_bits: self.frac_bits + other.frac_bits }
    }

    pub fn mul_int(&self, factor: u128) -> Dyadic {
        Dyadic { num: &self.num * BigUint::from(factor), frac_bits: self.frac_bits }
    }

    pub fn add(&self, other: &Dyadic) -> Dyadic {
        let frac_bits = self.frac_bits.max(other.frac_bits);
        let a = &self.num << (frac_bits - self.frac_bits);
        let b = &other.num << (frac_bits - other.frac_bits);
        Dyadic { num: a + b, frac_bits }
    }

    /// The smallest integer that is at least this value.
    pub fn ceil(&self) -> BigUint {
        let floor = &self.num >> self.frac_bits;
        if (&floor << self.frac_bits) == self.num {
            floor
        } else {
            floor + 1u32
        }
    }

    pub fn to_f64(&self) -> f64 {
        // Good to f64 precision for the magnitudes used here.
        let bits = self.num.bits();
        if bits <= 1000 {
            self.num.to_f64().unwrap_or(f64::INFINITY) / f64::from(self.frac_bits).exp2()
        } else {
            f64::INFINITY
        }
    }

    fn to_sigma(&self) -> Result<GaussianSigma> {
        let num = self.num.to_u128().ok_or_else(|| Error::invalid("Gaussian width is too large"))?;
        GaussianSigma::new(num, self.frac_bits)
    }

    fn from_sigma(sigma: GaussianSigma) -> Self {
        Dyadic { num: BigUint::from(sigma.num), frac_bits: sigma.frac_bits }
    }

    fn absorb(&self, xof: &mut Xof) {
        xof.absorb(self.num.to_string().as_bytes()).absorb_u64(u64::from(self.frac_bits));
    }
}

/// The purpose of a profile and the kind of claim it may carry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Category {
    /// Small, exhaustively checked examples.  No claimed security level.
    Functional,
    /// Our own matrices and parameters.
    Independent,
    /// The paper's selected instances, as published.
    Author,
    /// A variant we derived from an author instance.
    AuthorVariant,
    /// A standalone SEC parameter set of the SEC paper [46].  Privacy-only: such a
    /// profile supports dealing, reconstruction and leakage responses, and makes
    /// no signature claim.
    SecOnly,
}

impl Category {
    fn tag(self) -> &'static str {
        match self {
            Category::Functional => "functional",
            Category::Independent => "independent",
            Category::Author => "author",
            Category::AuthorVariant => "author-variant",
            Category::SecOnly => "sec-only",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RingSpec {
    /// Ring degree `d`, a power of two.
    pub d: usize,
    /// The prime modulus, `q = 1 (mod 2d)`.
    #[serde(with = "dec_u128")]
    pub q: u128,
    /// The primitive `2d`-th root of unity that defines the NTT and the expansion
    /// of the public matrix: `g^((q-1)/(2d))` for the smallest valid `g >= 2`.
    #[serde(with = "dec_u128")]
    pub psi: u128,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SecSpec {
    pub n: usize,
    pub k: usize,
    /// Nonzero entries per row of `G_hat`.
    pub c_exp: usize,
    /// `|g|_inf`.
    pub beta_g: u32,
    /// Privacy threshold `k_p`, which is the unforgeability threshold `t_uf`.
    pub t_uf: usize,
    /// Operational signing threshold `t_sig = k_c`.
    pub t_sig: usize,
    /// Size from which every set is authorized, if established.
    pub k_c_det: Option<usize>,
    pub matrix_name: String,
    pub matrix_source: Option<MatrixSource>,
    /// The matrix `G = [g | G_hat]`, one row per party.  Party `i` of the paper
    /// (one-based) is row `i - 1`.
    pub g: Vec<Vec<i8>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Budgets {
    /// Random-oracle queries assumed by the analysis.
    #[serde(with = "dec_u128")]
    pub q_h: u128,
    /// Global signing sessions assumed by the analysis.  Not enforced locally.
    #[serde(with = "dec_u128")]
    pub q_sign_global: u128,
    /// Sessions each signer will take part in per key.  Enforced by every signer.
    #[serde(with = "dec_u128")]
    pub q_participate: u128,
    /// How the two signing budgets relate.
    pub accounting: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GaussianSpec {
    pub convention: String,
    pub sigma_s: Dyadic,
    pub sigma_big_r: Dyadic,
    pub sigma_r: Dyadic,
    pub sampler: String,
    pub sampler_statistical_distance: String,
    pub sampler_timing: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChallengeSpec {
    /// Hamming weight of a challenge.  A signature carries the challenge itself.
    pub omega_c: u32,
    /// Length of a round-1 commitment.
    pub commitment_bytes: usize,
}

/// Golomb-Rice parameters of the Gaussian-distributed encodings, from
/// [`rice_parameter`] applied to the coefficient variances.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EncodingSpec {
    /// Parameter for a share coefficient of a party with `|g_i| = j`, at index `j`
    /// (`0 <= j <= beta_g`).  The variance is `g_i^2 sigma_s^2 + c_exp sigma_R^2`.
    pub rice_share: Vec<u32>,
    /// Parameter for a partial-response coefficient.  The variance is
    /// `sigma_r^2 + omega_c (beta_g^2 sigma_s^2 + c_exp sigma_R^2)`.
    pub rice_response: u32,
}

/// The Golomb-Rice parameter for a centred Gaussian of variance `sigma^2`: the
/// largest `k >= 0` with `1.2863 * 2^k <= sigma` (and `0` below that).
///
/// A codeword is `k` low bits, the high part `|x| >> k` in unary and a sign bit
/// for `x != 0`.  This is a large-width rule.  The width at which `k` and `k + 1`
/// give equal expected length tends to `2.5726 * 2^k` from below as `k` grows,
/// and comes earlier for small widths (`sigma = 2` is coded shorter with `k = 1`
/// than with the rule's `k = 0`).  `tools/rice_parameters.py` derives the constant
/// and finds a loss against the optimal parameter of at most 0.064 bits per
/// coefficient for `sigma >= 8` (at most 0.001 bits from `k = 8` on), an excess
/// over the entropy of at most 0.17 bits per coefficient for `sigma >= 2^15`, and,
/// modelling each component as a discrete Gaussian of its variance, that every
/// stored profile parameter minimizes the expected length, with every width at
/// least 4% from a switch point.  The rule is evaluated
/// in exact rational arithmetic, so it does not depend on floating point.
pub fn rice_parameter(variance: &Dyadic) -> u32 {
    // (12863/10000 * 2^k)^2 <= num / 2^frac  <=>  12863^2 * 4^k * 2^frac <= 10^8 * num.
    let lhs_base = BigUint::from(12863u32 * 12863u32) << variance.frac_bits;
    let rhs = &variance.num * BigUint::from(100_000_000u32);
    let mut k = 0u32;
    while (&lhs_base << (2 * (k + 1))) <= rhs {
        k += 1;
    }
    k
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TailSpec {
    /// `closed-form`, `exact-inversion` or `explicit`.
    pub rule: String,
    /// Tail budget of the masking matrix entries, `kappa + ceil(lg((k-1) D))`.
    pub kappa_big_r: u32,
    /// Tail budget of the signing masks, `kappa + ceil(lg(n Q_Sign))`.
    pub kappa_r: u32,
    pub theta_s: Dyadic,
    pub theta_big_r: Dyadic,
    pub theta_r: Dyadic,
}

/// The admissibility predicate in force (Section 3.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "predicate", rename_all = "kebab-case")]
pub enum Admissibility {
    /// The paper's predicate: `|lambda|_inf <= beta_lambda` and
    /// `wt(lambda) <= omega_lambda`.  Implies `|lambda|_1 <= beta_lambda * omega_lambda`.
    Paper { beta_lambda: u32, omega_lambda: u32 },
    /// The general predicate: `|lambda|_1 <= l1_cap` (and `|lambda|_inf < q`).
    L1 { l1_cap: u32 },
}

impl Admissibility {
    /// The bound on `|lambda|_inf` that the predicate implies, which also bounds the
    /// encoding of `lambda_act` in a session message.
    pub fn linf_bound(&self) -> u32 {
        match *self {
            Admissibility::Paper { beta_lambda, .. } => beta_lambda,
            Admissibility::L1 { l1_cap } => l1_cap,
        }
    }

    /// The integer `L` with `B2 = L * B1`.
    pub fn l1_bound(&self) -> u64 {
        match *self {
            Admissibility::Paper { beta_lambda, omega_lambda } => u64::from(beta_lambda) * u64::from(omega_lambda),
            Admissibility::L1 { l1_cap } => u64::from(l1_cap),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundSpec {
    /// `ceil(beta_sh)`: bound on the absolute value of a share coefficient.
    #[serde(with = "dec_big")]
    pub beta_sh: BigUint,
    /// `l_2` radius of a partial response.
    #[serde(with = "dec_big")]
    pub b1: BigUint,
    /// `l_2` radius of a signature response, `L * B1`.
    #[serde(with = "dec_big")]
    pub b2: BigUint,
    /// `B_MSIS^2 = 4 (B2^2 + omega_c)`.
    #[serde(with = "dec_big")]
    pub b_msis_squared: BigUint,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReconSpec {
    pub algorithm: String,
    pub lll_delta_num: u32,
    pub lll_delta_den: u32,
    pub passes: u32,
    pub window: u32,
}

impl ReconSpec {
    pub fn params(&self) -> ReconParams {
        ReconParams {
            delta: LllDelta { num: self.lll_delta_num, den: self.lll_delta_den },
            passes: self.passes,
            window: self.window,
        }
    }
}

/// Matrix constants with their evaluation method.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MatrixConstantsRecord {
    pub alpha_kp: f64,
    pub norm_g_hat: f64,
    pub norm_g: f64,
    pub beta_q: f64,
    pub beta_kp: f64,
    pub gamma_kp: f64,
    pub method: String,
    pub certified: bool,
}

/// One lattice-estimator result.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EstimateRecord {
    /// `MQ`, `AK` or `MSIS`.
    pub instance: String,
    pub bits: f64,
    pub target_bits: f64,
    pub source: String,
}

/// Reconstruction statistics behind the admissibility bounds.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReconStatistics {
    pub observed_linf_max: u32,
    pub observed_weight_max: u32,
    pub pipeline: String,
    pub note: String,
}

/// Fields that document a profile without affecting the protocol.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Evidence {
    /// What may be claimed for this profile.
    pub claim: String,
    pub matrix_constants: Option<MatrixConstantsRecord>,
    pub recon_statistics: Option<ReconStatistics>,
    pub estimates: Vec<EstimateRecord>,
    pub notes: Vec<String>,
}

/// A complete parameter profile.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Profile {
    pub format: String,
    pub id: String,
    pub category: Category,
    pub description: String,
    pub provenance: Vec<String>,
    /// Target security level in bits.  For functional profiles this only sizes the
    /// tail budgets and hash outputs.
    pub kappa: u32,
    pub ring: RingSpec,
    /// Rows of the public matrix `A = [I_p | A']`.
    pub p: usize,
    /// Columns of `A`; secrets, shares and responses live in `R^ell`.
    pub ell: usize,
    pub sec: SecSpec,
    pub budgets: Budgets,
    pub gaussian: GaussianSpec,
    pub challenge: ChallengeSpec,
    pub encoding: EncodingSpec,
    pub tails: TailSpec,
    pub admissibility: Admissibility,
    pub bounds: BoundSpec,
    pub recon: ReconSpec,
    pub xof: String,
    pub codec_version: u32,
    pub evidence: Evidence,
    /// Hex digest of the protocol-relevant fields; see [`Profile::compute_digest`].
    pub digest: String,
}

/// Inputs from which [`Profile::derive`] builds a profile.
#[derive(Clone, Debug)]
pub struct ProfileInputs {
    pub id: String,
    pub category: Category,
    pub description: String,
    pub provenance: Vec<String>,
    pub kappa: u32,
    pub d: usize,
    pub q: u128,
    pub p: usize,
    pub ell: usize,
    pub matrix_name: String,
    pub matrix_source: Option<MatrixSource>,
    pub g: Vec<Vec<i64>>,
    pub t_uf: usize,
    pub t_sig: usize,
    pub k_c_det: Option<usize>,
    pub q_h: u128,
    pub q_sign_global: u128,
    pub q_participate: u128,
    pub accounting: String,
    /// Real-valued widths; rounded upward to dyadic values.
    pub sigma_s: f64,
    pub sigma_big_r: f64,
    pub sigma_r: f64,
    pub omega_c: u32,
    pub tail_rule: TailRule,
    pub admissibility: Admissibility,
    pub recon: ReconParams,
    pub evidence: Evidence,
}

/// One security-side condition evaluated on a profile.
#[derive(Clone, Debug, PartialEq)]
pub struct Check {
    pub name: &'static str,
    pub ok: bool,
    pub detail: String,
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// `ceil(sqrt(D) * 2^32) / 2^32`.
fn sqrt_dim_up(dim: usize) -> Dyadic {
    let scaled = BigUint::from(dim) << (2 * FIXED_FRAC_BITS);
    let floor = scaled.sqrt();
    let num = if &floor * &floor == scaled { floor } else { floor + 1u32 };
    Dyadic { num, frac_bits: FIXED_FRAC_BITS }
}

fn tail_rule_tag(rule: TailRule) -> &'static str {
    match rule {
        TailRule::ClosedForm => "closed-form",
        TailRule::ExactInversion => "exact-inversion",
        TailRule::October => "october-invalid",
    }
}

impl Profile {
    /// Builds a profile, deriving the root of unity, tail factors, bounds and digest.
    pub fn derive(inputs: ProfileInputs) -> Result<Profile> {
        if inputs.tail_rule == TailRule::October {
            return Err(Error::invalid("the tail formula of the earlier draft does not certify its budget"));
        }
        let ring = Ring::new(inputs.d, inputs.q)?;
        let n = inputs.g.len();
        let k = inputs.g.first().map_or(0, |r| r.len());
        if n == 0 || k < 2 || inputs.g.iter().any(|r| r.len() != k) {
            return Err(Error::invalid("G must be a nonempty rectangular matrix with at least two columns"));
        }
        let g: Vec<Vec<i8>> = inputs
            .g
            .iter()
            .map(|row| row.iter().map(|&x| i8::try_from(x)).collect::<std::result::Result<Vec<_>, _>>())
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|_| Error::invalid("entries of G must fit in eight bits"))?;
        let c_exp = g[0][1..].iter().filter(|&&x| x != 0).count();
        let beta_g = g.iter().map(|r| u32::from(r[0].unsigned_abs())).max().unwrap_or(0);
        let dim = inputs.d * inputs.ell;
        let kappa_big_r = inputs.kappa + ceil_lg(((k - 1) * dim) as u128);
        let kappa_r = inputs.kappa + ceil_lg(n as u128 * inputs.q_sign_global);
        let theta = |kappa_tail: u32, dimension: usize| {
            Dyadic::round_up_from_f64(tail_theta(inputs.tail_rule, kappa_tail, dimension), FIXED_FRAC_BITS)
        };
        let width = |sigma: f64| -> Result<Dyadic> {
            // Sixteen fractional bits for narrow widths; wide ones are integers.
            let frac_bits = if sigma < 4294967296.0 { 16 } else { 0 };
            Ok(Dyadic::from_sigma(GaussianSigma::round_up_from_f64(sigma, frac_bits)?))
        };
        let commitment_bytes = (inputs.kappa as usize).div_ceil(4);
        let mut profile = Profile {
            format: PROFILE_FORMAT.to_string(),
            id: inputs.id,
            category: inputs.category,
            description: inputs.description,
            provenance: inputs.provenance,
            kappa: inputs.kappa,
            ring: RingSpec { d: inputs.d, q: inputs.q, psi: ring.psi() },
            p: inputs.p,
            ell: inputs.ell,
            sec: SecSpec {
                n,
                k,
                c_exp,
                beta_g,
                t_uf: inputs.t_uf,
                t_sig: inputs.t_sig,
                k_c_det: inputs.k_c_det,
                matrix_name: inputs.matrix_name,
                matrix_source: inputs.matrix_source,
                g,
            },
            budgets: Budgets {
                q_h: inputs.q_h,
                q_sign_global: inputs.q_sign_global,
                q_participate: inputs.q_participate,
                accounting: inputs.accounting,
            },
            gaussian: GaussianSpec {
                convention: GAUSSIAN_CONVENTION.to_string(),
                sigma_s: width(inputs.sigma_s)?,
                sigma_big_r: width(inputs.sigma_big_r)?,
                sigma_r: width(inputs.sigma_r)?,
                sampler: SAMPLER_ID.to_string(),
                sampler_statistical_distance: "below exp(-2^48): exact integer arithmetic given uniform random \
                                               bytes, with the support restricted to |x| < 2^126"
                    .to_string(),
                sampler_timing: "variable-time reference; not side-channel protected".to_string(),
            },
            challenge: ChallengeSpec { omega_c: inputs.omega_c, commitment_bytes },
            encoding: EncodingSpec { rice_share: Vec::new(), rice_response: 0 },
            tails: TailSpec {
                rule: tail_rule_tag(inputs.tail_rule).to_string(),
                kappa_big_r,
                kappa_r,
                theta_s: theta(inputs.kappa, dim)?,
                theta_big_r: theta(kappa_big_r, 1)?,
                theta_r: theta(kappa_r, dim)?,
            },
            admissibility: inputs.admissibility,
            bounds: BoundSpec {
                beta_sh: BigUint::zero(),
                b1: BigUint::zero(),
                b2: BigUint::zero(),
                b_msis_squared: BigUint::zero(),
            },
            recon: ReconSpec {
                algorithm: RECON_ALGORITHM_ID.to_string(),
                lll_delta_num: inputs.recon.delta.num,
                lll_delta_den: inputs.recon.delta.den,
                passes: inputs.recon.passes,
                window: inputs.recon.window,
            },
            xof: XOF_ID.to_string(),
            codec_version: CODEC_VERSION,
            evidence: inputs.evidence,
            digest: String::new(),
        };
        profile.encoding = profile.derive_encoding();
        profile.bounds = profile.derive_bounds();
        profile.digest = hex(&profile.compute_digest());
        profile.validate()?;
        Ok(profile)
    }

    /// `D = d * ell`, the coefficient dimension of a vector in `R^ell`.
    pub fn dim(&self) -> usize {
        self.ring.d * self.ell
    }

    /// The exact bound derivation described in the module documentation.
    pub fn derive_bounds(&self) -> BoundSpec {
        let sqrt_d = sqrt_dim_up(self.dim());
        let g = &self.gaussian;
        let beta_s = self.tails.theta_s.mul(&g.sigma_s).mul(&sqrt_d);
        let beta_big_r = self.tails.theta_big_r.mul(&g.sigma_big_r);
        let beta_sh = beta_s.mul_int(u128::from(self.sec.beta_g)).add(&beta_big_r.mul_int(self.sec.c_exp as u128));
        let beta_r = self.tails.theta_r.mul(&g.sigma_r).mul(&sqrt_d);
        let b1 = sqrt_d.mul(&beta_sh).mul_int(u128::from(self.challenge.omega_c)).add(&beta_r).ceil();
        let b2 = &b1 * BigUint::from(self.admissibility.l1_bound());
        let b_msis_squared = (&b2 * &b2 + BigUint::from(self.challenge.omega_c)) << 2u32;
        BoundSpec { beta_sh: beta_sh.ceil(), b1, b2, b_msis_squared }
    }

    /// The Golomb-Rice parameters of [`EncodingSpec`].
    pub fn derive_encoding(&self) -> EncodingSpec {
        let g = &self.gaussian;
        let share_variance = |abs_g: u32| {
            let from_s = g.sigma_s.mul(&g.sigma_s).mul_int(u128::from(abs_g * abs_g));
            from_s.add(&g.sigma_big_r.mul(&g.sigma_big_r).mul_int(self.sec.c_exp as u128))
        };
        let rice_share = (0..=self.sec.beta_g).map(|j| rice_parameter(&share_variance(j))).collect();
        let response_variance =
            g.sigma_r.mul(&g.sigma_r).add(&share_variance(self.sec.beta_g).mul_int(u128::from(self.challenge.omega_c)));
        EncodingSpec { rice_share, rice_response: rice_parameter(&response_variance) }
    }

    /// The digest of every field that affects protocol behavior or the bounds.
    pub fn compute_digest(&self) -> Vec<u8> {
        let mut xof = Xof::new(domain::PROFILE);
        xof.absorb(self.format.as_bytes()).absorb(self.id.as_bytes()).absorb(self.category.tag().as_bytes());
        xof.absorb_u64(u64::from(self.kappa));
        xof.absorb_u64(self.ring.d as u64);
        xof.absorb(self.ring.q.to_string().as_bytes()).absorb(self.ring.psi.to_string().as_bytes());
        xof.absorb_u64(self.p as u64).absorb_u64(self.ell as u64);
        let s = &self.sec;
        xof.absorb_u64(s.n as u64).absorb_u64(s.k as u64).absorb_u64(s.c_exp as u64);
        xof.absorb_u64(u64::from(s.beta_g)).absorb_u64(s.t_uf as u64).absorb_u64(s.t_sig as u64);
        xof.absorb_u64(s.k_c_det.map_or(0, |v| v as u64 + 1));
        let flat: Vec<u8> = s.g.iter().flatten().map(|&x| x as u8).collect();
        xof.absorb(&flat);
        let b = &self.budgets;
        for value in [b.q_h, b.q_sign_global, b.q_participate] {
            xof.absorb(value.to_string().as_bytes());
        }
        let g = &self.gaussian;
        xof.absorb(g.convention.as_bytes()).absorb(g.sampler.as_bytes());
        for width in [&g.sigma_s, &g.sigma_big_r, &g.sigma_r] {
            width.absorb(&mut xof);
        }
        let c = &self.challenge;
        xof.absorb_u64(u64::from(c.omega_c)).absorb_u64(c.commitment_bytes as u64);
        let e = &self.encoding;
        xof.absorb_u64(e.rice_share.len() as u64);
        for &k in &e.rice_share {
            xof.absorb_u64(u64::from(k));
        }
        xof.absorb_u64(u64::from(e.rice_response));
        let t = &self.tails;
        xof.absorb(t.rule.as_bytes()).absorb_u64(u64::from(t.kappa_big_r)).absorb_u64(u64::from(t.kappa_r));
        for theta in [&t.theta_s, &t.theta_big_r, &t.theta_r] {
            theta.absorb(&mut xof);
        }
        match self.admissibility {
            Admissibility::Paper { beta_lambda, omega_lambda } => {
                xof.absorb(b"paper").absorb_u64(u64::from(beta_lambda)).absorb_u64(u64::from(omega_lambda));
            }
            Admissibility::L1 { l1_cap } => {
                xof.absorb(b"l1").absorb_u64(u64::from(l1_cap));
            }
        }
        for bound in [&self.bounds.beta_sh, &self.bounds.b1, &self.bounds.b2, &self.bounds.b_msis_squared] {
            xof.absorb(bound.to_string().as_bytes());
        }
        let r = &self.recon;
        xof.absorb(r.algorithm.as_bytes());
        for value in [r.lll_delta_num, r.lll_delta_den, r.passes, r.window] {
            xof.absorb_u64(u64::from(value));
        }
        xof.absorb(self.xof.as_bytes()).absorb_u64(u64::from(self.codec_version));
        xof.digest(DIGEST_BYTES)
    }

    /// Recomputes the derived bounds and the digest after an edit, for building
    /// variants of an existing profile.  The result still has to pass
    /// [`Profile::validate`].
    pub fn refresh(&mut self) {
        self.encoding = self.derive_encoding();
        self.bounds = self.derive_bounds();
        self.digest = hex(&self.compute_digest());
    }

    /// The exact Gaussian widths as sampler parameters.
    pub fn sigmas(&self) -> Result<(GaussianSigma, GaussianSigma, GaussianSigma)> {
        let g = &self.gaussian;
        Ok((g.sigma_s.to_sigma()?, g.sigma_big_r.to_sigma()?, g.sigma_r.to_sigma()?))
    }

    /// Structural validation.  A profile that passes can be executed; whether its
    /// parameters support a security claim is a separate matter ([`Profile::checks`]).
    pub fn validate(&self) -> Result<()> {
        let fail = |msg: &str| Err(Error::invalid(format!("profile {}: {msg}", self.id)));
        if self.format != PROFILE_FORMAT {
            return fail("unsupported format");
        }
        if self.gaussian.convention != GAUSSIAN_CONVENTION {
            return fail("unsupported Gaussian convention");
        }
        if self.gaussian.sampler != SAMPLER_ID || self.recon.algorithm != RECON_ALGORITHM_ID {
            return fail("unknown sampler or reconstruction algorithm");
        }
        if self.xof != XOF_ID || self.codec_version != CODEC_VERSION {
            return fail("unknown hash construction or codec version");
        }
        let ring = Ring::new(self.ring.d, self.ring.q)?;
        if ring.psi() != self.ring.psi {
            return fail("psi is not the canonical root of unity");
        }
        if self.p == 0 || self.p >= self.ell {
            return fail("need 1 <= p < ell");
        }
        let s = &self.sec;
        if s.n == 0 || s.n > usize::from(u16::MAX) || s.k < 2 || s.g.len() != s.n || s.g.iter().any(|r| r.len() != s.k)
        {
            return fail("G does not match n and k");
        }
        if s.g.iter().any(|r| r[1..].iter().filter(|&&x| x != 0).count() != s.c_exp) {
            return fail("a row of G_hat does not have c_exp nonzero entries");
        }
        if s.g.iter().any(|r| r[1..].iter().any(|&x| !(-1..=1).contains(&x))) {
            return fail("G_hat has an entry outside {-1, 0, 1}");
        }
        if s.g.iter().map(|r| u32::from(r[0].unsigned_abs())).max() != Some(s.beta_g) {
            return fail("beta_g is not the largest absolute entry of g");
        }
        if s.t_uf == 0 || s.t_uf >= s.t_sig || s.t_sig > s.n {
            return fail("need 1 <= t_uf < t_sig <= n");
        }
        if s.k_c_det.is_some_and(|v| v < s.t_sig || v > s.n) {
            return fail("k_c_det must lie between t_sig and n");
        }
        let c = &self.challenge;
        if c.omega_c == 0 || c.omega_c as usize > self.ring.d {
            return fail("challenge weight out of range");
        }
        if c.commitment_bytes != (self.kappa as usize).div_ceil(4) {
            return fail("commitments must be 2 kappa bits");
        }
        if self.encoding != self.derive_encoding() {
            return fail("Golomb-Rice parameters do not follow from the widths");
        }
        if self.budgets.q_participate == 0 || self.budgets.q_sign_global == 0 || self.budgets.q_h == 0 {
            return fail("query budgets must be positive");
        }
        let (sigma_s, sigma_big_r, sigma_r) = self.sigmas()?;
        for sigma in [sigma_s, sigma_big_r, sigma_r] {
            if sigma.floor() == 0 {
                return fail("Gaussian widths must be at least 1");
            }
        }
        let dim = self.dim();
        let t = &self.tails;
        if t.kappa_big_r != self.kappa + ceil_lg(((s.k - 1) * dim) as u128)
            || t.kappa_r != self.kappa + ceil_lg(s.n as u128 * self.budgets.q_sign_global)
        {
            return fail("tail budgets do not match kappa, k, D, n and Q_Sign");
        }
        // The tail factors must certify their budgets: N ((theta^2 - 1)/2 - ln theta) >= kappa ln 2.
        for (name, theta, kappa_tail, dimension) in [
            ("theta_s", &t.theta_s, self.kappa, dim),
            ("theta_R", &t.theta_big_r, t.kappa_big_r, 1),
            ("theta_r", &t.theta_r, t.kappa_r, dim),
        ] {
            let value = theta.to_f64();
            if value.is_nan() || value <= 1.0 || tail_exponent_bits(value, dimension as f64) < f64::from(kappa_tail) {
                return fail(&format!("{name} does not certify its tail budget"));
            }
        }
        match self.admissibility {
            Admissibility::Paper { beta_lambda, omega_lambda } => {
                if beta_lambda == 0 || omega_lambda == 0 || u128::from(beta_lambda) >= self.ring.q {
                    return fail("need 1 <= beta_lambda < q and omega_lambda >= 1");
                }
            }
            Admissibility::L1 { l1_cap } => {
                if l1_cap == 0 {
                    return fail("the l1 cap must be positive");
                }
            }
        }
        if self.recon.passes > 16 || self.recon.window > 16 {
            return fail("greedy refinement parameters out of range");
        }
        if 4 * self.recon.lll_delta_num <= self.recon.lll_delta_den
            || self.recon.lll_delta_num >= self.recon.lll_delta_den
        {
            return fail("LLL delta must lie in (1/4, 1)");
        }
        if self.bounds != self.derive_bounds() {
            return fail("stored bounds differ from the exact derivation");
        }
        // Integer arithmetic limits of the reference backend: coefficients are i128
        // and a combined response coefficient is at most B2 in absolute value.
        if self.bounds.b2.bits() > 120 {
            return fail("B2 exceeds the reference backend's coefficient range");
        }
        if self.digest != hex(&self.compute_digest()) {
            return fail("digest does not match the profile contents");
        }
        Ok(())
    }

    /// Security-side conditions of the paper, evaluated for this profile.  They are
    /// reported rather than enforced: functional profiles fail several by design.
    pub fn checks(&self) -> Vec<Check> {
        let mut out = Vec::new();
        let q = BigUint::from(self.ring.q);
        let kappa = self.kappa;
        let b = &self.budgets;
        let two_kappa = BigUint::one() << kappa;

        let space = params::challenge_space_size(self.ring.d, self.challenge.omega_c);
        out.push(Check {
            name: "T_5: Q_H / |C| <= 2^-kappa",
            ok: space >= BigUint::from(b.q_h) * &two_kappa,
            detail: format!("lg |C| = {:.1}", space.to_f64().unwrap_or(f64::INFINITY).log2()),
        });
        let minimal = params::challenge_weight(self.ring.d, b.q_h, kappa);
        out.push(Check {
            name: "challenge weight is the smallest admissible",
            ok: minimal == Some(self.challenge.omega_c),
            detail: format!("omega_c = {}, smallest = {minimal:?}", self.challenge.omega_c),
        });

        if self.category == Category::SecOnly {
            out.push(Check {
                name: "SEC-only profile: signature conditions (T_1, T_2, MSIS, sigma_r) are not applicable",
                ok: true,
                detail: String::new(),
            });
            return out;
        }
        // T_1: (n + 1)(Q_H + 1) Q_Sign 2^(-d + 1) <= 2^-kappa.
        let t1 = BigUint::from(self.sec.n as u64 + 1) * (BigUint::from(b.q_h) + 1u32) * BigUint::from(b.q_sign_global);
        out.push(Check {
            name: "T_1: (n+1)(Q_H+1) Q_Sign 2^(1-d) <= 2^-kappa",
            ok: (t1 * &two_kappa) <= (BigUint::one() << (self.ring.d - 1)),
            detail: String::new(),
        });
        // T_2: (Q_H^2 + t_uf Q_Sign Q_H) 2^(-2 kappa) <= 2^-kappa.
        let t2 = BigUint::from(b.q_h) * BigUint::from(b.q_h)
            + BigUint::from(self.sec.t_uf as u64) * BigUint::from(b.q_sign_global) * BigUint::from(b.q_h);
        out.push(Check {
            name: "T_2: (Q_H^2 + t_uf Q_Sign Q_H) 2^(-2 kappa) <= 2^-kappa",
            ok: t2 <= two_kappa,
            detail: String::new(),
        });

        let b_msis_sq = &self.bounds.b_msis_squared;
        let lg_b_msis = b_msis_sq.to_f64().unwrap_or(f64::INFINITY).log2() / 2.0;
        out.push(Check {
            name: "Euclidean MSIS bound: B_MSIS < q",
            ok: b_msis_sq < &(&q * &q),
            detail: format!("lg B_MSIS = {lg_b_msis:.4}, lg q = {:.4}", (self.ring.q as f64).log2()),
        });
        // 2 B_MSIS + 1 < q  <=>  4 B_MSIS^2 < (q - 1)^2.
        let q_minus_one = &q - 1u32;
        out.push(Check {
            name: "paper's rule (pinned estimator): q > 2 B_MSIS + 1",
            ok: (b_msis_sq << 2u32) < &q_minus_one * &q_minus_one,
            detail: format!("B_MSIS / q = {:.3}", lg_b_msis.exp2() / self.ring.q as f64),
        });

        // sigma_r > 2d q^(1/ell + 2/(d (ell - p))).
        let exponent = 1.0 / self.ell as f64 + 2.0 / (self.ring.d as f64 * (self.ell - self.p) as f64);
        let threshold = 2.0 * self.ring.d as f64 * (self.ring.q as f64).powf(exponent);
        let sigma_r = self.gaussian.sigma_r.to_f64();
        out.push(Check {
            name: "sigma_r > 2d q^(1/ell + 2/(d(ell-p)))",
            ok: sigma_r > threshold,
            detail: format!("lg sigma_r = {:.4}, lg threshold = {:.4}", sigma_r.log2(), threshold.log2()),
        });
        out.push(Check {
            name: "local caps give at most Q_Sign sessions: n Q_Participate <= Q_Sign",
            ok: BigUint::from(self.sec.n as u64) * BigUint::from(b.q_participate) <= BigUint::from(b.q_sign_global),
            detail: String::new(),
        });
        if let Admissibility::Paper { omega_lambda, .. } = self.admissibility {
            out.push(Check {
                name: "note: omega_lambda < t_sig, so every admissible set has idle signers",
                ok: true,
                detail: format!("omega_lambda = {omega_lambda}, t_sig = {}", self.sec.t_sig),
            });
        }
        out
    }

    pub fn from_json(text: &str) -> Result<Profile> {
        let profile: Profile = serde_json::from_str(text)?;
        profile.validate()?;
        Ok(profile)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Profile> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|e| Error::Io(path.display().to_string(), e))?;
        Self::from_json(&text)
    }

    /// Pretty JSON with each row of small integers on one line.
    pub fn to_json(&self) -> String {
        let pretty = serde_json::to_string_pretty(self).expect("profile serializes");
        let mut out = String::with_capacity(pretty.len());
        let mut pending: Option<String> = None;
        for line in pretty.lines() {
            let trimmed = line.trim();
            match pending.as_mut() {
                Some(buffer) => {
                    let is_number = trimmed.trim_end_matches(',').parse::<i64>().is_ok();
                    if is_number {
                        if !buffer.ends_with('[') {
                            buffer.push(' ');
                        }
                        buffer.push_str(trimmed);
                    } else if trimmed == "]" || trimmed == "]," {
                        buffer.push_str(trimmed);
                        out.push_str(buffer);
                        out.push('\n');
                        pending = None;
                    } else {
                        // Not a row of integers after all: flush unchanged.
                        out.push_str(buffer);
                        out.push('\n');
                        if trimmed.ends_with('[') {
                            *buffer = line.to_string();
                        } else {
                            out.push_str(line);
                            out.push('\n');
                            pending = None;
                        }
                    }
                }
                None => {
                    if trimmed.ends_with('[') {
                        pending = Some(line.to_string());
                    } else {
                        out.push_str(line);
                        out.push('\n');
                    }
                }
            }
        }
        if let Some(buffer) = pending {
            out.push_str(&buffer);
            out.push('\n');
        }
        out
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        std::fs::write(path, self.to_json()).map_err(|e| Error::Io(path.display().to_string(), e))
    }

    /// `(lg sigma_s, lg sigma_R, lg sigma_r, lg B1, lg B2, lg B_MSIS)` for display.
    pub fn display_logs(&self) -> [f64; 6] {
        let lg = |x: &BigUint| x.to_f64().unwrap_or(f64::INFINITY).log2();
        [
            self.gaussian.sigma_s.to_f64().log2(),
            self.gaussian.sigma_big_r.to_f64().log2(),
            self.gaussian.sigma_r.to_f64().log2(),
            lg(&self.bounds.b1),
            lg(&self.bounds.b2),
            lg(&self.bounds.b_msis_squared) / 2.0,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_inputs() -> ProfileInputs {
        // A 10 x 6 SEC matrix of [46] (fixtures/author/G-10-5-3.json).
        let g = vec![
            vec![1, 0, -1, 1, 1, 0],
            vec![1, -1, 0, 1, 1, 0],
            vec![1, 0, 0, 1, 1, 1],
            vec![-1, -1, -1, 0, 1, 0],
            vec![0, 0, -1, 0, 1, 1],
            vec![1, -1, 0, 0, 1, 1],
            vec![-1, -1, -1, 1, 0, 0],
            vec![1, 0, -1, 1, 0, 1],
            vec![1, -1, 0, 1, 0, 1],
            vec![0, -1, -1, 0, 0, 1],
        ];
        ProfileInputs {
            id: "unit-test".into(),
            category: Category::Functional,
            description: "unit test profile".into(),
            provenance: vec![],
            kappa: 128,
            d: 16,
            q: crate::ring::largest_ntt_prime_below(40, 16).unwrap(),
            p: 2,
            ell: 4,
            matrix_name: "author-10-5-3".into(),
            matrix_source: None,
            g,
            t_uf: 2,
            t_sig: 7,
            k_c_det: Some(9),
            q_h: 1 << 20,
            q_sign_global: 1 << 20,
            q_participate: 1 << 10,
            accounting: "test".into(),
            sigma_s: 4.0,
            sigma_big_r: 4.0,
            sigma_r: 1048576.0,
            omega_c: 4,
            tail_rule: TailRule::ClosedForm,
            admissibility: Admissibility::L1 { l1_cap: 12 },
            recon: ReconParams::DEFAULT,
            evidence: Evidence { claim: "none".into(), ..Evidence::default() },
        }
    }

    #[test]
    fn dyadic_arithmetic() {
        let a = Dyadic { num: BigUint::from(5u32), frac_bits: 1 }; // 2.5
        let b = Dyadic { num: BigUint::from(3u32), frac_bits: 2 }; // 0.75
        assert_eq!(a.mul(&b), Dyadic { num: BigUint::from(15u32), frac_bits: 3 });
        assert_eq!(a.add(&b), Dyadic { num: BigUint::from(13u32), frac_bits: 2 });
        assert_eq!(a.ceil(), BigUint::from(3u32));
        assert_eq!(Dyadic::from_integer(7).ceil(), BigUint::from(7u32));
        assert_eq!(a.mul_int(4).ceil(), BigUint::from(10u32));
        assert!((a.to_f64() - 2.5).abs() < 1e-15);
        let s = sqrt_dim_up(6144);
        assert!(s.to_f64() >= 6144f64.sqrt() && s.to_f64() < 6144f64.sqrt() + 1e-9);
        assert_eq!(sqrt_dim_up(64), Dyadic { num: BigUint::from(8u64 << 32), frac_bits: 32 });
    }

    #[test]
    fn derive_validate_and_round_trip() {
        let profile = Profile::derive(toy_inputs()).unwrap();
        profile.validate().unwrap();
        let json = profile.to_json();
        let back = Profile::from_json(&json).unwrap();
        assert_eq!(back, profile);
        assert!(json.contains("[1, 0, -1, 1, 1, 0]"), "matrix rows should be on one line");
        // Exact bound relations.
        assert_eq!(profile.bounds.b2, &profile.bounds.b1 * 12u32);
        assert_eq!(profile.bounds.b_msis_squared, (&profile.bounds.b2 * &profile.bounds.b2 + 4u32) * 4u32);
        // B1 is at least the real-valued formula and within a hair of it.
        let dim = 64f64;
        let (ts, tr, tm) =
            (profile.tails.theta_s.to_f64(), profile.tails.theta_big_r.to_f64(), profile.tails.theta_r.to_f64());
        let beta_sh = ts * 4.0 * dim.sqrt() + 3.0 * tr * 4.0;
        let b1 = 4.0 * dim.sqrt() * beta_sh + tm * 1048576.0 * dim.sqrt();
        let stored = profile.bounds.b1.to_f64().unwrap();
        assert!(stored >= b1 && stored < b1 * (1.0 + 1e-8) + 1.0);
        assert_eq!(profile.sec.c_exp, 3);
        assert_eq!(profile.sec.beta_g, 1);
    }

    #[test]
    fn tampering_is_detected() {
        let profile = Profile::derive(toy_inputs()).unwrap();
        let mut bad = profile.clone();
        bad.bounds.b1 += 1u32;
        assert!(bad.validate().is_err(), "bounds must match the derivation");
        let mut bad = profile.clone();
        bad.sec.g[0][0] = 0;
        bad.sec.g[3][0] = 1;
        assert!(bad.validate().is_err(), "digest covers the matrix");
        let mut bad = profile.clone();
        bad.sec.t_sig = 8;
        assert!(bad.validate().is_err(), "digest covers the thresholds");
        let mut bad = profile.clone();
        bad.tails.theta_r = Dyadic { num: BigUint::from(1u64 << 32), frac_bits: 32 };
        assert!(bad.validate().is_err(), "theta = 1 certifies nothing");
        let mut bad = profile.clone();
        bad.ring.psi += 1;
        assert!(bad.validate().is_err());
        let mut bad = profile.clone();
        bad.gaussian.convention = "rho-s".into();
        assert!(bad.validate().is_err());
        // Evidence is not covered by the digest.
        let mut noted = profile.clone();
        noted.evidence.notes.push("a remark".into());
        noted.validate().unwrap();
    }

    #[test]
    fn rejects_bad_inputs() {
        let mut inputs = toy_inputs();
        inputs.tail_rule = TailRule::October;
        assert!(Profile::derive(inputs).is_err());
        let mut inputs = toy_inputs();
        inputs.t_sig = 2;
        assert!(Profile::derive(inputs).is_err());
        let mut inputs = toy_inputs();
        inputs.q = 97; // prime but not 1 mod 32... 97 = 1 mod 32, so use a non-NTT prime
        inputs.q = 101;
        assert!(Profile::derive(inputs).is_err());
        let mut inputs = toy_inputs();
        inputs.g[2][3] = 2;
        assert!(Profile::derive(inputs).is_err());
        let mut inputs = toy_inputs();
        inputs.p = 4;
        assert!(Profile::derive(inputs).is_err());
        assert!(Profile::from_json("{}").is_err());
    }

    #[test]
    fn functional_profile_fails_security_checks_by_design() {
        let profile = Profile::derive(toy_inputs()).unwrap();
        let checks = profile.checks();
        assert!(checks.iter().any(|c| c.name.starts_with("T_5") && !c.ok));
        assert!(checks.iter().any(|c| c.name.starts_with("Euclidean MSIS") && c.ok));
    }
}
