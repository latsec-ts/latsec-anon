//! Randomness sources and a discrete Gaussian sampler in exact integer arithmetic.
//!
//! # Gaussian convention
//!
//! Widths are in the paper's `sigma-exp-half` convention (Section 2.1):
//! `Pr[x] ~ exp(-x^2 / (2 sigma^2))`.  A source that uses `rho_s(x) = exp(-pi x^2 / s^2)`
//! must be converted with `s = sqrt(2 pi) * sigma`; this module never does that
//! implicitly.
//!
//! # Sampler
//!
//! [`DiscreteGaussian`] implements Algorithms 1-3 of Canonne, Kamath and Steinke,
//! "The Discrete Gaussian for Differential Privacy" (2020).  It uses only integer
//! arithmetic and uniform random integers, with a rational `sigma^2`.
//!
//! # Error budget
//!
//! The algorithms draw unbounded integers; this implementation uses machine
//! integers where the deviation can be bounded, and big integers elsewhere.  Given
//! uniform random bytes, and except for a counter overflow of probability below
//! `exp(-2^128)` (second item), the output distribution is exactly `D_{Z, sigma}`
//! conditioned on `|x| < 2^126`:
//!
//! * A Laplace proposal of magnitude `2^126` or more is rejected (the magnitude is
//!   computed with checked arithmetic, and the geometric counter saturates).
//!   Rejection sampling on a restricted proposal yields the target restricted to
//!   the same support, so this is the only approximation.  For every supported
//!   width (`sigma < 2^100 + 1`) the removed mass, and hence the statistical
//!   distance to `D_{Z, sigma}`, is below `exp(-2^48)`.
//! * The counters of the `Bernoulli(exp(-x))` loops (Algorithm 1) are `u128`
//!   values starting at 1.  Such a counter exceeds `m` with probability at most
//!   `1 / m!`.  It overflows only by exceeding `2^128 - 1`, so an overflow has
//!   probability at most `1 / (2^128 - 1)! < exp(-2^128)` per loop.
//! * The `exp(-whole)` part of a large exponent is sampled with a big-integer
//!   counter and is exact.
//!
//! **Timing status: variable time.**  Running time and the number of random bytes
//! consumed depend on the sampled value.  This sampler is a correctness reference
//! and must not be used where sampling time is observable.  Its big-integer
//! temporaries are not erased.

use num_bigint::BigUint;
use num_integer::Integer;
use num_traits::Zero;
use zeroize::Zeroize;

use crate::error::{Error, Result};
use crate::xof::{domain, Xof, XofOutput};

/// A source of uniform random bytes.
pub trait RandomSource {
    fn fill(&mut self, buf: &mut [u8]);
}

/// Operating-system randomness, buffered.
pub struct OsRandom {
    buf: [u8; 4096],
    pos: usize,
}

impl OsRandom {
    pub fn new() -> Self {
        OsRandom { buf: [0; 4096], pos: 4096 }
    }
}

impl Default for OsRandom {
    fn default() -> Self {
        Self::new()
    }
}

impl RandomSource for OsRandom {
    fn fill(&mut self, out: &mut [u8]) {
        let mut done = 0;
        while done < out.len() {
            if self.pos == self.buf.len() {
                getrandom::getrandom(&mut self.buf).expect("operating-system randomness is unavailable");
                self.pos = 0;
            }
            let take = (out.len() - done).min(self.buf.len() - self.pos);
            out[done..done + take].copy_from_slice(&self.buf[self.pos..self.pos + take]);
            // Bytes handed out are never served again.
            self.buf[self.pos..self.pos + take].zeroize();
            self.pos += take;
            done += take;
        }
    }
}

impl Drop for OsRandom {
    fn drop(&mut self) {
        self.buf.zeroize();
    }
}

/// Deterministic randomness expanded from a seed with SHAKE256.  For fixtures,
/// reproducible instance generation and tests only.
pub struct SeededRandom {
    stream: XofOutput,
}

impl SeededRandom {
    /// A stream determined by `seed` and a purpose label.
    pub fn new(seed: &[u8], label: &str) -> Self {
        let mut xof = Xof::new(domain::FIXTURE_RANDOMNESS);
        xof.absorb(label.as_bytes()).absorb(seed);
        SeededRandom { stream: xof.finalize() }
    }
}

impl RandomSource for SeededRandom {
    fn fill(&mut self, buf: &mut [u8]) {
        self.stream.read(buf);
    }
}

/// A uniform integer in `[0, bound)` by rejection sampling; `bound >= 1`.
pub fn uniform_below(rng: &mut dyn RandomSource, bound: u128) -> u128 {
    assert!(bound >= 1);
    if bound == 1 {
        return 0;
    }
    let bits = 128 - (bound - 1).leading_zeros();
    let nbytes = bits.div_ceil(8) as usize;
    let mask = if bits == 128 { u128::MAX } else { (1u128 << bits) - 1 };
    loop {
        let mut bytes = [0u8; 16];
        rng.fill(&mut bytes[..nbytes]);
        let x = u128::from_le_bytes(bytes) & mask;
        bytes.zeroize();
        if x < bound {
            return x;
        }
    }
}

/// A uniform integer in `[0, bound)` for a big `bound >= 1`.
pub fn uniform_below_big(rng: &mut dyn RandomSource, bound: &BigUint) -> BigUint {
    BigBound::new(bound.clone()).sample(rng)
}

/// A fixed big bound with the constants of its rejection sampler, for bounds that
/// are sampled below many times.  [`BigBound::sample`] reads the same bytes from
/// the randomness source as [`uniform_below_big`] and returns the same value.
#[derive(Clone, Debug)]
pub struct BigBound {
    bound: BigUint,
    /// Bytes read per attempt: `ceil(bits(bound - 1) / 8)`; zero for `bound = 1`.
    nbytes: usize,
    /// Mask for the most significant byte read.
    top_mask: u8,
}

impl BigBound {
    /// Bounds of up to this many bytes are sampled without a heap allocation for
    /// the random bytes.
    const INLINE_BYTES: usize = 128;

    pub fn new(bound: BigUint) -> Self {
        assert!(!bound.is_zero());
        let bits = (&bound - 1u32).bits();
        let nbytes = bits.div_ceil(8) as usize;
        let top_mask = if nbytes == 0 { 0 } else { 0xffu8 >> (8 * nbytes as u64 - bits) };
        BigBound { bound, nbytes, top_mask }
    }

    pub fn bound(&self) -> &BigUint {
        &self.bound
    }

    /// A uniform integer in `[0, bound)` by rejection sampling.
    pub fn sample(&self, rng: &mut dyn RandomSource) -> BigUint {
        if self.nbytes == 0 {
            return BigUint::zero();
        }
        let mut inline = [0u8; Self::INLINE_BYTES];
        let mut heap = Vec::new();
        let bytes: &mut [u8] = if self.nbytes <= Self::INLINE_BYTES {
            &mut inline[..self.nbytes]
        } else {
            heap.resize(self.nbytes, 0);
            &mut heap
        };
        loop {
            rng.fill(bytes);
            bytes[self.nbytes - 1] &= self.top_mask;
            let x = BigUint::from_bytes_le(bytes);
            if x < self.bound {
                bytes.zeroize();
                return x;
            }
        }
    }
}

/// One uniform bit.
pub fn random_bit(rng: &mut dyn RandomSource) -> bool {
    let mut byte = [0u8; 1];
    rng.fill(&mut byte);
    byte[0] & 1 == 1
}

/// `Bernoulli(exp(-1))`, exactly.
fn bernoulli_exp_minus_one(rng: &mut dyn RandomSource) -> bool {
    // CKS Algorithm 1 with gamma = 1: Bernoulli(1 / K) until the first failure.
    let mut k: u128 = 1;
    loop {
        if uniform_below(rng, k) != 0 {
            break;
        }
        k += 1;
    }
    k % 2 == 1
}

/// `Bernoulli(exp(-num / den))` for `0 <= num < den`, exactly.
fn bernoulli_exp_small(rng: &mut dyn RandomSource, num: u128, den: u128) -> bool {
    debug_assert!(num < den);
    let mut k: u128 = 1;
    loop {
        // Bernoulli(num / (den * K)): a uniform value below den * K is below num
        // exactly when its quotient by den is zero and its remainder is below num.
        if uniform_below(rng, k) != 0 || uniform_below(rng, den) >= num {
            break;
        }
        k += 1;
    }
    k % 2 == 1
}

/// `Bernoulli(exp(-num / den))` for big `num >= 0`, `den > 0`, exactly.
fn bernoulli_exp_big(rng: &mut dyn RandomSource, num: &BigUint, den: &BigBound) -> bool {
    let (whole, rem) = num.div_rem(den.bound());
    // exp(-whole) as a product of exp(-1) trials, stopping at the first failure.
    // The counter is a big integer, so no part of the exponent is dropped.
    let mut done = BigUint::zero();
    while done < whole {
        if !bernoulli_exp_minus_one(rng) {
            return false;
        }
        done += 1u32;
    }
    let mut k: u128 = 1;
    loop {
        if uniform_below(rng, k) != 0 || den.sample(rng) >= rem {
            break;
        }
        k += 1;
    }
    k % 2 == 1
}

/// A Gaussian width `sigma = num / 2^frac_bits` in the `sigma-exp-half` convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GaussianSigma {
    pub num: u128,
    pub frac_bits: u32,
}

impl GaussianSigma {
    /// The largest supported width is below `2^100`.
    pub const MAX_BITS: u32 = 100;

    pub fn new(num: u128, frac_bits: u32) -> Result<Self> {
        if num == 0 || frac_bits > 64 || (num >> frac_bits) >> Self::MAX_BITS != 0 || num >> 126 != 0 {
            return Err(Error::invalid("unsupported Gaussian width"));
        }
        Ok(GaussianSigma { num, frac_bits })
    }

    /// The smallest representable width at least `sigma`, with `frac_bits`
    /// fractional bits.  A relative guard of `2^-40` covers the rounding error of
    /// the `f64` computations that produce `sigma`.
    pub fn round_up_from_f64(sigma: f64, frac_bits: u32) -> Result<Self> {
        if !(sigma.is_finite() && sigma > 0.0) {
            return Err(Error::invalid("Gaussian width must be positive and finite"));
        }
        let guarded = sigma * (1.0 + (-40f64).exp2());
        let scaled = (guarded * (frac_bits as f64).exp2()).ceil();
        if scaled >= 126f64.exp2() {
            return Err(Error::invalid("Gaussian width is too large"));
        }
        Self::new(scaled as u128, frac_bits)
    }

    pub fn to_f64(&self) -> f64 {
        self.num as f64 / (self.frac_bits as f64).exp2()
    }

    /// `floor(sigma)`.
    pub fn floor(&self) -> u128 {
        self.num >> self.frac_bits
    }
}

/// A sampler for `D_{Z, sigma}`, `Pr[x] ~ exp(-x^2 / (2 sigma^2))`, exact up to the
/// support restriction `|x| < 2^126` described in the module documentation.
#[derive(Clone, Debug)]
pub struct DiscreteGaussian {
    sigma: GaussianSigma,
    /// `t = floor(sigma) + 1`, the scale of the Laplace proposal.
    t: u128,
    /// `sigma^2 = a / b` with `b = 4^frac_bits`.
    a: BigUint,
    /// `t * b`.
    tb: BigUint,
    /// `2 * a * b * t^2`, the denominator of the acceptance exponent, with the
    /// constants for sampling below it.
    den: BigBound,
}

impl DiscreteGaussian {
    /// Proposals of this magnitude or more are rejected.
    pub const MAGNITUDE_LIMIT: u128 = 1 << 126;

    pub fn new(sigma: GaussianSigma) -> Self {
        let t = sigma.floor() + 1;
        let a = BigUint::from(sigma.num) * BigUint::from(sigma.num);
        let b = BigUint::from(1u8) << (2 * sigma.frac_bits);
        let tb = BigUint::from(t) * &b;
        let den = (&a << 1u32) * &b * BigUint::from(t) * BigUint::from(t);
        DiscreteGaussian { sigma, t, a, tb, den: BigBound::new(den) }
    }

    pub fn sigma(&self) -> GaussianSigma {
        self.sigma
    }

    /// The acceptance exponent of a proposal of magnitude `y` as `(num, den)`:
    /// `(y - sigma^2 / t)^2 / (2 sigma^2) = (y t b - a)^2 / (2 a b t^2)`.
    fn acceptance_exponent(&self, y: u128) -> (BigUint, &BigBound) {
        let ytb = BigUint::from(y) * &self.tb;
        let diff = if ytb >= self.a { ytb - &self.a } else { &self.a - ytb };
        (&diff * &diff, &self.den)
    }

    /// Draws one sample.
    pub fn sample(&self, rng: &mut dyn RandomSource) -> i128 {
        loop {
            // Discrete Laplace of scale t (CKS Algorithm 2 with s = 1).
            let u = uniform_below(rng, self.t);
            if !bernoulli_exp_small(rng, u, self.t) {
                continue;
            }
            let mut v: u128 = 0;
            while bernoulli_exp_minus_one(rng) {
                v = v.saturating_add(1);
            }
            // |y| = u + t v, rejected at or above the magnitude limit (see the
            // module documentation for the resulting error bound).
            let y = match self.t.checked_mul(v).and_then(|tv| tv.checked_add(u)) {
                Some(y) if y < Self::MAGNITUDE_LIMIT => y,
                _ => continue,
            };
            let negative = random_bit(rng);
            if negative && y == 0 {
                continue;
            }
            // Accept with probability exp(-(|y| - sigma^2 / t)^2 / (2 sigma^2)).
            let (num, den) = self.acceptance_exponent(y);
            if bernoulli_exp_big(rng, &num, den) {
                let y = y as i128;
                return if negative { -y } else { y };
            }
        }
    }

    /// Fills `out` with independent samples.
    pub fn sample_into(&self, rng: &mut dyn RandomSource, out: &mut [i128]) {
        for x in out.iter_mut() {
            *x = self.sample(rng);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;
    use num_rational::BigRational;

    #[test]
    fn seeded_stream_is_deterministic() {
        let mut a = SeededRandom::new(b"seed", "test");
        let mut b = SeededRandom::new(b"seed", "test");
        let mut c = SeededRandom::new(b"seed", "other");
        let (mut x, mut y, mut z) = ([0u8; 40], [0u8; 40], [0u8; 40]);
        a.fill(&mut x);
        b.fill(&mut y);
        c.fill(&mut z);
        assert_eq!(x, y);
        assert_ne!(x, z);
    }

    #[test]
    fn os_randomness_changes() {
        let mut rng = OsRandom::new();
        let (mut x, mut y) = ([0u8; 5000], [0u8; 5000]);
        rng.fill(&mut x);
        rng.fill(&mut y);
        assert_ne!(x, y);
        assert!(x.iter().any(|&b| b != 0));
    }

    #[test]
    fn uniform_sampling_is_in_range_and_covers_it() {
        let mut rng = SeededRandom::new(b"u", "uniform");
        let mut seen = [0u32; 7];
        for _ in 0..7000 {
            seen[uniform_below(&mut rng, 7) as usize] += 1;
        }
        // Each cell has mean 1000 and standard deviation about 29.
        assert!(seen.iter().all(|&c| (850..1150).contains(&c)), "{seen:?}");
        assert_eq!(uniform_below(&mut rng, 1), 0);
        let bound = (BigUint::from(1u8) << 200u32) + 12345u32;
        for _ in 0..200 {
            assert!(uniform_below_big(&mut rng, &bound) < bound);
        }
        assert!(uniform_below(&mut rng, u128::MAX) < u128::MAX);
    }

    #[test]
    fn fixed_bound_sampler_reads_the_same_bytes() {
        // Bounds of 1, of one byte, at byte boundaries, and beyond the inline buffer.
        let bounds = [
            BigUint::from(1u8),
            BigUint::from(2u8),
            BigUint::from(255u8),
            BigUint::from(256u16),
            BigUint::from(257u16),
            (BigUint::from(1u8) << 640u32) + 12345u32,
            (BigUint::from(3u8) << 1100u32) - 1u32,
        ];
        for bound in bounds {
            let fixed = BigBound::new(bound.clone());
            let mut a = SeededRandom::new(b"fixed bound", "test");
            let mut b = SeededRandom::new(b"fixed bound", "test");
            for _ in 0..50 {
                // The reference: bits(bound - 1) random bits per attempt, read as
                // whole bytes with the top bits masked, until the value is below.
                let bits = (&bound - 1u32).bits();
                let nbytes = bits.div_ceil(8) as usize;
                let expected = loop {
                    let mut bytes = vec![0u8; nbytes];
                    a.fill(&mut bytes);
                    if nbytes > 0 {
                        bytes[nbytes - 1] &= 0xffu8 >> (8 * nbytes as u64 - bits);
                    }
                    let x = BigUint::from_bytes_le(&bytes);
                    if x < bound {
                        break x;
                    }
                };
                assert_eq!(fixed.sample(&mut b), expected);
            }
            // Both streams are at the same position afterwards.
            let (mut x, mut y) = ([0u8; 8], [0u8; 8]);
            a.fill(&mut x);
            b.fill(&mut y);
            assert_eq!(x, y);
        }
    }

    #[test]
    fn bernoulli_exp_frequencies() {
        let mut rng = SeededRandom::new(b"b", "bernoulli");
        let trials = 40_000;
        let rate = |hits: u32| f64::from(hits) / f64::from(trials);
        let one = (0..trials).filter(|_| bernoulli_exp_minus_one(&mut rng)).count() as u32;
        assert!((rate(one) - (-1f64).exp()).abs() < 0.01);
        let small = (0..trials).filter(|_| bernoulli_exp_small(&mut rng, 3, 10)).count() as u32;
        assert!((rate(small) - (-0.3f64).exp()).abs() < 0.01);
        let (num, den) = (BigUint::from(27u32), BigUint::from(10u32));
        let big = (0..trials).filter(|_| bernoulli_exp_big(&mut rng, &num, &BigBound::new(den.clone()))).count() as u32;
        assert!((rate(big) - (-2.7f64).exp()).abs() < 0.01);
    }

    /// The proposal mass exp(-y / t) times the acceptance probability must be
    /// proportional to exp(-y^2 / (2 sigma^2)).  Checked in exact arithmetic: the
    /// exponent `y / t + accept(y) - y^2 / (2 sigma^2)` is the same for every y.
    #[test]
    fn acceptance_exponent_gives_the_gaussian_mass_ratio() {
        for (num, frac_bits) in [(3u128, 0u32), (45, 4), (0x1234_5678_9abc, 16), ((1 << 48) + 12345, 16)] {
            let sampler = DiscreteGaussian::new(GaussianSigma::new(num, frac_bits).unwrap());
            let sigma_sq =
                BigRational::new(BigInt::from(num) * BigInt::from(num), BigInt::from(1u8) << (2 * frac_bits));
            let t = BigRational::from_integer(BigInt::from(sampler.t));
            let exponent = |y: u128| {
                let (n, d) = sampler.acceptance_exponent(y);
                let accept = BigRational::new(BigInt::from(n), BigInt::from(d.bound().clone()));
                let yq = BigRational::from_integer(BigInt::from(y));
                &yq / &t + accept - &yq * &yq / (BigRational::from_integer(BigInt::from(2)) * &sigma_sq)
            };
            let reference = exponent(0);
            for y in [1u128, 2, 3, 17, sampler.t - 1, sampler.t, sampler.t + 1, 5 * sampler.t + 3] {
                assert_eq!(exponent(y), reference, "sigma = {num} / 2^{frac_bits}, y = {y}");
            }
        }
    }

    #[test]
    fn small_width_matches_the_exact_distribution() {
        // sigma = 1.5: compare empirical frequencies with the exact probabilities.
        let sigma = GaussianSigma::new(3, 1).unwrap();
        let sampler = DiscreteGaussian::new(sigma);
        let mut rng = SeededRandom::new(b"g", "gaussian-small");
        let trials = 200_000usize;
        let mut counts = std::collections::BTreeMap::new();
        for _ in 0..trials {
            *counts.entry(sampler.sample(&mut rng)).or_insert(0usize) += 1;
        }
        let s2 = 1.5f64 * 1.5;
        let norm: f64 = (-60..=60).map(|x| (-(f64::from(x) * f64::from(x)) / (2.0 * s2)).exp()).sum();
        for x in -5i128..=5 {
            let p = (-((x * x) as f64) / (2.0 * s2)).exp() / norm;
            let observed = *counts.get(&x).unwrap_or(&0) as f64 / trials as f64;
            let sd = (p * (1.0 - p) / trials as f64).sqrt();
            assert!((observed - p).abs() < 5.0 * sd, "x = {x}: observed {observed}, expected {p}");
        }
        assert!(counts.keys().all(|x| x.abs() < 20));
    }

    #[test]
    fn large_width_moments() {
        // sigma = 2^45 + 0.5: mean near zero and second moment near sigma^2.
        let sigma = GaussianSigma::new((1 << 46) + 1, 1).unwrap();
        let sampler = DiscreteGaussian::new(sigma);
        let mut rng = SeededRandom::new(b"g", "gaussian-large");
        let n = 20_000;
        let mut samples = vec![0i128; n];
        sampler.sample_into(&mut rng, &mut samples);
        let s = sigma.to_f64();
        let mean = samples.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
        let second = samples.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 5.0 * s / (n as f64).sqrt(), "mean {mean}");
        // The second moment of n samples has relative standard deviation sqrt(2 / n).
        assert!(
            (second / (s * s) - 1.0).abs() < 5.0 * (2.0 / n as f64).sqrt(),
            "second moment ratio {}",
            second / (s * s)
        );
        let max = samples.iter().map(|x| x.unsigned_abs()).max().unwrap() as f64;
        assert!(max < 8.0 * s);
    }

    #[test]
    fn sigma_rounding_is_upward() {
        let sigma = GaussianSigma::round_up_from_f64(45193.123456789, 16).unwrap();
        assert!(sigma.to_f64() >= 45193.123456789);
        assert!(sigma.to_f64() < 45193.123456789 * (1.0 + 1e-9));
        assert_eq!(sigma.floor(), 45193);
        assert!(GaussianSigma::round_up_from_f64(0.0, 16).is_err());
        assert!(GaussianSigma::round_up_from_f64(f64::NAN, 16).is_err());
        assert!(GaussianSigma::new(0, 0).is_err());
    }
}
