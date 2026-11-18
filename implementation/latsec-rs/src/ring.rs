//! Arithmetic in `R = Z[X]/(X^d + 1)` and `R_q = R/qR`.
//!
//! * Residues are `u128` values in `[0, q)` for an odd modulus `q < 2^126`.  All six
//!   published LaTSEC primes fit, including the 66-bit `2^66 - 16383`.
//! * Multiplication mod `q` uses Montgomery reduction with `R = 2^128`, so no
//!   special form of `q` is assumed.
//! * The negacyclic NTT of length `d` works directly mod `q`, which requires
//!   `q = 1 (mod 2d)`.  A schoolbook product is kept as an independent reference.
//! * Integer polynomials (secrets, shares, masks, responses) use `i128`
//!   coefficients.  Callers are responsible for bounds; profile validation checks
//!   that no intermediate value can overflow.
//!
//! This backend is variable-time and is not meant to protect secret operands.

use crate::error::{Error, Result};

/// A polynomial with coefficients in `[0, q)`.
pub type PolyQ = Vec<u128>;
/// A polynomial with integer coefficients.
pub type PolyZ = Vec<i128>;

/// Full 128 x 128 -> 256-bit product as `(high, low)`.
#[inline]
fn mul_wide(a: u128, b: u128) -> (u128, u128) {
    const MASK: u128 = (1 << 64) - 1;
    let (a0, a1) = (a & MASK, a >> 64);
    let (b0, b1) = (b & MASK, b >> 64);
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;
    let mid = (p00 >> 64) + (p01 & MASK) + (p10 & MASK);
    let lo = (p00 & MASK) | (mid << 64);
    let hi = p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64);
    (hi, lo)
}

/// An odd modulus with precomputed Montgomery constants.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Modulus {
    q: u128,
    /// `-q^-1 mod 2^128`.
    q_inv_neg: u128,
    /// `2^256 mod q`.
    r2: u128,
}

impl Modulus {
    /// The largest supported modulus is below `2^126`.
    pub const MAX_BITS: u32 = 126;

    pub fn new(q: u128) -> Result<Self> {
        if q < 3 || q % 2 == 0 || q >> Self::MAX_BITS != 0 {
            return Err(Error::invalid("modulus must be odd, at least 3 and below 2^126"));
        }
        // Newton iteration for q^-1 mod 2^128; q is its own inverse mod 8.
        let mut inv = q;
        for _ in 0..6 {
            inv = inv.wrapping_mul(2u128.wrapping_sub(q.wrapping_mul(inv)));
        }
        debug_assert_eq!(q.wrapping_mul(inv), 1);
        // 2^128 mod q, then 128 doublings give 2^256 mod q.
        let mut r2 = (u128::MAX % q + 1) % q;
        for _ in 0..128 {
            r2 <<= 1;
            if r2 >= q {
                r2 -= q;
            }
        }
        Ok(Modulus { q, q_inv_neg: inv.wrapping_neg(), r2 })
    }

    #[inline]
    pub fn q(&self) -> u128 {
        self.q
    }

    /// The number of bits of `q`.
    pub fn bits(&self) -> u32 {
        128 - self.q.leading_zeros()
    }

    #[inline]
    pub fn add(&self, a: u128, b: u128) -> u128 {
        let s = a + b;
        if s >= self.q {
            s - self.q
        } else {
            s
        }
    }

    #[inline]
    pub fn sub(&self, a: u128, b: u128) -> u128 {
        if a >= b {
            a - b
        } else {
            a + self.q - b
        }
    }

    #[inline]
    pub fn neg(&self, a: u128) -> u128 {
        if a == 0 {
            0
        } else {
            self.q - a
        }
    }

    /// Montgomery reduction of `hi * 2^128 + lo`, which must be below `q * 2^128`.
    #[inline]
    fn redc(&self, hi: u128, lo: u128) -> u128 {
        let m = lo.wrapping_mul(self.q_inv_neg);
        let (mh, ml) = mul_wide(m, self.q);
        let (sum, carry) = lo.overflowing_add(ml);
        debug_assert_eq!(sum, 0);
        let t = hi + mh + carry as u128;
        if t >= self.q {
            t - self.q
        } else {
            t
        }
    }

    /// `a * b / 2^128 mod q`.  If one operand is in Montgomery form and the other is
    /// not, the result is the plain product.
    #[inline]
    pub fn mont_mul(&self, a: u128, b: u128) -> u128 {
        let (hi, lo) = mul_wide(a, b);
        self.redc(hi, lo)
    }

    /// `a * 2^128 mod q`.
    #[inline]
    pub fn to_mont(&self, a: u128) -> u128 {
        self.mont_mul(a, self.r2)
    }

    /// `a * b mod q` for reduced operands.
    #[inline]
    pub fn mul(&self, a: u128, b: u128) -> u128 {
        self.mont_mul(self.mont_mul(a, b), self.r2)
    }

    pub fn pow(&self, base: u128, mut exp: u128) -> u128 {
        let mut acc = self.to_mont(1 % self.q);
        let mut b = self.to_mont(base);
        while exp != 0 {
            if exp & 1 == 1 {
                acc = self.mont_mul(acc, b);
            }
            b = self.mont_mul(b, b);
            exp >>= 1;
        }
        self.mont_mul(acc, 1)
    }

    /// The inverse of `a` when `q` is prime and `a` is nonzero.
    pub fn inv(&self, a: u128) -> u128 {
        self.pow(a, self.q - 2)
    }

    /// The residue of a signed integer.
    #[inline]
    pub fn reduce_i128(&self, x: i128) -> u128 {
        let r = x.rem_euclid(self.q as i128);
        r as u128
    }

    /// The representative in `(-q/2, q/2]`.
    #[inline]
    pub fn center(&self, a: u128) -> i128 {
        if a > self.q / 2 {
            a as i128 - self.q as i128
        } else {
            a as i128
        }
    }

    /// Miller-Rabin with the first thirteen primes as bases.  This is a proof of
    /// primality for `q < 3.3 * 10^24` (about `2^81`) and a strong probable-prime
    /// test above that.
    pub fn is_prime(&self) -> bool {
        const BASES: [u128; 13] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41];
        let q = self.q;
        for &b in &BASES {
            if q % b == 0 {
                return q == b;
            }
        }
        let twos = (q - 1).trailing_zeros();
        let odd = (q - 1) >> twos;
        'bases: for &b in &BASES {
            let mut x = self.pow(b, odd);
            if x == 1 || x == q - 1 {
                continue;
            }
            for _ in 1..twos {
                x = self.mul(x, x);
                if x == q - 1 {
                    continue 'bases;
                }
            }
            return false;
        }
        true
    }
}

/// The largest prime below `2^bits` that is `1 mod 2d`, the modulus rule of Section 7.2.
pub fn largest_ntt_prime_below(bits: u32, d: usize) -> Result<u128> {
    if !(4..=Modulus::MAX_BITS).contains(&bits) || !d.is_power_of_two() {
        return Err(Error::invalid("unsupported modulus size or ring degree"));
    }
    let step = 2 * d as u128;
    let mut q = (1u128 << bits) - 1;
    q -= (q - 1) % step;
    while q > step {
        if Modulus::new(q)?.is_prime() {
            return Ok(q);
        }
        q -= step;
    }
    Err(Error::invalid("no NTT-friendly prime of that size"))
}

/// A fixed-weight ternary challenge polynomial.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Challenge {
    coeffs: Vec<i8>,
}

impl Challenge {
    /// Builds a challenge from dense coefficients in `{-1, 0, 1}`.
    pub fn from_coeffs(coeffs: Vec<i8>) -> Result<Self> {
        if coeffs.iter().any(|c| !(-1..=1).contains(c)) {
            return Err(Error::invalid("challenge coefficient outside {-1, 0, 1}"));
        }
        Ok(Challenge { coeffs })
    }

    pub fn coeffs(&self) -> &[i8] {
        &self.coeffs
    }

    pub fn weight(&self) -> usize {
        self.coeffs.iter().filter(|&&c| c != 0).count()
    }

    /// `c * a` over the integers, in `Z[X]/(X^d + 1)`.
    pub fn mul_z(&self, a: &[i128]) -> PolyZ {
        let d = self.coeffs.len();
        assert_eq!(a.len(), d);
        let mut out = vec![0i128; d];
        for (shift, &c) in self.coeffs.iter().enumerate() {
            if c == 0 {
                continue;
            }
            // X^shift * a: the first d - shift coefficients of a move up by shift,
            // the others wrap around with the opposite sign (X^d = -1).
            let (straight, wrapped) = a.split_at(d - shift);
            let (low, high) = out.split_at_mut(shift);
            if c > 0 {
                high.iter_mut().zip(straight).for_each(|(o, &x)| *o += x);
                low.iter_mut().zip(wrapped).for_each(|(o, &x)| *o -= x);
            } else {
                high.iter_mut().zip(straight).for_each(|(o, &x)| *o -= x);
                low.iter_mut().zip(wrapped).for_each(|(o, &x)| *o += x);
            }
        }
        out
    }

    /// `c * a` in `R_q`.
    pub fn mul_q(&self, modulus: &Modulus, a: &[u128]) -> PolyQ {
        let d = self.coeffs.len();
        assert_eq!(a.len(), d);
        let mut out = vec![0u128; d];
        for (shift, &c) in self.coeffs.iter().enumerate() {
            if c == 0 {
                continue;
            }
            // As in `mul_z`, with the sign chosen outside the loops.
            let (straight, wrapped) = a.split_at(d - shift);
            let (low, high) = out.split_at_mut(shift);
            if c > 0 {
                high.iter_mut().zip(straight).for_each(|(o, &x)| *o = modulus.add(*o, x));
                low.iter_mut().zip(wrapped).for_each(|(o, &x)| *o = modulus.sub(*o, x));
            } else {
                high.iter_mut().zip(straight).for_each(|(o, &x)| *o = modulus.sub(*o, x));
                low.iter_mut().zip(wrapped).for_each(|(o, &x)| *o = modulus.add(*o, x));
            }
        }
        out
    }
}

/// The ring `R_q` with its negacyclic NTT.
#[derive(Clone, Debug)]
pub struct Ring {
    d: usize,
    modulus: Modulus,
    psi: u128,
    /// `zetas[k] = psi^bitrev(k)` in Montgomery form.
    zetas: Vec<u128>,
    /// `d^-1` in Montgomery form.
    d_inv: u128,
}

impl Ring {
    /// Builds the ring for a prime `q = 1 (mod 2d)`, with the canonical root of unity:
    /// `psi = g^((q-1)/(2d))` for the smallest integer `g >= 2` such that `psi^d = -1`.
    pub fn new(d: usize, q: u128) -> Result<Self> {
        let modulus = Modulus::new(q)?;
        if !d.is_power_of_two() || d < 2 {
            return Err(Error::invalid("ring degree must be a power of two, at least 2"));
        }
        if !modulus.is_prime() {
            return Err(Error::invalid("modulus is not prime"));
        }
        if (q - 1) % (2 * d as u128) != 0 {
            return Err(Error::invalid("modulus is not 1 mod 2d"));
        }
        let exponent = (q - 1) / (2 * d as u128);
        let psi = (2..q)
            .map(|g| modulus.pow(g, exponent))
            .find(|&psi| modulus.pow(psi, d as u128) == q - 1)
            .ok_or_else(|| Error::invalid("no primitive 2d-th root of unity"))?;
        Self::with_root(d, q, psi)
    }

    /// Builds the ring with a given primitive `2d`-th root of unity.
    pub fn with_root(d: usize, q: u128, psi: u128) -> Result<Self> {
        let modulus = Modulus::new(q)?;
        if !d.is_power_of_two() || d < 2 {
            return Err(Error::invalid("ring degree must be a power of two, at least 2"));
        }
        if psi >= q || modulus.pow(psi, d as u128) != q - 1 {
            return Err(Error::invalid("psi is not a primitive 2d-th root of unity"));
        }
        let log_d = d.trailing_zeros();
        let mut powers = Vec::with_capacity(d);
        let mut acc = modulus.to_mont(1);
        let psi_mont = modulus.to_mont(psi);
        for _ in 0..d {
            powers.push(acc);
            acc = modulus.mont_mul(acc, psi_mont);
        }
        let zetas = (0..d).map(|k| powers[k.reverse_bits() >> (usize::BITS - log_d)]).collect();
        let d_inv = modulus.to_mont(modulus.inv(d as u128 % q));
        Ok(Ring { d, modulus, psi, zetas, d_inv })
    }

    pub fn degree(&self) -> usize {
        self.d
    }

    pub fn modulus(&self) -> &Modulus {
        &self.modulus
    }

    pub fn q(&self) -> u128 {
        self.modulus.q
    }

    /// The primitive `2d`-th root of unity defining the NTT.
    pub fn psi(&self) -> u128 {
        self.psi
    }

    pub fn zero(&self) -> PolyQ {
        vec![0; self.d]
    }

    /// Forward negacyclic NTT, in place.  Input and output are plain residues.
    pub fn ntt(&self, a: &mut [u128]) {
        assert_eq!(a.len(), self.d);
        let m = &self.modulus;
        let mut k = 0;
        let mut len = self.d / 2;
        while len >= 1 {
            for start in (0..self.d).step_by(2 * len) {
                k += 1;
                let zeta = self.zetas[k];
                for j in start..start + len {
                    let t = m.mont_mul(a[j + len], zeta);
                    a[j + len] = m.sub(a[j], t);
                    a[j] = m.add(a[j], t);
                }
            }
            len /= 2;
        }
    }

    /// Inverse negacyclic NTT, in place.
    pub fn intt(&self, a: &mut [u128]) {
        assert_eq!(a.len(), self.d);
        let m = &self.modulus;
        let mut k = self.d;
        let mut len = 1;
        while len < self.d {
            for start in (0..self.d).step_by(2 * len) {
                k -= 1;
                let zeta = m.neg(self.zetas[k]);
                for j in start..start + len {
                    let t = a[j];
                    a[j] = m.add(t, a[j + len]);
                    a[j + len] = m.mont_mul(m.sub(t, a[j + len]), zeta);
                }
            }
            len *= 2;
        }
        for x in a.iter_mut() {
            *x = m.mont_mul(*x, self.d_inv);
        }
    }

    /// Product in `R_q` through the NTT.
    pub fn mul(&self, a: &[u128], b: &[u128]) -> PolyQ {
        let mut fa = a.to_vec();
        let mut fb = b.to_vec();
        self.ntt(&mut fa);
        self.ntt(&mut fb);
        for (x, y) in fa.iter_mut().zip(&fb) {
            *x = self.modulus.mul(*x, *y);
        }
        self.intt(&mut fa);
        fa
    }

    /// Schoolbook negacyclic product, the reference for the NTT.
    pub fn mul_schoolbook(&self, a: &[u128], b: &[u128]) -> PolyQ {
        assert_eq!(a.len(), self.d);
        assert_eq!(b.len(), self.d);
        let m = &self.modulus;
        let mut out = vec![0u128; self.d];
        for (i, &x) in a.iter().enumerate() {
            if x == 0 {
                continue;
            }
            for (j, &y) in b.iter().enumerate() {
                let prod = m.mul(x, y);
                let idx = i + j;
                if idx < self.d {
                    out[idx] = m.add(out[idx], prod);
                } else {
                    out[idx - self.d] = m.sub(out[idx - self.d], prod);
                }
            }
        }
        out
    }

    pub fn add(&self, a: &[u128], b: &[u128]) -> PolyQ {
        a.iter().zip(b).map(|(&x, &y)| self.modulus.add(x, y)).collect()
    }

    pub fn sub(&self, a: &[u128], b: &[u128]) -> PolyQ {
        a.iter().zip(b).map(|(&x, &y)| self.modulus.sub(x, y)).collect()
    }

    /// Reduces an integer polynomial mod `q`.
    pub fn reduce(&self, a: &[i128]) -> PolyQ {
        a.iter().map(|&x| self.modulus.reduce_i128(x)).collect()
    }

    /// `acc += scalar * a` for a small signed integer scalar.
    pub fn add_scaled(&self, acc: &mut [u128], scalar: i64, a: &[u128]) {
        if scalar == 0 {
            return;
        }
        let s = self.modulus.to_mont(self.modulus.reduce_i128(i128::from(scalar)));
        for (x, &y) in acc.iter_mut().zip(a) {
            *x = self.modulus.add(*x, self.modulus.mont_mul(y, s));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigUint;
    use num_traits::ToPrimitive;

    const PRIMES: [(u32, u128); 6] = [
        (32, (1 << 32) - 10239),
        (61, (1 << 61) - 10239),
        (63, (1 << 63) - 17407),
        (64, (1 << 64) - 1023),
        (66, (1 << 66) - 16383),
        (65, (1 << 65) - 10239),
    ];

    struct Lcg(u128);
    impl Lcg {
        fn next(&mut self) -> u128 {
            self.0 = self
                .0
                .wrapping_mul(0x2360ed051fc65da44385df649fccf645)
                .wrapping_add(0x5851f42d4c957f2d14057b7ef767814f);
            self.0 ^ (self.0 >> 61)
        }
    }

    #[test]
    fn published_primes_are_the_largest_ntt_primes() {
        for (bits, q) in PRIMES {
            assert_eq!(largest_ntt_prime_below(bits, 512).unwrap(), q);
            let m = Modulus::new(q).unwrap();
            assert!(m.is_prime());
            assert_eq!(m.bits(), bits);
            assert_eq!(q % 1024, 1);
        }
        assert!(!Modulus::new((1 << 61) - 10239 + 1024).unwrap().is_prime());
        assert!(!Modulus::new(561).unwrap().is_prime());
        assert!(Modulus::new(97).unwrap().is_prime());
    }

    #[test]
    fn modular_multiplication_matches_bigint() {
        let mut rng = Lcg(42);
        for (_, q) in PRIMES.into_iter().chain([(7, 97u128), (125, (1 << 125) - 735)]) {
            let m = Modulus::new(q).unwrap();
            let big_q = BigUint::from(q);
            let edge = [0, 1, 2, q - 1, q - 2, q / 2, q / 2 + 1];
            let mut operands: Vec<u128> = edge.to_vec();
            operands.extend((0..200).map(|_| rng.next() % q));
            for &a in &operands {
                for &b in operands.iter().take(40) {
                    let expected = (BigUint::from(a) * BigUint::from(b) % &big_q).to_u128().unwrap();
                    assert_eq!(m.mul(a, b), expected, "q = {q}, a = {a}, b = {b}");
                    assert_eq!(m.mont_mul(m.to_mont(a), b), expected);
                    assert_eq!(m.add(a, b), (a + b) % q);
                    assert_eq!(m.sub(a, b), (a + q - b) % q);
                }
                if a != 0 && m.is_prime() {
                    assert_eq!(m.mul(a, m.inv(a)), 1);
                }
                assert_eq!(m.reduce_i128(m.center(a)), a);
                assert_eq!(m.reduce_i128(-(a as i128)), m.neg(a));
            }
        }
    }

    #[test]
    fn ntt_matches_schoolbook() {
        let mut rng = Lcg(7);
        for (_, q) in PRIMES {
            for d in [16usize, 512] {
                let ring = Ring::new(d, q).unwrap();
                assert_eq!(ring.modulus().pow(ring.psi(), d as u128), q - 1);
                let a: PolyQ = (0..d).map(|_| rng.next() % q).collect();
                // Coefficients near q exercise the widest products.
                let b: PolyQ = (0..d).map(|i| if i % 3 == 0 { q - 1 - (i as u128) } else { rng.next() % q }).collect();
                let mut fa = a.clone();
                ring.ntt(&mut fa);
                ring.intt(&mut fa);
                assert_eq!(fa, a, "NTT round trip");
                assert_eq!(ring.mul(&a, &b), ring.mul_schoolbook(&a, &b), "q = {q}, d = {d}");
            }
        }
    }

    #[test]
    fn negacyclic_wraparound() {
        let ring = Ring::new(16, 97).unwrap();
        let mut x15 = ring.zero();
        x15[15] = 1;
        let mut x1 = ring.zero();
        x1[1] = 1;
        let mut minus_one = ring.zero();
        minus_one[0] = 96;
        assert_eq!(ring.mul(&x15, &x1), minus_one);
        assert_eq!(ring.mul_schoolbook(&x15, &x1), minus_one);
    }

    #[test]
    fn sparse_challenge_products() {
        let mut rng = Lcg(99);
        for (_, q) in [PRIMES[1], PRIMES[4]] {
            let d = 512;
            let ring = Ring::new(d, q).unwrap();
            let mut coeffs = vec![0i8; d];
            for i in 0..30 {
                coeffs[(i * 17 + 3) % d] = if i % 2 == 0 { 1 } else { -1 };
            }
            let c = Challenge::from_coeffs(coeffs.clone()).unwrap();
            assert_eq!(c.weight(), 30);
            let a_z: PolyZ = (0..d).map(|_| (rng.next() % (1 << 50)) as i128 - (1 << 49)).collect();
            let prod_z = c.mul_z(&a_z);
            let c_q: PolyQ = coeffs.iter().map(|&x| ring.modulus().reduce_i128(i128::from(x))).collect();
            let a_q = ring.reduce(&a_z);
            assert_eq!(ring.reduce(&prod_z), ring.mul(&c_q, &a_q));
            assert_eq!(c.mul_q(ring.modulus(), &a_q), ring.mul(&c_q, &a_q));
        }
        assert!(Challenge::from_coeffs(vec![0, 2]).is_err());
    }

    #[test]
    fn rejects_bad_parameters() {
        assert!(Ring::new(512, (1 << 61) - 1).is_err(), "prime but not 1 mod 1024");
        assert!(Ring::new(512, (1 << 61) - 10239 + 1024).is_err(), "not prime");
        assert!(Ring::new(500, (1 << 61) - 10239).is_err(), "degree not a power of two");
        assert!(Ring::with_root(16, 97, 1).is_err());
        assert!(Modulus::new(1 << 126).is_err());
        assert!(Modulus::new(96).is_err());
    }
}
