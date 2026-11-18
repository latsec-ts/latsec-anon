//! Canonical binary encodings (codec version 3).
//!
//! Version 1 stored a `2 kappa`-bit challenge digest in a signature and version 2
//! the challenge polynomial at two bits per coefficient, both with fixed-width
//! Gaussian components.  Decoders accept only the current version.
//!
//! Every encoded object starts with a two-byte header: a type tag and the codec
//! version.  All multi-bit fields are little-endian at both levels: bit 0 of a
//! field is written first, into the least significant free bit of the current
//! byte, and multi-byte integers are stored least significant byte first.
//!
//! * Residues mod `q` use `ceil(lg q)` bits and must be below `q`.
//! * Integers bounded by `B` in absolute value (fixed width) are stored as `x + B`
//!   in `bitlen(2B)` bits and must not exceed `2B`.
//! * Gaussian-distributed integers are Golomb-Rice coded with a parameter `k`: the
//!   `k` low bits of `|x|`, then `|x| >> k` in unary (that many one bits and a zero
//!   bit), then a sign bit (`1` for negative) if `x != 0`.  Decoders reject
//!   `|x| > B` for the object's bound `B`.
//! * A challenge with `omega_c` nonzero coefficients is its support's rank in the
//!   combinatorial number system, `sum_j binom(p_j, j)` over the positions
//!   `p_1 < ... < p_omega` (zero-based), in `bitlen(binom(d, omega_c) - 1)` bits,
//!   followed by one sign bit per nonzero coefficient in increasing position
//!   order.  That is `ceil(omega_c + lg binom(d, omega_c))` bits, the `b_c` of
//!   Section 7.3 of the paper.  Decoders reject ranks `>= binom(d, omega_c)`.
//!
//! Decoders reject wrong tags or versions, out-of-range values, nonzero padding
//! bits, and any length other than the exact expected one.  There is therefore
//! exactly one encoding of each value.

use num_bigint::BigUint;
use num_traits::{One, Zero};

use crate::error::{Error, Result};
use crate::ring::{Challenge, PolyQ, PolyZ};

/// Type tags of encoded objects.
pub mod tag {
    pub const VERIFICATION_KEY: u8 = 0x01;
    pub const ABORT_KEY: u8 = 0x02;
    pub const SECRET_SHARE: u8 = 0x03;
    pub const OPENING: u8 = 0x04;
    pub const PARTIAL_RESPONSE: u8 = 0x05;
    pub const SIGNATURE: u8 = 0x06;
    pub const SESSION_MESSAGE: u8 = 0x07;
}

/// The codec version written into every header.
pub const VERSION: u8 = crate::profile::CODEC_VERSION as u8;

/// Packs fixed-width fields, least-significant bit first.
#[derive(Default)]
pub struct BitWriter {
    out: Vec<u8>,
    cur: u8,
    nbits: u32,
}

impl BitWriter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_header(tag: u8) -> Self {
        BitWriter { out: vec![tag, VERSION], cur: 0, nbits: 0 }
    }

    /// Appends whole bytes.  Only valid on a byte boundary.
    pub fn bytes(&mut self, data: &[u8]) {
        assert_eq!(self.nbits, 0, "byte field written off a byte boundary");
        self.out.extend_from_slice(data);
    }

    /// Appends the low `width` bits of `value`; the higher bits must be zero.
    pub fn bits(&mut self, value: u128, width: u32) {
        debug_assert!(width == 128 || value >> width == 0);
        let (mut v, mut w) = (value, width);
        while w > 0 {
            let take = (8 - self.nbits).min(w);
            let chunk = (v & ((1u128 << take) - 1)) as u8;
            self.cur |= chunk << self.nbits;
            self.nbits += take;
            v >>= take;
            w -= take;
            if self.nbits == 8 {
                self.out.push(self.cur);
                self.cur = 0;
                self.nbits = 0;
            }
        }
    }

    /// Pads the last byte with zero bits and returns the encoding.
    pub fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            self.out.push(self.cur);
        }
        self.out
    }
}

/// Reads fields written by [`BitWriter`].
pub struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    nbits: u32,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        BitReader { data, pos: 0, nbits: 0 }
    }

    /// Checks the two-byte header.
    pub fn with_header(data: &'a [u8], tag: u8) -> Result<Self> {
        if data.len() < 2 || data[0] != tag {
            return Err(Error::decode("wrong object type"));
        }
        if data[1] != VERSION {
            return Err(Error::decode("unsupported codec version"));
        }
        Ok(BitReader { data, pos: 2, nbits: 0 })
    }

    pub fn bytes(&mut self, len: usize) -> Result<&'a [u8]> {
        if self.nbits != 0 {
            return Err(Error::decode("byte field read off a byte boundary"));
        }
        let end = self.pos.checked_add(len).filter(|&e| e <= self.data.len());
        let end = end.ok_or_else(|| Error::decode("encoding is too short"))?;
        let out = &self.data[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    pub fn bits(&mut self, width: u32) -> Result<u128> {
        let (mut value, mut got) = (0u128, 0u32);
        while got < width {
            let byte = *self.data.get(self.pos).ok_or_else(|| Error::decode("encoding is too short"))?;
            let take = (8 - self.nbits).min(width - got);
            let chunk = (byte >> self.nbits) & (((1u16 << take) - 1) as u8);
            value |= u128::from(chunk) << got;
            got += take;
            self.nbits += take;
            if self.nbits == 8 {
                self.nbits = 0;
                self.pos += 1;
            }
        }
        Ok(value)
    }

    /// Requires zero padding bits up to the next byte boundary and returns the
    /// number of bytes read; later bytes are left to the caller.
    pub fn finish_prefix(mut self) -> Result<usize> {
        if self.nbits != 0 {
            let padding = self.data[self.pos] >> self.nbits;
            if padding != 0 {
                return Err(Error::decode("nonzero padding bits"));
            }
            self.pos += 1;
            self.nbits = 0;
        }
        Ok(self.pos)
    }

    /// Requires zero padding bits and no trailing bytes.
    pub fn finish(mut self) -> Result<()> {
        if self.nbits != 0 {
            let padding = self.data[self.pos] >> self.nbits;
            if padding != 0 {
                return Err(Error::decode("nonzero padding bits"));
            }
            self.pos += 1;
        }
        if self.pos != self.data.len() {
            return Err(Error::decode("trailing bytes"));
        }
        Ok(())
    }
}

/// The width in bits of a residue mod `q`.
pub fn residue_width(q: u128) -> u32 {
    128 - (q - 1).leading_zeros()
}

/// The width in bits of an integer bounded by `bound` in absolute value.
pub fn signed_width(bound: u128) -> u32 {
    128 - (2 * bound).leading_zeros()
}

/// The number of bytes of `count` fields of `width` bits.
pub fn packed_len(count: usize, width: u32) -> usize {
    (count * width as usize).div_ceil(8)
}

/// Writes residue polynomials.
pub fn write_residues(writer: &mut BitWriter, polys: &[PolyQ], q: u128) {
    let width = residue_width(q);
    for &x in polys.iter().flatten() {
        debug_assert!(x < q);
        writer.bits(x, width);
    }
}

/// Reads `count` residue polynomials of degree `d`, rejecting values `>= q`.
pub fn read_residues(reader: &mut BitReader<'_>, count: usize, d: usize, q: u128) -> Result<Vec<PolyQ>> {
    let width = residue_width(q);
    let mut polys = Vec::with_capacity(count);
    for _ in 0..count {
        let mut poly = Vec::with_capacity(d);
        for _ in 0..d {
            let x = reader.bits(width)?;
            if x >= q {
                return Err(Error::decode("residue is not reduced"));
            }
            poly.push(x);
        }
        polys.push(poly);
    }
    Ok(polys)
}

/// Writes bounded integer polynomials.  Returns an error if a coefficient exceeds
/// `bound` in absolute value, since it would have no encoding.
pub fn write_bounded(writer: &mut BitWriter, polys: &[PolyZ], bound: u128) -> Result<()> {
    let width = signed_width(bound);
    for &x in polys.iter().flatten() {
        if x.unsigned_abs() > bound {
            return Err(Error::invalid("coefficient exceeds its encoding bound"));
        }
        writer.bits((x + bound as i128) as u128, width);
    }
    Ok(())
}

/// Reads `count` bounded integer polynomials of degree `d`.
pub fn read_bounded(reader: &mut BitReader<'_>, count: usize, d: usize, bound: u128) -> Result<Vec<PolyZ>> {
    let width = signed_width(bound);
    let mut polys = Vec::with_capacity(count);
    for _ in 0..count {
        let mut poly = Vec::with_capacity(d);
        for _ in 0..d {
            let x = reader.bits(width)?;
            if x > 2 * bound {
                return Err(Error::decode("coefficient exceeds its bound"));
            }
            poly.push(x as i128 - bound as i128);
        }
        polys.push(poly);
    }
    Ok(polys)
}

/// `binom(n, k)` as a big integer.
fn binomial(n: usize, k: usize) -> BigUint {
    if k > n {
        return BigUint::zero();
    }
    let mut acc = BigUint::one();
    for i in 0..k {
        acc = acc * BigUint::from(n - i) / BigUint::from(i + 1);
    }
    acc
}

/// The number of bits of the challenge rank: `bitlen(binom(d, weight) - 1)`.
pub fn challenge_rank_bits(d: usize, weight: usize) -> u32 {
    let count = binomial(d, weight);
    (count - 1u32).bits() as u32
}

/// Writes `value` in `width` bits, least significant bit first.
fn write_big(writer: &mut BitWriter, value: &BigUint, width: u32) {
    debug_assert!(value.bits() <= u64::from(width));
    let digits = value.to_u64_digits();
    let mut left = width;
    let mut index = 0;
    while left > 0 {
        let take = left.min(64);
        let digit = digits.get(index).copied().unwrap_or(0);
        writer.bits(u128::from(digit) & ((1u128 << take) - 1), take);
        left -= take;
        index += 1;
    }
}

fn read_big(reader: &mut BitReader<'_>, width: u32) -> Result<BigUint> {
    let mut value = BigUint::zero();
    let mut shift = 0u32;
    while shift < width {
        let take = (width - shift).min(64);
        value |= BigUint::from(reader.bits(take)?) << shift;
        shift += take;
    }
    Ok(value)
}

/// Writes a challenge: the rank of its support, then its signs.
pub fn write_challenge(writer: &mut BitWriter, c: &Challenge) {
    let positions: Vec<usize> = c.coeffs().iter().enumerate().filter(|(_, &x)| x != 0).map(|(i, _)| i).collect();
    let rank = positions.iter().enumerate().fold(BigUint::zero(), |acc, (j, &p)| acc + binomial(p, j + 1));
    write_big(writer, &rank, challenge_rank_bits(c.coeffs().len(), positions.len()));
    for &p in &positions {
        writer.bits(u128::from(c.coeffs()[p] < 0), 1);
    }
}

/// Reads a challenge of degree `d` and weight `weight`, rejecting ranks that are
/// not below `binom(d, weight)`.
pub fn read_challenge(reader: &mut BitReader<'_>, d: usize, weight: usize) -> Result<Challenge> {
    if weight == 0 || weight > d {
        return Err(Error::decode("challenge weight out of range"));
    }
    let mut rank = read_big(reader, challenge_rank_bits(d, weight))?;
    if rank >= binomial(d, weight) {
        return Err(Error::decode("challenge rank out of range"));
    }
    // Unrank greedily: p_j is the largest p below p_{j+1} with binom(p, j) <= the
    // remaining rank.  The binomial follows p and j by exact recurrences:
    // binom(p - 1, j) = binom(p, j) (p - j) / p and
    // binom(p - 1, j - 1) = binom(p, j) j / p.  The search for p_j stops at
    // p = j - 1 at the latest, where the binomial is zero, so p >= 1 whenever a
    // recurrence is applied (for j = 1 and p = 0 none is needed).
    let mut positions = vec![0usize; weight];
    let mut p = d - 1;
    let mut b = binomial(p, weight);
    for j in (1..=weight).rev() {
        while b > rank {
            b = b * (p - j) / p;
            p -= 1;
        }
        rank -= &b;
        positions[j - 1] = p;
        if j > 1 {
            b = b * j / p;
            p -= 1;
        }
    }
    let mut coeffs = vec![0i8; d];
    for &p in &positions {
        coeffs[p] = if reader.bits(1)? == 1 { -1 } else { 1 };
    }
    Ok(Challenge::from_coeffs(coeffs).expect("ternary by construction"))
}

/// The length in bits of the Golomb-Rice codewords of `values` with parameter `k`.
pub fn rice_length(values: impl Iterator<Item = u128>, k: u32) -> u128 {
    values.map(|a| u128::from(k) + 1 + (a >> k) + u128::from(a != 0)).sum()
}

/// An upper bound on the Golomb-Rice length in bits, with parameter `k`, of
/// `count` integers of absolute value at most `bound`: a codeword has at most
/// `k + 2 + (|x| >> k)` bits.
pub fn rice_max_bits_linf(count: usize, k: u32, bound: u128) -> BigUint {
    BigUint::from(count) * (BigUint::from(k + 2) + (bound >> k.min(127)))
}

/// An upper bound on the Golomb-Rice length in bits, with parameter `k`, of
/// `count` integers of squared Euclidean norm at most `norm_squared`:
/// `sum_i (|x_i| >> k) <= (sum_i |x_i|) >> k` and
/// `sum_i |x_i| <= floor(sqrt(count * norm_squared))` (Cauchy-Schwarz).
pub fn rice_max_bits_l2(count: usize, k: u32, norm_squared: &BigUint) -> BigUint {
    rice_max_bits_l1(count, k, &l1_bound(count, norm_squared))
}

/// `floor(sqrt(count * norm_squared))`: an upper bound on the `l1` norm of `count`
/// integers of squared Euclidean norm at most `norm_squared`.
pub fn l1_bound(count: usize, norm_squared: &BigUint) -> BigUint {
    (BigUint::from(count) * norm_squared).sqrt()
}

/// [`rice_max_bits_l2`] with the `l1` bound of [`l1_bound`] given.
pub fn rice_max_bits_l1(count: usize, k: u32, l1: &BigUint) -> BigUint {
    BigUint::from(count) * (k + 2) + (l1 >> k)
}

/// Writes Golomb-Rice coded integer polynomials.  Returns an error if a
/// coefficient exceeds `bound` in absolute value.
pub fn write_rice(writer: &mut BitWriter, polys: &[PolyZ], k: u32, bound: u128) -> Result<()> {
    for &x in polys.iter().flatten() {
        let a = x.unsigned_abs();
        if a > bound {
            return Err(Error::invalid("coefficient exceeds its encoding bound"));
        }
        if k > 0 {
            writer.bits(a & ((1u128 << k) - 1), k);
        }
        for _ in 0..(a >> k) {
            writer.bits(1, 1);
        }
        writer.bits(0, 1);
        if a != 0 {
            writer.bits(u128::from(x < 0), 1);
        }
    }
    Ok(())
}

/// Reads `count` Golomb-Rice coded polynomials of degree `d`, rejecting
/// coefficients above `bound` in absolute value.
pub fn read_rice(reader: &mut BitReader<'_>, count: usize, d: usize, k: u32, bound: u128) -> Result<Vec<PolyZ>> {
    if k >= 127 {
        return Err(Error::decode("Rice parameter out of range"));
    }
    let max_high = bound >> k;
    let mut polys = Vec::with_capacity(count);
    for _ in 0..count {
        let mut poly = Vec::with_capacity(d);
        for _ in 0..d {
            let low = if k > 0 { reader.bits(k)? } else { 0 };
            let mut high = 0u128;
            while reader.bits(1)? == 1 {
                high += 1;
                if high > max_high {
                    return Err(Error::decode("coefficient exceeds its bound"));
                }
            }
            let a = (high << k) | low;
            if a > bound {
                return Err(Error::decode("coefficient exceeds its bound"));
            }
            let negative = a != 0 && reader.bits(1)? == 1;
            poly.push(if negative { -(a as i128) } else { a as i128 });
        }
        polys.push(poly);
    }
    Ok(polys)
}

/// The Golomb-Rice parameter that minimizes the coded length of `polys`, the
/// smallest one on ties.  The length is convex in `k`, so a decoder can confirm
/// the choice from the neighbouring parameters ([`is_shortest_rice`]).
pub fn shortest_rice(polys: &[PolyZ]) -> u32 {
    let values = || polys.iter().flatten().map(|x| x.unsigned_abs());
    let mut best = (rice_length(values(), 0), 0u32);
    for k in 1..127 {
        let length = rice_length(values(), k);
        if length >= best.0 {
            break;
        }
        best = (length, k);
    }
    best.1
}

/// Whether `k` is the parameter [`shortest_rice`] chooses for `polys`.
pub fn is_shortest_rice(polys: &[PolyZ], k: u32) -> bool {
    let values = || polys.iter().flatten().map(|x| x.unsigned_abs());
    let here = rice_length(values(), k);
    (k == 0 || rice_length(values(), k - 1) > here) && (k >= 126 || rice_length(values(), k + 1) >= here)
}

/// Residue polynomials as bytes, without a header.  Used as hash input.
pub fn residues_to_bytes(polys: &[PolyQ], q: u128) -> Vec<u8> {
    let mut writer = BitWriter::new();
    write_residues(&mut writer, polys, q);
    writer.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_fields_round_trip() {
        let mut w = BitWriter::with_header(tag::SIGNATURE);
        w.bytes(&[0xaa, 0xbb]);
        let fields: [(u128, u32); 6] = [(1, 1), (0x3ff, 10), (0, 7), ((1 << 66) - 5, 67), (5, 3), (u128::MAX, 128)];
        for (v, width) in fields {
            w.bits(v, width);
        }
        let data = w.finish();
        let mut r = BitReader::with_header(&data, tag::SIGNATURE).unwrap();
        assert_eq!(r.bytes(2).unwrap(), &[0xaa, 0xbb]);
        for (v, width) in fields {
            assert_eq!(r.bits(width).unwrap(), v);
        }
        r.finish().unwrap();
        assert!(BitReader::with_header(&data, tag::OPENING).is_err());
        let mut wrong_version = data.clone();
        wrong_version[1] ^= 1;
        assert!(BitReader::with_header(&wrong_version, tag::SIGNATURE).is_err());
    }

    #[test]
    fn widths() {
        assert_eq!(residue_width((1 << 61) - 10239), 61);
        assert_eq!(residue_width((1 << 66) - 16383), 66);
        assert_eq!(residue_width(97), 7);
        assert_eq!(signed_width(1), 2);
        assert_eq!(signed_width(3), 3);
        assert_eq!(signed_width(4), 4);
        assert_eq!(packed_len(512, 61), 3904);
    }

    #[test]
    fn residues_are_canonical() {
        let q = 97u128;
        // Six fields of seven bits: 42 bits, so the sixth byte has six padding bits.
        let polys = vec![vec![0u128, 1, 96], vec![50, 7, 8]];
        let mut w = BitWriter::new();
        write_residues(&mut w, &polys, q);
        let data = w.finish();
        assert_eq!(data.len(), packed_len(6, 7));
        let mut r = BitReader::new(&data);
        assert_eq!(read_residues(&mut r, 2, 3, q).unwrap(), polys);
        r.finish().unwrap();
        // 97 itself is not a residue.
        let mut w = BitWriter::new();
        w.bits(97, 7);
        let bad = w.finish();
        assert!(read_residues(&mut BitReader::new(&bad), 1, 1, q).is_err());
        // Padding and length are checked.
        let mut padded = data.clone();
        *padded.last_mut().unwrap() |= 0x80;
        let mut r = BitReader::new(&padded);
        assert_eq!(read_residues(&mut r, 2, 3, q).unwrap(), polys);
        assert!(r.finish().is_err());
        let mut long = data.clone();
        long.push(0);
        let mut r = BitReader::new(&long);
        read_residues(&mut r, 2, 3, q).unwrap();
        assert!(r.finish().is_err());
        assert!(read_residues(&mut BitReader::new(&data[..5]), 2, 3, q).is_err());
    }

    #[test]
    fn bounded_integers_are_canonical() {
        let bound = 1000u128;
        let polys = vec![vec![-1000i128, 1000, 0, -1, 1, 999]];
        let mut w = BitWriter::new();
        write_bounded(&mut w, &polys, bound).unwrap();
        let data = w.finish();
        let mut r = BitReader::new(&data);
        assert_eq!(read_bounded(&mut r, 1, 6, bound).unwrap(), polys);
        r.finish().unwrap();
        assert!(write_bounded(&mut BitWriter::new(), &[vec![1001]], bound).is_err());
        let mut w = BitWriter::new();
        w.bits(2001, signed_width(bound));
        assert!(read_bounded(&mut BitReader::new(&w.finish()), 1, 1, bound).is_err());
        // A wide bound, as for combined responses.
        let bound = (1u128 << 64) - 3;
        let polys = vec![vec![-(bound as i128), bound as i128, 12345678901234567890i128]];
        let mut w = BitWriter::new();
        write_bounded(&mut w, &polys, bound).unwrap();
        let data = w.finish();
        assert_eq!(read_bounded(&mut BitReader::new(&data), 1, 3, bound).unwrap(), polys);
    }

    #[test]
    fn rice_codewords_are_little_endian() {
        // -5 with k = 1: low bit 1, high part 2 in unary (1, 1, 0), sign 1.
        // Bits in order 1, 1, 1, 0, 1 fill the byte from bit 0: 0b0001_0111.
        let mut w = BitWriter::new();
        write_rice(&mut w, &[vec![-5]], 1, 100).unwrap();
        assert_eq!(w.finish(), vec![0x17]);
        // 0 with k = 2: two low bits and the terminating zero, no sign bit.
        // 6 with k = 2: low bits 1, 0 (value 2), high part 1 (1, 0), sign 0.
        let mut w = BitWriter::new();
        write_rice(&mut w, &[vec![0, 6]], 2, 100).unwrap();
        // Bits: 0 0 0 | 0 1 1 0 0  ->  byte 0b0011_0000 = 0x30, nothing in the next.
        assert_eq!(w.finish(), vec![0x30]);
        assert_eq!(rice_length([0u128, 6].into_iter(), 2), 3 + 5);
    }

    #[test]
    fn rice_round_trip_and_rejection() {
        let polys = vec![vec![0i128, 1, -1, 1000, -65535, 7], vec![123456789, -2, 0, 0, 3, -4]];
        for k in [0u32, 1, 3, 8, 20] {
            let mut w = BitWriter::new();
            write_rice(&mut w, &polys, k, 1 << 30).unwrap();
            let data = w.finish();
            let mut r = BitReader::new(&data);
            assert_eq!(read_rice(&mut r, 2, 6, k, 1 << 30).unwrap(), polys);
            r.finish().unwrap();
            // A tighter bound rejects the large coefficient.
            assert!(read_rice(&mut BitReader::new(&data), 2, 6, k, 1 << 20).is_err());
        }
        assert!(write_rice(&mut BitWriter::new(), &polys, 3, 1000).is_err());
        // An endless unary run is cut off at the bound.
        assert!(read_rice(&mut BitReader::new(&[0xff; 64]), 1, 1, 0, 100).is_err());
        // The shortest parameter is found and confirmed; its neighbours are not.
        let k = shortest_rice(&polys);
        assert!(is_shortest_rice(&polys, k));
        assert!(!is_shortest_rice(&polys, k + 1));
        if k > 0 {
            assert!(!is_shortest_rice(&polys, k - 1));
        }
        let lengths: Vec<u128> =
            (0..40).map(|j| rice_length(polys.iter().flatten().map(|x| x.unsigned_abs()), j)).collect();
        assert_eq!(lengths.iter().min(), Some(&lengths[k as usize]));
        assert!(lengths[..k as usize].iter().all(|&l| l > lengths[k as usize]));
    }

    #[test]
    fn challenge_ranks() {
        // binom(5, 2) = 10 supports: {0, 1} has rank 0 and {3, 4} rank
        // binom(3, 1) + binom(4, 2) = 9, the last one.
        let challenge = |coeffs: Vec<i8>| Challenge::from_coeffs(coeffs).unwrap();
        assert_eq!(challenge_rank_bits(5, 2), 4);
        for (coeffs, rank, signs) in [
            (vec![1i8, -1, 0, 0, 0], 0u128, 0b10u128),
            (vec![0, 0, 0, -1, -1], 9, 0b11),
            (vec![0, 1, 0, 1, 0], 1 + 3, 0b00),
        ] {
            let mut w = BitWriter::new();
            write_challenge(&mut w, &challenge(coeffs.clone()));
            let data = w.finish();
            let mut r = BitReader::new(&data);
            assert_eq!(r.bits(4).unwrap(), rank);
            assert_eq!(r.bits(2).unwrap(), signs);
            assert_eq!(read_challenge(&mut BitReader::new(&data), 5, 2).unwrap(), challenge(coeffs));
        }
        // Every rank of a small set unranks to a support that ranks back to it, in
        // increasing colexicographic order of the supports.
        for (d, weight) in [(1usize, 1usize), (6, 1), (6, 6), (7, 3), (9, 4), (10, 9)] {
            let bits = challenge_rank_bits(d, weight);
            let count = binomial(d, weight).to_u64_digits().first().copied().unwrap_or(0);
            let mut previous: Option<Vec<usize>> = None;
            for rank in 0..count {
                let mut w = BitWriter::new();
                w.bits(u128::from(rank), bits);
                w.bits(0, weight as u32);
                let data = w.finish();
                let c = read_challenge(&mut BitReader::new(&data), d, weight).unwrap();
                assert_eq!(c.weight(), weight);
                let mut again = BitWriter::new();
                write_challenge(&mut again, &c);
                assert_eq!(again.finish(), data, "d = {d}, weight = {weight}, rank = {rank}");
                let mut support: Vec<usize> = (0..d).filter(|&i| c.coeffs()[i] != 0).collect();
                support.reverse();
                assert!(previous.as_ref().is_none_or(|p| p < &support));
                previous = Some(support);
            }
        }
        // Ranks 10 to 15 do not name a support.
        let mut w = BitWriter::new();
        w.bits(10, 4);
        w.bits(0, 2);
        assert!(read_challenge(&mut BitReader::new(&w.finish()), 5, 2).is_err());
        // b_c of Section 7.3 for d = 512: 192, 257 and 323 bits for omega_c = 30, 44, 60.
        for (omega, b_c) in [(30usize, 192u32), (44, 257), (60, 323)] {
            assert_eq!(challenge_rank_bits(512, omega) + omega as u32, b_c);
        }
    }
}
