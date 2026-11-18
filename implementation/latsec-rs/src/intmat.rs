//! Exact integer linear algebra for SEC reconstruction.
//!
//! Everything here is public-data arithmetic over arbitrary-precision integers:
//! row Hermite normal forms with a transformation, saturated integer kernels, an
//! exact integral LLL, Babai's nearest-plane algorithm and the greedy `l_inf`
//! refinement of [46, Algorithm 1].
//!
//! # Conventions
//!
//! * **Row HNF.** For a matrix `A`, `row_hnf` returns `H = U A` with `U` unimodular.
//!   The nonzero rows of `H` come first.  Their pivots (leftmost nonzero entries) are
//!   positive and sit in strictly increasing columns, and every entry above a pivot
//!   lies in `[0, pivot)`.  `H` is unique; `U` is not when `A` has dependent rows.
//! * **Gram-Schmidt data.** For a basis `b_1..b_r` the integral data are
//!   `d_0 = 1`, `d_i = prod_{j <= i} |b*_j|^2` and `lam[i][j] = d_j * mu_{i,j}`
//!   (Cohen, *A Course in Computational Algebraic Number Theory*, section 2.6.3).
//!   All of them are integers, so no rational arithmetic is needed.
//! * **LLL.** Cohen's integral LLL (Algorithm 2.6.7) with a rational `delta`; see
//!   [`lll_reduce`] for the exact order of operations and tie rules.

use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, Signed, Zero};

/// A row vector of arbitrary-precision integers.
pub type Row = Vec<BigInt>;

/// A dense integer matrix stored by rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntMatrix {
    ncols: usize,
    rows: Vec<Row>,
}

impl IntMatrix {
    /// Builds a matrix from rows; all rows must have `ncols` entries.
    pub fn from_rows(ncols: usize, rows: Vec<Row>) -> Self {
        assert!(rows.iter().all(|r| r.len() == ncols), "ragged matrix");
        IntMatrix { ncols, rows }
    }

    /// Builds a matrix from machine integers.
    pub fn from_i64(ncols: usize, rows: &[Vec<i64>]) -> Self {
        Self::from_rows(ncols, rows.iter().map(|r| r.iter().map(|&x| BigInt::from(x)).collect()).collect())
    }

    pub fn zero(nrows: usize, ncols: usize) -> Self {
        IntMatrix { ncols, rows: vec![vec![BigInt::zero(); ncols]; nrows] }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zero(n, n);
        for (i, row) in m.rows.iter_mut().enumerate() {
            row[i] = BigInt::one();
        }
        m
    }

    pub fn nrows(&self) -> usize {
        self.rows.len()
    }

    pub fn ncols(&self) -> usize {
        self.ncols
    }

    pub fn rows(&self) -> &[Row] {
        &self.rows
    }

    pub fn into_rows(self) -> Vec<Row> {
        self.rows
    }

    pub fn row(&self, i: usize) -> &Row {
        &self.rows[i]
    }

    /// The submatrix made of the listed rows, in the listed order.
    pub fn select_rows(&self, indices: &[usize]) -> Self {
        IntMatrix { ncols: self.ncols, rows: indices.iter().map(|&i| self.rows[i].clone()).collect() }
    }

    /// `v^T * self` for a row vector `v` with one entry per row.
    pub fn left_mul_vec(&self, v: &[BigInt]) -> Row {
        assert_eq!(v.len(), self.nrows());
        let mut out = vec![BigInt::zero(); self.ncols];
        for (coeff, row) in v.iter().zip(&self.rows) {
            if coeff.is_zero() {
                continue;
            }
            for (acc, entry) in out.iter_mut().zip(row) {
                if !entry.is_zero() {
                    *acc += coeff * entry;
                }
            }
        }
        out
    }

    /// Matrix product `self * other`.
    pub fn mul(&self, other: &IntMatrix) -> IntMatrix {
        assert_eq!(self.ncols, other.nrows());
        let rows = self.rows.iter().map(|r| other.left_mul_vec(r)).collect();
        IntMatrix { ncols: other.ncols, rows }
    }

    pub fn is_zero(&self) -> bool {
        self.rows.iter().all(|r| r.iter().all(|x| x.is_zero()))
    }
}

/// Inner product of two integer vectors.
pub fn dot(a: &[BigInt], b: &[BigInt]) -> BigInt {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = BigInt::zero();
    for (x, y) in a.iter().zip(b) {
        if !x.is_zero() && !y.is_zero() {
            acc += x * y;
        }
    }
    acc
}

/// `row -= q * other`.
fn sub_mul(row: &mut [BigInt], q: &BigInt, other: &[BigInt]) {
    if q.is_zero() {
        return;
    }
    for (x, y) in row.iter_mut().zip(other) {
        if !y.is_zero() {
            *x -= q * y;
        }
    }
}

/// `floor(num / den + 1/2)` for `den > 0`: nearest integer, ties toward `+inf`.
pub fn round_half_up(num: &BigInt, den: &BigInt) -> BigInt {
    debug_assert!(den.is_positive());
    let two_den: BigInt = den << 1usize;
    let shifted: BigInt = num << 1usize;
    (shifted + den).div_floor(&two_den)
}

/// The `l_inf` norm of an integer vector.
pub fn linf_norm(v: &[BigInt]) -> BigInt {
    v.iter().map(|x| x.abs()).max().unwrap_or_else(BigInt::zero)
}

/// The `l_1` norm of an integer vector.
pub fn l1_norm(v: &[BigInt]) -> BigInt {
    v.iter().map(|x| x.abs()).sum()
}

/// The number of nonzero entries.
pub fn hamming_weight(v: &[BigInt]) -> usize {
    v.iter().filter(|x| !x.is_zero()).count()
}

/// Row-reduces `rows` to Hermite normal form on their first `main_cols` columns.
///
/// Every row operation is applied to the whole row, so trailing columns can carry a
/// transformation.  Returns the pivot columns; the rank is their number.  Rows from
/// the rank on are zero on the first `main_cols` columns.
///
/// The elimination in a column repeatedly moves the entry of least absolute value to
/// the pivot position and reduces the other entries by it with nearest-integer
/// quotients, which keeps the cofactors comparatively small.
fn hnf_in_place(rows: &mut [Row], main_cols: usize) -> Vec<usize> {
    let nrows = rows.len();
    let mut pivots = Vec::new();
    let mut rank = 0;
    for col in 0..main_cols {
        if rank == nrows {
            break;
        }
        loop {
            // The remaining row with the smallest nonzero entry in this column.
            let mut best: Option<usize> = None;
            let mut others = false;
            for i in rank..nrows {
                if rows[i][col].is_zero() {
                    continue;
                }
                match best {
                    None => best = Some(i),
                    Some(b) => {
                        others = true;
                        if rows[i][col].magnitude() < rows[b][col].magnitude() {
                            best = Some(i);
                        }
                    }
                }
            }
            let Some(best) = best else { break };
            rows.swap(rank, best);
            if !others {
                break;
            }
            let (head, tail) = rows.split_at_mut(rank + 1);
            let pivot_row = &head[rank];
            let pivot_abs = pivot_row[col].abs();
            for row in tail.iter_mut() {
                if row[col].is_zero() {
                    continue;
                }
                let mut q = round_half_up(&row[col], &pivot_abs);
                if pivot_row[col].is_negative() {
                    q = -q;
                }
                sub_mul(row, &q, pivot_row);
            }
        }
        if rows[rank][col].is_zero() {
            continue;
        }
        if rows[rank][col].is_negative() {
            for x in rows[rank].iter_mut() {
                *x = -std::mem::take(x);
            }
        }
        // Reduce the entries above the pivot into [0, pivot).
        let (above, rest) = rows.split_at_mut(rank);
        let pivot_row = &rest[0];
        for row in above.iter_mut() {
            let q = row[col].div_floor(&pivot_row[col]);
            sub_mul(row, &q, pivot_row);
        }
        pivots.push(col);
        rank += 1;
    }
    pivots
}

/// The result of [`row_hnf`].
#[derive(Clone, Debug)]
pub struct Hnf {
    /// `H = U * A`, with zero rows last.
    pub hnf: IntMatrix,
    /// The unimodular transformation `U`.
    pub transform: IntMatrix,
    /// Pivot column of each nonzero row of `H`.
    pub pivots: Vec<usize>,
}

impl Hnf {
    pub fn rank(&self) -> usize {
        self.pivots.len()
    }

    /// The rows of `U` that map to zero.  They form a basis of the saturated left
    /// kernel `{x : x^T A = 0}` because `U` is unimodular.
    pub fn kernel_rows(&self) -> &[Row] {
        &self.transform.rows()[self.rank()..]
    }
}

/// Row Hermite normal form with a unimodular transformation, `H = U * A`.
pub fn row_hnf(a: &IntMatrix) -> Hnf {
    let (nrows, ncols) = (a.nrows(), a.ncols());
    let mut rows: Vec<Row> = a
        .rows()
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let mut row = r.clone();
            row.extend((0..nrows).map(|j| if i == j { BigInt::one() } else { BigInt::zero() }));
            row
        })
        .collect();
    let pivots = hnf_in_place(&mut rows, ncols);
    let mut hnf_rows = Vec::with_capacity(nrows);
    let mut u_rows = Vec::with_capacity(nrows);
    for mut row in rows {
        let u = row.split_off(ncols);
        hnf_rows.push(row);
        u_rows.push(u);
    }
    Hnf { hnf: IntMatrix::from_rows(ncols, hnf_rows), transform: IntMatrix::from_rows(nrows, u_rows), pivots }
}

/// Row Hermite normal form of a lattice basis (or generating set), without zero rows.
pub fn hnf_basis(generators: &[Row], ncols: usize) -> IntMatrix {
    let mut rows = generators.to_vec();
    let pivots = hnf_in_place(&mut rows, ncols);
    rows.truncate(pivots.len());
    IntMatrix::from_rows(ncols, rows)
}

/// The unique HNF basis of the saturated left kernel `{x in Z^t : x^T A = 0}`.
pub fn left_kernel_hnf(a: &IntMatrix) -> IntMatrix {
    let hnf = row_hnf(a);
    hnf_basis(hnf.kernel_rows(), a.nrows())
}

/// Integral Gram-Schmidt data of a basis (see the module documentation).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gso {
    /// `d[0] = 1` and `d[i] = prod_{j <= i} |b*_j|^2` for `1 <= i <= r`.
    pub d: Vec<BigInt>,
    /// `lam[i][j] = d[j + 1] * mu_{i+1, j+1}` for `j < i` (zero-based storage).
    pub lam: Vec<Vec<BigInt>>,
}

fn exact_div(num: BigInt, den: &BigInt) -> BigInt {
    debug_assert!((&num % den).is_zero(), "inexact division in integral Gram-Schmidt");
    num / den
}

/// Fills row `k` (zero-based) of the Gram-Schmidt data from rows `0..k`.
fn gso_extend(basis: &[Row], gso: &mut Gso, k: usize) {
    for j in 0..=k {
        let mut u = dot(&basis[k], &basis[j]);
        for i in 0..j {
            u = exact_div(&gso.d[i + 1] * u - &gso.lam[k][i] * &gso.lam[j][i], &gso.d[i]);
        }
        if j < k {
            gso.lam[k][j] = u;
        } else {
            assert!(!u.is_zero(), "basis vectors are linearly dependent");
            gso.d[k + 1] = u;
        }
    }
}

/// Integral Gram-Schmidt data of a linearly independent family of rows.
pub fn integral_gso(basis: &[Row]) -> Gso {
    let r = basis.len();
    let mut gso = Gso { d: vec![BigInt::one(); r + 1], lam: vec![vec![BigInt::zero(); r]; r] };
    for k in 0..r {
        gso_extend(basis, &mut gso, k);
    }
    gso
}

/// The LLL parameter `delta = num / den`, with `1/4 < delta < 1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LllDelta {
    pub num: u32,
    pub den: u32,
}

impl LllDelta {
    /// `delta = 99/100`, the value used throughout this crate.
    pub const DEFAULT: LllDelta = LllDelta { num: 99, den: 100 };
}

/// Exact LLL reduction of a basis, in place.  Returns the Gram-Schmidt data of the
/// reduced basis.
///
/// This is Cohen's integral LLL, Algorithm 2.6.7, with these fixed conventions:
///
/// 1. Vectors are processed with an index `k` starting at the second vector.  When
///    `k` reaches a vector for the first time, its Gram-Schmidt row is computed.
/// 2. `b_k` is size-reduced against `b_{k-1}` only, then the Lovasz condition
///    `|b*_k|^2 >= (delta - mu_{k,k-1}^2) |b*_{k-1}|^2` is tested exactly as
///    `den * (d_k d_{k-2} + lam_{k,k-1}^2) >= num * d_{k-1}^2`.
/// 3. If it fails, `b_k` and `b_{k-1}` are swapped and `k` decreases by one (not
///    below the second vector).  Otherwise `b_k` is size-reduced against
///    `b_{k-2}, ..., b_1` in that order and `k` increases by one.
/// 4. Size reduction of `b_k` against `b_l` does nothing when `|mu_{k,l}| <= 1/2`
///    (both `+1/2` and `-1/2` are left alone).  Otherwise it subtracts
///    `floor(mu_{k,l} + 1/2)` times `b_l`.
///
/// The output is a function of the input basis alone.
pub fn lll_reduce(basis: &mut [Row], delta: LllDelta) -> Gso {
    assert!(4 * delta.num > delta.den && delta.num < delta.den, "delta must lie in (1/4, 1)");
    let r = basis.len();
    let mut gso = Gso { d: vec![BigInt::one(); r + 1], lam: vec![vec![BigInt::zero(); r]; r] };
    if r == 0 {
        return gso;
    }
    gso_extend(basis, &mut gso, 0);
    let num = BigInt::from(delta.num);
    let den = BigInt::from(delta.den);
    // `k` is the zero-based index of the vector being inserted; `kmax` the largest
    // index whose Gram-Schmidt row is known.
    let mut k = 1;
    let mut kmax = 0;
    while k < r {
        if k > kmax {
            kmax = k;
            gso_extend(basis, &mut gso, k);
        }
        loop {
            size_reduce(basis, &mut gso, k, k - 1);
            let lam = &gso.lam[k][k - 1];
            let lhs = &den * (&gso.d[k + 1] * &gso.d[k - 1] + lam * lam);
            let rhs = &num * &gso.d[k] * &gso.d[k];
            if lhs < rhs {
                swap_adjacent(basis, &mut gso, k, kmax);
                if k > 1 {
                    k -= 1;
                }
            } else {
                for l in (0..k.saturating_sub(1)).rev() {
                    size_reduce(basis, &mut gso, k, l);
                }
                k += 1;
                break;
            }
        }
    }
    gso
}

/// Cohen's `REDI(k, l)` with zero-based indices.
fn size_reduce(basis: &mut [Row], gso: &mut Gso, k: usize, l: usize) {
    let d_l = &gso.d[l + 1];
    let twice: BigInt = &gso.lam[k][l] << 1usize;
    if twice.magnitude() <= d_l.magnitude() {
        return;
    }
    let q = round_half_up(&gso.lam[k][l], d_l);
    let (low, high) = basis.split_at_mut(k);
    sub_mul(&mut high[0], &q, &low[l]);
    let (lam_low, lam_high) = gso.lam.split_at_mut(k);
    lam_high[0][l] -= &q * d_l;
    for i in 0..l {
        lam_high[0][i] -= &q * &lam_low[l][i];
    }
}

/// Cohen's `SWAPI(k)` with zero-based indices: exchanges `b_k` and `b_{k-1}`.
fn swap_adjacent(basis: &mut [Row], gso: &mut Gso, k: usize, kmax: usize) {
    basis.swap(k, k - 1);
    let (lam_low, lam_high) = gso.lam.split_at_mut(k);
    for j in 0..k - 1 {
        std::mem::swap(&mut lam_low[k - 1][j], &mut lam_high[0][j]);
    }
    let lam = gso.lam[k][k - 1].clone();
    let b = exact_div(&gso.d[k - 1] * &gso.d[k + 1] + &lam * &lam, &gso.d[k]);
    for i in k + 1..=kmax {
        let t = gso.lam[i][k].clone();
        let new_ik = exact_div(&gso.d[k + 1] * &gso.lam[i][k - 1] - &lam * &t, &gso.d[k]);
        gso.lam[i][k - 1] = exact_div(&b * &t + &lam * &new_ik, &gso.d[k + 1]);
        gso.lam[i][k] = new_ik;
    }
    gso.d[k] = b;
}

/// Exact Babai nearest-plane reduction of `target` modulo the lattice spanned by
/// `basis`, whose Gram-Schmidt data are `gso`.
///
/// Planes are visited from the last basis vector to the first, and each coefficient
/// is rounded with ties toward `+inf`.  The result lies in `target + L(basis)` and
/// depends only on that coset and on the ordered basis.
// The index loops follow the recurrences of the integral Gram-Schmidt process.
#[allow(clippy::needless_range_loop)]
pub fn babai_nearest_plane(basis: &[Row], gso: &Gso, target: &[BigInt]) -> Row {
    let r = basis.len();
    let mut w = target.to_vec();
    if r == 0 {
        return w;
    }
    // nu[j] = d_{j+1} * <w, b*_j> / |b*_j|^2, an integer.
    let mut nu: Vec<BigInt> = Vec::with_capacity(r);
    for j in 0..r {
        let mut u = dot(&w, &basis[j]);
        for i in 0..j {
            u = exact_div(&gso.d[i + 1] * u - &nu[i] * &gso.lam[j][i], &gso.d[i]);
        }
        nu.push(u);
    }
    for i in (0..r).rev() {
        let c = round_half_up(&nu[i], &gso.d[i + 1]);
        if c.is_zero() {
            continue;
        }
        sub_mul(&mut w, &c, &basis[i]);
        for j in 0..i {
            nu[j] -= &c * &gso.lam[i][j];
        }
    }
    w
}

/// Greedy `l_inf` refinement ([46, Algorithm 1], second stage).
///
/// For each basis vector in order, the coordinate of largest absolute value (smallest
/// index on ties) is targeted.  All shifts `alpha` in `[alpha0 - window, alpha0 + window]`
/// are scanned in increasing order and the first one attaining the smallest `l_inf`
/// norm is kept if it is a strict improvement.  Stops after `passes` passes or after
/// a pass without improvement.
pub fn greedy_reduce_linf(lambda: &[BigInt], basis: &[Row], passes: u32, window: u32) -> Row {
    let mut lam = lambda.to_vec();
    if basis.is_empty() || lam.is_empty() {
        return lam;
    }
    let mut best = linf_norm(&lam);
    let window = BigInt::from(window);
    for _ in 0..passes {
        let mut improved = false;
        for b in basis {
            let mut j = 0;
            for (idx, x) in lam.iter().enumerate() {
                if x.magnitude() > lam[j].magnitude() {
                    j = idx;
                }
            }
            if b[j].is_zero() {
                continue;
            }
            // floor(-lam[j] / b[j] + 1/2) with a positive denominator.
            let (num, den) = if b[j].is_negative() { (lam[j].clone(), -&b[j]) } else { (-&lam[j], b[j].clone()) };
            let alpha0 = round_half_up(&num, &den);
            let mut norm_best = best.clone();
            let mut lam_best: Option<Row> = None;
            let mut alpha = &alpha0 - &window;
            let last = &alpha0 + &window;
            while alpha <= last {
                let cand: Row = lam.iter().zip(b).map(|(x, y)| x + &alpha * y).collect();
                let cn = linf_norm(&cand);
                if cn < norm_best {
                    norm_best = cn;
                    lam_best = Some(cand);
                }
                alpha += 1;
            }
            if let Some(cand) = lam_best {
                lam = cand;
                best = norm_best;
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }
    lam
}

/// Parameters of the reconstruction-vector algorithm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReconParams {
    pub delta: LllDelta,
    pub passes: u32,
    pub window: u32,
}

impl ReconParams {
    /// `delta = 99/100`, `passes = window = 2`, as in the statistics of [46].
    pub const DEFAULT: ReconParams = ReconParams { delta: LllDelta::DEFAULT, passes: 2, window: 2 };
}

/// Intermediate values of [`recon_vector_detailed`].
#[derive(Clone, Debug)]
pub struct ReconDetail {
    /// The HNF witness `lambda_0` (a row of the HNF transformation).
    pub witness: Row,
    /// The unique HNF basis of the kernel lattice `K_T`.
    pub kernel_hnf: IntMatrix,
    /// The LLL-reduced kernel basis used by Babai and the greedy stage.
    pub kernel_reduced: IntMatrix,
    /// The vector after Babai's nearest plane.
    pub after_babai: Row,
    /// The final reconstruction vector.
    pub lambda: Row,
}

/// The first unit row vector `e_1^T` of length `ncols`.
fn e1(ncols: usize) -> Row {
    let mut v = vec![BigInt::zero(); ncols];
    if ncols > 0 {
        v[0] = BigInt::one();
    }
    v
}

/// [46, Algorithm 1]: the reconstruction vector of the rows `g_t`.
///
/// Returns `None` if `e_1^T` is not in the integer row span (the set is not
/// authorized).  Otherwise the result satisfies `lambda^T g_t = e_1^T`, which is
/// checked before returning.
pub fn recon_vector_detailed(g_t: &IntMatrix, params: ReconParams) -> Option<ReconDetail> {
    if g_t.nrows() == 0 || g_t.ncols() == 0 {
        return None;
    }
    let target = e1(g_t.ncols());
    let hnf = row_hnf(g_t);
    if hnf.rank() == 0 || hnf.hnf.row(0) != &target {
        return None;
    }
    let witness = hnf.transform.row(0).clone();
    let kernel_hnf = hnf_basis(hnf.kernel_rows(), g_t.nrows());
    let mut reduced = kernel_hnf.rows().to_vec();
    let gso = if reduced.len() > 1 { lll_reduce(&mut reduced, params.delta) } else { integral_gso(&reduced) };
    let after_babai = babai_nearest_plane(&reduced, &gso, &witness);
    let lambda = greedy_reduce_linf(&after_babai, &reduced, params.passes, params.window);
    assert!(g_t.left_mul_vec(&lambda) == target, "reconstruction vector failed certification");
    Some(ReconDetail {
        witness,
        kernel_hnf,
        kernel_reduced: IntMatrix::from_rows(g_t.nrows(), reduced),
        after_babai,
        lambda,
    })
}

/// The reconstruction vector of the rows `g_t`, or `None` if the set is not authorized.
pub fn recon_vector(g_t: &IntMatrix, params: ReconParams) -> Option<Row> {
    recon_vector_detailed(g_t, params).map(|d| d.lambda)
}

/// Whether `e_1^T` lies in the integer row span of `g_t`.
pub fn is_authorized(g_t: &IntMatrix) -> bool {
    if g_t.nrows() == 0 || g_t.ncols() == 0 {
        return false;
    }
    let mut rows = g_t.rows().to_vec();
    let pivots = hnf_in_place(&mut rows, g_t.ncols());
    !pivots.is_empty() && rows[0] == e1(g_t.ncols())
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_rational::BigRational;

    fn mat(rows: &[&[i64]]) -> IntMatrix {
        let ncols = rows.first().map_or(0, |r| r.len());
        IntMatrix::from_i64(ncols, &rows.iter().map(|r| r.to_vec()).collect::<Vec<_>>())
    }

    fn ints(v: &[i64]) -> Row {
        v.iter().map(|&x| BigInt::from(x)).collect()
    }

    /// Small deterministic generator for test matrices.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn range(&mut self, lo: i64, hi: i64) -> i64 {
            lo + (self.next() % (hi - lo + 1) as u64) as i64
        }
    }

    fn random_matrix(rng: &mut Lcg, nrows: usize, ncols: usize, bound: i64) -> IntMatrix {
        let rows: Vec<Vec<i64>> = (0..nrows).map(|_| (0..ncols).map(|_| rng.range(-bound, bound)).collect()).collect();
        IntMatrix::from_i64(ncols, &rows)
    }

    fn det(m: &IntMatrix) -> BigInt {
        // Fraction-free (Bareiss) elimination.
        let n = m.nrows();
        assert_eq!(n, m.ncols());
        let mut a: Vec<Row> = m.rows().to_vec();
        let mut sign = BigInt::one();
        let mut prev = BigInt::one();
        for k in 0..n {
            if a[k][k].is_zero() {
                match (k + 1..n).find(|&i| !a[i][k].is_zero()) {
                    Some(i) => {
                        a.swap(k, i);
                        sign = -sign;
                    }
                    None => return BigInt::zero(),
                }
            }
            for i in k + 1..n {
                for j in k + 1..n {
                    a[i][j] = (&a[i][j] * &a[k][k] - &a[i][k] * &a[k][j]) / &prev;
                }
            }
            prev = a[k][k].clone();
        }
        sign * &a[n - 1][n - 1]
    }

    fn check_hnf_shape(h: &IntMatrix, pivots: &[usize]) {
        for (i, &c) in pivots.iter().enumerate() {
            let row = h.row(i);
            assert!(row[..c].iter().all(|x| x.is_zero()));
            assert!(row[c].is_positive());
            for above in 0..i {
                let x = &h.row(above)[c];
                assert!(!x.is_negative() && x < &row[c]);
            }
            if i > 0 {
                assert!(pivots[i - 1] < c);
            }
        }
        for i in pivots.len()..h.nrows() {
            assert!(h.row(i).iter().all(|x| x.is_zero()));
        }
    }

    #[test]
    fn hnf_small_example() {
        let a = mat(&[&[2, 4, 4], &[-6, 6, 12], &[10, 4, 16]]);
        let res = row_hnf(&a);
        check_hnf_shape(&res.hnf, &res.pivots);
        assert_eq!(res.transform.mul(&a), res.hnf);
        assert_eq!(det(&res.transform).abs(), BigInt::one());
        // SageMath 10.9: Matrix(ZZ, [[2,4,4],[-6,6,12],[10,4,16]]).hermite_form()
        assert_eq!(res.hnf, mat(&[&[2, 0, 120], &[0, 2, 20], &[0, 0, 156]]));
    }

    #[test]
    fn hnf_random_matrices() {
        let mut rng = Lcg(1);
        for trial in 0..200 {
            let nrows = 1 + (trial % 7);
            let ncols = 1 + (trial % 5);
            let a = random_matrix(&mut rng, nrows, ncols, 6);
            let res = row_hnf(&a);
            check_hnf_shape(&res.hnf, &res.pivots);
            assert_eq!(res.transform.mul(&a), res.hnf);
            assert_eq!(det(&res.transform).abs(), BigInt::one(), "transform must be unimodular");
            // The HNF does not depend on the generating set: permute and combine rows.
            let mut rows = a.rows().to_vec();
            rows.reverse();
            if rows.len() > 1 {
                let extra: Row = rows[0].iter().zip(&rows[1]).map(|(x, y)| x * 3 - y * 2).collect();
                rows.push(extra);
            }
            let again = hnf_basis(&rows, ncols);
            assert_eq!(again.rows(), &res.hnf.rows()[..res.rank()]);
        }
    }

    #[test]
    fn kernel_is_saturated() {
        // Rows (2, 0) and (0, 2) and (1, 1): kernel of the 3 x 2 matrix is spanned by
        // (1, 1, -2); clearing denominators of a rational basis could give a multiple.
        let a = mat(&[&[2, 0], &[0, 2], &[1, 1]]);
        let k = left_kernel_hnf(&a);
        assert_eq!(k, mat(&[&[1, 1, -2]]));

        let mut rng = Lcg(7);
        for _ in 0..100 {
            let a = random_matrix(&mut rng, 6, 3, 4);
            let k = left_kernel_hnf(&a);
            assert!(k.mul(&a).is_zero());
            let rank = row_hnf(&a).rank();
            assert_eq!(k.nrows(), 6 - rank);
            // Saturation: the gcd of the maximal minors of a saturated basis is 1.
            // Equivalent check here: extending by integer rows reaches determinant 1,
            // i.e. the HNF of [K ; anything] restricted to pivots has pivot product 1
            // when K is completed.  Use the Smith-style test through the HNF of K^T.
            let kt_rows: Vec<Row> =
                (0..k.ncols()).map(|c| (0..k.nrows()).map(|r| k.row(r)[c].clone()).collect()).collect();
            let h = hnf_basis(&kt_rows, k.nrows());
            for (i, row) in h.rows().iter().enumerate() {
                assert!(row[i].is_one(), "kernel basis is not saturated");
            }
        }
    }

    fn rational_gso(basis: &[Row]) -> Vec<Vec<BigRational>> {
        let mut bs: Vec<Vec<BigRational>> = Vec::new();
        for b in basis {
            let bq: Vec<BigRational> = b.iter().map(|x| BigRational::from_integer(x.clone())).collect();
            let mut v = bq.clone();
            for prev in &bs {
                let d: BigRational = prev.iter().map(|x| x * x).sum();
                if d.is_zero() {
                    continue;
                }
                let num: BigRational = bq.iter().zip(prev).map(|(x, y)| x * y).sum();
                let mu = num / d;
                for (x, y) in v.iter_mut().zip(prev) {
                    *x -= &mu * y;
                }
            }
            bs.push(v);
        }
        bs
    }

    /// `babai_exact` of the SageMath pipeline of [46], over the rationals.
    fn babai_rational(basis: &[Row], target: &[BigInt]) -> Row {
        let bs = rational_gso(basis);
        let half = BigRational::new(BigInt::one(), BigInt::from(2));
        let mut w: Vec<BigRational> = target.iter().map(|x| BigRational::from_integer(x.clone())).collect();
        for i in (0..basis.len()).rev() {
            let d: BigRational = bs[i].iter().map(|x| x * x).sum();
            let c: BigRational = w.iter().zip(&bs[i]).map(|(x, y)| x * y).sum::<BigRational>() / d;
            let ch = (c + &half).floor();
            for (x, y) in w.iter_mut().zip(&basis[i]) {
                *x -= &ch * BigRational::from_integer(y.clone());
            }
        }
        w.into_iter().map(|x| x.to_integer()).collect()
    }

    fn check_lll_reduced(basis: &[Row], delta: LllDelta) {
        let bs = rational_gso(basis);
        let norms: Vec<BigRational> = bs.iter().map(|v| v.iter().map(|x| x * x).sum()).collect();
        let half = BigRational::new(BigInt::one(), BigInt::from(2));
        let delta_q = BigRational::new(BigInt::from(delta.num), BigInt::from(delta.den));
        for i in 0..basis.len() {
            for j in 0..i {
                let num: BigRational =
                    basis[i].iter().zip(&bs[j]).map(|(x, y)| BigRational::from_integer(x.clone()) * y).sum();
                let mu = num / &norms[j];
                assert!(mu.abs() <= half, "not size-reduced");
                if j + 1 == i {
                    assert!(norms[i] >= (&delta_q - &mu * &mu) * &norms[j], "Lovasz condition fails");
                }
            }
        }
    }

    #[test]
    fn lll_reduces_and_preserves_lattice() {
        let mut rng = Lcg(11);
        for trial in 0..60 {
            let r = 2 + trial % 5;
            let dim = r + trial % 3;
            let mut basis: Vec<Row> = loop {
                let m = random_matrix(&mut rng, r, dim, 40);
                if row_hnf(&m).rank() == r {
                    break m.into_rows();
                }
            };
            let before = hnf_basis(&basis, dim);
            let gso = lll_reduce(&mut basis, LllDelta::DEFAULT);
            check_lll_reduced(&basis, LllDelta::DEFAULT);
            assert_eq!(hnf_basis(&basis, dim), before, "LLL changed the lattice");
            assert_eq!(gso, integral_gso(&basis), "incremental Gram-Schmidt data drifted");
        }
    }

    #[test]
    fn lll_known_example() {
        // SageMath 10.9: Matrix(ZZ, [[1,1,1],[-1,0,2],[3,5,6]]).LLL(delta=0.75)
        let mut basis = vec![ints(&[1, 1, 1]), ints(&[-1, 0, 2]), ints(&[3, 5, 6])];
        lll_reduce(&mut basis, LllDelta { num: 3, den: 4 });
        check_lll_reduced(&basis, LllDelta { num: 3, den: 4 });
        assert_eq!(basis, vec![ints(&[0, 1, 0]), ints(&[1, 0, 1]), ints(&[-1, 0, 2])]);
    }

    #[test]
    fn babai_matches_rational_reference() {
        let mut rng = Lcg(23);
        for trial in 0..80 {
            let r = 1 + trial % 5;
            let dim = r + 1 + trial % 3;
            let mut basis: Vec<Row> = loop {
                let m = random_matrix(&mut rng, r, dim, 9);
                if row_hnf(&m).rank() == r {
                    break m.into_rows();
                }
            };
            if trial % 2 == 0 {
                lll_reduce(&mut basis, LllDelta::DEFAULT);
            }
            let gso = integral_gso(&basis);
            let target: Row = (0..dim).map(|_| BigInt::from(rng.range(-100_000, 100_000))).collect();
            let got = babai_nearest_plane(&basis, &gso, &target);
            assert_eq!(got, babai_rational(&basis, &target));
            // Coset invariance: shifting the target by a lattice vector changes nothing.
            let mut shifted = target.clone();
            for b in &basis {
                let c = BigInt::from(rng.range(-50, 50));
                for (x, y) in shifted.iter_mut().zip(b) {
                    *x += &c * y;
                }
            }
            assert_eq!(babai_nearest_plane(&basis, &gso, &shifted), got);
        }
    }

    #[test]
    fn babai_ties_round_up() {
        // One basis vector (2); target 1 has coefficient exactly 1/2 and must round up.
        let basis = vec![ints(&[2])];
        let gso = integral_gso(&basis);
        assert_eq!(babai_nearest_plane(&basis, &gso, &ints(&[1])), ints(&[-1]));
        assert_eq!(babai_nearest_plane(&basis, &gso, &ints(&[-1])), ints(&[-1]));
        assert_eq!(babai_nearest_plane(&basis, &gso, &ints(&[3])), ints(&[-1]));
    }

    #[test]
    fn recon_toy_matrix() {
        // The self-check matrix of the SageMath pipeline: rows e1, e2, e3, e1+e2, e1+e3, e2+e3.
        let g = mat(&[&[1, 0, 0], &[0, 1, 0], &[0, 0, 1], &[1, 1, 0], &[1, 0, 1], &[0, 1, 1]]);
        let all = recon_vector(&g, ReconParams::DEFAULT).expect("authorized");
        assert_eq!(g.left_mul_vec(&all), ints(&[1, 0, 0]));
        // {r4, r5, r6}: 2 e1 = r4 + r5 - r6 only, so not authorized.
        assert!(recon_vector(&g.select_rows(&[3, 4, 5]), ReconParams::DEFAULT).is_none());
        assert!(!is_authorized(&g.select_rows(&[3, 4, 5])));
        let small = recon_vector(&g.select_rows(&[0, 1, 2]), ReconParams::DEFAULT).unwrap();
        assert_eq!(small, ints(&[1, 0, 0]));
        assert!(is_authorized(&g.select_rows(&[1, 3])));
        assert_eq!(recon_vector(&g.select_rows(&[1, 3]), ReconParams::DEFAULT).unwrap(), ints(&[-1, 1]));
    }

    #[test]
    fn greedy_never_increases_and_stays_in_coset() {
        let mut rng = Lcg(5);
        let mut authorized = 0;
        for _ in 0..50 {
            let g = random_matrix(&mut rng, 7, 3, 2);
            let Some(detail) = recon_vector_detailed(&g, ReconParams::DEFAULT) else { continue };
            authorized += 1;
            assert!(linf_norm(&detail.lambda) <= linf_norm(&detail.after_babai));
            assert_eq!(g.left_mul_vec(&detail.after_babai), e1(3));
            assert_eq!(g.left_mul_vec(&detail.lambda), e1(3));
            // Deterministic.
            assert_eq!(recon_vector(&g, ReconParams::DEFAULT).unwrap(), detail.lambda);
        }
        assert!(authorized >= 10, "only {authorized} of 50 random matrices were authorized");
    }
}
