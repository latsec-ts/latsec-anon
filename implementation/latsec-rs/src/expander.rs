//! Structural checks on an SEC matrix: expansion of the underlying bipartite graph,
//! [46, Algorithm 3] peeling, and float64 spectral statistics of its submatrices.
//!
//! The expansion and peeling checks are exact and exhaustive.  The spectral
//! statistics reproduce the float64 values of the SEC-geometry notebooks; they are **not** certified
//! bounds (no interval arithmetic, no error enclosure).
//!
//! The graph has one left vertex per party (row of `G_hat`) and one right vertex per
//! column of `G_hat`; row `i` is adjacent to the columns where it is nonzero.  At
//! most 64 parties and 64 columns are supported, which covers `n <= 60`.

use crate::error::{Error, Result};

/// An exact nonnegative rational, parsed from `"5/40"` or `"0.48"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ratio {
    pub num: u64,
    pub den: u64,
}

impl Ratio {
    pub fn parse(text: &str) -> Result<Ratio> {
        let bad = || Error::invalid(format!("cannot parse {text:?} as a rational"));
        let ratio = if let Some((a, b)) = text.split_once('/') {
            Ratio { num: a.trim().parse().map_err(|_| bad())?, den: b.trim().parse().map_err(|_| bad())? }
        } else if let Some((whole, frac)) = text.split_once('.') {
            if frac.is_empty() || frac.len() > 15 || !frac.bytes().all(|b| b.is_ascii_digit()) {
                return Err(bad());
            }
            let den = 10u64.pow(frac.len() as u32);
            let whole: u64 = if whole.is_empty() { 0 } else { whole.parse().map_err(|_| bad())? };
            Ratio { num: whole * den + frac.parse::<u64>().map_err(|_| bad())?, den }
        } else {
            Ratio { num: text.trim().parse().map_err(|_| bad())?, den: 1 }
        };
        if ratio.den == 0 {
            return Err(bad());
        }
        Ok(ratio)
    }

    /// `floor(self * x)`.
    pub fn floor_mul(&self, x: u64) -> u64 {
        self.num * x / self.den
    }
}

/// `ceil((1 - eps) * x)` for `0 <= eps <= 1`.
fn ceil_one_minus(eps: Ratio, x: u64) -> u64 {
    ((eps.den - eps.num) * x).div_ceil(eps.den)
}

/// The bipartite graph of `G_hat`, as one column bitmask per party.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Graph {
    pub neighbors: Vec<u64>,
    pub right_vertices: usize,
}

impl Graph {
    /// Builds the graph from `G = [g | G_hat]`, ignoring the first column.
    pub fn from_matrix(g: &[Vec<i64>]) -> Result<Graph> {
        let n = g.len();
        let m = g.first().map_or(0, |r| r.len()).saturating_sub(1);
        if n == 0 || n > 64 || m == 0 || m > 64 || g.iter().any(|r| r.len() != m + 1) {
            return Err(Error::invalid("expander checks support 1..=64 parties and 1..=64 columns"));
        }
        let neighbors = g
            .iter()
            .map(|row| {
                row[1..].iter().enumerate().fold(0u64, |mask, (j, &x)| if x != 0 { mask | 1 << j } else { mask })
            })
            .collect();
        Ok(Graph { neighbors, right_vertices: m })
    }

    pub fn n(&self) -> usize {
        self.neighbors.len()
    }
}

/// The paper's privacy threshold `k_p = ceil(2 (1 - eps) floor(gamma n)) - 1`.
pub fn privacy_threshold(n: usize, gamma: Ratio, eps: Ratio) -> u64 {
    ceil_one_minus(eps, 2 * gamma.floor_mul(n as u64)).saturating_sub(1)
}

/// The notebooks' variant `max(floor(gamma n), floor(2 gamma (1 - eps) n) - 1)`.
pub fn privacy_threshold_notebook(n: usize, gamma: Ratio, eps: Ratio) -> u64 {
    let second = (2 * gamma.num * (eps.den - eps.num) * n as u64) / (gamma.den * eps.den);
    gamma.floor_mul(n as u64).max(second.saturating_sub(1))
}

/// The outcome of [`certify_expansion`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpansionReport {
    /// `floor(gamma n)`, the largest set size checked.
    pub max_size: usize,
    pub subsets_checked: u64,
    /// A set `S` with `|N(S)| < ceil((1 - eps) c |S|)`, if one exists.
    pub violation: Option<Vec<usize>>,
}

/// Visits every nonempty subset of `{0, .., n-1}` of size at most `max_size`,
/// in depth-first lexicographic order, with the union of its neighbor masks.
/// The visitor returns `false` to stop the search.
fn for_each_subset(graph: &Graph, max_size: usize, visit: &mut dyn FnMut(&[usize], u64) -> bool) -> bool {
    fn recurse(
        graph: &Graph,
        max_size: usize,
        start: usize,
        set: &mut Vec<usize>,
        union: u64,
        visit: &mut dyn FnMut(&[usize], u64) -> bool,
    ) -> bool {
        for next in start..graph.n() {
            set.push(next);
            let merged = union | graph.neighbors[next];
            if !visit(set, merged) {
                return false;
            }
            if set.len() < max_size && !recurse(graph, max_size, next + 1, set, merged, visit) {
                return false;
            }
            set.pop();
        }
        true
    }
    recurse(graph, max_size, 0, &mut Vec::with_capacity(max_size), 0, visit)
}

/// Checks `|N(S)| >= ceil((1 - eps) c |S|)` for every `S` with `1 <= |S| <= floor(gamma n)`.
pub fn certify_expansion(graph: &Graph, c_exp: usize, gamma: Ratio, eps: Ratio) -> ExpansionReport {
    let max_size = gamma.floor_mul(graph.n() as u64) as usize;
    let thresholds: Vec<u32> = (0..=max_size).map(|s| ceil_one_minus(eps, (c_exp * s) as u64) as u32).collect();
    let mut checked = 0u64;
    let mut violation = None;
    if max_size > 0 {
        for_each_subset(graph, max_size, &mut |set, union| {
            checked += 1;
            if union.count_ones() < thresholds[set.len()] {
                violation = Some(set.to_vec());
                return false;
            }
            true
        });
    }
    ExpansionReport { max_size, subsets_checked: checked, violation }
}

/// [46, Algorithm 3] on the rows `set`: repeatedly takes the smallest column that meets
/// exactly one remaining row and removes that row.  Returns the columns in the
/// order they were chosen, or `None` if peeling gets stuck.
pub fn peel(graph: &Graph, set: &[usize]) -> Option<Vec<usize>> {
    let mut rows: Vec<u64> = set.iter().map(|&i| graph.neighbors[i]).collect();
    let mut columns = Vec::with_capacity(rows.len());
    while !rows.is_empty() {
        let (mut once, mut more) = (0u64, 0u64);
        for &mask in &rows {
            more |= once & mask;
            once |= mask;
        }
        let unique = once & !more;
        if unique == 0 {
            return None;
        }
        let column = unique.trailing_zeros() as usize;
        let position = rows.iter().position(|&mask| mask >> column & 1 == 1).expect("a row has the column");
        rows.remove(position);
        columns.push(column);
    }
    Some(columns)
}

/// The outcome of [`certify_peeling`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PeelingReport {
    pub max_size: usize,
    pub subsets_checked: u64,
    /// A set on which [46, Algorithm 3] gets stuck, if one exists.
    pub failure: Option<Vec<usize>>,
}

/// Runs [46, Algorithm 3] on every nonempty set of at most `max_size` parties.  Success
/// on all of them is a constructive certificate that every such `G_hat_U` has a
/// unimodular `|U| x |U|` minor.
pub fn certify_peeling(graph: &Graph, max_size: usize) -> PeelingReport {
    let mut checked = 0u64;
    let mut failure = None;
    if max_size > 0 {
        for_each_subset(graph, max_size, &mut |set, _| {
            checked += 1;
            if peel(graph, set).is_none() {
                failure = Some(set.to_vec());
                return false;
            }
            true
        });
    }
    PeelingReport { max_size, subsets_checked: checked, failure }
}

/// Eigenvalues of a symmetric matrix by cyclic Jacobi rotations, ascending.
/// `a` holds the matrix row by row and is destroyed.
pub fn symmetric_eigenvalues(a: &mut [f64], n: usize) -> Vec<f64> {
    assert_eq!(a.len(), n * n);
    for _sweep in 0..64 {
        let off: f64 = (0..n)
            .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[i * n + j].powi(2))
            .sum();
        let diag: f64 = (0..n).map(|i| a[i * n + i].powi(2)).sum();
        if off <= 1e-30 * diag.max(1e-300) {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
            }
        }
    }
    let mut eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    eigenvalues.sort_by(|x, y| x.partial_cmp(y).unwrap());
    eigenvalues
}

/// `(sigma_min, sigma_max)` of a matrix with at least as many columns as rows,
/// from the eigenvalues of its Gram matrix `M M^T`.
fn singular_extremes(rows: &[&[i64]]) -> (f64, f64) {
    let r = rows.len();
    let mut gram = vec![0f64; r * r];
    for i in 0..r {
        for j in 0..r {
            gram[i * r + j] = rows[i].iter().zip(rows[j]).map(|(x, y)| x * y).sum::<i64>() as f64;
        }
    }
    let eig = symmetric_eigenvalues(&mut gram, r);
    (eig[0].max(0.0).sqrt(), eig[r - 1].max(0.0).sqrt())
}

/// The largest singular value of a matrix, from `M^T M` or `M M^T`.
pub fn spectral_norm(rows: &[Vec<i64>]) -> f64 {
    let ncols = rows.first().map_or(0, |r| r.len());
    if rows.len() <= ncols {
        let refs: Vec<&[i64]> = rows.iter().map(|r| r.as_slice()).collect();
        return singular_extremes(&refs).1;
    }
    let columns: Vec<Vec<i64>> = (0..ncols).map(|j| rows.iter().map(|r| r[j]).collect()).collect();
    let refs: Vec<&[i64]> = columns.iter().map(|c| c.as_slice()).collect();
    singular_extremes(&refs).1
}

/// Float64 spectral statistics of an SEC matrix (Table 3 of the paper).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpectralStatistics {
    /// `min_{|V| = k_p} sigma_min(G_hat_V)`.
    pub alpha_kp: f64,
    /// `max_{|U| = k_p} |G_hat_U|_2`.
    pub beta_kp: f64,
    /// `max_{|U| = k_p} |G_U|_2`.
    pub delta_kp: f64,
    /// `min_{1 <= |U| <= k_p} sigma_min(M_U)` over the peeling minors.
    pub gamma_kp: f64,
    pub norm_g_hat: f64,
    pub norm_g: f64,
    /// `1 + sqrt(1 + delta_kp^2) / gamma_kp`.
    pub beta_q: f64,
    pub subsets: u64,
}

/// All `size`-subsets of `{0, .., n-1}` whose smallest element is `first`.
fn subsets_starting_at(n: usize, size: usize, first: usize, visit: &mut dyn FnMut(&[usize])) {
    fn recurse(n: usize, size: usize, start: usize, set: &mut Vec<usize>, visit: &mut dyn FnMut(&[usize])) {
        if set.len() == size {
            visit(set);
            return;
        }
        for next in start..n {
            set.push(next);
            recurse(n, size, next + 1, set, visit);
            set.pop();
        }
    }
    let mut set = vec![first];
    recurse(n, size, first + 1, &mut set, visit);
}

/// Evaluates the spectral statistics exhaustively over all sets of at most `k_p`
/// parties, on `threads` threads.  Requires peeling to succeed on every set.
pub fn spectral_statistics(g: &[Vec<i64>], k_p: usize, threads: usize) -> Result<SpectralStatistics> {
    let graph = Graph::from_matrix(g)?;
    let n = g.len();
    if k_p == 0 || k_p > n {
        return Err(Error::invalid("k_p out of range"));
    }
    let g_hat: Vec<Vec<i64>> = g.iter().map(|r| r[1..].to_vec()).collect();
    #[derive(Clone, Copy)]
    struct Partial {
        alpha: f64,
        beta: f64,
        delta: f64,
        gamma: f64,
        subsets: u64,
        stuck: bool,
    }
    let work = |firsts: Vec<usize>| -> Partial {
        let mut acc =
            Partial { alpha: f64::INFINITY, beta: 0.0, delta: 0.0, gamma: f64::INFINITY, subsets: 0, stuck: false };
        for first in firsts {
            for size in 1..=k_p {
                subsets_starting_at(n, size, first, &mut |set| {
                    acc.subsets += 1;
                    let Some(columns) = peel(&graph, set) else {
                        acc.stuck = true;
                        return;
                    };
                    let minor: Vec<Vec<i64>> =
                        set.iter().map(|&i| columns.iter().map(|&j| g_hat[i][j]).collect()).collect();
                    let refs: Vec<&[i64]> = minor.iter().map(|r| r.as_slice()).collect();
                    acc.gamma = acc.gamma.min(singular_extremes(&refs).0);
                    if size == k_p {
                        let hat: Vec<&[i64]> = set.iter().map(|&i| g_hat[i].as_slice()).collect();
                        let (lo, hi) = singular_extremes(&hat);
                        acc.alpha = acc.alpha.min(lo);
                        acc.beta = acc.beta.max(hi);
                        let full: Vec<&[i64]> = set.iter().map(|&i| g[i].as_slice()).collect();
                        acc.delta = acc.delta.max(singular_extremes(&full).1);
                    }
                });
            }
        }
        acc
    };
    let threads = threads.clamp(1, n);
    // Small first elements own many more subsets; deal them out round-robin.
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); threads];
    for first in 0..n {
        buckets[first % threads].push(first);
    }
    let parts: Vec<Partial> = std::thread::scope(|scope| {
        let handles: Vec<_> = buckets.into_iter().map(|b| scope.spawn(|| work(b))).collect();
        handles.into_iter().map(|h| h.join().expect("spectral worker panicked")).collect()
    });
    if parts.iter().any(|p| p.stuck) {
        return Err(Error::invalid("peeling failed on some set; the structural condition does not hold"));
    }
    let alpha_kp = parts.iter().map(|p| p.alpha).fold(f64::INFINITY, f64::min);
    let beta_kp = parts.iter().map(|p| p.beta).fold(0.0, f64::max);
    let delta_kp = parts.iter().map(|p| p.delta).fold(0.0, f64::max);
    let gamma_kp = parts.iter().map(|p| p.gamma).fold(f64::INFINITY, f64::min);
    Ok(SpectralStatistics {
        alpha_kp,
        beta_kp,
        delta_kp,
        gamma_kp,
        norm_g_hat: spectral_norm(&g_hat),
        norm_g: spectral_norm(g),
        beta_q: 1.0 + (1.0 + delta_kp * delta_kp).sqrt() / gamma_kp,
        subsets: parts.iter().map(|p| p.subsets).sum(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ratios() {
        assert_eq!(Ratio::parse("5/40").unwrap(), Ratio { num: 5, den: 40 });
        assert_eq!(Ratio::parse("0.48").unwrap(), Ratio { num: 48, den: 100 });
        assert_eq!(Ratio::parse("0.450").unwrap(), Ratio { num: 450, den: 1000 });
        assert_eq!(Ratio::parse("1/3").unwrap(), Ratio { num: 1, den: 3 });
        assert!(Ratio::parse("1/0").is_err());
        assert!(Ratio::parse("abc").is_err());
        assert_eq!(Ratio::parse("5/40").unwrap().floor_mul(40), 5);
        // ceil((1 - 0.48) * 3 * s): 2, 4, 5, 7, 8 for s = 1..5.
        let eps = Ratio::parse("0.48").unwrap();
        assert_eq!((1..=5).map(|s| ceil_one_minus(eps, 3 * s)).collect::<Vec<_>>(), vec![2, 4, 5, 7, 8]);
    }

    #[test]
    fn thresholds_of_the_published_instances() {
        let cases = [(40, "5/40", "0.48", 5), (60, "6/60", "0.45", 6), (10, "2/10", "1/3", 2), (20, "3/20", "0.49", 3)];
        for (n, gamma, eps, k_p) in cases {
            let (gamma, eps) = (Ratio::parse(gamma).unwrap(), Ratio::parse(eps).unwrap());
            assert_eq!(privacy_threshold(n, gamma, eps), k_p);
            assert_eq!(privacy_threshold_notebook(n, gamma, eps), k_p);
        }
    }

    #[test]
    fn peeling_and_expansion_on_toy_graphs() {
        // Three parties, each with a private column: peels in column order.
        let g = vec![vec![1, 1, 0, 0, 1], vec![0, 0, 1, 0, 1], vec![1, 0, 0, 1, 1]];
        let graph = Graph::from_matrix(&g).unwrap();
        assert_eq!(peel(&graph, &[0, 1, 2]), Some(vec![0, 1, 2]));
        assert!(certify_peeling(&graph, 3).failure.is_none());
        assert_eq!(certify_peeling(&graph, 3).subsets_checked, 7);
        // Two parties with identical neighborhoods cannot be peeled.
        let stuck = vec![vec![1, 1, 1], vec![0, 1, 1], vec![0, 1, 0]];
        let graph = Graph::from_matrix(&stuck).unwrap();
        assert_eq!(peel(&graph, &[0, 1]), None);
        assert_eq!(certify_peeling(&graph, 2).failure, Some(vec![0, 1]));
        // Expansion: sets {0, 1} have one neighbor... two columns, fewer than ceil(0.9 * 2 * 2) = 4.
        let report = certify_expansion(&graph, 2, Ratio::parse("2/3").unwrap(), Ratio::parse("0.1").unwrap());
        assert_eq!(report.max_size, 2);
        assert_eq!(report.violation, Some(vec![0, 1]));
    }

    #[test]
    fn jacobi_eigenvalues() {
        let mut a = vec![2.0, 1.0, 1.0, 2.0];
        let eig = symmetric_eigenvalues(&mut a, 2);
        assert!((eig[0] - 1.0).abs() < 1e-14 && (eig[1] - 3.0).abs() < 1e-14);
        // Tridiagonal (2, -1): eigenvalues 2 - 2 cos(k pi / (n + 1)).
        let n = 6;
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            a[i * n + i] = 2.0;
            if i + 1 < n {
                a[i * n + i + 1] = -1.0;
                a[(i + 1) * n + i] = -1.0;
            }
        }
        let eig = symmetric_eigenvalues(&mut a, n);
        for (k, value) in eig.iter().enumerate() {
            let expected = 2.0 - 2.0 * ((k + 1) as f64 * std::f64::consts::PI / (n as f64 + 1.0)).cos();
            assert!((value - expected).abs() < 1e-13, "eigenvalue {k}: {value} vs {expected}");
        }
        assert!((spectral_norm(&[vec![3, 0], vec![0, 4], vec![0, 0]]) - 4.0).abs() < 1e-14);
        assert!((spectral_norm(&[vec![1, 1], vec![1, 1]]) - 2.0).abs() < 1e-14);
    }
}
