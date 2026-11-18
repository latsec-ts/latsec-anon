//! Reconstruction-vector statistics over signer sets of a given size: how many
//! sets are authorized, the distributions of `|lambda|_inf`, `wt(lambda)` and
//! `|lambda|_1`, and how many authorized sets a predicate rejects.
//!
//! This is offline analysis.  The numbers describe this crate's reconstruction
//! algorithm; they are not interchangeable with the SageMath statistics of [46].

use std::collections::BTreeMap;
use std::time::Instant;

use num_traits::ToPrimitive;

use crate::intmat::{recon_vector, IntMatrix, ReconParams};
use crate::params::binomial;
use crate::profile::Admissibility;
use crate::sampling::{uniform_below, RandomSource, SeededRandom};

/// Statistics for one set size.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SizeStatistics {
    pub size: usize,
    /// Whether every set of this size was tested.
    pub exhaustive: bool,
    pub tested: u64,
    pub authorized: u64,
    /// Authorized sets whose vector fails the predicate, if one was given.
    pub rejected_by_bounds: u64,
    pub linf: BTreeMap<u64, u64>,
    pub weight: BTreeMap<u64, u64>,
    pub l1: BTreeMap<u64, u64>,
    /// Mean wall-clock time of one reconstruction, in microseconds.
    pub micros_per_set: f64,
    /// Up to [`SizeStatistics::MAX_RECORDED`] authorized sets rejected by the
    /// predicate, with their reconstruction vectors.
    pub rejected_sets: Vec<(Vec<usize>, Vec<i64>)>,
}

impl SizeStatistics {
    /// How many rejected sets are kept per size.
    pub const MAX_RECORDED: usize = 16;

    fn merge(&mut self, other: &SizeStatistics) {
        for item in &other.rejected_sets {
            if self.rejected_sets.len() < Self::MAX_RECORDED {
                self.rejected_sets.push(item.clone());
            }
        }
        self.tested += other.tested;
        self.authorized += other.authorized;
        self.rejected_by_bounds += other.rejected_by_bounds;
        for (dst, src) in [(&mut self.linf, &other.linf), (&mut self.weight, &other.weight), (&mut self.l1, &other.l1)]
        {
            for (&key, &count) in src {
                *dst.entry(key).or_insert(0) += count;
            }
        }
        self.micros_per_set += other.micros_per_set;
    }

    pub fn unauthorized(&self) -> u64 {
        self.tested - self.authorized
    }

    pub fn max_of(hist: &BTreeMap<u64, u64>) -> u64 {
        hist.keys().next_back().copied().unwrap_or(0)
    }

    pub fn mean_of(hist: &BTreeMap<u64, u64>) -> f64 {
        let total: u64 = hist.values().sum();
        if total == 0 {
            return 0.0;
        }
        hist.iter().map(|(&k, &c)| k as f64 * c as f64).sum::<f64>() / total as f64
    }

    /// The fraction of authorized sets with `value <= bound`.
    pub fn fraction_at_most(hist: &BTreeMap<u64, u64>, bound: u64) -> f64 {
        let total: u64 = hist.values().sum();
        if total == 0 {
            return 0.0;
        }
        hist.range(..=bound).map(|(_, &c)| c).sum::<u64>() as f64 / total as f64
    }
}

/// Whether a vector passes the coefficient bounds of a predicate.
pub fn passes_bounds(lambda: &[i64], predicate: &Admissibility) -> bool {
    let linf = lambda.iter().map(|x| x.unsigned_abs()).max().unwrap_or(0);
    let weight = lambda.iter().filter(|&&x| x != 0).count() as u64;
    let l1: u64 = lambda.iter().map(|x| x.unsigned_abs()).sum();
    match *predicate {
        Admissibility::Paper { beta_lambda, omega_lambda } => {
            linf <= u64::from(beta_lambda) && weight <= u64::from(omega_lambda)
        }
        Admissibility::L1 { l1_cap } => l1 <= u64::from(l1_cap),
    }
}

/// The next `size`-subset of `{0, .., n-1}` in lexicographic order, in place.
fn next_combination(set: &mut [usize], n: usize) -> bool {
    let size = set.len();
    let mut i = size;
    while i > 0 {
        i -= 1;
        if set[i] != i + n - size {
            set[i] += 1;
            for j in i + 1..size {
                set[j] = set[j - 1] + 1;
            }
            return true;
        }
    }
    false
}

/// A uniform `size`-subset of `{0, .., n-1}`, sorted.
pub fn random_subset(rng: &mut dyn RandomSource, n: usize, size: usize) -> Vec<usize> {
    let mut pool: Vec<usize> = (0..n).collect();
    for i in 0..size {
        let j = i + uniform_below(rng, (n - i) as u128) as usize;
        pool.swap(i, j);
    }
    pool.truncate(size);
    pool.sort_unstable();
    pool
}

fn tally(g: &IntMatrix, params: ReconParams, predicate: Option<&Admissibility>, sets: &[Vec<usize>]) -> SizeStatistics {
    let mut stats = SizeStatistics::default();
    let started = Instant::now();
    for set in sets {
        stats.tested += 1;
        let Some(lambda) = recon_vector(&g.select_rows(set), params) else { continue };
        stats.authorized += 1;
        let small: Vec<i64> = lambda.iter().map(|x| x.to_i64().unwrap_or(i64::MAX)).collect();
        let linf = small.iter().map(|x| x.unsigned_abs()).max().unwrap_or(0);
        let weight = small.iter().filter(|&&x| x != 0).count() as u64;
        let l1 = small.iter().fold(0u64, |acc, x| acc.saturating_add(x.unsigned_abs()));
        *stats.linf.entry(linf).or_insert(0) += 1;
        *stats.weight.entry(weight).or_insert(0) += 1;
        *stats.l1.entry(l1).or_insert(0) += 1;
        if predicate.is_some_and(|p| !passes_bounds(&small, p)) {
            stats.rejected_by_bounds += 1;
            if stats.rejected_sets.len() < SizeStatistics::MAX_RECORDED {
                stats.rejected_sets.push((set.clone(), small));
            }
        }
    }
    stats.micros_per_set = started.elapsed().as_secs_f64() * 1e6;
    stats
}

/// Statistics for sets of `size` rows of `g`.  If the number of such sets is at
/// most `samples`, all of them are tested; otherwise `samples` distinct sets are
/// drawn from a stream determined by `seed` and `size`.  Work is split over
/// `threads` threads; the result does not depend on their number.
pub fn size_statistics(
    g: &IntMatrix,
    params: ReconParams,
    predicate: Option<&Admissibility>,
    size: usize,
    samples: u64,
    seed: &[u8],
    threads: usize,
) -> SizeStatistics {
    let n = g.nrows();
    assert!(size >= 1 && size <= n);
    let total = binomial(n as u64, size as u64);
    let exhaustive = total <= num_bigint::BigUint::from(samples);
    let sets: Vec<Vec<usize>> = if exhaustive {
        let mut out = Vec::with_capacity(total.to_usize().unwrap_or(0));
        let mut current: Vec<usize> = (0..size).collect();
        loop {
            out.push(current.clone());
            if !next_combination(&mut current, n) {
                break;
            }
        }
        out
    } else {
        let mut rng = SeededRandom::new(seed, &format!("recon-statistics/size-{size}"));
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::with_capacity(samples as usize);
        while (out.len() as u64) < samples {
            let set = random_subset(&mut rng, n, size);
            if seen.insert(set.clone()) {
                out.push(set);
            }
        }
        out
    };
    let threads = threads.max(1).min(sets.len().max(1));
    let chunk = sets.len().div_ceil(threads).max(1);
    let parts: Vec<SizeStatistics> = std::thread::scope(|scope| {
        let handles: Vec<_> =
            sets.chunks(chunk).map(|part| scope.spawn(move || tally(g, params, predicate, part))).collect();
        handles.into_iter().map(|h| h.join().expect("statistics worker panicked")).collect()
    });
    let mut stats = SizeStatistics { size, exhaustive, ..SizeStatistics::default() };
    for part in &parts {
        stats.merge(part);
    }
    stats.rejected_sets.sort();
    stats.micros_per_set /= stats.tested.max(1) as f64;
    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy() -> IntMatrix {
        IntMatrix::from_i64(
            3,
            &[vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1], vec![1, 1, 0], vec![1, 0, 1], vec![0, 1, 1]],
        )
    }

    #[test]
    fn combinations_enumerate_all_subsets() {
        let mut set = vec![0, 1, 2];
        let mut count = 1;
        while next_combination(&mut set, 6) {
            count += 1;
        }
        assert_eq!(count, 20);
        assert_eq!(set, vec![3, 4, 5]);
    }

    #[test]
    fn exhaustive_and_thread_independent() {
        let g = toy();
        let one = size_statistics(&g, ReconParams::DEFAULT, None, 3, 1000, b"s", 1);
        let four = size_statistics(&g, ReconParams::DEFAULT, None, 3, 1000, b"s", 4);
        assert!(one.exhaustive);
        assert_eq!(one.tested, 20);
        assert_eq!((one.authorized, &one.linf, &one.weight), (four.authorized, &four.linf, &four.weight));
        // Sets of three rows that contain e1 in their integer span.
        assert!(one.authorized > 0 && one.authorized < 20);
        assert_eq!(one.linf.values().sum::<u64>(), one.authorized);
    }

    #[test]
    fn sampling_is_reproducible_and_distinct() {
        let g = toy();
        let a = size_statistics(&g, ReconParams::DEFAULT, None, 3, 10, b"seed", 2);
        let b = size_statistics(&g, ReconParams::DEFAULT, None, 3, 10, b"seed", 3);
        assert!(!a.exhaustive);
        assert_eq!(a.tested, 10);
        assert_eq!((a.authorized, a.linf.clone()), (b.authorized, b.linf.clone()));
        let predicate = Admissibility::L1 { l1_cap: 1 };
        let strict = size_statistics(&g, ReconParams::DEFAULT, Some(&predicate), 4, 1000, b"seed", 2);
        assert!(strict.rejected_by_bounds > 0);
        assert!(passes_bounds(&[1, 0, -1], &Admissibility::Paper { beta_lambda: 1, omega_lambda: 2 }));
        assert!(!passes_bounds(&[1, 1, -1], &Admissibility::Paper { beta_lambda: 1, omega_lambda: 2 }));
    }
}
