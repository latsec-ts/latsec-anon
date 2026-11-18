//! Revalidation of the published SEC matrices:
//! shape, rank, expansion, [46, Algorithm 3] peeling, planted supports and witnesses,
//! and the float64 matrix constants used by the parameter selection.

mod common;

use std::time::Instant;

use common::repo_path;
use latsec::expander::{
    certify_expansion, certify_peeling, privacy_threshold, privacy_threshold_notebook, spectral_statistics, Graph,
    Ratio,
};
use latsec::fixture::AuthorMatrix;
use latsec::intmat::{is_authorized, recon_vector, row_hnf, ReconParams};
use latsec::params::binomial;
use latsec::published;
use num_bigint::{BigInt, BigUint};
use num_traits::{One, Zero};

fn load(tag: &str) -> AuthorMatrix {
    AuthorMatrix::load(repo_path(&format!("fixtures/author/G-{tag}.json"))).unwrap()
}

fn subsets_up_to(n: usize, max: usize) -> BigUint {
    (1..=max).map(|s| binomial(n as u64, s as u64)).sum()
}

fn check_structure(tag: &str) {
    let m = load(tag);
    let g = m.matrix();
    assert_eq!((g.nrows(), g.ncols()), (m.n, m.k));
    assert!(m.g.iter().flatten().all(|x| (-1..=1).contains(x)));
    assert!(m.g.iter().all(|row| row[1..].iter().filter(|&&x| x != 0).count() == m.c_exp));
    assert_eq!(row_hnf(&g).rank(), m.k, "G has full column rank");

    let (gamma, eps) = (Ratio::parse(&m.gamma).unwrap(), Ratio::parse(&m.eps_exp).unwrap());
    assert_eq!(privacy_threshold(m.n, gamma, eps), m.k_p as u64);
    assert_eq!(privacy_threshold_notebook(m.n, gamma, eps), m.k_p as u64);

    let graph = Graph::from_matrix(&m.g).unwrap();
    let started = Instant::now();
    let expansion = certify_expansion(&graph, m.c_exp, gamma, eps);
    assert_eq!(expansion.violation, None, "{tag}: expansion fails");
    assert_eq!(BigUint::from(expansion.subsets_checked), subsets_up_to(m.n, expansion.max_size));
    let expansion_time = started.elapsed();

    // [46, Algorithm 3] succeeds on every set of at most k_p parties: a constructive
    // certificate of the unimodular-minor condition.
    let started = Instant::now();
    let peeling = certify_peeling(&graph, m.k_p);
    assert_eq!(peeling.failure, None, "{tag}: peeling fails");
    assert_eq!(BigUint::from(peeling.subsets_checked), subsets_up_to(m.n, m.k_p));
    println!(
        "{tag}: expansion over {} subsets in {:.2} s, peeling over {} subsets in {:.2} s",
        expansion.subsets_checked,
        expansion_time.as_secs_f64(),
        peeling.subsets_checked,
        started.elapsed().as_secs_f64()
    );

    // Planted supports are disjoint and their witnesses reconstruct.
    let mut seen = vec![false; m.n];
    let mut e1 = vec![BigInt::zero(); m.k];
    e1[0] = BigInt::one();
    for (support, witness) in m.planted_supports.iter().zip(&m.planted_witnesses) {
        assert_eq!(support.len(), witness.len());
        assert!(support.windows(2).all(|w| w[0] < w[1]) && support.iter().all(|&i| i < m.n));
        for &i in support {
            assert!(!std::mem::replace(&mut seen[i], true), "supports overlap at party {i}");
        }
        let lambda: Vec<BigInt> = witness.iter().map(|&x| BigInt::from(x)).collect();
        assert_eq!(g.select_rows(support).left_mul_vec(&lambda), e1, "witness does not reconstruct");
        // Dropping the zero coefficients leaves an authorized set of that weight.
        let small: Vec<usize> = support.iter().zip(witness).filter(|(_, &w)| w != 0).map(|(&i, _)| i).collect();
        assert!(is_authorized(&g.select_rows(&small)));
        let ours = recon_vector(&g.select_rows(&small), ReconParams::DEFAULT).unwrap();
        assert_eq!(g.select_rows(&small).left_mul_vec(&ours), e1);
    }
    // Parties outside every planted support get a fixed value in {-1, 0, 1}
    // ([46, Algorithm 2]).  It is zero except for two parties of the n = 60 instance.
    let free_nonzero: Vec<usize> = (0..m.n).filter(|&i| !seen[i] && m.g[i][0] != 0).collect();
    assert_eq!(free_nonzero, if m.n == 60 { vec![24, 59] } else { vec![] });
}

#[test]
fn structure_10_5_3() {
    check_structure("10-5-3");
}

#[test]
fn structure_20_9_3() {
    check_structure("20-9-3");
}

#[test]
fn structure_40_19_3() {
    check_structure("40-19-3");
    let m = load("40-19-3");
    // Smallest authorized sets implied by the witnesses: six parties.
    let weights: Vec<usize> = m.planted_witnesses.iter().map(|w| w.iter().filter(|&&x| x != 0).count()).collect();
    assert_eq!(weights, vec![6, 6, 9]);
    assert_eq!(m.planted_supports.iter().map(|s| s.len()).collect::<Vec<_>>(), vec![9, 6, 12]);
}

#[test]
fn structure_60_39_3() {
    check_structure("60-39-3");
    let m = load("60-39-3");
    let weights: Vec<usize> = m.planted_witnesses.iter().map(|w| w.iter().filter(|&&x| x != 0).count()).collect();
    assert_eq!(weights, vec![10, 32]);
    assert_eq!(m.planted_supports.iter().map(|s| s.len()).collect::<Vec<_>>(), vec![21, 35]);
    // The notebook reports that gamma = 7/60 fails certification for this graph.
    let graph = Graph::from_matrix(&m.g).unwrap();
    let wider = certify_expansion(&graph, 3, Ratio::parse("7/60").unwrap(), Ratio::parse(&m.eps_exp).unwrap());
    assert_eq!(wider.max_size, 7);
    let violation = wider.violation.expect("gamma = 7/60 should fail");
    assert_eq!(violation.len(), 7);
}

fn check_constants(n: usize, tolerance: f64) {
    let geo = published::geometry(n);
    let m = load(geo.fixture.trim_start_matches("G-").trim_end_matches(".json"));
    let threads = std::thread::available_parallelism().map_or(1, |t| t.get());
    let started = Instant::now();
    let stats = spectral_statistics(&m.g, m.k_p, threads).unwrap();
    println!(
        "n = {n}: {} subsets in {:.1} s on {threads} threads: {stats:?}",
        stats.subsets,
        started.elapsed().as_secs_f64()
    );
    let c = &geo.constants;
    for (name, ours, theirs) in [
        ("alpha_kp", stats.alpha_kp, c.alpha_kp),
        ("beta_kp", stats.beta_kp, c.beta_kp),
        ("gamma_kp", stats.gamma_kp, c.gamma_kp),
        ("norm_g_hat", stats.norm_g_hat, c.norm_g_hat),
        ("norm_g", stats.norm_g, c.norm_g),
        ("beta_q", stats.beta_q, c.beta_q),
    ] {
        assert!((ours / theirs - 1.0).abs() < tolerance, "n = {n}: {name} = {ours}, published {theirs}");
    }
    // delta_kp is published to three decimals (Table 3); six are in the notebooks.
    let delta = if n == 40 { 3.645751 } else { 3.788450 };
    assert!((stats.delta_kp - delta).abs() < 1e-6, "delta_kp = {}", stats.delta_kp);
}

#[test]
fn matrix_constants_40() {
    check_constants(40, 1e-11);
}

/// About 56 million subsets; run with `cargo test --release -- --ignored`.
#[test]
#[ignore]
fn matrix_constants_60() {
    check_constants(60, 1e-11);
}
