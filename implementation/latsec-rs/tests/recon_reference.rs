//! Reconstruction against reference vectors from the SageMath pipeline of [46]
//! (`tools/gen_recon_fixtures.py`).

use std::time::Instant;

use latsec::fixture::AuthorMatrix;
use latsec::intmat::{
    babai_nearest_plane, greedy_reduce_linf, hamming_weight, integral_gso, is_authorized, left_kernel_hnf, linf_norm,
    recon_vector_detailed, row_hnf, IntMatrix, ReconParams, Row,
};
use num_bigint::BigInt;
use num_traits::One;
use serde_json::Value;

fn fixture_path(rel: &str) -> String {
    format!("{}/../fixtures/{}", env!("CARGO_MANIFEST_DIR"), rel)
}

fn rows_of(value: &Value) -> Vec<Row> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|x| match x {
                    Value::String(s) => s.parse::<BigInt>().unwrap(),
                    other => BigInt::from(other.as_i64().unwrap()),
                })
                .collect()
        })
        .collect()
}

fn ints_of(value: &Value) -> Row {
    value.as_array().unwrap().iter().map(|x| BigInt::from(x.as_i64().unwrap())).collect()
}

fn check_matrix(tag: &str) {
    let author = AuthorMatrix::load(fixture_path(&format!("author/G-{tag}.json"))).unwrap();
    let g = author.matrix();
    let text = std::fs::read_to_string(fixture_path(&format!("sage/recon-{tag}.json"))).unwrap();
    let reference: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(reference["format"], "latsec-recon-reference-v1");
    assert_eq!(reference["matrix_source_sha256"].as_str().unwrap(), author.source.cell_sources_sha256);
    let params = ReconParams::DEFAULT;
    assert_eq!(reference["passes"].as_u64().unwrap(), u64::from(params.passes));
    assert_eq!(reference["window"].as_u64().unwrap(), u64::from(params.window));

    let mut e1 = vec![BigInt::from(0); author.k];
    e1[0] = BigInt::one();
    let (mut authorized, mut same_lambda, mut same_norms) = (0usize, 0usize, 0usize);
    let started = Instant::now();
    let cases = reference["cases"].as_array().unwrap();
    for case in cases {
        let set: Vec<usize> = case["set"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
        let g_t = g.select_rows(&set);
        let expected_auth = case["authorized"].as_bool().unwrap();
        assert_eq!(is_authorized(&g_t), expected_auth, "authorization differs for {set:?}");

        // The canonical kernel basis is a function of the lattice alone.
        let kernel = left_kernel_hnf(&g_t);
        assert_eq!(kernel.rows(), rows_of(&case["kernel_hnf"]).as_slice(), "kernel HNF differs for {set:?}");

        let detail = recon_vector_detailed(&g_t, params);
        assert_eq!(detail.is_some(), expected_auth);
        let Some(detail) = detail else { continue };
        authorized += 1;
        assert_eq!(g_t.left_mul_vec(&detail.lambda), e1);

        // Same reduced basis and same coset: Babai and the greedy stage must agree
        // with Sage's exact-rational implementation bit for bit, although our HNF
        // witness differs from Sage's.
        let sage_basis = rows_of(&case["kernel_lll"]);
        let gso = integral_gso(&sage_basis);
        let witness = row_hnf(&g_t).transform.row(0).clone();
        let babai = babai_nearest_plane(&sage_basis, &gso, &witness);
        assert_eq!(babai, ints_of(&case["lambda_babai"]), "Babai differs for {set:?}");
        let lambda = greedy_reduce_linf(&babai, &sage_basis, params.passes, params.window);
        let sage_lambda = ints_of(&case["lambda"]);
        assert_eq!(lambda, sage_lambda, "greedy stage differs for {set:?}");

        // With our own LLL the vector may differ; it is still a valid reconstruction.
        if detail.lambda == sage_lambda {
            same_lambda += 1;
        } else {
            println!(
                "{tag}: set of {} parties {set:?}: this crate (linf {}, weight {}), SageMath (linf {}, weight {})",
                set.len(),
                linf_norm(&detail.lambda),
                hamming_weight(&detail.lambda),
                linf_norm(&sage_lambda),
                hamming_weight(&sage_lambda)
            );
        }
        if linf_norm(&detail.lambda) == linf_norm(&sage_lambda)
            && hamming_weight(&detail.lambda) == hamming_weight(&sage_lambda)
        {
            same_norms += 1;
        }
        // Our reduced basis spans the same lattice as Sage's.
        let ours = latsec::intmat::hnf_basis(detail.kernel_reduced.rows(), set.len());
        assert_eq!(ours, IntMatrix::from_rows(set.len(), rows_of(&case["kernel_hnf"])));
    }
    println!(
        "{tag}: {} sets, {authorized} authorized, identical lambda in {same_lambda}, \
         same (linf, weight) in {same_norms}, {:.1} ms per set",
        cases.len(),
        started.elapsed().as_secs_f64() * 1e3 / cases.len() as f64
    );
}

/// Sets whose reconstruction vector exceeds the maxima observed in the samples
/// of [46].  SageMath's pipeline and this crate agree on every one of them.
fn check_notable(tag: &str, observed: (u64, usize), enforced: (u64, usize), expected: &[(usize, u64, usize)]) {
    let author = AuthorMatrix::load(fixture_path(&format!("author/G-{tag}.json"))).unwrap();
    let g = author.matrix();
    let text = std::fs::read_to_string(fixture_path(&format!("sage/recon-{tag}.json"))).unwrap();
    let reference: Value = serde_json::from_str(&text).unwrap();
    let notable = reference["notable"].as_array().unwrap();
    assert_eq!(notable.len(), expected.len());
    let mut rejected = 0;
    for (case, &(size, linf, weight)) in notable.iter().zip(expected) {
        let set: Vec<usize> = case["set"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
        assert_eq!(set.len(), size);
        let sage = ints_of(&case["lambda"]);
        let ours = recon_vector_detailed(&g.select_rows(&set), ReconParams::DEFAULT).unwrap().lambda;
        assert_eq!(ours, sage, "{tag}: this crate and SageMath disagree on {set:?}");
        let got_linf: u64 = linf_norm(&ours).try_into().unwrap();
        let got_weight = hamming_weight(&ours);
        assert_eq!((got_linf, got_weight), (linf, weight));
        assert!(got_linf > observed.0 || got_weight > observed.1, "exceeds an observed maximum");
        if got_linf > enforced.0 || got_weight > enforced.1 {
            rejected += 1;
        }
    }
    // n = 40: one authorized 27-set has weight 26 > omega_lambda = 25 and is
    // rejected by the paper's predicate.  n = 60: none is rejected, but one set
    // attains beta_lambda = 9 exactly.
    assert_eq!(rejected, if tag == "40-19-3" { 1 } else { 0 });
}

#[test]
fn notable_sets_40_19_3() {
    check_notable("40-19-3", (4, 24), (5, 25), &[(26, 4, 25), (26, 5, 21), (27, 1, 26), (27, 2, 25), (27, 2, 25)]);
}

#[test]
fn notable_sets_60_39_3() {
    check_notable("60-39-3", (8, 42), (9, 45), &[(46, 5, 44), (46, 9, 37)]);
}

#[test]
fn reference_10_5_3() {
    check_matrix("10-5-3");
}

#[test]
fn reference_20_9_3() {
    check_matrix("20-9-3");
}

#[test]
fn reference_40_19_3() {
    check_matrix("40-19-3");
}

#[test]
fn reference_60_39_3() {
    check_matrix("60-39-3");
}
