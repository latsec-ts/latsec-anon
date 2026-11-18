//! The committed profiles and test vectors: validity, agreement with the builder,
//! agreement with the published tables, and reproducibility of the test vectors.

mod common;

use std::sync::Arc;

use common::{load_profile, repo_path, scheme};
use latsec::builder;
use latsec::kat::TestVector;
use latsec::params::{ideal_sizes, solve_widths};
use latsec::profile::{Admissibility, Category, Profile};
use latsec::published::{self, PUBLISHED_ROWS};
use num_traits::ToPrimitive;

const FUNCTIONAL: [&str; 3] = ["functional-n10-d16", "functional-n20-d16", "functional-n20-d512"];

fn relative_difference(a: f64, b: f64) -> f64 {
    (a / b - 1.0).abs()
}

#[test]
fn committed_profiles_match_the_builder() {
    let built = builder::standard_profiles(repo_path("fixtures")).unwrap();
    assert_eq!(built.len(), 17);
    for fresh in built {
        let stored = load_profile(&fresh.id);
        stored.validate().unwrap();
        // Widths and tail factors come from floating-point formulas and are rounded
        // upward to dyadic values; another platform's libm may round a last bit
        // differently.  Everything else must be identical, and everything derived
        // from the stored widths is exact (checked by validate above).
        for (a, b) in [
            (&stored.gaussian.sigma_s, &fresh.gaussian.sigma_s),
            (&stored.gaussian.sigma_big_r, &fresh.gaussian.sigma_big_r),
            (&stored.gaussian.sigma_r, &fresh.gaussian.sigma_r),
            (&stored.tails.theta_s, &fresh.tails.theta_s),
            (&stored.tails.theta_big_r, &fresh.tails.theta_big_r),
            (&stored.tails.theta_r, &fresh.tails.theta_r),
        ] {
            assert!(relative_difference(a.to_f64(), b.to_f64()) < 1e-9, "{}: width drifted", fresh.id);
        }
        let mut patched = fresh.clone();
        patched.gaussian = stored.gaussian.clone();
        patched.tails = stored.tails.clone();
        patched.bounds = stored.bounds.clone();
        patched.digest = stored.digest.clone();
        assert_eq!(patched, stored, "{}: committed profile differs from the builder", fresh.id);
        // The committed JSON is exactly what the serializer writes.
        let text = std::fs::read_to_string(repo_path(&format!("profiles/{}.json", stored.id))).unwrap();
        assert_eq!(text, stored.to_json(), "{}: file is not in canonical form", stored.id);
    }
}

#[test]
fn author_profiles_reproduce_tables_4_and_5() {
    for row in PUBLISHED_ROWS {
        let profile = load_profile(&builder::author_profile_id(row.n, row.kappa));
        let geo = published::geometry(row.n);
        assert_eq!(profile.category, Category::Author);
        assert_eq!((profile.ring.d, profile.ring.q), (512, row.q));
        assert_eq!((profile.p, profile.ell, profile.kappa), (row.p, row.ell, row.kappa));
        assert_eq!(
            (profile.sec.n, profile.sec.k, profile.sec.t_uf, profile.sec.t_sig),
            (geo.n, geo.k, geo.k_p, geo.k_c)
        );
        assert_eq!(profile.sec.k_c_det, Some(geo.k_c_det));
        assert_eq!(
            profile.admissibility,
            Admissibility::Paper { beta_lambda: geo.beta_lambda, omega_lambda: geo.omega_lambda }
        );
        let [lg_s, lg_big_r, lg_r, lg_b1, lg_b2, lg_msis] = profile.display_logs();
        for (name, value, expected) in [
            ("lg sigma_s", lg_s, row.lg_sigma_s),
            ("lg sigma_R", lg_big_r, row.lg_sigma_s),
            ("lg sigma_r", lg_r, row.lg_sigma_r),
            ("lg B1", lg_b1, row.lg_b1),
            ("lg B2", lg_b2, row.lg_b2),
            ("lg B_MSIS", lg_msis, row.lg_b2 + 1.0),
        ] {
            assert!((value - expected).abs() < 6e-5, "{}: {name} = {value}, published {expected}", profile.id);
        }
        // Rounding the widths upward moves the exact bounds by a negligible amount.
        let real = solve_widths(&published::width_inputs(row)).unwrap();
        let b1 = profile.bounds.b1.to_f64().unwrap();
        // Less than a relative 2^-30.
        assert!(b1 >= real.b1 && relative_difference(b1, real.b1) < 9.4e-10, "{}: B1 moved", profile.id);
        // The modulus is the smallest allowed by the paper's rule q > 2 B_MSIS + 1.
        assert_eq!(lg_msis.floor() as u32 + 2, row.modulus_bits);
        // Every security-side condition the paper states holds.
        for check in profile.checks() {
            assert!(check.ok, "{}: {} fails ({})", profile.id, check.name, check.detail);
        }
        // Table 5 from the profile's own (rounded) values.
        let sizes = ideal_sizes(
            512,
            row.ell,
            row.p,
            row.n,
            geo.c_exp,
            row.kappa,
            profile.challenge.omega_c,
            row.modulus_bits,
            profile.gaussian.sigma_s.to_f64(),
            b1,
            profile.bounds.b2.to_f64().unwrap(),
        );
        for (value, expected) in sizes.kib().iter().zip(row.sizes_kib) {
            assert!((value - expected).abs() < 6e-4, "{}: size {value} KiB, published {expected}", profile.id);
        }
        // Published estimates are recorded with their targets.
        let ak = profile.evidence.estimates.iter().find(|e| e.instance == "AK").unwrap();
        assert_eq!(ak.bits, row.security.ak);
        assert!(profile.evidence.estimates.iter().all(|e| e.bits >= e.target_bits));
        assert!(!profile.evidence.matrix_constants.as_ref().unwrap().certified);
    }
}

#[test]
fn variant_profiles_shorten_the_modulus_by_one_bit() {
    // q for b_q - 1, from tools/results (largest NTT-friendly primes).
    let expected: [(usize, u32, u32, u128); 6] = [
        (40, 128, 60, (1 << 60) - 16383),
        (40, 192, 62, (1 << 62) - 21503),
        (40, 256, 63, (1 << 63) - 17407),
        (60, 128, 63, (1 << 63) - 17407),
        (60, 192, 65, (1 << 65) - 10239),
        (60, 256, 65, (1 << 65) - 10239),
    ];
    for (n, kappa, bits, q) in expected {
        let variant = load_profile(&builder::variant_profile_id(n, kappa, bits));
        let author = load_profile(&builder::author_profile_id(n, kappa));
        assert_eq!(variant.category, Category::AuthorVariant);
        assert_eq!(variant.ring.q, q);
        assert_eq!(128 - author.ring.q.leading_zeros(), bits + 1);
        // Same widths, tails and bounds; only the modulus (and its root) differ.
        assert_eq!(variant.gaussian, author.gaussian);
        assert_eq!(variant.tails, author.tails);
        assert_eq!(variant.bounds, author.bounds);
        assert_eq!(variant.sec, author.sec);
        assert_ne!(variant.digest, author.digest);
        let checks = variant.checks();
        let euclidean = checks.iter().find(|c| c.name.starts_with("Euclidean MSIS")).unwrap();
        let papers_rule = checks.iter().find(|c| c.name.starts_with("paper's rule")).unwrap();
        assert!(euclidean.ok, "B_MSIS < q must hold");
        assert!(!papers_rule.ok, "q > 2 B_MSIS + 1 fails by construction");
        assert!(checks.iter().filter(|c| !c.ok).count() == 1);
        // The recorded estimates still meet their targets, with less MSIS margin.
        for (ours, theirs) in variant.evidence.estimates.iter().zip(&author.evidence.estimates) {
            assert_eq!(ours.instance, theirs.instance);
            assert!(ours.bits >= ours.target_bits);
            if ours.instance == "MSIS" {
                assert!(ours.bits < theirs.bits);
            } else {
                assert!(ours.bits > theirs.bits);
            }
        }
        assert!(variant.evidence.claim.starts_with("Our variant"));
    }
}

#[test]
fn functional_profiles_claim_nothing() {
    for id in FUNCTIONAL {
        let profile = load_profile(id);
        assert_eq!(profile.category, Category::Functional);
        assert!(profile.evidence.claim.starts_with("No security claim"));
        assert!(profile.evidence.estimates.is_empty());
        // Still executable with headroom: the response bound stays below the modulus.
        assert!(profile.checks().iter().any(|c| c.name.starts_with("Euclidean MSIS") && c.ok));
    }
    // The full-degree functional profile really uses the wide paths.
    let wide = load_profile("functional-n20-d512");
    assert_eq!(wide.ring.q, (1u128 << 66) - 16383);
    assert!(wide.bounds.b2.bits() > 64);
}

#[test]
fn profiles_reject_edits_to_protocol_fields() {
    let text = std::fs::read_to_string(repo_path("profiles/author-n40-k128-nov2026.json")).unwrap();
    assert!(Profile::from_json(&text).is_ok());
    // Changing the challenge weight without updating bounds and digest is caught.
    let edited = text.replace("\"omega_c\": 30", "\"omega_c\": 29");
    assert_ne!(edited, text);
    assert!(Profile::from_json(&edited).is_err());
    // So is changing one entry of G.
    let edited = text.replacen("[1, 0, 0, 0, 0, 1, 0, 0, 1,", "[1, 0, 0, 0, 0, 1, 0, 0, -1,", 1);
    assert_ne!(edited, text);
    assert!(Profile::from_json(&edited).is_err());
    // A leading zero is not a canonical integer.
    let edited = text.replace("\"q\": \"2305843009213683713\"", "\"q\": \"02305843009213683713\"");
    assert_ne!(edited, text);
    assert!(Profile::from_json(&edited).is_err());
}

/// The signature length limit of the decoder against the ideal size of a
/// response at the bound `B2` (Table 5 of the paper, column
/// `|sig|`, in KiB): the limit must lie above the estimate, and stays within 3%.
#[test]
fn signature_length_limits_are_close_to_the_size_estimates() {
    let table_5 = [
        ("author-n40-k128-nov2026", 40.750),
        ("author-n40-k192-nov2026", 56.147),
        ("author-n40-k256-nov2026", 70.981),
        ("author-n60-k128-nov2026", 42.622),
        ("author-n60-k192-nov2026", 58.641),
        ("author-n60-k256-nov2026", 74.098),
    ];
    for (id, estimate) in table_5 {
        let limit = latsec::latsec::limits::max_signature_len(&scheme(id)) as f64 / 1024.0;
        assert!(limit > estimate && limit < 1.03 * estimate, "{id}: limit {limit:.3} KiB, estimate {estimate} KiB");
    }
}

fn check_vector(id: &str) {
    let stored = TestVector::load(repo_path(&format!("fixtures/kat/{id}.json"))).unwrap();
    let scheme = scheme(id);
    assert_eq!(stored.profile_digest, scheme.profile().digest, "{id}: vector is for another profile version");
    let fresh = TestVector::generate(Arc::clone(&scheme), &stored.seed).unwrap();
    assert_eq!(fresh, stored, "{id}: test vector changed");
}

#[test]
fn test_vectors_functional() {
    for id in FUNCTIONAL {
        check_vector(id);
    }
    // Short encodings are stored in full and can be decoded and verified directly.
    let stored = TestVector::load(repo_path("fixtures/kat/functional-n10-d16.json")).unwrap();
    let scheme = scheme("functional-n10-d16");
    let unhex = |text: &str| -> Vec<u8> {
        (0..text.len()).step_by(2).map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap()).collect()
    };
    let vk = latsec::latsec::VerificationKey::from_bytes(&scheme, &unhex(stored.vk.as_ref().unwrap())).unwrap();
    let sig = latsec::latsec::Signature::from_bytes(&scheme, &unhex(stored.signature.as_ref().unwrap())).unwrap();
    let msg = unhex(&stored.msg);
    assert!(latsec::latsec::verify(&scheme, &vk, &msg, &sig));
    let parsed = latsec::latsec::SessionMessage::from_bytes(&scheme, &msg).unwrap();
    assert_eq!(parsed.sid.0.to_vec(), unhex(&stored.session_id));
    assert_eq!(parsed.act.parties(), stored.signers.as_slice());
    assert_eq!(parsed.lambda, stored.lambda);
    assert_eq!(parsed.payload, stored.payload.as_bytes());
    let mut other = parsed.clone();
    other.payload = b"other".to_vec();
    assert!(!latsec::latsec::verify(&scheme, &vk, &other.to_bytes(&scheme).unwrap(), &sig));
}

#[test]
fn test_vectors_author_n40() {
    for kappa in [128, 192, 256] {
        check_vector(&builder::author_profile_id(40, kappa));
    }
}

#[test]
fn test_vectors_author_n60() {
    for kappa in [128, 192, 256] {
        check_vector(&builder::author_profile_id(60, kappa));
    }
}

#[test]
fn test_vectors_variants() {
    for (n, kappa, bits) in [(40, 128, 60), (40, 256, 63), (60, 128, 63), (60, 192, 65)] {
        check_vector(&builder::variant_profile_id(n, kappa, bits));
    }
}
