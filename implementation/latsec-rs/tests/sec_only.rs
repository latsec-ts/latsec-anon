//! SEC on its own at full degree, with the standalone parameter sets of the SEC paper [46]
//! and the 32-bit hard-core modulus `q = 2^32 - 10239`.

mod common;

use std::sync::Arc;

use common::{load_profile, scheme};
use latsec::latsec::{keygen, sample_challenge, Signer};
use latsec::profile::Category;
use latsec::sampling::{RandomSource, SeededRandom};
use latsec::scheme::{combine_z, norm_squared, ActiveSet, Inadmissible, PartyId, PublicMatrix};
use latsec::stats::random_subset;

fn check(id: &str) {
    let scheme = scheme(id);
    let profile = scheme.profile();
    assert_eq!(profile.category, Category::SecOnly);
    assert_eq!(profile.ring.q, (1u128 << 32) - 10239);
    assert_eq!((profile.ring.d, profile.ell, profile.p), (512, 6, 3));
    assert!(!scheme.supports_signing());
    let (d, ell, n) = (512, profile.ell, profile.sec.n);
    let mut rng = SeededRandom::new(id.as_bytes(), "sec-only");

    // Deal a Gaussian secret; every share coefficient respects the tail bound.
    let secret: Vec<Vec<i128>> = (0..ell)
        .map(|_| {
            let mut poly = vec![0i128; d];
            scheme.secret_sampler().sample_into(&mut rng, &mut poly);
            poly
        })
        .collect();
    let shares = scheme.deal(&secret, &mut rng);
    assert_eq!(shares.len(), n);
    let largest = shares.iter().flatten().flatten().map(|x| x.unsigned_abs()).max().unwrap();
    assert!(largest <= scheme.beta_sh(), "share coefficient {largest} exceeds beta_sh = {}", scheme.beta_sh());
    // Shares are a few bits wide: sigma_s and sigma_R are below 2^6.
    assert!(largest < 1 << 10);

    // The hard-core map: public images t_i = A s_i mod q reconstruct to t = A s.
    let mut seed = [0u8; 32];
    rng.fill(&mut seed);
    let matrix = PublicMatrix::expand(scheme.ring(), profile.p, ell, &seed);
    let t = matrix.mul_z(scheme.ring(), &secret);
    let images: Vec<Vec<Vec<u128>>> = shares.iter().map(|s| matrix.mul_z(scheme.ring(), s)).collect();

    let (mut authorized, mut tried) = (0, 0);
    while authorized < 12 {
        tried += 1;
        assert!(tried < 200);
        let size = profile.sec.t_sig + (tried % (n - profile.sec.t_sig + 1));
        let parties: Vec<PartyId> = random_subset(&mut rng, n, size).into_iter().map(|p| p as PartyId).collect();
        let set = ActiveSet::new(&parties).unwrap();
        let picked: Vec<&[Vec<i128>]> = parties.iter().map(|&p| shares[usize::from(p)].as_slice()).collect();
        match scheme.reconstruct(&set, &picked) {
            Ok(recovered) => {
                authorized += 1;
                assert_eq!(recovered, secret, "{id}: reconstruction is not exact");
                let lambda = scheme.recon_vector(&set).unwrap();
                assert_eq!(combine_z(&lambda, &picked), secret);
                let refs: Vec<&[Vec<u128>]> = parties.iter().map(|&p| images[usize::from(p)].as_slice()).collect();
                assert_eq!(scheme.combine_q(&lambda, &refs), t, "{id}: reconstruction does not commute with A");
            }
            Err(Inadmissible::NotAuthorized) => {}
            Err(other) => panic!("unexpected: {other}"),
        }
    }
    // Below the reconstruction threshold SEC.Rec refuses.
    let few: Vec<PartyId> = (0..profile.sec.t_sig as PartyId - 1).collect();
    let picked: Vec<&[Vec<i128>]> = few.iter().map(|&p| shares[usize::from(p)].as_slice()).collect();
    assert!(matches!(
        scheme.reconstruct(&ActiveSet::new(&few).unwrap(), &picked),
        Err(Inadmissible::TooFewSigners { .. })
    ));

    // A leakage query of the multi-query privacy model, c s_i + r_i, stays within B1.
    let challenge = sample_challenge(&scheme, &mut SeededRandom::new(b"a leakage query", "sec-only-test"));
    assert_eq!(challenge.weight(), profile.challenge.omega_c as usize);
    for share in shares.iter().take(4) {
        let response: Vec<Vec<i128>> = share
            .iter()
            .map(|poly| {
                let mut mask = vec![0i128; d];
                scheme.mask_sampler().sample_into(&mut rng, &mut mask);
                challenge.mul_z(poly).iter().zip(&mask).map(|(a, b)| a + b).collect()
            })
            .collect();
        assert!(&norm_squared(&response) <= scheme.b1_squared());
    }

    // Signing is disabled for SEC-only profiles.
    let material = keygen(&scheme, &mut rng);
    let vk = Arc::new(material.vk);
    let share = material.shares.into_iter().next().unwrap();
    assert!(Signer::new(Arc::clone(&scheme), vk, share).is_err());
}

#[test]
fn sec_only_n40() {
    check("sec-only-n40-k128-q32");
}

#[test]
fn sec_only_n60() {
    check("sec-only-n60-k128-q32");
}

#[test]
fn sec_only_widths_match_the_sec_paper_tables() {
    // Table 7 of the SEC paper: n = 60, kappa = 128, Q = 2^64: lg sigma_s = 2.665, lg sigma_R = 5.706, lg sigma_r = 43.929.
    let profile = load_profile("sec-only-n60-k128-q32");
    let [lg_s, lg_big_r, lg_r, ..] = profile.display_logs();
    assert!((lg_s - 2.6652).abs() < 1.5e-4 && (lg_big_r - 5.7063).abs() < 1.5e-4 && (lg_r - 43.9290).abs() < 1.5e-4);
    let profile = load_profile("sec-only-n40-k128-q32");
    let [lg_s, lg_big_r, lg_r, ..] = profile.display_logs();
    assert!((lg_s - 3.7238).abs() < 1.5e-4 && (lg_big_r - 5.2172).abs() < 1.5e-4 && (lg_r - 43.6412).abs() < 1.5e-4);
    assert_eq!(profile.challenge.omega_c, 19);
    // The signature-side checks are reported as not applicable rather than failed.
    assert!(profile.checks().iter().all(|c| c.ok));
}
