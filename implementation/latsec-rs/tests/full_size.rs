//! Full-degree (`d = 512`) sessions: the wide functional profile and the largest
//! and smallest author profiles, with fresh sampled masks.

mod common;

use std::sync::Arc;

use common::{deployment, set, sid};
use latsec::latsec::{combine, verify, CombineOutput, PartialResponse, SessionMessage, Signature};
use latsec::sampling::RandomSource;
use latsec::scheme::{norm_squared, ActiveSet, PartyId};
use latsec::sim::SessionOutcome;
use latsec::stats::random_subset;
use num_traits::ToPrimitive;

fn admissible_set(dep: &latsec::sim::Deployment, rng: &mut dyn RandomSource, size: usize) -> ActiveSet {
    for _ in 0..1000 {
        let parties: Vec<PartyId> =
            random_subset(rng, dep.scheme.n(), size).into_iter().map(|p| p as PartyId).collect();
        let act = ActiveSet::new(&parties).unwrap();
        if dep.scheme.lambda(&act).is_ok() {
            return act;
        }
    }
    panic!("no admissible set of size {size}");
}

#[test]
fn wide_functional_profile() {
    let (mut dep, mut rng) = deployment("functional-n20-d512", "wide");
    let scheme = Arc::clone(&dep.scheme);
    assert_eq!(scheme.ring().q(), (1u128 << 66) - 16383);
    let act = set(&(0..20).collect::<Vec<PartyId>>());
    let (outcome, _) = dep.sign(sid(1), &act, b"wide", &mut rng).unwrap();
    let SessionOutcome::Signed { signature, transcript } = outcome else { panic!("honest session aborted") };
    assert!(verify(&scheme, &dep.vk, &transcript.msg, &signature));

    // Challenge: weight 30 in degree 512.
    assert_eq!(signature.challenge.weight(), 30);
    assert_eq!(signature.challenge.coeffs().len(), 512);

    // Masks are near 2^56: partial responses do not fit in 56 bits but respect B1.
    let largest =
        transcript.responses.iter().flat_map(|r| r.z.iter().flatten()).map(|x| x.unsigned_abs()).max().unwrap();
    assert!(largest > 1 << 56 && largest < scheme.b1());
    for response in &transcript.responses {
        assert!(&norm_squared(&response.z) <= scheme.b1_squared());
        let bytes = response.to_bytes(&scheme).unwrap();
        assert_eq!(PartialResponse::from_bytes(&scheme, &bytes).unwrap(), *response);
    }
    // The signature bound exceeds 2^64.  The response is Rice coded, after the
    // parameter byte and the ranked challenge of ceil(30 + lg binom(512, 30)) = 192 bits.
    assert!(scheme.b2() > 1 << 64);
    assert_eq!(latsec::codec::challenge_rank_bits(512, 30) + 30, 192);
    let bytes = signature.to_bytes(&scheme).unwrap();
    let k = u32::from(bytes[2]);
    let z_bits = latsec::codec::rice_length(signature.z.iter().flatten().map(|x| x.unsigned_abs()), k) as usize;
    assert_eq!(bytes.len(), 3 + (192 + z_bits).div_ceil(8));
    assert_eq!(Signature::from_bytes(&scheme, &bytes).unwrap(), signature);

    // A coefficient beyond 64 bits (but within B2, about 2^64.95) survives the codec.
    let mut big = signature.clone();
    big.z[0][0] = -((1i128 << 64) + 12345);
    let round_trip = Signature::from_bytes(&scheme, &big.to_bytes(&scheme).unwrap()).unwrap();
    assert_eq!(round_trip.z[0][0], -((1i128 << 64) + 12345));
    assert!(!verify(&scheme, &dep.vk, &transcript.msg, &round_trip));

    // Abort attribution at full degree.
    let mut tampered = transcript.clone();
    tampered.responses[7].z[3][511] += 1;
    assert_eq!(
        combine(&scheme, &dep.vk, &dep.abort_keys, &tampered).unwrap(),
        CombineOutput::Abort { blamed: vec![7] }
    );
}

fn author_session(id: &str, size: usize) {
    let (mut dep, mut rng) = deployment(id, "author-session");
    let scheme = Arc::clone(&dep.scheme);
    let act = admissible_set(&dep, &mut rng, size);
    let lambda = scheme.lambda(&act).unwrap();
    let msg = b"author profile session";
    let (outcome, _) = dep.sign(sid(1), &act, msg, &mut rng).unwrap();
    let SessionOutcome::Signed { signature, transcript } = outcome else { panic!("{id}: honest session aborted") };
    assert!(verify(&scheme, &dep.vk, &transcript.msg, &signature));
    let other = SessionMessage::new(&scheme, sid(2), &act, msg).unwrap().to_bytes(&scheme).unwrap();
    assert!(!verify(&scheme, &dep.vk, &other, &signature));

    // Honest norms sit far inside the verification radii.
    let b1 = scheme.b1() as f64;
    for response in &transcript.responses {
        let ratio = norm_squared(&response.z).to_f64().unwrap().sqrt() / b1;
        assert!(ratio > 0.5 && ratio < 1.0, "{id}: |z_i| / B1 = {ratio}");
    }
    let ratio = norm_squared(&signature.z).to_f64().unwrap().sqrt() / scheme.b2() as f64;
    assert!(ratio < 0.2, "{id}: |z| / B2 = {ratio}");

    // omega_lambda < t_sig: some signers have a zero coefficient yet took part.
    let idle = lambda.iter().filter(|&&x| x == 0).count();
    assert!(idle >= 1, "{id}: no idle signer");
    assert_eq!(transcript.responses.len(), act.len());

    // The signature carries the challenge in b_c bits and is shorter than the
    // fixed-width encoding of z at B2 would be.
    let profile = scheme.profile();
    let dim = profile.ell * 512;
    let sig_bytes = signature.to_bytes(&scheme).unwrap();
    let omega_c = profile.challenge.omega_c as usize;
    let b_c = latsec::codec::challenge_rank_bits(512, omega_c) as usize + omega_c;
    let k = u32::from(sig_bytes[2]);
    let z_bits = latsec::codec::rice_length(signature.z.iter().flatten().map(|x| x.unsigned_abs()), k) as usize;
    assert_eq!(sig_bytes.len(), 3 + (b_c + z_bits).div_ceil(8));
    assert!(z_bits < dim * latsec::codec::signed_width(scheme.b2()) as usize);
    assert_eq!(Signature::from_bytes(&scheme, &sig_bytes).unwrap(), signature);
    let modulus_bits = 128 - profile.ring.q.leading_zeros() as usize;
    assert_eq!(dep.vk.to_bytes(&scheme).len(), 2 + 64 + 32 + profile.p * 512 * modulus_bits / 8);

    // A signer set that fails the paper's predicate is refused without blame.
    let small: Vec<PartyId> = act.parties()[..profile.sec.t_sig - 1].to_vec();
    assert!(dep.sign(sid(3), &ActiveSet::new(&small).unwrap(), msg, &mut rng).is_err());
}

#[test]
fn author_n40_kappa128_at_the_signing_threshold() {
    author_session("author-n40-k128-nov2026", 26);
}

#[test]
fn author_n40_kappa256_all_parties() {
    author_session("author-n40-k256-nov2026", 40);
}

#[test]
fn author_n60_kappa192_at_the_signing_threshold() {
    author_session("author-n60-k192-nov2026", 46);
}

#[test]
fn author_n60_kappa256_at_the_signing_threshold() {
    author_session("author-n60-k256-nov2026", 46);
}
