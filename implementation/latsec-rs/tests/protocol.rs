//! Protocol behavior on the small functional profiles: honest sessions, rejection
//! of inadmissible signer sets, identifiable aborts, session lifecycle, encodings.

mod common;

use std::sync::Arc;

use common::{deployment, load_profile, scheme, set, sid, subsets};
use latsec::codec::{self, tag, BitWriter};
use latsec::latsec::{
    admissible, combine, partial_verify, verify, AbortKey, CombineOutput, Commitment, InvocationError, Opening,
    PartialResponse, Round3Output, SecretShare, SessionMessage, SessionStatus, Signature, Signer, Transcript,
    VerificationKey,
};
use latsec::ring::Challenge;
use latsec::sampling::{RandomSource, SeededRandom};
use latsec::scheme::{combine_z, ActiveSet, Inadmissible, PartyId, Scheme};
use latsec::sim::{Deployment, SessionOutcome};

const SMALL: &str = "functional-n10-d16";
const MEDIUM: &str = "functional-n20-d16";

/// The encoded session message for `(sid(tag), act)` with this crate's
/// reconstruction vector, as the session initiator builds it.
fn session_msg(scheme: &Scheme, tag: u64, act: &ActiveSet, payload: &[u8]) -> Vec<u8> {
    SessionMessage::new(scheme, sid(tag), act, payload).unwrap().to_bytes(scheme).unwrap()
}

/// The encoded session message with a given reconstruction vector.
fn session_msg_with(scheme: &Scheme, tag: u64, act: &ActiveSet, lambda: &[i64], payload: &[u8]) -> Vec<u8> {
    let message =
        SessionMessage { sid: sid(tag), act: act.clone(), lambda: lambda.to_vec(), payload: payload.to_vec() };
    message.to_bytes(scheme).unwrap()
}

/// The `lambda_act` a session message carries.
fn lambda_of(scheme: &Scheme, msg: &[u8]) -> Vec<i64> {
    SessionMessage::from_bytes(scheme, msg).unwrap().lambda
}

/// Runs the three rounds by hand and returns the transcript.  `tamper` may alter
/// the messages after each round, as a malicious party's broadcast would.
struct Rounds {
    commitments: Vec<(PartyId, Commitment)>,
    openings: Vec<(PartyId, Opening)>,
    round3: Vec<(PartyId, Round3Output)>,
}

#[allow(clippy::too_many_arguments)]
fn run_rounds(
    dep: &mut Deployment,
    rng: &mut dyn RandomSource,
    session: latsec::latsec::SessionId,
    act: &ActiveSet,
    msg: &[u8],
    tamper_commitments: impl FnOnce(&mut Vec<(PartyId, Commitment)>),
    tamper_openings: impl FnOnce(&mut Vec<(PartyId, Opening)>),
    malicious: &[PartyId],
) -> Rounds {
    let mut commitments = Vec::new();
    for &party in act.parties() {
        commitments.push((party, dep.signers[usize::from(party)].round1(session, act, msg, rng).unwrap()));
    }
    tamper_commitments(&mut commitments);
    let mut openings = Vec::new();
    for &party in act.parties() {
        match dep.signers[usize::from(party)].round2(&session, act, &commitments) {
            Ok(opening) => openings.push((party, opening)),
            // A malicious party notices nothing wrong with its own lie; model its
            // broadcast by an arbitrary well-formed opening.
            Err(InvocationError::OwnCommitmentMismatch) if malicious.contains(&party) => {
                let d = dep.scheme.ring().degree();
                openings.push((party, Opening { w: vec![vec![1u128; d]; dep.scheme.profile().p] }));
            }
            Err(other) => panic!("round 2 failed for party {party}: {other}"),
        }
    }
    tamper_openings(&mut openings);
    let mut round3 = Vec::new();
    for &party in act.parties() {
        if malicious.contains(&party) {
            continue;
        }
        round3.push((party, dep.signers[usize::from(party)].round3(&session, act, &openings).unwrap()));
    }
    Rounds { commitments, openings, round3 }
}

fn honest_transcript(
    dep: &mut Deployment,
    rng: &mut dyn RandomSource,
    tag: u64,
    act: &ActiveSet,
    payload: &[u8],
) -> Transcript {
    let msg = session_msg(&dep.scheme, tag, act, payload);
    let rounds = run_rounds(dep, rng, sid(tag), act, &msg, |_| {}, |_| {}, &[]);
    let responses = rounds
        .round3
        .into_iter()
        .map(|(_, out)| match out {
            Round3Output::Response(r) => r,
            Round3Output::Abort { blamed } => panic!("honest session aborted, blaming {blamed:?}"),
        })
        .collect();
    Transcript {
        sid: sid(tag),
        act: act.clone(),
        msg,
        commitments: rounds.commitments.into_iter().map(|(_, c)| c).collect(),
        openings: rounds.openings.into_iter().map(|(_, o)| o).collect(),
        responses,
    }
}

fn signature_of(dep: &Deployment, tr: &Transcript) -> Signature {
    match combine(&dep.scheme, &dep.vk, &dep.abort_keys, tr).unwrap() {
        CombineOutput::Signature(sig) => sig,
        CombineOutput::Abort { blamed } => panic!("combine aborted, blaming {blamed:?}"),
    }
}

#[test]
fn every_admissible_set_signs_and_verifies() {
    let (mut dep, mut rng) = deployment(SMALL, "exhaustive");
    let (mut signed, mut unauthorized) = (0, 0);
    let mut tag = 0;
    for size in 7..=10 {
        for parties in subsets(10, size) {
            let act = set(&parties);
            tag += 1;
            let msg = format!("message {tag}");
            match dep.sign(sid(tag), &act, msg.as_bytes(), &mut rng) {
                Ok((SessionOutcome::Signed { signature, transcript }, _)) => {
                    signed += 1;
                    assert!(verify(&dep.scheme, &dep.vk, &transcript.msg, &signature));
                    // Every partial response individually verifies.
                    for &party in act.parties() {
                        assert!(partial_verify(&dep.scheme, &dep.vk, &dep.abort_keys, &transcript, party).unwrap());
                    }
                }
                Ok((SessionOutcome::Aborted { blamed, stage }, _)) => {
                    panic!("honest session aborted at stage {stage}, blaming {blamed:?}")
                }
                Err(InvocationError::Inadmissible(Inadmissible::NotAuthorized)) => unauthorized += 1,
                Err(other) => panic!("unexpected error: {other}"),
            }
        }
    }
    // Exhaustive counts for the 10 x 6 matrix: 24 of 120 seven-sets and 3 of 45
    // eight-sets cannot reconstruct.
    assert_eq!(unauthorized, 27);
    assert_eq!(signed, 120 + 45 + 10 + 1 - 27);
}

#[test]
fn inadmissible_sets_are_rejected_before_round_one_without_blame() {
    let (mut dep, mut rng) = deployment(SMALL, "inadmissible");
    let scheme = Arc::clone(&dep.scheme);

    let unauthorized = subsets(10, 7).into_iter().map(|p| set(&p)).find(|a| scheme.lambda(a).is_err()).unwrap();
    assert_eq!(scheme.lambda(&unauthorized), Err(Inadmissible::NotAuthorized));
    let err = dep.sign(sid(1), &unauthorized, b"m", &mut rng).unwrap_err();
    assert_eq!(err, InvocationError::Inadmissible(Inadmissible::NotAuthorized));
    // Nothing was consumed: no budget unit, no session identifier, no state.
    for signer in &dep.signers {
        assert_eq!(signer.participations(), 0);
        assert_eq!(signer.session_status(&sid(1)), SessionStatus::Unused);
    }

    // An authorized set below the signing threshold.
    let small = subsets(10, 6).into_iter().map(|p| set(&p)).find(|a| scheme.recon_vector(a).is_ok()).unwrap();
    assert_eq!(scheme.lambda(&small), Err(Inadmissible::TooFewSigners { have: 6, need: 7 }));
    assert!(matches!(
        dep.sign(sid(2), &small, b"m", &mut rng),
        Err(InvocationError::Inadmissible(Inadmissible::TooFewSigners { .. }))
    ));

    // A party that does not exist.
    let unknown = set(&[0, 1, 2, 3, 4, 5, 6, 10]);
    assert_eq!(scheme.lambda(&unknown), Err(Inadmissible::UnknownParty(10)));

    // Authorized, large enough, but outside the coefficient bounds.
    let medium = common::scheme(MEDIUM);
    let mut found = None;
    for parties in subsets(20, 14) {
        let act = set(&parties);
        if let Err(Inadmissible::L1TooLarge { l1, bound }) = medium.lambda(&act) {
            assert!(l1 > bound && bound == 8);
            // The set is authorized; only the bound rejects it.
            let vector = medium.recon_vector(&act).unwrap();
            assert_eq!(vector.iter().map(|x| x.unsigned_abs()).sum::<u64>(), l1);
            found = Some(act);
            break;
        }
    }
    let out_of_bounds = found.expect("some authorized 14-set exceeds the l1 cap");
    let (mut dep20, mut rng20) = deployment(MEDIUM, "inadmissible");
    assert!(matches!(
        dep20.sign(sid(3), &out_of_bounds, b"m", &mut rng20),
        Err(InvocationError::Inadmissible(Inadmissible::L1TooLarge { .. }))
    ));
    assert!(dep20.signers.iter().all(|s| s.participations() == 0));

    // Duplicates and the empty set are not signer sets at all.
    assert!(ActiveSet::new(&[1, 2, 2]).is_err());
    assert!(ActiveSet::new(&[]).is_err());
    // Order does not matter; the set is canonicalized.
    assert_eq!(ActiveSet::new(&[5, 1, 3]).unwrap().parties(), &[1, 3, 5]);
}

#[test]
fn tampered_opening_is_blamed_by_every_honest_signer() {
    let (mut dep, mut rng) = deployment(SMALL, "opening");
    let act = set(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let cheater: PartyId = 4;
    let q = dep.scheme.ring().q();
    let msg = session_msg(&dep.scheme, 1, &act, b"m");
    let rounds = run_rounds(
        &mut dep,
        &mut rng,
        sid(1),
        &act,
        &msg,
        |_| {},
        |openings| {
            let opening = &mut openings.iter_mut().find(|(p, _)| *p == cheater).unwrap().1;
            opening.w[0][3] = (opening.w[0][3] + 1) % q;
        },
        &[cheater],
    );
    assert_eq!(rounds.round3.len(), 9);
    for (party, out) in rounds.round3 {
        assert_eq!(out, Round3Output::Abort { blamed: vec![cheater] }, "signer {party}");
        // The aborted session is finished and its budget unit stays consumed.
        assert_eq!(dep.signers[usize::from(party)].session_status(&sid(1)), SessionStatus::Finished);
        assert_eq!(dep.signers[usize::from(party)].participations(), 1);
    }
}

#[test]
fn tampered_commitment_is_blamed_by_every_honest_signer() {
    let (mut dep, mut rng) = deployment(SMALL, "commitment");
    let act = set(&[0, 1, 2, 3, 5, 6, 7, 8, 9]);
    let cheater: PartyId = 7;
    let msg = session_msg(&dep.scheme, 1, &act, b"m");
    let rounds = run_rounds(
        &mut dep,
        &mut rng,
        sid(1),
        &act,
        &msg,
        |commitments| commitments.iter_mut().find(|(p, _)| *p == cheater).unwrap().1 .0[0] ^= 1,
        |_| {},
        &[cheater],
    );
    for (_, out) in rounds.round3 {
        assert_eq!(out, Round3Output::Abort { blamed: vec![cheater] });
    }
    // An honest signer whose own commitment was altered in transit does not blame
    // anyone: it reports a delivery fault.
    let (mut dep, mut rng) = deployment(SMALL, "commitment-2");
    let msg = session_msg(&dep.scheme, 2, &act, b"m");
    let mut commitments = Vec::new();
    for &party in act.parties() {
        commitments.push((party, dep.signers[usize::from(party)].round1(sid(2), &act, &msg, &mut rng).unwrap()));
    }
    commitments[0].1 .0[5] ^= 0x40;
    let victim = usize::from(act.parties()[0]);
    assert_eq!(dep.signers[victim].round2(&sid(2), &act, &commitments), Err(InvocationError::OwnCommitmentMismatch));
}

/// Figure 1: a missing or unparsable peer message fails the check that uses it,
/// so its sender is blamed.
#[test]
fn missing_and_malformed_round_messages_are_blamed_in_round_3() {
    let act = set(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let cheater: PartyId = 4;
    type TamperC = fn(&mut Vec<(PartyId, Commitment)>);
    type TamperO = fn(&mut Vec<(PartyId, Opening)>);
    let none_c: TamperC = |_| {};
    let none_o: TamperO = |_| {};
    let cases: [(&str, TamperC, TamperO); 6] = [
        ("missing commitment", |c| c.retain(|(p, _)| *p != 4), none_o),
        (
            "short commitment",
            |c| {
                c.iter_mut().find(|(p, _)| *p == 4).unwrap().1 .0.pop();
            },
            none_o,
        ),
        ("empty commitment", |c| c.iter_mut().find(|(p, _)| *p == 4).unwrap().1 .0.clear(), none_o),
        ("missing opening", none_c, |o| o.retain(|(p, _)| *p != 4)),
        ("truncated opening", none_c, |o| {
            o.iter_mut().find(|(p, _)| *p == 4).unwrap().1.w.pop();
        }),
        ("unreduced opening", none_c, |o| o.iter_mut().find(|(p, _)| *p == 4).unwrap().1.w[1][2] = u128::MAX),
    ];
    for (name, tamper_commitments, tamper_openings) in cases {
        let (mut dep, mut rng) = deployment(SMALL, name);
        let msg = session_msg(&dep.scheme, 1, &act, b"m");
        let rounds =
            run_rounds(&mut dep, &mut rng, sid(1), &act, &msg, tamper_commitments, tamper_openings, &[cheater]);
        assert_eq!(rounds.round3.len(), 9, "{name}");
        for (party, out) in rounds.round3 {
            assert_eq!(out, Round3Output::Abort { blamed: vec![cheater] }, "{name}, signer {party}");
            assert_eq!(dep.signers[usize::from(party)].session_status(&sid(1)), SessionStatus::Finished);
        }
    }

    // With two faulty signers, every honest signer names the first in the
    // canonical order of act.
    let (mut dep, mut rng) = deployment(SMALL, "two faulty");
    let msg = session_msg(&dep.scheme, 1, &act, b"m");
    let rounds = run_rounds(
        &mut dep,
        &mut rng,
        sid(1),
        &act,
        &msg,
        |c| c.retain(|(p, _)| *p != 7),
        |o| o.retain(|(p, _)| *p != 2),
        &[2, 7],
    );
    assert_eq!(rounds.round3.len(), 8);
    for (_, out) in rounds.round3 {
        assert_eq!(out, Round3Output::Abort { blamed: vec![2] });
    }
}

#[test]
fn missing_and_malformed_transcript_entries_are_blamed_by_combine() {
    let (mut dep, mut rng) = deployment(SMALL, "transcript entries");
    let act = set(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let honest = honest_transcript(&mut dep, &mut rng, 1, &act, b"m");
    let run = |tr: &Transcript| combine(&dep.scheme, &dep.vk, &dep.abort_keys, tr).unwrap();
    let check = |tr: &Transcript, party| partial_verify(&dep.scheme, &dep.vk, &dep.abort_keys, tr, party);

    // A missing response.
    let mut tr = honest.clone();
    tr.responses[6] = PartialResponse::missing();
    assert_eq!(run(&tr), CombineOutput::Abort { blamed: vec![6] });
    assert_eq!(check(&tr, 6), Ok(false));
    assert_eq!(check(&tr, 5), Ok(true));

    // A missing opening: there is no challenge, and only its sender is blamed.
    let mut tr = honest.clone();
    tr.openings[3] = Opening::missing();
    assert_eq!(run(&tr), CombineOutput::Abort { blamed: vec![3] });
    assert_eq!(check(&tr, 3), Ok(false));
    assert_eq!(check(&tr, 5), Err(InvocationError::MalformedMessage(3)));

    // An unreduced opening, and with it a missing and an overlong response.
    let mut tr = honest.clone();
    tr.openings[8].w[0][0] = dep.scheme.ring().q();
    assert_eq!(run(&tr), CombineOutput::Abort { blamed: vec![8] });
    tr.responses[1] = PartialResponse::missing();
    tr.responses[9].z[0][0] += 1i128 << 100;
    assert_eq!(run(&tr), CombineOutput::Abort { blamed: vec![1, 8, 9] });
    // A response that only fails the linear relation cannot be judged without a
    // challenge, so its sender is not named here.
    tr.responses[2].z[0][0] += 1;
    assert_eq!(run(&tr), CombineOutput::Abort { blamed: vec![1, 8, 9] });

    // Mixed faults: a party whose own response is missing, malformed or too long
    // fails PartialVerify even when another party's opening is missing, and
    // Combine names both.
    let q = dep.scheme.ring().q() as i128;
    for (name, response) in [
        ("missing", PartialResponse::missing()),
        ("truncated", PartialResponse { z: honest.responses[0].z[1..].to_vec() }),
        ("oversized", {
            let mut r = honest.responses[0].clone();
            r.z[0][0] += 1i128 << 100;
            r
        }),
    ] {
        let mut tr = honest.clone();
        tr.responses[0] = response;
        tr.openings[1] = Opening::missing();
        assert_eq!(check(&tr, 0), Ok(false), "{name}");
        assert_eq!(check(&tr, 1), Ok(false), "{name}");
        assert_eq!(check(&tr, 2), Err(InvocationError::MalformedMessage(1)), "{name}");
        assert_eq!(run(&tr), CombineOutput::Abort { blamed: vec![0, 1] }, "{name}");
    }
    // A response that is in bounds but fails the relation mod q needs the
    // challenge: with another opening missing there is no verdict on it.
    let mut tr = honest.clone();
    tr.responses[0].z[0][0] += 1;
    tr.openings[1] = Opening::missing();
    assert_eq!(check(&tr, 0), Err(InvocationError::MalformedMessage(1)));
    assert_eq!(run(&tr), CombineOutput::Abort { blamed: vec![1] });
    // The target's own malformed opening fails first, whatever else is wrong.
    let mut tr = honest.clone();
    tr.openings[0].w[0][0] = q as u128;
    tr.openings[1] = Opening::missing();
    assert_eq!(check(&tr, 0), Ok(false));
    assert_eq!(run(&tr), CombineOutput::Abort { blamed: vec![0, 1] });

    // The commitments are not used by Combine.
    let mut tr = honest.clone();
    tr.commitments[0].0.clear();
    assert!(matches!(run(&tr), CombineOutput::Signature(_)));
}

#[test]
fn bad_partial_responses_are_blamed_by_combine() {
    let (mut dep, mut rng) = deployment(SMALL, "response");
    let act = set(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let honest = honest_transcript(&mut dep, &mut rng, 1, &act, b"m");
    let signature = signature_of(&dep, &honest);
    assert!(verify(&dep.scheme, &dep.vk, &honest.msg, &signature));
    let q = dep.scheme.ring().q() as i128;

    // A response that fails the linear relation.
    let mut tr = honest.clone();
    tr.responses[2].z[0][0] += 1;
    assert_eq!(combine(&dep.scheme, &dep.vk, &dep.abort_keys, &tr).unwrap(), CombineOutput::Abort { blamed: vec![2] });
    assert!(!partial_verify(&dep.scheme, &dep.vk, &dep.abort_keys, &tr, 2).unwrap());
    assert!(partial_verify(&dep.scheme, &dep.vk, &dep.abort_keys, &tr, 3).unwrap());

    // A response that satisfies the relation mod q but is too long.
    let mut tr = honest.clone();
    tr.responses[5].z[1][7] += q;
    assert_eq!(combine(&dep.scheme, &dep.vk, &dep.abort_keys, &tr).unwrap(), CombineOutput::Abort { blamed: vec![5] });

    // Several bad responses are all named, in canonical order.
    let mut tr = honest.clone();
    tr.responses[8].z[3][15] -= 5;
    tr.responses[1].z[2][2] += q;
    assert_eq!(
        combine(&dep.scheme, &dep.vk, &dep.abort_keys, &tr).unwrap(),
        CombineOutput::Abort { blamed: vec![1, 8] }
    );

    // A response of the wrong shape counts as a failed check, not a crash.
    let mut tr = honest.clone();
    tr.responses[0].z.pop();
    assert_eq!(combine(&dep.scheme, &dep.vk, &dep.abort_keys, &tr).unwrap(), CombineOutput::Abort { blamed: vec![0] });

    // A signer with an idle coefficient is still checked and blamed.
    let idle = lambda_of(&dep.scheme, &honest.msg).iter().position(|&x| x == 0).expect("a zero coefficient");
    let mut tr = honest.clone();
    tr.responses[idle].z[0][0] ^= 1;
    assert_eq!(
        combine(&dep.scheme, &dep.vk, &dep.abort_keys, &tr).unwrap(),
        CombineOutput::Abort { blamed: vec![act.parties()[idle]] }
    );

    // Malformed transcripts are invocation errors.
    let mut tr = honest.clone();
    tr.responses.pop();
    assert_eq!(combine(&dep.scheme, &dep.vk, &dep.abort_keys, &tr), Err(InvocationError::MessageSetMismatch));
    let mut tr = honest.clone();
    let mut lambda = lambda_of(&dep.scheme, &honest.msg);
    lambda[0] += 1;
    tr.msg = session_msg_with(&dep.scheme, 1, &act, &lambda, b"m");
    assert!(matches!(combine(&dep.scheme, &dep.vk, &dep.abort_keys, &tr), Err(InvocationError::Inadmissible(_))));
    let mut tr = honest.clone();
    tr.msg = session_msg(&dep.scheme, 2, &act, b"m");
    assert_eq!(
        combine(&dep.scheme, &dep.vk, &dep.abort_keys, &tr),
        Err(InvocationError::Inadmissible(Inadmissible::ContextMismatch))
    );
    assert_eq!(combine(&dep.scheme, &dep.vk, &dep.abort_keys[1..], &honest), Err(InvocationError::KeyMismatch));
}

#[test]
fn session_lifecycle() {
    let (mut dep, mut rng) = deployment(SMALL, "lifecycle");
    let act = set(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let other = set(&[0, 1, 2, 3, 4, 5, 6, 7, 8]);
    assert!(dep.scheme.lambda(&other).is_ok());
    let m1 = session_msg(&dep.scheme, 1, &act, b"m");
    let m1_other_set = session_msg(&dep.scheme, 1, &other, b"m");
    let m1_other_payload = session_msg(&dep.scheme, 1, &act, b"other");

    // Round 2 and 3 need a running session.
    assert_eq!(dep.signers[0].round2(&sid(1), &act, &[]), Err(InvocationError::UnknownSession));
    assert_eq!(dep.signers[0].round3(&sid(1), &act, &[]), Err(InvocationError::UnknownSession));

    // A signer outside the set refuses.
    assert_eq!(dep.signers[9].round1(sid(1), &other, &m1_other_set, &mut rng), Err(InvocationError::NotInActiveSet));
    // The message must name this session and signer set.
    assert_eq!(
        dep.signers[0].round1(sid(1), &other, &m1, &mut rng),
        Err(InvocationError::Inadmissible(Inadmissible::ContextMismatch))
    );
    assert_eq!(
        dep.signers[0].round1(sid(1), &act, b"not a session message", &mut rng),
        Err(InvocationError::Inadmissible(Inadmissible::MalformedMessage))
    );
    assert_eq!(dep.signers[0].session_status(&sid(1)), SessionStatus::Unused);
    assert_eq!(dep.signers[9].session_status(&sid(1)), SessionStatus::Unused);

    let mut commitments = Vec::new();
    for &party in act.parties() {
        commitments.push((party, dep.signers[usize::from(party)].round1(sid(1), &act, &m1, &mut rng).unwrap()));
    }
    assert_eq!(dep.signers[0].session_status(&sid(1)), SessionStatus::Running { round: 1 });
    // The identifier cannot be used again, even with another message or set.
    assert_eq!(dep.signers[0].round1(sid(1), &act, &m1_other_payload, &mut rng), Err(InvocationError::SessionIdReused));
    assert_eq!(dep.signers[0].round1(sid(1), &other, &m1_other_set, &mut rng), Err(InvocationError::SessionIdReused));
    // Round 3 before round 2.
    assert_eq!(dep.signers[0].round3(&sid(1), &act, &[]), Err(InvocationError::WrongRound { expected: 2, found: 1 }));
    // The signer set is fixed in round 1.
    assert_eq!(dep.signers[0].round2(&sid(1), &other, &commitments), Err(InvocationError::ActiveSetMismatch));
    // Two messages from one party, or one from outside the set, are delivery faults.
    let mut duplicated = commitments.clone();
    duplicated[9] = duplicated[8].clone();
    assert_eq!(dep.signers[0].round2(&sid(1), &act, &duplicated), Err(InvocationError::MessageSetMismatch));
    let mut foreign = commitments.clone();
    foreign[9].0 = 19;
    assert_eq!(dep.signers[0].round2(&sid(1), &act, &foreign), Err(InvocationError::MessageSetMismatch));
    // The signer's own commitment must have been delivered.
    assert_eq!(dep.signers[0].round2(&sid(1), &act, &commitments[1..]), Err(InvocationError::OwnCommitmentMismatch));
    // Failed invocations leave the session where it was.
    assert_eq!(dep.signers[0].session_status(&sid(1)), SessionStatus::Running { round: 1 });

    // Messages may arrive in any order.
    let mut shuffled = commitments.clone();
    shuffled.reverse();
    let mut openings = Vec::new();
    for &party in act.parties() {
        openings.push((party, dep.signers[usize::from(party)].round2(&sid(1), &act, &shuffled).unwrap()));
    }
    assert_eq!(dep.signers[0].session_status(&sid(1)), SessionStatus::Running { round: 2 });
    assert_eq!(
        dep.signers[0].round2(&sid(1), &act, &commitments),
        Err(InvocationError::WrongRound { expected: 1, found: 2 })
    );
    // Own opening altered in transit: a delivery fault, not a blame.
    let mut altered = openings.clone();
    altered[0].1.w[0][0] ^= 1;
    assert_eq!(dep.signers[0].round3(&sid(1), &act, &altered), Err(InvocationError::OwnOpeningMismatch));
    assert_eq!(dep.signers[0].round3(&sid(1), &act, &openings[1..]), Err(InvocationError::OwnOpeningMismatch));
    assert_eq!(dep.signers[0].session_status(&sid(1)), SessionStatus::Running { round: 2 });

    assert!(matches!(dep.signers[0].round3(&sid(1), &act, &openings), Ok(Round3Output::Response(_))));
    // After round 3 the ephemeral state is gone and the identifier stays used.
    assert_eq!(dep.signers[0].session_status(&sid(1)), SessionStatus::Finished);
    assert_eq!(dep.signers[0].round3(&sid(1), &act, &openings), Err(InvocationError::UnknownSession));
    assert_eq!(dep.signers[0].round1(sid(1), &act, &m1, &mut rng), Err(InvocationError::SessionIdReused));

    // An explicitly abandoned session cannot be resumed or restarted.
    dep.signers[1].abort_session(&sid(1));
    assert_eq!(dep.signers[1].session_status(&sid(1)), SessionStatus::Finished);
    assert_eq!(dep.signers[1].round3(&sid(1), &act, &openings), Err(InvocationError::UnknownSession));
    assert_eq!(dep.signers[1].round1(sid(1), &act, &m1, &mut rng), Err(InvocationError::SessionIdReused));

    // Concurrent sessions with different identifiers are independent.
    let m2 = session_msg(&dep.scheme, 2, &act, b"a");
    let m3 = session_msg(&dep.scheme, 3, &act, b"a");
    let c2 = dep.signers[2].round1(sid(2), &act, &m2, &mut rng).unwrap();
    let c3 = dep.signers[2].round1(sid(3), &act, &m3, &mut rng).unwrap();
    assert_ne!(c2, c3, "fresh masks give different commitments");
    assert_eq!(dep.signers[2].participations(), 3);
}

#[test]
fn participation_cap_counts_every_started_session() {
    let mut profile = load_profile(SMALL);
    profile.id = "functional-n10-d16-cap2".into();
    profile.budgets.q_participate = 2;
    profile.refresh();
    let scheme = Arc::new(Scheme::new(profile).unwrap());
    let mut rng = SeededRandom::new(b"cap", "test");
    let mut dep = Deployment::keygen(scheme, &mut rng);
    let act = set(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

    assert!(matches!(dep.sign(sid(1), &act, b"one", &mut rng).unwrap().0, SessionOutcome::Signed { .. }));
    // A session that is started and abandoned consumes a unit too.
    let two = session_msg(&dep.scheme, 2, &act, b"two");
    let three = session_msg(&dep.scheme, 3, &act, b"three");
    dep.signers[0].round1(sid(2), &act, &two, &mut rng).unwrap();
    dep.signers[0].abort_session(&sid(2));
    assert_eq!(dep.signers[0].participations(), 2);
    assert_eq!(dep.signers[0].round1(sid(3), &act, &three, &mut rng), Err(InvocationError::ParticipationCapReached));
    // The refused invocation did not burn the identifier.
    assert_eq!(dep.signers[0].session_status(&sid(3)), SessionStatus::Unused);
    // Other signers still have one unit, then they are exhausted as well.
    let without_zero = set(&[1, 2, 3, 4, 5, 6, 7, 8, 9]);
    assert!(matches!(dep.sign(sid(4), &without_zero, b"four", &mut rng).unwrap().0, SessionOutcome::Signed { .. }));
    assert_eq!(
        dep.sign(sid(5), &without_zero, b"five", &mut rng).unwrap_err(),
        InvocationError::ParticipationCapReached
    );
}

#[test]
fn simulator_erases_started_sessions_on_an_early_error() {
    let (mut dep, mut rng) = deployment(SMALL, "cleanup");
    let without_zero = set(&[1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let all = set(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    assert!(matches!(dep.sign(sid(1), &without_zero, b"one", &mut rng).unwrap().0, SessionOutcome::Signed { .. }));

    // Party 0 starts Round 1 of a second session with the same identifier, then
    // party 1 refuses it.  The simulator must not leave party 0's mask behind.
    assert_eq!(dep.sign(sid(1), &all, b"again", &mut rng).unwrap_err(), InvocationError::SessionIdReused);
    assert_eq!(dep.signers[0].participations(), 1, "the started session counts");
    for signer in &dep.signers {
        assert_eq!(signer.session_status(&sid(1)), SessionStatus::Finished);
    }
    assert!(matches!(dep.sign(sid(2), &all, b"two", &mut rng).unwrap().0, SessionOutcome::Signed { .. }));
}

#[test]
fn verification_rejects_anything_but_the_signed_context() {
    let (mut dep, mut rng) = deployment(SMALL, "verify");
    let act = set(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let tr = honest_transcript(&mut dep, &mut rng, 1, &act, b"message");
    let sig = signature_of(&dep, &tr);
    let (scheme, vk) = (Arc::clone(&dep.scheme), Arc::clone(&dep.vk));
    assert!(verify(&scheme, &vk, &tr.msg, &sig));

    // Every part of the session message is bound: payload, identifier, signer
    // set and reconstruction vector.
    assert!(!verify(&scheme, &vk, &session_msg(&scheme, 1, &act, b"messagf"), &sig));
    assert!(!verify(&scheme, &vk, &session_msg(&scheme, 2, &act, b"message"), &sig));
    let other_set = set(&[0, 1, 2, 3, 4, 5, 6, 7, 8]);
    assert!(!verify(&scheme, &vk, &session_msg(&scheme, 1, &other_set, b"message"), &sig));
    let mut lambda = lambda_of(&scheme, &tr.msg);
    lambda[0] = if lambda[0] == 0 { 1 } else { 0 };
    assert!(!verify(&scheme, &vk, &session_msg_with(&scheme, 1, &act, &lambda, b"message"), &sig));
    assert!(!verify(&scheme, &vk, b"message", &sig), "a message that does not parse");
    let msg = tr.msg.clone();

    let mut bad = sig.clone();
    bad.z[0][0] += 1;
    assert!(!verify(&scheme, &vk, &msg, &bad));
    let edited = |edit: &dyn Fn(&mut Vec<i8>)| {
        let mut coeffs = sig.challenge.coeffs().to_vec();
        edit(&mut coeffs);
        Challenge::from_coeffs(coeffs).unwrap()
    };
    let first = sig.challenge.coeffs().iter().position(|&x| x != 0).unwrap();
    let mut bad = sig.clone();
    bad.challenge = edited(&|c| c[first] = -c[first]);
    assert!(!verify(&scheme, &vk, &msg, &bad), "flipped sign");
    let mut bad = sig.clone();
    bad.challenge = edited(&|c| c[first] = 0);
    assert!(!verify(&scheme, &vk, &msg, &bad), "wrong weight");
    let mut bad = sig.clone();
    bad.challenge = edited(&|c| {
        c.pop();
    });
    assert!(!verify(&scheme, &vk, &msg, &bad), "wrong degree");
    let mut bad = sig.clone();
    bad.z.pop();
    assert!(!verify(&scheme, &vk, &msg, &bad));

    // Adding a multiple of q keeps the linear relation and the hash but breaks the
    // norm bound.  The amount is chosen to exceed B2.
    let q = scheme.ring().q() as i128;
    let multiples = (scheme.b2() as i128) / q + 1;
    let mut bad = sig.clone();
    bad.z[2][9] += multiples * q;
    assert!(!verify(&scheme, &vk, &msg, &bad));

    // Another key, and another profile with the same key material.
    let (other, _) = deployment(SMALL, "verify-other-key");
    assert!(!verify(&scheme, &other.vk, &msg, &sig));
    let mut renamed = load_profile(SMALL);
    renamed.id = "functional-n10-d16-renamed".into();
    renamed.refresh();
    let renamed = Scheme::new(renamed).unwrap();
    let vk_bytes = vk.to_bytes(&scheme);
    assert!(VerificationKey::from_bytes(&renamed, &vk_bytes).is_err(), "keys are bound to their profile");
}

#[test]
fn signature_norm_is_far_below_the_bound() {
    let (mut dep, mut rng) = deployment(SMALL, "norm");
    let act = set(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let tr = honest_transcript(&mut dep, &mut rng, 1, &act, b"m");
    let sig = signature_of(&dep, &tr);
    let norm = latsec::scheme::norm_squared(&sig.z);
    assert!(&norm <= dep.scheme.b2_squared());
    // The combined response equals the integer combination of the partial ones.
    let items: Vec<&[Vec<i128>]> = tr.responses.iter().map(|r| r.z.as_slice()).collect();
    assert_eq!(sig.z, combine_z(&lambda_of(&dep.scheme, &tr.msg), &items));
}

#[test]
fn coordinator_supplied_reconstruction_vectors() {
    let (mut dep, mut rng) = deployment(SMALL, "supplied");
    let scheme = Arc::clone(&dep.scheme);
    let act = set(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let canonical = scheme.lambda(&act).unwrap().as_ref().clone();
    // Another vector within the bounds: search the ternary vectors of weight four.
    let mut alternatives = Vec::new();
    for support in subsets(10, 4) {
        for signs in 0..16u32 {
            let mut candidate = vec![0i64; 10];
            for (bit, &party) in support.iter().enumerate() {
                candidate[usize::from(party)] = if signs >> bit & 1 == 1 { -1 } else { 1 };
            }
            if candidate != canonical && scheme.check_supplied_lambda(&act, &candidate).is_ok() {
                alternatives.push(candidate);
            }
        }
    }
    assert!(!alternatives.is_empty(), "the full set has more than one short reconstruction vector");
    let alternative = alternatives[0].clone();
    assert_ne!(alternative, canonical);

    // All signers use the supplied vector: the session completes and verifies.
    let msg = session_msg_with(&scheme, 1, &act, &alternative, b"m");
    assert_eq!(admissible(&scheme, &sid(1), &act, &msg).unwrap().lambda, alternative);
    let mut commitments = Vec::new();
    for &p in act.parties() {
        commitments.push((p, dep.signers[usize::from(p)].round1(sid(1), &act, &msg, &mut rng).unwrap()));
    }
    let mut openings = Vec::new();
    for &p in act.parties() {
        openings.push((p, dep.signers[usize::from(p)].round2(&sid(1), &act, &commitments).unwrap()));
    }
    let mut responses = Vec::new();
    for &p in act.parties() {
        match dep.signers[usize::from(p)].round3(&sid(1), &act, &openings).unwrap() {
            Round3Output::Response(r) => responses.push(r),
            Round3Output::Abort { blamed } => panic!("aborted, blaming {blamed:?}"),
        }
    }
    let tr = Transcript {
        sid: sid(1),
        act: act.clone(),
        msg: msg.clone(),
        commitments: commitments.into_iter().map(|(_, c)| c).collect(),
        openings: openings.into_iter().map(|(_, o)| o).collect(),
        responses,
    };
    let sig = signature_of(&dep, &tr);
    assert!(verify(&scheme, &dep.vk, &msg, &sig));

    // Vectors that do not reconstruct, or break the bounds, are refused up front.
    let mut wrong = alternative.clone();
    wrong[0] += 1; // adds the nonzero row 0 of G to the combination
    let wrong_msg = session_msg_with(&scheme, 2, &act, &wrong, b"m");
    assert_eq!(
        dep.signers[0].round1(sid(2), &act, &wrong_msg, &mut rng),
        Err(InvocationError::Inadmissible(Inadmissible::NotAReconstructionVector))
    );
    assert_eq!(scheme.check_supplied_lambda(&act, &alternative[..9]), Err(Inadmissible::NotAReconstructionVector));
    // canonical + 2 * (alternative - canonical) is still a reconstruction vector
    // but has a larger l1 norm than the cap of 4.
    let stretched: Vec<i64> = canonical.iter().zip(&alternative).map(|(c, a)| 2 * a - c).collect();
    assert!(matches!(scheme.check_supplied_lambda(&act, &stretched), Err(Inadmissible::L1TooLarge { .. })));
    let stretched_msg = session_msg_with(&scheme, 2, &act, &stretched, b"m");
    assert!(matches!(admissible(&scheme, &sid(2), &act, &stretched_msg), Err(Inadmissible::L1TooLarge { .. })));

    // A signer that was handed a different session message computes another
    // challenge, and Combine (on the initiator's message) blames it.
    let canonical_msg = session_msg(&scheme, 3, &act, b"m");
    let other_msg = session_msg_with(&scheme, 3, &act, &alternative, b"m");
    let mut commitments = Vec::new();
    for &p in act.parties() {
        let msg = if p == 0 { &other_msg } else { &canonical_msg };
        commitments.push((p, dep.signers[usize::from(p)].round1(sid(3), &act, msg, &mut rng).unwrap()));
    }
    let mut openings = Vec::new();
    for &p in act.parties() {
        openings.push((p, dep.signers[usize::from(p)].round2(&sid(3), &act, &commitments).unwrap()));
    }
    let mut responses = Vec::new();
    for &p in act.parties() {
        match dep.signers[usize::from(p)].round3(&sid(3), &act, &openings).unwrap() {
            Round3Output::Response(r) => responses.push(r),
            Round3Output::Abort { blamed } => panic!("aborted, blaming {blamed:?}"),
        }
    }
    let tr = Transcript {
        sid: sid(3),
        act: act.clone(),
        msg: canonical_msg,
        commitments: commitments.into_iter().map(|(_, c)| c).collect(),
        openings: openings.into_iter().map(|(_, o)| o).collect(),
        responses,
    };
    assert_eq!(combine(&dep.scheme, &dep.vk, &dep.abort_keys, &tr).unwrap(), CombineOutput::Abort { blamed: vec![0] });
}

#[test]
fn encodings_round_trip_and_reject_malformed_input() {
    let (mut dep, mut rng) = deployment(SMALL, "codec");
    let scheme = Arc::clone(&dep.scheme);
    let act = set(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let tr = honest_transcript(&mut dep, &mut rng, 1, &act, b"m");
    let sig = signature_of(&dep, &tr);

    let vk_bytes = dep.vk.to_bytes(&scheme);
    assert_eq!(&VerificationKey::from_bytes(&scheme, &vk_bytes).unwrap(), dep.vk.as_ref());
    let ak_bytes = dep.abort_keys[3].to_bytes(&scheme);
    assert_eq!(AbortKey::from_bytes(&scheme, &ak_bytes).unwrap(), dep.abort_keys[3]);
    let share_bytes = dep.signers[3].share().to_bytes(&scheme).unwrap();
    assert_eq!(&SecretShare::from_bytes(&scheme, &share_bytes).unwrap(), dep.signers[3].share());
    let opening_bytes = tr.openings[0].to_bytes(&scheme);
    assert_eq!(Opening::from_bytes(&scheme, &opening_bytes).unwrap(), tr.openings[0]);
    let response_bytes = tr.responses[0].to_bytes(&scheme).unwrap();
    assert_eq!(PartialResponse::from_bytes(&scheme, &response_bytes).unwrap(), tr.responses[0]);
    let sig_bytes = sig.to_bytes(&scheme).unwrap();
    let decoded = Signature::from_bytes(&scheme, &sig_bytes).unwrap();
    assert_eq!(decoded, sig);
    assert!(verify(&scheme, &dep.vk, &tr.msg, &decoded));
    assert_eq!(ActiveSet::decode(&act.encode()).unwrap(), act);
    let message = SessionMessage::from_bytes(&scheme, &tr.msg).unwrap();
    assert_eq!(message.to_bytes(&scheme).unwrap(), tr.msg);
    assert_eq!((message.sid, &message.act, message.payload.as_slice()), (sid(1), &act, &b"m"[..]));

    // Sizes: header, then packed fields.  The signature has the Rice parameter
    // byte, the ranked challenge (b_c bits) and the Rice-coded response.
    let (d, p) = (16, 2);
    let omega_c = scheme.profile().challenge.omega_c as usize;
    assert_eq!(vk_bytes.len(), 2 + 64 + 32 + (p * d * 30usize).div_ceil(8));
    assert_eq!(ak_bytes.len(), 2 + 2 + (p * d * 30usize).div_ceil(8));
    let k = sig_bytes[2] as u32;
    let z_bits = codec::rice_length(sig.z.iter().flatten().map(|x| x.unsigned_abs()), k) as usize;
    let b_c = codec::challenge_rank_bits(d, omega_c) as usize + omega_c;
    assert_eq!(sig_bytes.len(), 3 + (b_c + z_bits).div_ceil(8));
    // A session message: header, sid, set, lambda at bitlen(2 * 4) = 4 bits each, payload.
    assert_eq!(tr.msg.len(), 2 + 32 + 2 + 2 * 10 + (10 * 4usize).div_ceil(8) + 1);

    // Truncation, extension, wrong type, wrong version.
    for bytes in [&vk_bytes, &ak_bytes, &share_bytes, &opening_bytes, &response_bytes, &sig_bytes] {
        let decode_all = |data: &[u8]| {
            [
                VerificationKey::from_bytes(&scheme, data).is_ok(),
                AbortKey::from_bytes(&scheme, data).is_ok(),
                SecretShare::from_bytes(&scheme, data).is_ok(),
                Opening::from_bytes(&scheme, data).is_ok(),
                PartialResponse::from_bytes(&scheme, data).is_ok(),
                Signature::from_bytes(&scheme, data).is_ok(),
            ]
        };
        assert_eq!(decode_all(bytes).iter().filter(|&&ok| ok).count(), 1, "exactly one type accepts an encoding");
        assert!(decode_all(&bytes[..bytes.len() - 1]).iter().all(|&ok| !ok), "truncated");
        let mut longer = bytes.to_vec();
        longer.push(0);
        assert!(decode_all(&longer).iter().all(|&ok| !ok), "trailing byte");
        for other in [2u8, 4] {
            let mut version = bytes.to_vec();
            version[1] = other;
            assert!(decode_all(&version).iter().all(|&ok| !ok), "codec version {other}");
        }
        assert!(decode_all(&[]).iter().all(|&ok| !ok));
    }

    // Out-of-range coefficients have no encoding and no decoding.
    let mut all_ones = sig_bytes.clone();
    for byte in all_ones[6..].iter_mut() {
        *byte = 0xff;
    }
    assert!(Signature::from_bytes(&scheme, &all_ones).is_err());
    // A challenge must be in C: weight omega_c.
    let mut heavier = sig.clone();
    let mut coeffs = heavier.challenge.coeffs().to_vec();
    let zero = coeffs.iter().position(|&x| x == 0).unwrap();
    coeffs[zero] = 1;
    heavier.challenge = Challenge::from_coeffs(coeffs).unwrap();
    assert!(heavier.to_bytes(&scheme).is_err());
    // Only the shortest Rice parameter is accepted for the response.
    for other_k in [k.wrapping_sub(1), k + 1] {
        if other_k > 64 {
            continue;
        }
        let mut w = BitWriter::with_header(tag::SIGNATURE);
        w.bytes(&[other_k as u8]);
        codec::write_challenge(&mut w, &sig.challenge);
        codec::write_rice(&mut w, &sig.z, other_k, scheme.b2()).unwrap();
        let noncanonical = w.finish();
        assert!(Signature::from_bytes(&scheme, &noncanonical).is_err(), "Rice parameter {other_k} instead of {k}");
    }
    // Session messages: lambda beyond its encoding bound, unknown parties, and a
    // malformed signer set are rejected; trailing bytes are payload.
    let mut too_big = message.clone();
    too_big.lambda[0] = 5;
    assert!(too_big.to_bytes(&scheme).is_err());
    let mut unknown = tr.msg.clone();
    unknown[34 + 2 + 2 * 9] = 10; // last party index
    assert!(SessionMessage::from_bytes(&scheme, &unknown).is_err());
    assert!(SessionMessage::from_bytes(&scheme, &tr.msg[..40]).is_err());
    let mut unreduced = opening_bytes.clone();
    for byte in unreduced[2..6].iter_mut() {
        *byte = 0xff;
    }
    assert!(Opening::from_bytes(&scheme, &unreduced).is_err());
    let mut oversized = sig.clone();
    oversized.z[0][0] = scheme.b2() as i128 + 1;
    assert!(oversized.to_bytes(&scheme).is_err());
    let mut bad_index = ak_bytes.clone();
    bad_index[2] = 10;
    assert!(AbortKey::from_bytes(&scheme, &bad_index).is_err());

    // Signer sets must be canonical.
    assert!(ActiveSet::decode(&[2, 0, 5, 0, 1, 0]).is_err(), "not increasing");
    assert!(ActiveSet::decode(&[2, 0, 1, 0, 1, 0]).is_err(), "duplicate");
    assert!(ActiveSet::decode(&[3, 0, 1, 0, 2, 0]).is_err(), "wrong count");
    assert!(ActiveSet::decode(&[0, 0]).is_err(), "empty");
}

#[test]
fn sec_reconstruction_is_exact_and_commutes_with_the_public_map() {
    let scheme = scheme(SMALL);
    let mut rng = SeededRandom::new(b"sec", "test");
    let (d, ell) = (scheme.ring().degree(), scheme.profile().ell);
    let secret: Vec<Vec<i128>> =
        (0..ell).map(|i| (0..d).map(|j| ((i * 31 + j * 7) as i128 % 23) - 11).collect()).collect();
    let shares = scheme.deal(&secret, &mut rng);
    assert_eq!(shares.len(), 10);
    let seed = [7u8; 32];
    let matrix = latsec::scheme::PublicMatrix::expand(scheme.ring(), scheme.profile().p, ell, &seed);
    let t = matrix.mul_z(scheme.ring(), &secret);

    let mut authorized = 0;
    for size in 4..=10 {
        for parties in subsets(10, size) {
            let act = set(&parties);
            let Ok(lambda) = scheme.recon_vector(&act) else { continue };
            authorized += 1;
            let picked: Vec<&[Vec<i128>]> = parties.iter().map(|&p| shares[usize::from(p)].as_slice()).collect();
            // Exact recovery over the integers.
            assert_eq!(combine_z(&lambda, &picked), secret);
            if size >= scheme.profile().sec.t_sig {
                assert_eq!(scheme.reconstruct(&act, &picked).unwrap(), secret);
            } else {
                assert!(matches!(scheme.reconstruct(&act, &picked), Err(Inadmissible::TooFewSigners { .. })));
            }
            // Reconstruction commutes with the public linear map A.
            let images: Vec<Vec<Vec<u128>>> = picked.iter().map(|s| matrix.mul_z(scheme.ring(), s)).collect();
            let image_refs: Vec<&[Vec<u128>]> = images.iter().map(|v| v.as_slice()).collect();
            assert_eq!(scheme.combine_q(&lambda, &image_refs), t);
        }
    }
    // All subsets of size >= 4: the exhaustive authorized count for this matrix.
    assert_eq!(authorized, 319);
    // A share of a party with g_i = 0 does not depend on the secret.
    let zero_party = scheme.profile().sec.g.iter().position(|row| row[0] == 0).unwrap();
    let mut rng_a = SeededRandom::new(b"same", "test");
    let mut rng_b = SeededRandom::new(b"same", "test");
    let other_secret: Vec<Vec<i128>> = secret.iter().map(|p| p.iter().map(|x| x + 5).collect()).collect();
    assert_eq!(scheme.deal(&secret, &mut rng_a)[zero_party], scheme.deal(&other_secret, &mut rng_b)[zero_party]);
}

#[test]
fn reconstruction_cache_is_bounded() {
    let scheme = scheme(SMALL);
    scheme.set_cache_capacity(3);
    let sets: Vec<ActiveSet> = subsets(10, 9).into_iter().map(|p| set(&p)).collect();
    for act in &sets {
        scheme.lambda(act).unwrap();
    }
    assert_eq!(scheme.cache_statistics(), (0, 10));
    // The three most recent sets are cached; the first one was evicted.
    scheme.lambda(&sets[9]).unwrap();
    scheme.lambda(&sets[0]).unwrap();
    assert_eq!(scheme.cache_statistics(), (1, 11));
    // Results do not depend on caching.
    scheme.set_cache_capacity(0);
    let fresh = scheme.lambda(&sets[4]).unwrap();
    scheme.set_cache_capacity(8);
    assert_eq!(scheme.lambda(&sets[4]).unwrap(), fresh);
    assert_eq!(scheme.lambda(&sets[4]).unwrap(), fresh);
}

#[test]
fn signers_reject_mismatched_shares() {
    let (dep, _) = deployment(SMALL, "mismatch");
    let (other, _) = deployment(MEDIUM, "mismatch");
    let share =
        SecretShare::from_bytes(&other.scheme, &other.signers[15].share().to_bytes(&other.scheme).unwrap()).unwrap();
    assert!(Signer::new(Arc::clone(&dep.scheme), Arc::clone(&dep.vk), share).is_err());
}

const TOO_LONG: &str = "encoding is longer than any valid one";

/// The reason a decoder gave for rejecting its input.
fn reason(result: latsec::error::Result<()>) -> String {
    match result {
        Err(latsec::error::Error::Decode(reason)) => reason,
        other => panic!("expected a decoding error, got {other:?}"),
    }
}

/// Decoders bound their work by the profile: an input of the wrong or of
/// excessive length is rejected before it is read, and every value that can be
/// valid still has an encoding within the limit.
#[test]
fn decoders_reject_inputs_beyond_the_profile_limits() {
    use latsec::latsec::limits;
    use num_bigint::BigUint;

    let (mut dep, mut rng) = deployment(SMALL, "limits");
    let scheme = Arc::clone(&dep.scheme);
    let act = set(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let tr = honest_transcript(&mut dep, &mut rng, 1, &act, b"payload");
    let sig = signature_of(&dep, &tr);
    let (d, ell) = (scheme.ring().degree(), scheme.profile().ell);
    let dim = d * ell;
    let constant = |value: i128| vec![vec![value; d]; ell];
    let spike = |value: i128| {
        let mut z = constant(0);
        z[0][0] = value;
        z
    };
    // floor(B / sqrt(D)): every coefficient equal maximizes the l1 norm, and with
    // it the unary part, among the vectors of norm at most B.
    let flat = |bound_squared: &BigUint| i128::try_from((bound_squared / BigUint::from(dim)).sqrt()).expect("fits");

    // Objects of one length.
    let fixed: [(Vec<u8>, usize); 3] = [
        (dep.vk.to_bytes(&scheme), limits::verification_key_len(&scheme)),
        (dep.abort_keys[3].to_bytes(&scheme), limits::abort_key_len(&scheme)),
        (tr.openings[0].to_bytes(&scheme), limits::opening_len(&scheme)),
    ];
    for (bytes, len) in &fixed {
        assert_eq!(bytes.len(), *len);
    }

    // Shares: honest ones, and the largest value a share can take.
    for (index, signer) in dep.signers.iter().enumerate() {
        let bytes = signer.share().to_bytes(&scheme).unwrap();
        let limit = limits::max_share_len(&scheme, index as PartyId);
        assert!(bytes.len() <= limit);
        let mut long = vec![0xffu8; limit + 1];
        long[..4].copy_from_slice(&bytes[..4]);
        assert_eq!(reason(SecretShare::from_bytes(&scheme, &long).map(|_| ())), TOO_LONG);
    }

    // Partial responses: honest ones and the extreme shapes of norm B1.
    let limit = limits::max_response_len(&scheme);
    let b1 = i128::try_from(scheme.b1()).unwrap();
    let mut shapes: Vec<Vec<Vec<i128>>> = tr.responses.iter().map(|r| r.z.clone()).collect();
    shapes.extend([constant(0), spike(b1), spike(-b1), constant(flat(scheme.b1_squared()))]);
    for z in shapes {
        let response = PartialResponse { z };
        let bytes = response.to_bytes(&scheme).unwrap();
        assert!(bytes.len() <= limit, "{} > {limit}", bytes.len());
        assert_eq!(PartialResponse::from_bytes(&scheme, &bytes).unwrap(), response);
    }
    let mut long = vec![0xffu8; limit + 1];
    long[..2].copy_from_slice(&tr.responses[0].to_bytes(&scheme).unwrap()[..2]);
    assert_eq!(reason(PartialResponse::from_bytes(&scheme, &long).map(|_| ())), TOO_LONG);

    // Signatures: the honest one and the extreme shapes of norm B2, each with the
    // parameter the encoder chooses.  The all-zero response has k = 0, which
    // stays a valid encoding.
    let limit = limits::max_signature_len(&scheme);
    let b2 = i128::try_from(scheme.b2()).unwrap();
    let mut parameters = Vec::new();
    for z in [sig.z.clone(), constant(0), constant(1), spike(b2), spike(-b2), constant(flat(scheme.b2_squared()))] {
        let signature = Signature { challenge: sig.challenge.clone(), z };
        let bytes = signature.to_bytes(&scheme).unwrap();
        assert!(bytes.len() <= limit, "{} > {limit}", bytes.len());
        assert_eq!(Signature::from_bytes(&scheme, &bytes).unwrap(), signature);
        parameters.push(bytes[2]);
    }
    assert_eq!(parameters[1], 0, "the all-zero response is coded with k = 0");
    assert!(parameters[5] > parameters[0], "a response at the bound takes a larger parameter");

    // k = 0 and a valid challenge, followed by an endless unary run.  The header,
    // the parameter byte and the challenge bits are those of the all-zero
    // signature; every later bit is one.
    let zero_bytes = Signature { challenge: sig.challenge.clone(), z: constant(0) }.to_bytes(&scheme).unwrap();
    let weight = scheme.profile().challenge.omega_c as usize;
    let challenge_bits = codec::challenge_rank_bits(d, weight) as usize + weight;
    let endless = |len: usize| {
        let whole = 3 + challenge_bits / 8;
        let mut run = vec![0xffu8; len];
        run[..whole].copy_from_slice(&zero_bytes[..whole]);
        if challenge_bits % 8 != 0 {
            run[whole] = zero_bytes[whole] | (0xff << (challenge_bits % 8));
        }
        run
    };
    let decode = |data: &[u8]| reason(Signature::from_bytes(&scheme, data).map(|_| ()));
    // Beyond the limit it is refused by its length, before anything is read.
    assert_eq!(decode(&endless(limit + 1)), TOO_LONG);
    assert_eq!(decode(&endless(100 * limit)), TOO_LONG);
    // At the limit the challenge is accepted and the run is followed, to the end
    // of the input and no further: the work is bounded by the limit.
    assert_eq!(decode(&endless(limit)), "encoding is too short");
    // The same prefix with the zero codewords in place decodes.
    assert!(Signature::from_bytes(&scheme, &zero_bytes).is_ok());

    // A parameter the input is too short for is rejected before any coefficient
    // is read: the challenge is valid, and the error is not one of the Rice
    // decoder.
    let sig_bytes = sig.to_bytes(&scheme).unwrap();
    for k in [sig_bytes[2] + 40, 126, 127, 255] {
        let mut other = sig_bytes.clone();
        other[2] = k;
        assert_eq!(decode(&other), "Rice parameter out of range for this encoding", "k = {k}");
    }

    // Session messages: the head has a fixed length; the payload limit is the
    // application's.
    assert_eq!(tr.msg.len(), limits::session_head_len(&scheme, act.len()) + b"payload".len());
    assert!(SessionMessage::from_bytes_with_limit(&scheme, &tr.msg, 7).is_ok());
    assert!(SessionMessage::from_bytes_with_limit(&scheme, &tr.msg, 6).is_err());
    assert!(SessionMessage::from_bytes(&scheme, &tr.msg[..tr.msg.len() - 8]).is_err(), "truncated head");
    let mut many = tr.msg.clone();
    many[34..36].copy_from_slice(&11u16.to_le_bytes());
    assert!(SessionMessage::from_bytes(&scheme, &many).is_err(), "more signers than parties");
}
