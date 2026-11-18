//! A local, in-process deployment: trusted key generation and a transcript
//! dispatcher that delivers every round message to every signer.
//!
//! This implements the paper's communication assumption (authenticated broadcast
//! with consistent delivery) by construction.  It is not a network transport:
//! equivocation, missing messages and timeouts cannot occur here and are not
//! handled.

use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::latsec::{
    combine, keygen, verify, AbortKey, CombineOutput, Commitment, InvocationError, Opening, PartialResponse,
    Round3Output, SessionId, SessionMessage, Signature, Signer, Transcript, VerificationKey,
};
use crate::sampling::RandomSource;
use crate::scheme::{ActiveSet, Inadmissible, PartyId, Scheme};

/// All parties of one key, in one process.
pub struct Deployment {
    pub scheme: Arc<Scheme>,
    pub vk: Arc<VerificationKey>,
    pub abort_keys: Vec<AbortKey>,
    pub signers: Vec<Signer>,
}

/// How a session ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SessionOutcome {
    /// All partial responses verified and the signature verifies.
    Signed { signature: Signature, transcript: Transcript },
    /// An identifiable abort, with the round in which it was raised (3 for a
    /// commitment mismatch, 4 for a failed partial verification in `Combine`).
    Aborted { blamed: Vec<PartyId>, stage: u8 },
}

/// Wall-clock time spent in each phase of a session, summed over signers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SessionTiming {
    /// Computing or fetching the reconstruction vector (the initiator's work).
    pub recon: Duration,
    pub round1: Duration,
    pub round2: Duration,
    pub round3: Duration,
    pub combine: Duration,
    pub verify: Duration,
}

impl Deployment {
    /// Runs trusted key generation and creates one signer per party.
    ///
    /// # Panics
    ///
    /// Panics if the profile does not support signing (a SEC-only profile).
    pub fn keygen(scheme: Arc<Scheme>, rng: &mut dyn RandomSource) -> Self {
        let material = keygen(&scheme, rng);
        let vk = Arc::new(material.vk);
        let signers = material
            .shares
            .into_iter()
            .map(|share| {
                Signer::new(Arc::clone(&scheme), Arc::clone(&vk), share).expect("dealt shares match the profile")
            })
            .collect();
        Deployment { scheme, vk, abort_keys: material.abort_keys, signers }
    }

    /// Runs one all-honest signing session for `payload` with the signers in
    /// `act`, acting as the session initiator: builds the [`SessionMessage`] with
    /// this crate's reconstruction vector, then calls [`Deployment::sign_message`].
    ///
    /// An inadmissible signer set is reported as an [`InvocationError`] before any
    /// signer samples a mask; it is not an abort and blames nobody.
    pub fn sign(
        &mut self,
        sid: SessionId,
        act: &ActiveSet,
        payload: &[u8],
        rng: &mut dyn RandomSource,
    ) -> Result<(SessionOutcome, SessionTiming), InvocationError> {
        let started = Instant::now();
        let message = SessionMessage::new(&self.scheme, sid, act, payload)?;
        let recon = started.elapsed();
        let msg = message
            .to_bytes(&self.scheme)
            .map_err(|_| InvocationError::Inadmissible(Inadmissible::MalformedMessage))?;
        let (outcome, mut timing) = self.sign_message(sid, act, &msg, rng)?;
        timing.recon = recon;
        Ok((outcome, timing))
    }

    /// Runs one all-honest signing session for an encoded session message.
    ///
    /// Each signer evaluates the admissibility predicate in Round 1.  Any
    /// invocation error ends the session for every signer in `act`: the sessions
    /// already started are erased, together with their masks.
    pub fn sign_message(
        &mut self,
        sid: SessionId,
        act: &ActiveSet,
        msg: &[u8],
        rng: &mut dyn RandomSource,
    ) -> Result<(SessionOutcome, SessionTiming), InvocationError> {
        let result = self.run(sid, act, msg, rng);
        if result.is_err() {
            for &party in act.parties() {
                if let Some(signer) = self.signers.get_mut(usize::from(party)) {
                    signer.abort_session(&sid);
                }
            }
        }
        result
    }

    fn run(
        &mut self,
        sid: SessionId,
        act: &ActiveSet,
        msg: &[u8],
        rng: &mut dyn RandomSource,
    ) -> Result<(SessionOutcome, SessionTiming), InvocationError> {
        let mut timing = SessionTiming::default();
        let started = Instant::now();
        let mut commitments: Vec<(PartyId, Commitment)> = Vec::with_capacity(act.len());
        for &party in act.parties() {
            commitments.push((party, self.signers[usize::from(party)].round1(sid, act, msg, rng)?));
        }
        timing.round1 = started.elapsed();

        let started = Instant::now();
        let mut openings: Vec<(PartyId, Opening)> = Vec::with_capacity(act.len());
        for &party in act.parties() {
            openings.push((party, self.signers[usize::from(party)].round2(&sid, act, &commitments)?));
        }
        timing.round2 = started.elapsed();

        let started = Instant::now();
        let mut responses: Vec<PartialResponse> = Vec::with_capacity(act.len());
        let mut blamed: Vec<PartyId> = Vec::new();
        for &party in act.parties() {
            match self.signers[usize::from(party)].round3(&sid, act, &openings)? {
                Round3Output::Response(response) => responses.push(response),
                Round3Output::Abort { blamed: parties } => blamed.extend(parties),
            }
        }
        timing.round3 = started.elapsed();
        if !blamed.is_empty() {
            blamed.sort_unstable();
            blamed.dedup();
            return Ok((SessionOutcome::Aborted { blamed, stage: 3 }, timing));
        }

        let transcript = Transcript {
            sid,
            act: act.clone(),
            msg: msg.to_vec(),
            commitments: commitments.into_iter().map(|(_, c)| c).collect(),
            openings: openings.into_iter().map(|(_, o)| o).collect(),
            responses,
        };
        let started = Instant::now();
        let combined = combine(&self.scheme, &self.vk, &self.abort_keys, &transcript)?;
        timing.combine = started.elapsed();
        match combined {
            CombineOutput::Abort { blamed } => Ok((SessionOutcome::Aborted { blamed, stage: 4 }, timing)),
            CombineOutput::Signature(signature) => {
                let started = Instant::now();
                let ok = verify(&self.scheme, &self.vk, msg, &signature);
                timing.verify = started.elapsed();
                assert!(ok, "a signature that passed all partial checks must verify");
                Ok((SessionOutcome::Signed { signature, transcript }, timing))
            }
        }
    }
}
