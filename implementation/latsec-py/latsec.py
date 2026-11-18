"""LaTSEC: Figure 1 of the paper.

    Setup / Keygen (trusted dealer)      keygen()
    ShareSign_1, ShareSign_2, ShareSign_3  Signer.share_sign_1 / _2 / _3
    PartialVerify                         partial_verify()
    Combine                               combine()
    Verify                                verify()
    P(tspar, vk, sid, act, msg)           admissible()

The signed message msg canonically encodes (sid, act, lambda_act) and the
application payload (Section 3.1): see encode_session_message().  Hashes take
a 64-byte key digest vkd = H(profile digest, seed, t) in place of vk, which
also binds every hash to the profile and its codec version.

Instantiation (Section 8.1):
  * H_cmt(sid, vk, act, i, w_i): SHAKE256 truncated to 2 kappa bits.
  * H_c(sid, vk, act, msg, w): a SHAKE256 stream feeding the sample-in-ball
    shuffle of ring.sample_in_ball, i.e. a random oracle into C directly.
  * A' is expanded from a 32-byte seed (ring.expand_a); Gaussians use the
    exact sampler of sampler.py; encodings are those of codec.py.

Missing and malformed peer messages (caption of Figure 1: "a missing or
unparsable peer message fails the check that uses it, so its sender is
blamed").  Round messages and transcripts are lists with one entry per signer,
in the order of act; None stands for a message that is missing or could not be
decoded.
  * A commitment is only used in Round 3, so ShareSign_2 stores whatever
    arrived and ShareSign_3 blames the first signer, in the order of act, whose
    commitment or opening is missing, malformed or inconsistent.
  * Combine blames the sender of every missing or malformed opening and of
    every response that fails PartialVerify.  With a malformed opening there
    is no aggregate w and no challenge; the blame set is then the senders of
    such openings and of responses that fail the checks needing no challenge
    (shape and |z_i|_2 <= B1).  PartialVerify makes those checks on the
    party's own messages before it looks at anyone else's.
  * A list without one entry per signer, and a signer's own message coming
    back altered, are malformed invocations: consistent delivery is assumed,
    and an honest signer must not blame itself.  (Appendix I of the full version has no such
    assertion in ShareSign_3.)

Integers: polynomials over R are lists of Python ints; over R_q, residues in
[0, q).  Party indices are zero-based (party i of the paper is index i - 1).
"""

import codec
from codec import BitReader, BitWriter, DecodeError
from ring import Ring, challenge_mul, expand_a, mul_a, sample_in_ball
from sampler import DiscreteGaussian
from sec import combine_q, combine_z, deal, is_reconstruction_vector, reconstruction_vector
from xof import Xof, DOMAIN_CHALLENGE, DOMAIN_COMMITMENT, DOMAIN_VERIFICATION_KEY, random_subset
from profiles import dyadic, l1_bound, linf_bound


class Inadmissible(Exception):
    """The session input fails P: the session is refused before Round 1 and
    nobody is blamed (Section 3.1)."""


class InvocationError(Exception):
    """A malformed invocation (Figure 1: a failed assertion returns bottom)."""


# --- Lengths of the encoded objects ------------------------------------------------
#
# Decoders reject an input of the wrong or of excessive length before reading
# it.  Keys and openings have one length.  For Golomb-Rice coded objects the
# limit is an upper bound on the encodings of the values that can be valid: a
# share has coefficients of absolute value at most beta_sh; a partial response
# that passes PartialVerify has |z_i|_2 <= B1; a signature that verifies has
# |z|_2 <= B2, and its parameter is the one giving the shortest code, so its
# length is at most the bound for any k.  A longer input encodes a value that
# would be rejected anyway.  The application payload of a session message is not
# covered: decode_session_message takes the application's limit.

HEADER_BYTES = 2


def bytes_of_bits(bits):
    return (bits + 7) // 8


def challenge_bits(prof):
    """b_c: the rank of the challenge support and its signs."""
    return codec.challenge_rank_bits(prof.d, prof.omega_c) + prof.omega_c


def residues_len(prof):
    return bytes_of_bits(prof.p * prof.d * codec.residue_width(prof.q))


def verification_key_len(prof):
    return HEADER_BYTES + 64 + 32 + residues_len(prof)


def abort_key_len(prof):
    return HEADER_BYTES + 2 + residues_len(prof)


def opening_len(prof):
    return HEADER_BYTES + residues_len(prof)


def session_head_len(prof, signers):
    """The length of a session message with `signers` signers and no payload."""
    width = codec.signed_width(linf_bound(prof.admissibility))
    return HEADER_BYTES + 32 + 2 + 2 * signers + bytes_of_bits(signers * width)


def max_share_len(prof, index):
    bits = codec.rice_max_bits_linf(prof.dim, prof.share_rice_parameter(index), prof.beta_sh)
    return HEADER_BYTES + 2 + bytes_of_bits(bits)


def max_response_len(prof):
    bits = codec.rice_max_bits_l2(prof.dim, prof.encoding["rice_response"], prof.b1 ** 2)
    return HEADER_BYTES + bytes_of_bits(bits)


def max_signature_len(prof):
    z_bits = min(codec.rice_max_bits_l2(prof.dim, k, prof.b2 ** 2) for k in range(127))
    return HEADER_BYTES + 1 + bytes_of_bits(z_bits + challenge_bits(prof))


def check_exact_len(data, length):
    if len(data) != length:
        raise DecodeError("encoding has the wrong length")


def check_max_len(data, length):
    if len(data) > length:
        raise DecodeError("encoding is longer than any valid one")


class Scheme:
    """A profile with its ring, samplers and reconstruction-vector cache."""

    def __init__(self, profile):
        self.profile = profile
        self.ring = Ring(profile.d, profile.q, profile.psi)
        self.gauss = {
            name: DiscreteGaussian(int(record["num"]), record["frac_bits"])
            for name, record in (("s", profile.sigma_s), ("R", profile.sigma_big_r), ("r", profile.sigma_r))
        }
        self._lambda_cache = {}

    def sample_poly(self, which, rng):
        return self.gauss[which].sample_poly(rng, self.profile.d)

    # -- SEC ------------------------------------------------------------------

    def recon_vector(self, act):
        """[46, Algorithm 1] for G_act, or None if act is not authorized."""
        g_t = [self.profile.g_rows[i] for i in act]
        return reconstruction_vector(g_t, *self.profile.recon_params)

    def check_bounds(self, lam):
        """|lambda|_inf <= beta_lambda and wt(lambda) <= omega_lambda (or the l1
        cap), and |lambda|_inf < q."""
        adm = self.profile.admissibility
        linf = max(abs(x) for x in lam)
        if linf >= self.profile.q:
            raise Inadmissible("coefficient reaches q")
        if adm["predicate"] == "paper":
            if linf > adm["beta_lambda"]:
                raise Inadmissible(f"|lambda|_inf = {linf} exceeds beta_lambda = {adm['beta_lambda']}")
            weight = sum(1 for x in lam if x)
            if weight > adm["omega_lambda"]:
                raise Inadmissible(f"wt(lambda) = {weight} exceeds omega_lambda = {adm['omega_lambda']}")
        elif sum(abs(x) for x in lam) > adm["l1_cap"]:
            raise Inadmissible("|lambda|_1 exceeds the cap")

    def check_parties(self, act):
        if any(not 0 <= i < self.profile.n for i in act):
            raise Inadmissible("unknown party")
        if len(act) < self.profile.t_sig:
            raise Inadmissible(f"{len(act)} signers, {self.profile.t_sig} required")

    def initiator_lambda(self, act):
        """The vector an initiator puts in the session message: Algorithm 1,
        provided the set is admissible."""
        act = tuple(act)
        if act not in self._lambda_cache:
            try:
                self.check_parties(act)
                lam = self.recon_vector(act)
                if lam is None:
                    raise Inadmissible("the set cannot reconstruct")
                self.check_bounds(lam)
                self._lambda_cache[act] = lam
            except Inadmissible as e:
                self._lambda_cache[act] = e
        result = self._lambda_cache[act]
        if isinstance(result, Inadmissible):
            raise result
        return result

    def check_supplied_lambda(self, act, lam):
        """Section 2.4, Item 3: exact checks of a supplied vector."""
        self.check_parties(act)
        if not is_reconstruction_vector(self.profile.g_rows, act, lam):
            raise Inadmissible("the vector does not reconstruct")
        self.check_bounds(lam)


# --- Keys ---------------------------------------------------------------------------

class VerificationKey:
    """vk = (A, t), with A = [I_p | A'] given by its seed."""

    def __init__(self, scheme, seed, t):
        prof = scheme.profile
        self.seed, self.t = bytes(seed), t
        self.a_hat = expand_a(scheme.ring, prof.p, prof.ell, self.seed)
        x = Xof(DOMAIN_VERIFICATION_KEY).absorb(prof.digest).absorb(self.seed)
        self.digest = x.absorb(codec.residues_to_bytes(t, prof.q)).digest(64)

    def encode(self, scheme):
        w = BitWriter(codec.TAG_VERIFICATION_KEY)
        w.bytes(scheme.profile.digest)
        w.bytes(self.seed)
        codec.write_residues(w, self.t, scheme.profile.q)
        return w.finish()

    @staticmethod
    def decode(scheme, data):
        prof = scheme.profile
        check_exact_len(data, verification_key_len(prof))
        r = BitReader(data, codec.TAG_VERIFICATION_KEY)
        if r.bytes(64) != prof.digest:
            raise DecodeError("verification key belongs to another profile")
        seed = r.bytes(32)
        t = codec.read_residues(r, prof.p, prof.d, prof.q)
        r.finish()
        return VerificationKey(scheme, seed, t)


def encode_abort_key(scheme, index, t_i):
    w = BitWriter(codec.TAG_ABORT_KEY)
    w.bytes(index.to_bytes(2, "little"))
    codec.write_residues(w, t_i, scheme.profile.q)
    return w.finish()


def decode_abort_key(scheme, data):
    prof = scheme.profile
    check_exact_len(data, abort_key_len(prof))
    r = BitReader(data, codec.TAG_ABORT_KEY)
    index = int.from_bytes(r.bytes(2), "little")
    if index >= prof.n:
        raise DecodeError("abort key for an unknown party")
    t_i = codec.read_residues(r, prof.p, prof.d, prof.q)
    r.finish()
    return index, t_i


def encode_share(scheme, index, s_i):
    prof = scheme.profile
    w = BitWriter(codec.TAG_SECRET_SHARE)
    w.bytes(index.to_bytes(2, "little"))
    codec.write_rice(w, s_i, prof.share_rice_parameter(index), prof.beta_sh)
    return w.finish()


def decode_share(scheme, data):
    prof = scheme.profile
    r = BitReader(data, codec.TAG_SECRET_SHARE)
    index = int.from_bytes(r.bytes(2), "little")
    if index >= prof.n:
        raise DecodeError("share for an unknown party")
    check_max_len(data, max_share_len(prof, index))
    s_i = codec.read_rice(r, prof.ell, prof.d, prof.share_rice_parameter(index), prof.beta_sh)
    r.finish()
    return index, s_i


def keygen(scheme, rng):
    """Keygen (Figure 1, trusted dealer):  A' <- R_q^(p x (ell-p)) from a seed,
    s <- D_{sigma_s}^ell, (s_i) <- SEC.Deal(s), t = A s, t_i = A s_i.

    Returns (vk, abort_keys, shares) with abort_keys[i] = t_i and shares[i] = s_i.
    """
    prof, ring = scheme.profile, scheme.ring
    seed = rng.read(32)
    a_hat = expand_a(ring, prof.p, prof.ell, seed)
    s = [scheme.sample_poly("s", rng) for _ in range(prof.ell)]
    shares = deal(prof.g_rows, s, lambda rng: scheme.sample_poly("R", rng), rng)
    t = mul_a(ring, a_hat, s)
    abort_keys = [mul_a(ring, a_hat, s_i) for s_i in shares]
    return VerificationKey(scheme, seed, t), abort_keys, shares


# --- The session message and the predicate P ----------------------------------------

def encode_session_message(scheme, sid, act, lam, payload):
    """msg: header, the 32-byte sid, the signer set, lambda_act (x + B in
    bitlen(2B) bits each, B the predicate's coefficient bound), zero padding to
    a byte, then the payload, which runs to the end."""
    if len(lam) != len(act):
        raise ValueError("lambda_act does not match the signer set")
    w = BitWriter(codec.TAG_SESSION_MESSAGE)
    w.bytes(sid)
    w.bytes(codec.encode_active_set(act))
    codec.write_bounded(w, lam, linf_bound(scheme.profile.admissibility))
    return w.finish() + bytes(payload)


def decode_session_message(scheme, msg, max_payload=None):
    """(sid, act, lambda_act, payload) from msg; raises DecodeError.

    Everything before the payload has a length fixed by the profile and the
    number of signers (session_head_len).  The payload runs to the end of msg
    and has no length field, so its size is the application's to bound: with
    `max_payload`, a longer payload is rejected before anything else is read."""
    r = BitReader(msg, codec.TAG_SESSION_MESSAGE)
    sid = r.bytes(32)
    count = int.from_bytes(r.bytes(2), "little")
    if count == 0 or count > scheme.profile.n:
        raise DecodeError("signer set is empty or larger than the number of parties")
    head = session_head_len(scheme.profile, count)
    if len(msg) < head:
        raise DecodeError("encoding is too short")
    if max_payload is not None and len(msg) - head > max_payload:
        raise DecodeError("payload exceeds the application's limit")
    act = tuple(int.from_bytes(r.bytes(2), "little") for _ in range(count))
    if count == 0 or any(a >= b for a, b in zip(act, act[1:])):
        raise DecodeError("signer set is empty or not strictly increasing")
    if any(i >= scheme.profile.n for i in act):
        raise DecodeError("signer set names an unknown party")
    lam = codec.read_bounded(r, count, linf_bound(scheme.profile.admissibility))
    end = r.finish_prefix()
    return sid, act, lam, bytes(msg[end:])


def admissible(scheme, sid, act, msg):
    """P(tspar, vk, sid, act, msg) of Figure 1.  Returns lambda_act or raises
    Inadmissible.

    1: parse (sid', act', lambda_act) from msg; fail if (sid', act') != (sid, act)
    2: fail if |act| < t_sig
    3: accept iff lambda_act^T G_act = e_1^T, |lambda_act|_inf <= beta_lambda and
       wt(lambda_act) <= omega_lambda
    """
    try:
        sid2, act2, lam, _ = decode_session_message(scheme, msg)
    except DecodeError as e:
        raise Inadmissible(f"the session message does not parse: {e}")
    if (sid2, act2) != (bytes(sid), tuple(act)):
        raise Inadmissible("the session message names another session or signer set")
    scheme.check_supplied_lambda(act, lam)
    return lam


# --- Hashes -------------------------------------------------------------------------

def commitment_hash(scheme, sid, vk, act, i, w_i):
    """cmt_i = H_cmt(sid, vk, act, i, w_i), 2 kappa bits."""
    x = Xof(DOMAIN_COMMITMENT).absorb(sid).absorb(vk.digest).absorb(codec.encode_active_set(act))
    x.absorb_u64(i).absorb(codec.residues_to_bytes(w_i, scheme.profile.q))
    return x.digest(scheme.profile.commitment_bytes)


def hash_to_challenge(scheme, sid, vk, act, msg, w):
    """c = H_c(sid, vk, act, msg, w) in C."""
    prof = scheme.profile
    x = Xof(DOMAIN_CHALLENGE).absorb(sid).absorb(vk.digest).absorb(codec.encode_active_set(act)).absorb(msg)
    x.absorb(codec.residues_to_bytes(w, prof.q)).absorb_u64(prof.d).absorb_u64(prof.omega_c)
    return sample_in_ball(x.stream(), prof.d, prof.omega_c)


def session_challenge(scheme, vk, sid, act, msg, lam, openings):
    """w = SEC.Rec_pp(act, (w_j)) = sum_j lambda_j w_j mod q, and c = H_c(..., w)."""
    w = combine_q(lam, openings, scheme.profile.q)
    return hash_to_challenge(scheme, sid, vk, act, msg, w)


# --- Signing ------------------------------------------------------------------------

class Signer:
    """Party i: its share s_i, its sessions, used identifiers, and a counter of
    the sessions it took part in."""

    def __init__(self, scheme, vk, index, share):
        self.scheme, self.vk, self.index, self.share = scheme, vk, index, share
        self.sessions = {}
        self.used = set()
        self.participations = 0

    def share_sign_1(self, sid, act, msg, rng):
        """ShareSign_1: assert sid is fresh, i in act and P(...) = 1;
        r_i <- D_{sigma_r}^ell; w_i = A r_i; return cmt_i = H_cmt(sid, vk, act, i, w_i)."""
        prof = self.scheme.profile
        if sid in self.used:
            raise InvocationError("session identifier reused")
        if self.index not in act:
            raise InvocationError("not in the signer set")
        lam = admissible(self.scheme, sid, act, msg)
        if self.participations >= prof.q_participate:
            raise InvocationError("participation budget spent")
        self.participations += 1
        self.used.add(sid)
        r_i = [self.scheme.sample_poly("r", rng) for _ in range(prof.ell)]
        w_i = mul_a(self.scheme.ring, self.vk.a_hat, r_i)
        cmt_i = commitment_hash(self.scheme, sid, self.vk, act, self.index, w_i)
        self.sessions[sid] = {"round": 1, "act": tuple(act), "msg": bytes(msg), "lambda": lam,
                              "r": r_i, "w": w_i, "cmt": cmt_i}
        return cmt_i

    def share_sign_2(self, sid, act, commitments):
        """ShareSign_2: assert round 1 and that its own entry equals cmt_i;
        store (cmt_j); return w_i.  `commitments` lists cmt_j in the order of
        act, None for a missing one; the others are checked in Round 3."""
        st = self._running(sid, act, 1)
        one_per_signer(act, commitments)
        if commitments[act.index(self.index)] != st["cmt"]:
            raise InvocationError("own commitment not delivered intact")
        st["commitments"], st["round"] = list(commitments), 2
        return st["w"]

    def share_sign_3(self, sid, act, openings):
        """ShareSign_3: check every opening against its commitment in the
        canonical order of act, aborting with the first failure; otherwise
        w = sum_j lambda_j w_j, c = H_c(sid, vk, act, msg, w), z_i = c s_i + r_i.

        `openings` lists w_j in the order of act, None for a missing one.  A
        signer whose commitment or opening is missing or malformed fails the
        check.  The session state, with its mask, is dropped either way.

        Returns ("response", z_i) or ("abort", [j])."""
        st = self._running(sid, act, 2)
        one_per_signer(act, openings)
        if openings[act.index(self.index)] != st["w"]:
            raise InvocationError("own opening not delivered intact")
        del self.sessions[sid]
        for j, w_j, cmt_j in zip(act, openings, st["commitments"]):
            if (cmt_j is None or not well_formed_w(self.scheme, w_j)
                    or commitment_hash(self.scheme, sid, self.vk, act, j, w_j) != cmt_j):
                return ("abort", [j])
        c = session_challenge(self.scheme, self.vk, sid, act, st["msg"], st["lambda"], openings)
        z_i = [[x + y for x, y in zip(challenge_mul(c, s), r)] for s, r in zip(self.share, st["r"])]
        return ("response", z_i)

    def abort_session(self, sid):
        """Drop the ephemeral state of a session.  The identifier stays used.
        The mask is only dereferenced: Python integers are not zeroized."""
        self.sessions.pop(sid, None)

    def _running(self, sid, act, expected_round):
        st = self.sessions.get(sid)
        if st is None:
            raise InvocationError("unknown session")
        if st["round"] != expected_round or st["act"] != tuple(act):
            raise InvocationError("wrong round or signer set")
        return st


# --- Verification ---------------------------------------------------------------------

def norm_squared(v):
    return sum(x * x for poly in v for x in poly)


def response_image(scheme, vk, z, c, t):
    """A z - c t mod q."""
    ring = scheme.ring
    az = mul_a(ring, vk.a_hat, z)
    return [ring.sub(a, challenge_mul(c, tp, ring.q)) for a, tp in zip(az, t)]


def one_per_signer(act, messages):
    """A list of round messages or transcript entries has one slot per signer."""
    if len(messages) != len(act):
        raise InvocationError("the messages do not correspond to the signer set")


def is_poly_vector(v, count, d):
    return (isinstance(v, (list, tuple)) and len(v) == count
            and all(isinstance(p, (list, tuple)) and len(p) == d and all(type(x) is int for x in p) for p in v))


def well_formed_z(scheme, z):
    """z is in R^ell; None (a missing response) is not."""
    return is_poly_vector(z, scheme.profile.ell, scheme.profile.d)


def well_formed_w(scheme, w):
    """w is in R_q^p with reduced coefficients; None (a missing opening) is not."""
    prof = scheme.profile
    return is_poly_vector(w, prof.p, prof.d) and all(0 <= x < prof.q for poly in w for x in poly)


def response_in_bounds(scheme, z_i):
    """The checks of PartialVerify that need no challenge: the shape of z_i and
    |z_i|_2 <= B1."""
    return well_formed_z(scheme, z_i) and norm_squared(z_i) <= scheme.profile.b1 ** 2


def partial_verify_with(scheme, vk, t_i, c, w_i, z_i):
    """1 iff A z_i - c t_i = w_i and |z_i|_2 <= B1."""
    if not response_in_bounds(scheme, z_i):
        return False
    return response_image(scheme, vk, z_i, c, t_i) == w_i


def partial_verify(scheme, vk, abort_keys, sid, act, msg, transcript, i):
    """PartialVerify(tspar, vk, ak_i, sid, act, msg, PM).  `transcript` is the
    list of (cmt_j, w_j, z_j) in the order of act, with None for a missing or
    undecodable w_j or z_j.

    The checks on party i's own messages that need no challenge come first:
    a missing or malformed opening, or a response that is missing, malformed
    or longer than B1, fails whatever the other signers sent.  Only then, if
    the opening of another signer is malformed, the challenge is undefined and
    the call is a malformed invocation; this is not a verdict on party i."""
    lam = admissible(scheme, sid, act, msg)
    one_per_signer(act, transcript)
    openings = [w for _, w, _ in transcript]
    _, w_i, z_i = transcript[act.index(i)]
    if not well_formed_w(scheme, w_i) or not response_in_bounds(scheme, z_i):
        return False
    for j, w_j in zip(act, openings):
        if not well_formed_w(scheme, w_j):
            raise InvocationError(f"the opening of party {j} is missing or malformed")
    c = session_challenge(scheme, vk, sid, act, msg, lam, openings)
    return partial_verify_with(scheme, vk, abort_keys[i], c, w_i, z_i)


def combine(scheme, vk, abort_keys, sid, act, msg, transcript):
    """Combine: S = {i : PartialVerify fails}; abort with S if nonempty,
    otherwise sig = (c, z) with z = SEC.Rec_pp(act, (z_j)).

    `transcript` has one entry (cmt_j, w_j, z_j) per signer, as for
    partial_verify.  With a missing or malformed opening there is no
    challenge; S is then the senders of such openings and of responses that
    are malformed or longer than B1."""
    lam = admissible(scheme, sid, act, msg)
    one_per_signer(act, transcript)
    openings = [w for _, w, _ in transcript]
    if not all(well_formed_w(scheme, w) for w in openings):
        return ("abort", [i for i, (_, w_i, z_i) in zip(act, transcript)
                          if not well_formed_w(scheme, w_i) or not response_in_bounds(scheme, z_i)])
    c = session_challenge(scheme, vk, sid, act, msg, lam, openings)
    blamed = [i for i, (_, w_i, z_i) in zip(act, transcript)
              if not partial_verify_with(scheme, vk, abort_keys[i], c, w_i, z_i)]
    if blamed:
        return ("abort", blamed)
    return ("signature", (c, combine_z(lam, [z for _, _, z in transcript])))


def verify(scheme, vk, msg, sig):
    """Verify(vk, msg, sig): parse (sid, act) from msg; w = A z - c t; accept iff
    c = H_c(sid, vk, act, msg, w) and |z|_2 <= B2."""
    try:
        sid, act, _, _ = decode_session_message(scheme, msg)
    except DecodeError:
        return False
    c, z = sig
    prof = scheme.profile
    if len(c) != prof.d or sum(1 for x in c if x) != prof.omega_c or not well_formed_z(scheme, z):
        return False
    if norm_squared(z) > prof.b2 ** 2:
        return False
    w = response_image(scheme, vk, z, c, vk.t)
    return hash_to_challenge(scheme, sid, vk, act, msg, w) == c


# --- Encodings of the round messages and the signature ---------------------------------

def encode_opening(scheme, w_i):
    w = BitWriter(codec.TAG_OPENING)
    codec.write_residues(w, w_i, scheme.profile.q)
    return w.finish()


def encode_response(scheme, z_i):
    prof = scheme.profile
    w = BitWriter(codec.TAG_PARTIAL_RESPONSE)
    codec.write_rice(w, z_i, prof.encoding["rice_response"], prof.b1)
    return w.finish()


def decode_opening(scheme, data):
    prof = scheme.profile
    check_exact_len(data, opening_len(prof))
    r = BitReader(data, codec.TAG_OPENING)
    w_i = codec.read_residues(r, prof.p, prof.d, prof.q)
    r.finish()
    return w_i


def decode_response(scheme, data):
    """Rejects an input longer than max_response_len before reading it."""
    prof = scheme.profile
    check_max_len(data, max_response_len(prof))
    r = BitReader(data, codec.TAG_PARTIAL_RESPONSE)
    z = codec.read_rice(r, prof.ell, prof.d, prof.encoding["rice_response"], prof.b1)
    r.finish()
    return z


def encode_signature(scheme, sig):
    """Header, the Rice parameter k of z (one byte), the ranked challenge, then
    z Rice-coded with k.  k is the shortest-code parameter (smallest on ties)."""
    c, z = sig
    prof = scheme.profile
    if len(c) != prof.d or sum(1 for x in c if x) != prof.omega_c:
        raise ValueError("challenge is not in C")
    k = codec.shortest_rice(z)
    w = BitWriter(codec.TAG_SIGNATURE)
    w.bytes([k])
    codec.write_challenge(w, c)
    codec.write_rice(w, z, k, prof.b2)
    return w.finish()


def decode_signature(scheme, data):
    """Rejects a parameter k other than the one encode_signature chooses.

    Two checks come before any coefficient is read: the input is no longer
    than max_signature_len, and it is long enough for its k (every codeword
    has at least k + 1 bits).  Together they bound the work of the decoder,
    including the unary runs, by the profile; k = 0, the canonical parameter
    of a short enough response, stays valid."""
    prof = scheme.profile
    check_max_len(data, max_signature_len(prof))
    r = BitReader(data, codec.TAG_SIGNATURE)
    k = r.bytes(1)[0]
    if k >= 127 or challenge_bits(prof) + prof.dim * (k + 1) > 8 * (len(data) - 3):
        raise DecodeError("Rice parameter out of range for this encoding")
    c = codec.read_challenge(r, prof.d, prof.omega_c)
    z = codec.read_rice(r, prof.ell, prof.d, k, prof.b2)
    r.finish()
    if not codec.is_shortest_rice(z, k):
        raise DecodeError("Rice parameter is not the canonical one")
    return c, z


# --- A whole session -----------------------------------------------------------------

def sign_session(scheme, vk, abort_keys, signers, sid, act, payload, rng):
    """An all-honest session, as the in-process simulator of the Rust crate runs
    it: the initiator builds msg with Algorithm 1, then every signer runs the
    three rounds in the order of act, and the combiner runs Combine.

    A malformed invocation in any round ends the session for every signer in
    act: the sessions already started are dropped, with their masks.

    Returns (msg, transcript, outcome) with outcome ("signature", sig) or
    ("abort", blamed)."""
    lam = scheme.initiator_lambda(act)
    msg = encode_session_message(scheme, sid, act, lam, payload)
    try:
        commitments = [signers[i].share_sign_1(sid, act, msg, rng) for i in act]
        openings = [signers[i].share_sign_2(sid, act, commitments) for i in act]
        responses, blamed = [], []
        for i in act:
            kind, value = signers[i].share_sign_3(sid, act, openings)
            if kind == "abort":
                blamed.extend(value)
                responses.append(None)
            else:
                responses.append(value)
    except (InvocationError, Inadmissible):
        for i in act:
            signers[i].abort_session(sid)
        raise
    transcript = list(zip(commitments, openings, responses))
    if blamed:
        return msg, transcript, ("abort", sorted(set(blamed)))
    return msg, transcript, combine(scheme, vk, abort_keys, sid, act, msg, transcript)


def first_admissible_set(scheme, rng, size, tries=10000):
    """Draw signer sets (random_subset) until the initiator finds one admissible."""
    for _ in range(tries):
        act = tuple(random_subset(rng, scheme.profile.n, size))
        try:
            scheme.initiator_lambda(act)
            return act
        except Inadmissible:
            continue
    raise Inadmissible(f"no admissible set of size {size} found")
