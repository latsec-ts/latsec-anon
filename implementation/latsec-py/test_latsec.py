"""Unit tests of the Python implementation.  Run: python3 -m unittest -v test_latsec

Expected values come from hand computation (codec), from the paper (b_c of
Section 7.3, the challenge weights), or from the SageMath pipeline of [46]
(fixtures/sage, written by tools/gen_recon_fixtures.py).  Interoperability
with the Rust implementation is tested by vectors.py --verify.
"""

import json
import os
import unittest
from unittest import mock
from fractions import Fraction

import codec
import latsec
import sec
from profiles import Profile, rice_parameter
from xof import SeededRandom

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fixture(*parts):
    with open(os.path.join(REPO, "fixtures", *parts)) as f:
        return json.load(f)


def profile(pid):
    return Profile.load(os.path.join(REPO, "profiles", pid + ".json"))


class CodecTest(unittest.TestCase):
    def test_rice_codewords_are_little_endian(self):
        # -5 with k = 1: low bit 1, unary 1 1 0, sign 1 -> bits 1 1 1 0 1 -> 0x17.
        w = codec.BitWriter()
        codec.write_rice(w, [[-5]], 1, 100)
        self.assertEqual(w.finish(), bytes([0x17]))
        # 0 with k = 2: 0 0 | 0.  6 with k = 2: 0 1 | 1 0 | 0 -> 0b0011_0000.
        w = codec.BitWriter()
        codec.write_rice(w, [[0, 6]], 2, 100)
        self.assertEqual(w.finish(), bytes([0x30]))

    def test_rice_round_trip_and_canonical_parameter(self):
        polys = [[0, 1, -1, 1000, -65535, 7], [123456789, -2, 0, 0, 3, -4]]
        for k in (0, 1, 3, 8, 20):
            w = codec.BitWriter()
            codec.write_rice(w, polys, k, 1 << 30)
            data = w.finish()
            r = codec.BitReader(data)
            self.assertEqual(codec.read_rice(r, 2, 6, k, 1 << 30), polys)
            r.finish()
            with self.assertRaises(codec.DecodeError):
                codec.read_rice(codec.BitReader(data), 2, 6, k, 1 << 20)
        k = codec.shortest_rice(polys)
        lengths = [codec.rice_length(polys, j) for j in range(40)]
        self.assertEqual(lengths[k], min(lengths))
        self.assertTrue(all(lengths[j] > lengths[k] for j in range(k)))
        self.assertTrue(codec.is_shortest_rice(polys, k))
        self.assertFalse(codec.is_shortest_rice(polys, k + 1))

    def test_challenge_ranks(self):
        # binom(5, 2) = 10: {0, 1} has rank 0, {3, 4} rank 3 + 6 = 9, {1, 3} rank 1 + 3.
        for c, rank, signs in (([1, -1, 0, 0, 0], 0, 0b10), ([0, 0, 0, -1, -1], 9, 0b11),
                               ([0, 1, 0, 1, 0], 4, 0b00)):
            w = codec.BitWriter()
            codec.write_challenge(w, c)
            data = w.finish()
            r = codec.BitReader(data)
            self.assertEqual((r.bits(4), r.bits(2)), (rank, signs))
            self.assertEqual(codec.read_challenge(codec.BitReader(data), 5, 2), c)
        w = codec.BitWriter()
        w.bits(10, 4)
        w.bits(0, 2)
        with self.assertRaises(codec.DecodeError):
            codec.read_challenge(codec.BitReader(w.finish()), 5, 2)
        # b_c of Section 7.3: 192, 257 and 323 bits for omega_c = 30, 44, 60.
        for omega, b_c in ((30, 192), (44, 257), (60, 323)):
            self.assertEqual(codec.challenge_rank_bits(512, omega) + omega, b_c)

    def test_rice_rule(self):
        # The rule switches to k at sigma = 1.2863 * 2^k exactly.
        for k in (3, 16, 45):
            boundary = Fraction(12863, 10000) * 2 ** k
            self.assertEqual(rice_parameter(boundary ** 2), k)
            self.assertEqual(rice_parameter(boundary ** 2 - Fraction(1, 10 ** 9)), k - 1)
        self.assertEqual(rice_parameter(Fraction(1)), 0)


class ReconstructionTest(unittest.TestCase):
    """Against the SageMath pipeline of [46] (fixtures/sage)."""

    def check_matrix(self, tag):
        g = fixture("author", f"G-{tag}.json")["g"]
        ref = fixture("sage", f"recon-{tag}.json")
        for case in ref["cases"]:
            g_t = [g[i] for i in case["set"]]
            lam = sec.reconstruction_vector(g_t)
            self.assertEqual(lam is not None, case["authorized"], case["set"])
            # The canonical kernel basis depends on the lattice alone.
            hnf, transform, pivots = sec.row_hnf(g_t)
            kernel = [list(r) for r in transform[len(pivots):]]
            kernel = kernel[:len(sec.hnf_in_place(kernel, len(g_t)))]
            self.assertEqual(kernel, [[int(x) for x in row] for row in case["kernel_hnf"]])
            if lam is None:
                continue
            # On Sage's reduced basis, Babai and the greedy stage agree exactly.
            basis = case["kernel_lll"]
            babai = sec.babai_nearest_plane(basis, sec.integral_gso(basis), transform[0])
            self.assertEqual(babai, case["lambda_babai"])
            self.assertEqual(sec.greedy_reduce_linf(babai, basis, 2, 2), case["lambda"])

    def test_reference_10(self):
        self.check_matrix("10-5-3")

    def test_reference_20(self):
        self.check_matrix("20-9-3")

    def test_reference_40(self):
        self.check_matrix("40-19-3")

    def test_notable_sets(self):
        # Sets beyond the maxima observed in [46]: this implementation and Sage
        # agree on each; at n = 40 one 27-party set has weight 26 > omega_lambda = 25.
        for tag, expected in (("40-19-3", [(26, 4, 25), (26, 5, 21), (27, 1, 26), (27, 2, 25), (27, 2, 25)]),
                              ("60-39-3", [(46, 5, 44), (46, 9, 37)])):
            g = fixture("author", f"G-{tag}.json")["g"]
            notable = fixture("sage", f"recon-{tag}.json")["notable"]
            got = []
            for case in notable:
                lam = sec.reconstruction_vector([g[i] for i in case["set"]])
                self.assertEqual(lam, case["lambda"])
                got.append((len(case["set"]), sec.linf(lam), sum(1 for x in lam if x)))
            self.assertEqual(got, expected)


class ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scheme = latsec.Scheme(profile("functional-n10-d16"))
        rng = SeededRandom(b"python-unit-test", "test")
        cls.vk, cls.abort_keys, cls.shares = latsec.keygen(cls.scheme, rng)
        cls.rng = rng

    def signers(self):
        return [latsec.Signer(self.scheme, self.vk, i, s) for i, s in enumerate(self.shares)]

    def test_every_admissible_set_signs(self):
        from itertools import combinations
        signed = unauthorized = 0
        signers = self.signers()
        for size in range(7, 11):
            for act in combinations(range(10), size):
                sid = bytes([size, len(act)]) + bytes(act) + bytes(30 - len(act))
                try:
                    msg, _, (kind, sig) = latsec.sign_session(
                        self.scheme, self.vk, self.abort_keys, signers, sid, act, b"m", self.rng)
                except latsec.Inadmissible:
                    unauthorized += 1
                    continue
                self.assertEqual(kind, "signature")
                self.assertTrue(latsec.verify(self.scheme, self.vk, msg, sig))
                signed += 1
        # As in the Rust tests: 24 of 120 seven-sets and 3 of 45 eight-sets fail.
        self.assertEqual(unauthorized, 27)
        self.assertEqual(signed, 120 + 45 + 10 + 1 - 27)

    def test_predicate_and_blame(self):
        act = tuple(range(10))
        sid = bytes(32)
        lam = self.scheme.initiator_lambda(act)
        msg = latsec.encode_session_message(self.scheme, sid, act, lam, b"payload")
        self.assertEqual(latsec.admissible(self.scheme, sid, act, msg), lam)
        with self.assertRaises(latsec.Inadmissible):
            latsec.admissible(self.scheme, bytes([1]) + bytes(31), act, msg)       # another sid
        wrong = list(lam)
        wrong[0] += 1
        bad = latsec.encode_session_message(self.scheme, sid, act, wrong, b"payload")
        with self.assertRaises(latsec.Inadmissible):
            latsec.admissible(self.scheme, sid, act, bad)
        # A tampered partial response is blamed by Combine.
        signers = self.signers()
        sid = bytes([7]) * 32
        msg, transcript, (kind, sig) = latsec.sign_session(
            self.scheme, self.vk, self.abort_keys, signers, sid, act, b"m", self.rng)
        self.assertEqual(kind, "signature")
        cmt, w, z = transcript[3]
        z = [list(p) for p in z]
        z[0][0] += 1
        transcript[3] = (cmt, w, z)
        self.assertEqual(latsec.combine(self.scheme, self.vk, self.abort_keys, sid, act, msg, transcript),
                         ("abort", [3]))
        # The signature binds the whole session message.
        other = latsec.encode_session_message(self.scheme, sid, act, self.scheme.initiator_lambda(act), b"n")
        self.assertFalse(latsec.verify(self.scheme, self.vk, other, sig))
        self.assertTrue(latsec.verify(self.scheme, self.vk, msg, sig))

    def rounds(self, signers, sid, act, tamper_commitments=None, tamper_openings=None, cheaters=()):
        """The three rounds by hand.  The tamper functions edit the broadcast
        lists; the parties in `cheaters` do not run Round 3."""
        lam = self.scheme.initiator_lambda(act)
        msg = latsec.encode_session_message(self.scheme, sid, act, lam, b"m")
        commitments = [signers[i].share_sign_1(sid, act, msg, self.rng) for i in act]
        if tamper_commitments:
            tamper_commitments(commitments)
        # A cheater broadcasts the opening it prepared without running Round 2.
        openings = [signers[i].sessions[sid]["w"] if i in cheaters
                    else signers[i].share_sign_2(sid, act, commitments) for i in act]
        if tamper_openings:
            tamper_openings(openings)
        outputs = [signers[i].share_sign_3(sid, act, openings) for i in act if i not in cheaters]
        return msg, commitments, openings, outputs

    def test_missing_and_malformed_round_messages_are_blamed(self):
        # Figure 1: a missing or unparsable peer message fails the check that
        # uses it, so its sender is blamed.
        act = tuple(range(10))
        q, pos = self.scheme.profile.q, 4

        def drop(messages):
            messages[pos] = None

        def shorten(messages):
            messages[pos] = messages[pos][:-1]

        def unreduce(openings):
            openings[pos] = [list(poly) for poly in openings[pos]]
            openings[pos][1][2] = q

        cases = [("missing commitment", drop, None), ("short commitment", shorten, None),
                 ("missing opening", None, drop), ("truncated opening", None, shorten),
                 ("unreduced opening", None, unreduce)]
        for number, (name, tamper_c, tamper_o) in enumerate(cases):
            signers = self.signers()
            sid = bytes([0x40 + number]) * 32
            _, _, _, outputs = self.rounds(signers, sid, act, tamper_c, tamper_o, cheaters=(pos,))
            self.assertEqual(outputs, [("abort", [pos])] * 9, name)
            self.assertTrue(all(sid not in s.sessions for i, s in enumerate(signers) if i != pos), name)

        # Two faulty signers: the first in the order of act is named.
        def drop_two(openings):
            openings[7] = openings[2] = None
        signers = self.signers()
        _, _, _, outputs = self.rounds(signers, bytes([0x50]) * 32, act, None, drop_two, cheaters=(2, 7))
        self.assertEqual(outputs, [("abort", [2])] * 8)

        # A list without one entry per signer is a malformed invocation, and so
        # is the signer's own message coming back altered.
        signers = self.signers()
        sid = bytes([0x51]) * 32
        lam = self.scheme.initiator_lambda(act)
        msg = latsec.encode_session_message(self.scheme, sid, act, lam, b"m")
        commitments = [signers[i].share_sign_1(sid, act, msg, self.rng) for i in act]
        with self.assertRaises(latsec.InvocationError):
            signers[0].share_sign_2(sid, act, commitments[:6])
        with self.assertRaises(latsec.InvocationError):
            signers[0].share_sign_2(sid, act, [None] + commitments[1:])
        openings = [signers[i].share_sign_2(sid, act, commitments) for i in act]
        with self.assertRaises(latsec.InvocationError):
            signers[0].share_sign_3(sid, act, openings[:6])
        with self.assertRaises(latsec.InvocationError):
            signers[0].share_sign_3(sid, act, [None] + openings[1:])
        self.assertEqual(signers[0].share_sign_3(sid, act, openings)[0], "response")

    def test_incomplete_transcripts_do_not_combine(self):
        act = tuple(range(10))
        sid = bytes([0x60]) * 32
        args = (self.scheme, self.vk, self.abort_keys, sid, act)
        msg, honest, (kind, _) = latsec.sign_session(
            self.scheme, self.vk, self.abort_keys, self.signers(), sid, act, b"m", self.rng)
        self.assertEqual(kind, "signature")
        lam = latsec.admissible(self.scheme, sid, act, msg)
        idle = [i for i, x in zip(act, lam) if x == 0]
        self.assertTrue(idle, "the test needs a signer with a zero coefficient")

        # Dropping the entries of trailing signers, whose coefficients may be
        # zero, must not give a signature.
        for keep in range(1, 10):
            with self.assertRaises(latsec.InvocationError):
                latsec.combine(*args, msg, honest[:keep])
            with self.assertRaises(latsec.InvocationError):
                latsec.partial_verify(*args, msg, honest[:keep], 0)

        # A missing response of an idle signer is blamed.
        i = idle[-1]
        tr = list(honest)
        tr[i] = (tr[i][0], tr[i][1], None)
        self.assertEqual(latsec.combine(*args, msg, tr), ("abort", [i]))
        self.assertFalse(latsec.partial_verify(*args, msg, tr, i))
        self.assertTrue(latsec.partial_verify(*args, msg, tr, (i + 1) % 10))

        # A missing opening: no challenge, and only its sender is blamed.
        tr = list(honest)
        tr[3] = (tr[3][0], None, tr[3][2])
        self.assertEqual(latsec.combine(*args, msg, tr), ("abort", [3]))
        self.assertFalse(latsec.partial_verify(*args, msg, tr, 3))
        with self.assertRaises(latsec.InvocationError):
            latsec.partial_verify(*args, msg, tr, 5)
        # With it, a missing and an overlong response are named; one that only
        # fails the linear relation cannot be judged without a challenge.
        tr[1] = (tr[1][0], tr[1][1], None)
        long_z = [list(p) for p in tr[9][2]]
        long_z[0][0] += 1 << 100
        tr[9] = (tr[9][0], tr[9][1], long_z)
        off_z = [list(p) for p in tr[2][2]]
        off_z[0][0] += 1
        tr[2] = (tr[2][0], tr[2][1], off_z)
        self.assertEqual(latsec.combine(*args, msg, tr), ("abort", [1, 3, 9]))

    def test_partial_verify_judges_own_messages_first(self):
        act = tuple(range(10))
        sid = bytes([0x61]) * 32
        args = (self.scheme, self.vk, self.abort_keys, sid, act)
        msg, honest, _ = latsec.sign_session(
            self.scheme, self.vk, self.abort_keys, self.signers(), sid, act, b"m", self.rng)
        cmt0, w0, z0 = honest[0]
        oversized = [list(p) for p in z0]
        oversized[0][0] += 1 << 100
        # Party 0's response is missing, malformed or too long, and party 1's
        # opening is missing: PartialVerify(0) fails, and Combine names both.
        for name, z in (("missing", None), ("truncated", z0[1:]), ("oversized", oversized)):
            tr = list(honest)
            tr[0] = (cmt0, w0, z)
            tr[1] = (tr[1][0], None, tr[1][2])
            self.assertFalse(latsec.partial_verify(*args, msg, tr, 0), name)
            self.assertFalse(latsec.partial_verify(*args, msg, tr, 1), name)
            with self.assertRaises(latsec.InvocationError):
                latsec.partial_verify(*args, msg, tr, 2)
            self.assertEqual(latsec.combine(*args, msg, tr), ("abort", [0, 1]), name)
        # In bounds but off the relation: that needs the challenge, so with
        # another opening missing there is no verdict.
        off = [list(p) for p in z0]
        off[0][0] += 1
        tr = list(honest)
        tr[0] = (cmt0, w0, off)
        tr[1] = (tr[1][0], None, tr[1][2])
        with self.assertRaises(latsec.InvocationError):
            latsec.partial_verify(*args, msg, tr, 0)
        self.assertEqual(latsec.combine(*args, msg, tr), ("abort", [1]))

    def test_refused_round_drops_started_sessions(self):
        act = tuple(range(10))
        sid = bytes([0x70]) * 32
        signers = self.signers()
        signers[6].participations = self.scheme.profile.q_participate      # budget spent
        with self.assertRaises(latsec.InvocationError):
            latsec.sign_session(self.scheme, self.vk, self.abort_keys, signers, sid, act, b"m", self.rng)
        self.assertTrue(all(not s.sessions for s in signers))
        # The identifier stays used for the signers that had started.
        self.assertIn(sid, signers[0].used)
        self.assertNotIn(sid, signers[6].used)

    def test_decoders_reject_inputs_beyond_the_profile_limits(self):
        from math import isqrt
        scheme, prof = self.scheme, self.scheme.profile
        act = tuple(range(10))
        sid = bytes([0x62]) * 32
        msg, transcript, (_, sig) = latsec.sign_session(
            scheme, self.vk, self.abort_keys, self.signers(), sid, act, b"payload", self.rng)
        c, z = sig

        def constant(value):
            return [[value] * prof.d for _ in range(prof.ell)]

        def spike(value):
            v = constant(0)
            v[0][0] = value
            return v

        def flat(bound):
            # floor(B / sqrt(D)): equal coefficients maximize the l1 norm, and
            # with it the unary part, among the vectors of norm at most B.
            return isqrt(bound * bound // prof.dim)

        # Coefficient decoding must not start on an input that the length checks
        # refuse: read_rice is replaced by a function that fails the test.
        too_long = "longer than any valid one"
        never_decoding = mock.patch.object(
            codec, "read_rice", side_effect=AssertionError("coefficient decoding was reached"))

        # Objects of one length.
        vk_bytes = self.vk.encode(scheme)
        self.assertEqual(len(vk_bytes), latsec.verification_key_len(prof))
        ak_bytes = latsec.encode_abort_key(scheme, 3, self.abort_keys[3])
        self.assertEqual(len(ak_bytes), latsec.abort_key_len(prof))
        self.assertEqual(latsec.decode_abort_key(scheme, ak_bytes), (3, self.abort_keys[3]))
        opening_bytes = latsec.encode_opening(scheme, transcript[0][1])
        self.assertEqual(len(opening_bytes), latsec.opening_len(prof))
        self.assertEqual(latsec.decode_opening(scheme, opening_bytes), transcript[0][1])
        for decode, data in ((latsec.VerificationKey.decode, vk_bytes), (latsec.decode_abort_key, ak_bytes),
                             (latsec.decode_opening, opening_bytes)):
            for other in (data[:-1], data + b"\0"):
                with self.assertRaises(codec.DecodeError):
                    decode(scheme, other)

        # Shares.
        for index, share in enumerate(self.shares):
            data = latsec.encode_share(scheme, index, share)
            limit = latsec.max_share_len(prof, index)
            self.assertLessEqual(len(data), limit)
            self.assertEqual(latsec.decode_share(scheme, data), (index, share))
            with never_decoding, self.assertRaisesRegex(codec.DecodeError, too_long):
                latsec.decode_share(scheme, data[:4] + b"\xff" * (limit - 3))

        # Partial responses: honest ones and the extreme shapes of norm B1.
        limit = latsec.max_response_len(prof)
        shapes = [z_i for _, _, z_i in transcript]
        shapes += [constant(0), spike(prof.b1), spike(-prof.b1), constant(flat(prof.b1))]
        for z_i in shapes:
            data = latsec.encode_response(scheme, z_i)
            self.assertLessEqual(len(data), limit)
            self.assertEqual(latsec.decode_response(scheme, data), z_i)
        with never_decoding, self.assertRaisesRegex(codec.DecodeError, too_long):
            latsec.decode_response(scheme, data[:2] + b"\xff" * (limit - 1))

        # Signatures: the honest one and the extreme shapes of norm B2.  The
        # all-zero response has k = 0, which stays a valid encoding.
        limit = latsec.max_signature_len(prof)
        parameters = []
        for other in (z, constant(0), constant(1), spike(prof.b2), spike(-prof.b2), constant(flat(prof.b2))):
            data = latsec.encode_signature(scheme, (c, other))
            self.assertLessEqual(len(data), limit)
            self.assertEqual(latsec.decode_signature(scheme, data), (c, other))
            parameters.append(data[2])
        self.assertEqual(parameters[1], 0)
        self.assertGreater(parameters[5], parameters[0])

        # k = 0 and a valid challenge, followed by an endless unary run.  The
        # header, the parameter byte and the challenge bits are those of the
        # all-zero signature; every later bit is one.
        zero_bytes = latsec.encode_signature(scheme, (c, constant(0)))
        c_bits = latsec.challenge_bits(prof)

        def endless(length):
            whole = 3 + c_bits // 8
            run = bytearray(b"\xff" * length)
            run[:whole] = zero_bytes[:whole]
            if c_bits % 8:
                run[whole] = zero_bytes[whole] | (0xff << (c_bits % 8)) & 0xff
            return bytes(run)

        # Beyond the limit it is refused by its length, before anything is read.
        for length in (limit + 1, 100 * limit):
            with never_decoding, self.assertRaisesRegex(codec.DecodeError, too_long):
                latsec.decode_signature(scheme, endless(length))
        # At the limit the challenge is accepted and the run is followed, to the
        # end of the input and no further: the work is bounded by the limit.
        with mock.patch.object(codec, "read_rice", wraps=codec.read_rice) as reached:
            with self.assertRaisesRegex(codec.DecodeError, "too short"):
                latsec.decode_signature(scheme, endless(limit))
        self.assertEqual(reached.call_count, 1)
        self.assertEqual(latsec.decode_signature(scheme, zero_bytes), (c, constant(0)))

        # A parameter the input is too short for is rejected before any
        # coefficient is read; the challenge is valid.
        sig_bytes = latsec.encode_signature(scheme, sig)
        for k in (sig_bytes[2] + 40, 126, 127, 255):
            with never_decoding, self.assertRaisesRegex(codec.DecodeError, "Rice parameter out of range"):
                latsec.decode_signature(scheme, sig_bytes[:2] + bytes([k]) + sig_bytes[3:])

        # Session messages: a fixed head, and the application's payload limit.
        self.assertEqual(len(msg), latsec.session_head_len(prof, len(act)) + len(b"payload"))
        self.assertEqual(latsec.decode_session_message(scheme, msg, max_payload=7)[3], b"payload")
        with self.assertRaises(codec.DecodeError):
            latsec.decode_session_message(scheme, msg, max_payload=6)
        with self.assertRaises(codec.DecodeError):
            latsec.decode_session_message(scheme, msg[:-8])
        with self.assertRaises(codec.DecodeError):
            latsec.decode_session_message(scheme, msg[:34] + (11).to_bytes(2, "little") + msg[36:])

    def test_signature_length_limits_are_close_to_the_size_estimates(self):
        # Table 5 of the paper, column |sig| in KiB: the ideal
        # size of a response at the bound B2.  The decoder's limit must lie
        # above it, and stays within 3%.
        table_5 = {(40, 128): 40.750, (40, 192): 56.147, (40, 256): 70.981,
                   (60, 128): 42.622, (60, 192): 58.641, (60, 256): 74.098}
        for (n, kappa), estimate in table_5.items():
            limit = latsec.max_signature_len(profile(f"author-n{n}-k{kappa}-nov2026")) / 1024
            self.assertTrue(estimate < limit < 1.03 * estimate, (n, kappa, limit))

    def test_signature_encoding_is_canonical(self):
        act = tuple(range(10))
        sid = bytes([9]) * 32
        msg, _, (_, sig) = latsec.sign_session(
            self.scheme, self.vk, self.abort_keys, self.signers(), sid, act, b"m", self.rng)
        data = latsec.encode_signature(self.scheme, sig)
        self.assertEqual(latsec.decode_signature(self.scheme, data), sig)
        k = data[2]
        for other in (k - 1, k + 1):
            if other < 0:
                continue
            w = codec.BitWriter(codec.TAG_SIGNATURE)
            w.bytes([other])
            codec.write_challenge(w, sig[0])
            codec.write_rice(w, sig[1], other, self.scheme.profile.b2)
            with self.assertRaises(codec.DecodeError):
                latsec.decode_signature(self.scheme, w.finish())
        with self.assertRaises(codec.DecodeError):
            latsec.decode_signature(self.scheme, data + b"\0")


if __name__ == "__main__":
    unittest.main()
