"""SHAKE256 with domain separation and framed inputs, and seeded randomness.

Every hash of the protocol is

    SHAKE256( frame(domain) || frame(x_1) || frame(x_2) || ... )

where frame(x) is the byte length of x as a little-endian 64-bit integer,
followed by x.  Distinct domains give independent functions, and distinct
input tuples never collide as byte strings.

The paper, Section 3.1: H_cmt and H_c are random oracles; Section 3.1
also suggests a seed-then-expand H_c, which this implementation does not use
(see latsec.py, hash_to_challenge).
"""

# PyCryptodome installs as `Crypto` from PyPI (pycryptodome) and as `Cryptodome`
# from the Debian package python3-pycryptodome and from PyPI's pycryptodomex.
try:
    from Crypto.Hash import SHAKE256
except ImportError:
    from Cryptodome.Hash import SHAKE256

# The domain strings, one per use.
DOMAIN_PROFILE = "LaTSEC-v1/profile"
DOMAIN_EXPAND_A = "LaTSEC-v1/expand-A"
DOMAIN_VERIFICATION_KEY = "LaTSEC-v1/vk"
DOMAIN_COMMITMENT = "LaTSEC-v1/commit"
DOMAIN_CHALLENGE = "LaTSEC-v1/challenge"
DOMAIN_FIXTURE_RANDOMNESS = "LaTSEC-v1/fixture-randomness"


def u64_le(value):
    """An integer as an eight-byte little-endian field."""
    return value.to_bytes(8, "little")


class Xof:
    """An absorbing SHAKE256 instance with framed inputs."""

    def __init__(self, domain):
        self._shake = SHAKE256.new()
        self.absorb(domain.encode())

    def absorb(self, data):
        """Absorbs one length-framed field."""
        data = bytes(data)
        self._shake.update(u64_le(len(data)))
        self._shake.update(data)
        return self

    def absorb_u64(self, value):
        """Absorbs an integer as an eight-byte little-endian field."""
        return self.absorb(u64_le(value))

    def stream(self):
        """Finalizes; the result is a byte stream with read(n)."""
        return self._shake

    def digest(self, length):
        """Finalizes and returns the first `length` output bytes."""
        return self._shake.read(length)


class SeededRandom:
    """Deterministic randomness expanded from a seed with SHAKE256.

    For test vectors and reproducible experiments only.  The stream is
    SHAKE256(frame("LaTSEC-v1/fixture-randomness") || frame(label) || frame(seed)).
    """

    def __init__(self, seed, label):
        xof = Xof(DOMAIN_FIXTURE_RANDOMNESS)
        xof.absorb(label.encode()).absorb(seed)
        self._stream = xof.stream()

    def read(self, length):
        return self._stream.read(length)


class OsRandom:
    """Operating-system randomness."""

    def read(self, length):
        import os
        return os.urandom(length)


def uniform_below(rng, bound):
    """A uniform integer in [0, bound) by rejection sampling.

    Reads ceil(bitlen(bound - 1) / 8) bytes per attempt, little-endian, and
    masks them to bitlen(bound - 1) bits.  bound = 1 reads nothing.
    """
    assert bound >= 1
    if bound == 1:
        return 0
    bits = (bound - 1).bit_length()
    nbytes = (bits + 7) // 8
    mask = (1 << bits) - 1
    while True:
        x = int.from_bytes(rng.read(nbytes), "little") & mask
        if x < bound:
            return x


def random_bit(rng):
    """One uniform bit: the low bit of one byte."""
    return rng.read(1)[0] & 1 == 1


def random_subset(rng, n, size):
    """A uniform subset of `size` elements of [0, n), sorted (partial Fisher-Yates)."""
    pool = list(range(n))
    for i in range(size):
        j = i + uniform_below(rng, n - i)
        pool[i], pool[j] = pool[j], pool[i]
    return sorted(pool[:size])
