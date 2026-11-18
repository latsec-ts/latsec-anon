"""Canonical binary encodings (codec version 3).

Every encoded object starts with two bytes: a type tag and the codec version.
All fields are little-endian at both levels: bit 0 of a field is written
first, into the least significant free bit of the current byte, and
multi-byte integers are stored least significant byte first.

  * residues mod q: ceil(lg q) bits each, below q;
  * integers bounded by B (fixed width): x + B in bitlen(2B) bits;
  * Gaussian integers: Golomb-Rice with parameter k -- the k low bits of |x|,
    |x| >> k in unary (that many one bits, then a zero bit), and a sign bit
    (1 = negative) if x != 0; |x| <= B for the object's bound B;
  * a challenge c in C: the rank of its support in the combinatorial number
    system, sum_j binom(p_j, j) over the nonzero positions p_1 < ... < p_omega,
    in bitlen(binom(d, omega_c) - 1) bits, then one sign bit per nonzero
    coefficient in increasing position order.  This takes
    b_c = ceil(omega_c + lg binom(d, omega_c)) bits (the paper, Section 7.3).

Decoders reject wrong tags or versions, out-of-range values, nonzero padding
bits, and any length but the expected one, so every value has exactly one
encoding.  A decoding error raises DecodeError.
"""

from math import comb, isqrt

CODEC_VERSION = 3

TAG_VERIFICATION_KEY = 0x01
TAG_ABORT_KEY = 0x02
TAG_SECRET_SHARE = 0x03
TAG_OPENING = 0x04
TAG_PARTIAL_RESPONSE = 0x05
TAG_SIGNATURE = 0x06
TAG_SESSION_MESSAGE = 0x07


class DecodeError(ValueError):
    pass


class BitWriter:
    def __init__(self, tag=None):
        self.out = bytearray() if tag is None else bytearray([tag, CODEC_VERSION])
        self.acc = 0       # pending bits, least significant first
        self.nbits = 0

    def bytes(self, data):
        assert self.nbits == 0, "byte field off a byte boundary"
        self.out += bytes(data)

    def bits(self, value, width):
        assert 0 <= value < (1 << width) or width == 0 and value == 0
        self.acc |= value << self.nbits
        self.nbits += width
        while self.nbits >= 8:
            self.out.append(self.acc & 0xff)
            self.acc >>= 8
            self.nbits -= 8

    def finish(self):
        """Pads with zero bits to a byte boundary and returns the bytes."""
        if self.nbits:
            self.out.append(self.acc)
            self.acc, self.nbits = 0, 0
        return bytes(self.out)


class BitReader:
    def __init__(self, data, tag=None):
        self.data = bytes(data)
        self.pos = 0       # bit position
        if tag is not None:
            if len(self.data) < 2 or self.data[0] != tag:
                raise DecodeError("wrong object type")
            if self.data[1] != CODEC_VERSION:
                raise DecodeError("unsupported codec version")
            self.pos = 16

    def bytes(self, length):
        if self.pos % 8:
            raise DecodeError("byte field off a byte boundary")
        start = self.pos // 8
        if start + length > len(self.data):
            raise DecodeError("encoding is too short")
        self.pos += 8 * length
        return self.data[start:start + length]

    def bit(self):
        byte = self.pos >> 3
        if byte >= len(self.data):
            raise DecodeError("encoding is too short")
        value = (self.data[byte] >> (self.pos & 7)) & 1
        self.pos += 1
        return value

    def bits(self, width):
        value = 0
        for i in range(width):
            value |= self.bit() << i
        return value

    def finish_prefix(self):
        """Requires zero padding to the next byte boundary; returns the number
        of bytes read."""
        while self.pos % 8:
            if self.bit():
                raise DecodeError("nonzero padding bits")
        return self.pos // 8

    def finish(self):
        """Requires zero padding and no trailing bytes."""
        if self.finish_prefix() != len(self.data):
            raise DecodeError("trailing bytes")


# --- Field encodings ---------------------------------------------------------------

def residue_width(q):
    return (q - 1).bit_length()


def signed_width(bound):
    return (2 * bound).bit_length()


def write_residues(w, polys, q):
    width = residue_width(q)
    for poly in polys:
        for x in poly:
            w.bits(x, width)


def read_residues(r, count, d, q):
    width = residue_width(q)
    polys = []
    for _ in range(count):
        poly = [r.bits(width) for _ in range(d)]
        if any(x >= q for x in poly):
            raise DecodeError("residue is not reduced")
        polys.append(poly)
    return polys


def residues_to_bytes(polys, q):
    """Residue polynomials as bytes without a header (a hash input)."""
    w = BitWriter()
    write_residues(w, polys, q)
    return w.finish()


def write_bounded(w, values, bound):
    width = signed_width(bound)
    for x in values:
        if abs(x) > bound:
            raise ValueError("coefficient exceeds its encoding bound")
        w.bits(x + bound, width)


def read_bounded(r, count, bound):
    width = signed_width(bound)
    values = [r.bits(width) for _ in range(count)]
    if any(x > 2 * bound for x in values):
        raise DecodeError("coefficient exceeds its bound")
    return [x - bound for x in values]


def write_rice(w, polys, k, bound):
    """Golomb-Rice coded integer polynomials."""
    for poly in polys:
        for x in poly:
            a = abs(x)
            if a > bound:
                raise ValueError("coefficient exceeds its encoding bound")
            w.bits(a & ((1 << k) - 1), k)
            for _ in range(a >> k):
                w.bits(1, 1)
            w.bits(0, 1)
            if a:
                w.bits(1 if x < 0 else 0, 1)


def read_rice(r, count, d, k, bound):
    if k >= 127:
        raise DecodeError("Rice parameter out of range")
    max_high = bound >> k
    polys = []
    for _ in range(count):
        poly = []
        for _ in range(d):
            low = r.bits(k)
            high = 0
            while r.bit():
                high += 1
                if high > max_high:
                    raise DecodeError("coefficient exceeds its bound")
            a = (high << k) | low
            if a > bound:
                raise DecodeError("coefficient exceeds its bound")
            poly.append(-a if a and r.bit() else a)
        polys.append(poly)
    return polys


def rice_max_bits_linf(count, k, bound):
    """An upper bound on the Golomb-Rice length in bits, with parameter k, of
    `count` integers of absolute value at most `bound`: a codeword has at most
    k + 2 + (|x| >> k) bits."""
    return count * (k + 2 + (bound >> k))


def rice_max_bits_l2(count, k, norm_squared):
    """An upper bound on the Golomb-Rice length in bits, with parameter k, of
    `count` integers of squared Euclidean norm at most `norm_squared`:
    sum_i (|x_i| >> k) <= (sum_i |x_i|) >> k and
    sum_i |x_i| <= floor(sqrt(count * norm_squared)) (Cauchy-Schwarz)."""
    return count * (k + 2) + (isqrt(count * norm_squared) >> k)


def rice_length(polys, k):
    """The length in bits of the Golomb-Rice codewords of `polys`."""
    return sum(k + 1 + (abs(x) >> k) + (1 if x else 0) for poly in polys for x in poly)


def shortest_rice(polys):
    """The parameter giving the shortest code, the smallest on ties.  The
    length is convex in k, so the first non-improvement ends the search."""
    best_k, best = 0, rice_length(polys, 0)
    for k in range(1, 127):
        length = rice_length(polys, k)
        if length >= best:
            break
        best_k, best = k, length
    return best_k


def is_shortest_rice(polys, k):
    here = rice_length(polys, k)
    left_ok = k == 0 or rice_length(polys, k - 1) > here
    right_ok = k >= 126 or rice_length(polys, k + 1) >= here
    return left_ok and right_ok


def challenge_rank_bits(d, weight):
    return (comb(d, weight) - 1).bit_length()


def write_challenge(w, c):
    positions = [i for i, x in enumerate(c) if x != 0]
    rank = sum(comb(p, j + 1) for j, p in enumerate(positions))
    w.bits(rank, challenge_rank_bits(len(c), len(positions)))
    for p in positions:
        w.bits(1 if c[p] < 0 else 0, 1)


def read_challenge(r, d, weight):
    if not 0 < weight <= d:
        raise DecodeError("challenge weight out of range")
    rank = r.bits(challenge_rank_bits(d, weight))
    if rank >= comb(d, weight):
        raise DecodeError("challenge rank out of range")
    # Unrank: p_j is the largest p with binom(p, j) <= the remaining rank.
    positions = [0] * weight
    upper = d
    for j in range(weight, 0, -1):
        p = upper - 1
        while comb(p, j) > rank:
            p -= 1
        rank -= comb(p, j)
        positions[j - 1] = p
        upper = p
    c = [0] * d
    for p in positions:
        c[p] = -1 if r.bit() else 1
    return c


def encode_active_set(act):
    """The signer set: a 16-bit count and 16-bit indices, little-endian."""
    out = len(act).to_bytes(2, "little")
    for party in act:
        out += party.to_bytes(2, "little")
    return out
