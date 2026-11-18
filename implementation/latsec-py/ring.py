"""The rings R = Z[X]/(X^d + 1) and R_q, the challenge set C, and A = [I_p | A'].

The paper, Section 2.1 (notation) and Section 3.1 (parameters, the public
matrix and the challenge set C = {c in R : |c|_inf = 1, |c|_1 = omega_c}).

Polynomials are lists of d coefficients, constant term first.  Elements of R_q
are kept as residues in [0, q).
"""

from xof import Xof, DOMAIN_EXPAND_A, u64_le, uniform_below, random_bit


# --- R and R_q ---------------------------------------------------------------

def poly_mul_negacyclic(a, b, q=None):
    """a * b in Z[X]/(X^d + 1) (schoolbook), reduced mod q if q is given."""
    d = len(a)
    out = [0] * d
    for i, x in enumerate(a):
        if x == 0:
            continue
        for j, y in enumerate(b):
            if i + j < d:
                out[i + j] += x * y
            else:
                out[i + j - d] -= x * y
    if q is not None:
        out = [c % q for c in out]
    return out


def centered(x, q):
    """The representative of x mod q in (-q/2, q/2]."""
    x %= q
    return x - q if x > q // 2 else x


def is_prime(q):
    """Miller-Rabin with the first thirteen primes as bases (deterministic for
    q < 3.3 * 10^24, and the moduli here are checked against the profiles)."""
    bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
    for b in bases:
        if q % b == 0:
            return q == b
    twos = ((q - 1) & -(q - 1)).bit_length() - 1
    odd = (q - 1) >> twos
    for b in bases:
        x = pow(b, odd, q)
        if x in (1, q - 1):
            continue
        for _ in range(twos - 1):
            x = x * x % q
            if x == q - 1:
                break
        else:
            return False
    return True


def largest_ntt_prime_below(bits, d):
    """The largest prime q < 2^bits with q = 1 mod 2d (Section 7.1)."""
    step = 2 * d
    q = (1 << bits) - 1
    q -= (q - 1) % step
    while q > step:
        if is_prime(q):
            return q
        q -= step
    raise ValueError("no NTT-friendly prime of that size")


def canonical_root(d, q):
    """psi = g^((q-1)/(2d)) for the smallest g >= 2 with psi^d = -1 mod q."""
    e = (q - 1) // (2 * d)
    g = 2
    while True:
        psi = pow(g, e, q)
        if pow(psi, d, q) == q - 1:
            return psi
        g += 1


def bit_reverse(k, bits):
    return int(format(k, "0{}b".format(bits))[::-1], 2) if bits else 0


class Ring:
    """R_q = Z_q[X]/(X^d + 1) with its negacyclic NTT.

    The NTT is the in-place Cooley-Tukey transform with zetas[k] = psi^bitrev(k)
    (as in ML-KEM and ML-DSA).  Its output order matters only because A' is
    sampled directly in the NTT domain.
    """

    def __init__(self, d, q, psi):
        assert pow(psi, d, q) == q - 1, "psi is not a primitive 2d-th root of unity"
        self.d, self.q, self.psi = d, q, psi
        log_d = d.bit_length() - 1
        self.zetas = [pow(psi, bit_reverse(k, log_d), q) for k in range(d)]
        self.d_inv = pow(d, -1, q)

    def ntt(self, a):
        a, q, d = list(a), self.q, self.d
        k, length = 0, d // 2
        while length >= 1:
            for start in range(0, d, 2 * length):
                k += 1
                zeta = self.zetas[k]
                for j in range(start, start + length):
                    t = a[j + length] * zeta % q
                    a[j + length] = (a[j] - t) % q
                    a[j] = (a[j] + t) % q
            length //= 2
        return a

    def intt(self, a):
        a, q, d = list(a), self.q, self.d
        k, length = d, 1
        while length < d:
            for start in range(0, d, 2 * length):
                k -= 1
                zeta = -self.zetas[k] % q
                for j in range(start, start + length):
                    t = a[j]
                    a[j] = (t + a[j + length]) % q
                    a[j + length] = (t - a[j + length]) * zeta % q
            length *= 2
        return [x * self.d_inv % q for x in a]

    def reduce(self, a):
        return [x % self.q for x in a]

    def add(self, a, b):
        return [(x + y) % self.q for x, y in zip(a, b)]

    def sub(self, a, b):
        return [(x - y) % self.q for x, y in zip(a, b)]


# --- The public matrix A = [I_p | A'] -----------------------------------------

def expand_a(ring, p, ell, seed):
    """A' in R_q^(p x (ell - p)), given in the NTT domain.

    Entry (i, j) is read from SHAKE256(frame("LaTSEC-v1/expand-A") || frame(seed)
    || frame(i) || frame(j)) by rejection sampling of ceil(lg q)-bit
    little-endian integers, each taking ceil(ceil(lg q) / 8) bytes.
    """
    q = ring.q
    bits = (q - 1).bit_length()
    nbytes = (bits + 7) // 8
    mask = (1 << bits) - 1
    a_hat = []
    for i in range(p):
        row = []
        for j in range(ell - p):
            stream = Xof(DOMAIN_EXPAND_A).absorb(seed).absorb_u64(i).absorb_u64(j).stream()
            poly = []
            while len(poly) < ring.d:
                x = int.from_bytes(stream.read(nbytes), "little") & mask
                if x < q:
                    poly.append(x)
            row.append(poly)
        a_hat.append(row)
    return a_hat


def mul_a(ring, a_hat, v):
    """A v mod q = v[:p] + A' v[p:] for v in R^ell (integer or residue
    coefficients)."""
    p = len(a_hat)
    q = ring.q
    tail = [ring.ntt(ring.reduce(poly)) for poly in v[p:]]
    out = []
    for i in range(p):
        acc = [0] * ring.d
        for a_ij, x in zip(a_hat[i], tail):
            acc = [(s + y * z) % q for s, y, z in zip(acc, a_ij, x)]
        out.append(ring.add(ring.intt(acc), ring.reduce(v[i])))
    return out


# --- The challenge set C -------------------------------------------------------

def sample_in_ball(stream, d, omega_c):
    """A uniform element of C from a stream of uniform bytes (Section 3.1).

    For i = d - omega_c, ..., d - 1: a uniform j in [0, i] and a sign bit, then
    c[i] = c[j] and c[j] = +-1.  This unbiased shuffle makes every element of C
    equally likely, so with a random-oracle stream H_c is a random oracle into C.
    """
    c = [0] * d
    for i in range(d - omega_c, d):
        j = uniform_below(stream, i + 1)
        negative = random_bit(stream)
        c[i] = c[j]
        c[j] = -1 if negative else 1
    return c


def challenge_mul(c, a, q=None):
    """c * a in R (or R_q) for a sparse ternary c."""
    d = len(c)
    out = [0] * d
    for shift, ci in enumerate(c):
        if ci == 0:
            continue
        for j, x in enumerate(a):
            idx = shift + j
            if idx < d:
                out[idx] += ci * x
            else:
                out[idx - d] -= ci * x
    if q is not None:
        out = [x % q for x in out]
    return out
