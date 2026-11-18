"""The discrete Gaussian sampler of Canonne, Kamath and Steinke, in exact integers.

Convention (the paper, Section 2.1): D_sigma has mass proportional to
exp(-x^2 / (2 sigma^2)), the "sigma-exp-half" convention of the profiles.

The algorithms are those of C. Canonne, G. Kamath and T. Steinke, "The Discrete
Gaussian for Differential Privacy" (2020): Algorithm 1 (Bernoulli(exp(-gamma))),
Algorithm 2 (discrete Laplace) and Algorithm 3 (discrete Gaussian).  Python
integers are unbounded, so the arithmetic is exact.  The one deviation is the
same as in the Rust implementation: Laplace proposals of magnitude 2^126 or
more are rejected, which conditions the output on |x| < 2^126 (statistical
distance below exp(-2^48) for every supported width).

The order in which random bytes are consumed is part of the test vectors.
"""

from xof import uniform_below, random_bit

MAGNITUDE_LIMIT = 1 << 126


def bernoulli_exp_minus_one(rng):
    """Bernoulli(exp(-1)): Bernoulli(1/k) trials for k = 1, 2, ... until the
    first failure; the outcome is whether that k is odd (CKS Algorithm 1)."""
    k = 1
    while uniform_below(rng, k) == 0:
        k += 1
    return k % 2 == 1


def bernoulli_exp(rng, num, den):
    """Bernoulli(exp(-num/den)) for num >= 0, den > 0 (CKS Algorithm 1).

    exp(-num/den) = exp(-1)^whole * exp(-rem/den) with whole = num // den.
    The second factor uses trials Bernoulli(rem / (den k)): a uniform value
    below den k is below rem exactly when its quotient by den is zero (a
    uniform value below k is zero) and its remainder is below rem.
    """
    whole, rem = divmod(num, den)
    for _ in range(whole):
        if not bernoulli_exp_minus_one(rng):
            return False
    k = 1
    while uniform_below(rng, k) == 0 and uniform_below(rng, den) < rem:
        k += 1
    return k % 2 == 1


class DiscreteGaussian:
    """D_{Z, sigma} for sigma = num / 2^frac_bits (CKS Algorithm 3)."""

    def __init__(self, num, frac_bits):
        self.num, self.frac_bits = num, frac_bits
        self.t = (num >> frac_bits) + 1          # Laplace scale floor(sigma) + 1
        self.a = num * num                       # sigma^2 = a / b
        self.b = 1 << (2 * frac_bits)
        self.den = 2 * self.a * self.b * self.t * self.t

    def sample(self, rng):
        t = self.t
        while True:
            # Discrete Laplace of scale t (CKS Algorithm 2 with s = 1).
            u = uniform_below(rng, t)
            if not bernoulli_exp(rng, u, t):
                continue
            v = 0
            while bernoulli_exp_minus_one(rng):
                v += 1
            y = u + t * v
            if y >= MAGNITUDE_LIMIT:
                continue
            negative = random_bit(rng)
            if negative and y == 0:
                continue
            # Accept with probability exp(-(|y| - sigma^2/t)^2 / (2 sigma^2))
            #   = exp(-(y t b - a)^2 / (2 a b t^2)).
            diff = y * t * self.b - self.a
            if bernoulli_exp(rng, diff * diff, self.den):
                return -y if negative else y

    def sample_poly(self, rng, d):
        return [self.sample(rng) for _ in range(d)]
