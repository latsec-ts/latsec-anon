"""Parameter profiles: loading, the derived values, and the digest.

A profile (profiles/*.json, format "latsec-profile-v1") fixes every parameter
of Table 2 of the paper for one instance, together with the
conventions of the implementation.  This module recomputes everything that
can be derived from the stored inputs -- the bounds of Section 4.1, the
Golomb-Rice parameters of the codec, and the digest that every hash is bound
to -- and rejects a profile whose stored values differ.  All arithmetic is
exact (Python integers and fractions).

Bounds (Section 4.1, Items 3 and 5; the tail factors theta are stored):

    sqrtD_up  = ceil(sqrt(D) 2^32) / 2^32,  D = d ell
    beta_s    = theta_s sigma_s sqrtD_up
    beta_R    = theta_R sigma_R
    beta_sh   = beta_g beta_s + c_exp beta_R           (stored rounded up)
    beta_r    = theta_r sigma_r sqrtD_up
    B1        = ceil(omega_c sqrtD_up beta_sh + beta_r)  (beta_sh unrounded)
    B2        = L B1,  L = beta_lambda omega_lambda (or the l1 cap)
    B_MSIS^2  = 4 (B2^2 + omega_c)
"""

import json
from fractions import Fraction
from math import isqrt

from codec import CODEC_VERSION
from xof import Xof, DOMAIN_PROFILE

PROFILE_FORMAT = "latsec-profile-v1"
GAUSSIAN_CONVENTION = "sigma-exp-half"
SAMPLER_ID = "cks-exact-v1"
RECON_ALGORITHM_ID = "xlatsec-recon-v1"
XOF_ID = "shake256-framed-v1"
DIGEST_BYTES = 64
FIXED_FRAC_BITS = 32
CATEGORY_TAGS = {"functional", "independent", "author", "author-variant", "sec-only"}

# The Golomb-Rice rule: k is the largest integer >= 0 with RICE_C 2^k <= sigma.
RICE_C = Fraction(12863, 10000)


def ceil_fraction(x):
    return -((-x.numerator) // x.denominator)


def dyadic(record):
    """A stored dyadic rational {"num": "...", "frac_bits": f} as a Fraction."""
    return Fraction(int(record["num"]), 1 << record["frac_bits"])


def sqrt_dim_up(dim):
    scaled = dim << (2 * FIXED_FRAC_BITS)
    root = isqrt(scaled)
    if root * root != scaled:
        root += 1
    return Fraction(root, 1 << FIXED_FRAC_BITS)


def rice_parameter(variance):
    """The Golomb-Rice parameter for a Gaussian of the given variance.

    k is the largest integer >= 0 with 1.2863 * 2^k <= sigma.  This is a
    large-width rule: the width at which k and k + 1 tie tends to 2.5726 * 2^k
    and comes earlier for small widths.  tools/rice_parameters.py derives it,
    finds a loss against the optimal parameter of at most 0.064 bits per
    coefficient for sigma >= 8, and checks the stored profiles against a
    discrete-Gaussian model of each component.  Evaluated exactly:
    (1.2863 * 2^k)^2 <= sigma^2.
    """
    k = 0
    while RICE_C * RICE_C * 4 ** (k + 1) <= variance:
        k += 1
    return k


def l1_bound(admissibility):
    """L with B2 = L B1."""
    if admissibility["predicate"] == "paper":
        return admissibility["beta_lambda"] * admissibility["omega_lambda"]
    return admissibility["l1_cap"]


def linf_bound(admissibility):
    """The bound on |lambda|_inf that the predicate implies."""
    if admissibility["predicate"] == "paper":
        return admissibility["beta_lambda"]
    return admissibility["l1_cap"]


def derive_bounds(data):
    """The bounds of the module documentation, as integers."""
    d, ell = data["ring"]["d"], data["ell"]
    sec, g, t = data["sec"], data["gaussian"], data["tails"]
    omega_c = data["challenge"]["omega_c"]
    sqrt_d = sqrt_dim_up(d * ell)
    beta_s = dyadic(t["theta_s"]) * dyadic(g["sigma_s"]) * sqrt_d
    beta_big_r = dyadic(t["theta_big_r"]) * dyadic(g["sigma_big_r"])
    beta_sh = sec["beta_g"] * beta_s + sec["c_exp"] * beta_big_r
    beta_r = dyadic(t["theta_r"]) * dyadic(g["sigma_r"]) * sqrt_d
    b1 = ceil_fraction(sqrt_d * beta_sh * omega_c + beta_r)
    b2 = b1 * l1_bound(data["admissibility"])
    return {
        "beta_sh": ceil_fraction(beta_sh),
        "b1": b1,
        "b2": b2,
        "b_msis_squared": 4 * (b2 * b2 + omega_c),
    }


def derive_encoding(data):
    """Rice parameters: shares of parties with |g_i| = j have variance
    j^2 sigma_s^2 + c_exp sigma_R^2; partial responses sigma_r^2 + omega_c
    times the largest share variance."""
    sec, g = data["sec"], data["gaussian"]
    s2 = dyadic(g["sigma_s"]) ** 2
    big_r2 = dyadic(g["sigma_big_r"]) ** 2

    def share_variance(j):
        return j * j * s2 + sec["c_exp"] * big_r2

    response = dyadic(g["sigma_r"]) ** 2 + data["challenge"]["omega_c"] * share_variance(sec["beta_g"])
    return {
        "rice_share": [rice_parameter(share_variance(j)) for j in range(sec["beta_g"] + 1)],
        "rice_response": rice_parameter(response),
    }


def compute_digest(d):
    """The SHAKE256 digest of every protocol-relevant field, in a fixed order.
    Integers that may exceed 64 bits are absorbed as decimal strings."""
    x = Xof(DOMAIN_PROFILE)
    x.absorb(d["format"].encode()).absorb(d["id"].encode()).absorb(d["category"].encode())
    x.absorb_u64(d["kappa"]).absorb_u64(d["ring"]["d"])
    x.absorb(d["ring"]["q"].encode()).absorb(d["ring"]["psi"].encode())
    x.absorb_u64(d["p"]).absorb_u64(d["ell"])
    s = d["sec"]
    for value in (s["n"], s["k"], s["c_exp"], s["beta_g"], s["t_uf"], s["t_sig"]):
        x.absorb_u64(value)
    x.absorb_u64(0 if s["k_c_det"] is None else s["k_c_det"] + 1)
    x.absorb(bytes(v & 0xff for row in s["g"] for v in row))
    b = d["budgets"]
    for key in ("q_h", "q_sign_global", "q_participate"):
        x.absorb(b[key].encode())
    g = d["gaussian"]
    x.absorb(g["convention"].encode()).absorb(g["sampler"].encode())
    for key in ("sigma_s", "sigma_big_r", "sigma_r"):
        x.absorb(g[key]["num"].encode()).absorb_u64(g[key]["frac_bits"])
    x.absorb_u64(d["challenge"]["omega_c"]).absorb_u64(d["challenge"]["commitment_bytes"])
    e = d["encoding"]
    x.absorb_u64(len(e["rice_share"]))
    for k in e["rice_share"]:
        x.absorb_u64(k)
    x.absorb_u64(e["rice_response"])
    t = d["tails"]
    x.absorb(t["rule"].encode()).absorb_u64(t["kappa_big_r"]).absorb_u64(t["kappa_r"])
    for key in ("theta_s", "theta_big_r", "theta_r"):
        x.absorb(t[key]["num"].encode()).absorb_u64(t[key]["frac_bits"])
    a = d["admissibility"]
    if a["predicate"] == "paper":
        x.absorb(b"paper").absorb_u64(a["beta_lambda"]).absorb_u64(a["omega_lambda"])
    else:
        x.absorb(b"l1").absorb_u64(a["l1_cap"])
    for key in ("beta_sh", "b1", "b2", "b_msis_squared"):
        x.absorb(d["bounds"][key].encode())
    r = d["recon"]
    x.absorb(r["algorithm"].encode())
    for key in ("lll_delta_num", "lll_delta_den", "passes", "window"):
        x.absorb_u64(r[key])
    x.absorb(d["xof"].encode()).absorb_u64(d["codec_version"])
    return x.digest(DIGEST_BYTES)


class Profile:
    """A loaded and checked profile.  The JSON record is kept in `data`."""

    def __init__(self, data):
        self.data = data
        self._check_conventions()
        self.id = data["id"]
        self.kappa = data["kappa"]
        self.d = data["ring"]["d"]
        self.q = int(data["ring"]["q"])
        self.psi = int(data["ring"]["psi"])
        self.p, self.ell = data["p"], data["ell"]
        sec = data["sec"]
        self.n, self.k = sec["n"], sec["k"]
        self.c_exp, self.beta_g = sec["c_exp"], sec["beta_g"]
        self.t_uf, self.t_sig = sec["t_uf"], sec["t_sig"]
        self.g_rows = sec["g"]
        self.omega_c = data["challenge"]["omega_c"]
        self.commitment_bytes = data["challenge"]["commitment_bytes"]
        self.admissibility = data["admissibility"]
        g = data["gaussian"]
        self.sigma_s, self.sigma_big_r, self.sigma_r = (g["sigma_s"], g["sigma_big_r"], g["sigma_r"])
        recon = data["recon"]
        self.recon_params = (recon["lll_delta_num"], recon["lll_delta_den"], recon["passes"], recon["window"])
        self.q_participate = int(data["budgets"]["q_participate"])
        self.supports_signing = data["category"] != "sec-only"

        self.encoding = derive_encoding(data)
        self.bounds = derive_bounds(data)
        if self.encoding != data["encoding"]:
            raise ValueError(f"profile {self.id}: Golomb-Rice parameters do not follow from the widths")
        if {key: str(v) for key, v in self.bounds.items()} != data["bounds"]:
            raise ValueError(f"profile {self.id}: stored bounds differ from the derived ones")
        self.digest = compute_digest(data)
        if self.digest.hex() != data["digest"]:
            raise ValueError(f"profile {self.id}: digest mismatch")
        self.beta_sh, self.b1, self.b2 = (self.bounds["beta_sh"], self.bounds["b1"], self.bounds["b2"])

    @staticmethod
    def load(path):
        with open(path) as f:
            return Profile(json.load(f))

    def _check_conventions(self):
        d = self.data
        if d["format"] != PROFILE_FORMAT or d["gaussian"]["convention"] != GAUSSIAN_CONVENTION:
            raise ValueError("unsupported profile format or Gaussian convention")
        if d["gaussian"]["sampler"] != SAMPLER_ID or d["recon"]["algorithm"] != RECON_ALGORITHM_ID:
            raise ValueError("unknown sampler or reconstruction algorithm")
        if d["xof"] != XOF_ID or d["codec_version"] != CODEC_VERSION:
            raise ValueError("unknown hash construction or codec version")
        if d["category"] not in CATEGORY_TAGS:
            raise ValueError("unknown profile category")

    @property
    def dim(self):
        return self.d * self.ell

    def share_rice_parameter(self, party):
        return self.encoding["rice_share"][abs(self.g_rows[party][0])]
