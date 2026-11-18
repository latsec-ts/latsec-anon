"""build_profiles.py -- re-derive the standard profiles from their inputs.

Every profile in profiles/ is rebuilt from the published inputs and the
matrix fixtures in fixtures/author/, independently of the Rust builder, and
its protocol-relevant fields (everything the profile digest covers) are
compared with the stored file:

  * author profiles: Table 4 of the paper (the rows of params.PUBLISHED),
    widths from the width optimizer of params.py (60-digit decimals);
  * author variants: the same widths with q the largest NTT-friendly prime
    below 2^(b_q - 1) (our variant, not a parameter set of the paper);
  * functional profiles: the hand-chosen test parameters below;
  * SEC-only profiles: the privacy-only sets of the SEC paper [46] for Q = 2^64 and a
    32-bit modulus, widths from solve_sec_widths() below.

Widths and tail factors are rounded upward to dyadic rationals exactly as
the profiles record them: ceil(x (1 + 2^-40) 2^f) / 2^f, with f = 16 for
widths below 2^32, f = 0 for wider ones, and f = 32 for tail factors.  The
relative guard 2^-40 absorbs floating-point differences between the two
implementations.

Usage:  python3 build_profiles.py [--profiles DIR] [--fixtures DIR]
Exit status 0 iff every stored profile is reproduced.
"""

import argparse
import json
import math
import os
import sys

import params
from codec import CODEC_VERSION
from profiles import (GAUSSIAN_CONVENTION, PROFILE_FORMAT, RECON_ALGORITHM_ID, SAMPLER_ID, XOF_ID,
                      compute_digest, derive_bounds, derive_encoding)
from ring import canonical_root, largest_ntt_prime_below

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LN_2 = math.log(2.0)

# k_c (the signing threshold) and k_c_det (from which every set is authorized)
# of the published instances; the rest of the geometry is params.GEOMETRY.
THRESHOLDS = {40: (26, 35), 60: (46, 56)}
MATRIX_FIXTURE = {10: "G-10-5-3.json", 20: "G-20-9-3.json", 40: "G-40-19-3.json", 60: "G-60-39-3.json"}

# Functional profiles: (id, n, d, modulus bits, p, ell, t_sig, k_c_det, lg Q_H,
# lg Q_Sign, sigma_s = sigma_R, sigma_r, omega_c, l1 cap).
FUNCTIONAL = [
    ("functional-n10-d16", 10, 16, 30, 2, 4, 7, 9, 20, 20, 8.0, 1048576.0, 4, 4),
    ("functional-n20-d16", 20, 16, 31, 2, 4, 14, 16, 20, 20, 8.0, 1048576.0, 4, 8),
    ("functional-n20-d512", 20, 512, 66, 2, 4, 14, 16, 63, 62, 46000.0, float(2**56), 30, 8),
]


# --- Rounding as stored in the profiles -------------------------------------------------

def round_up(x, frac_bits):
    """{"num", "frac_bits"} for ceil(x (1 + 2^-40) 2^frac_bits) / 2^frac_bits."""
    guarded = float(x) * (1.0 + 2.0 ** -40)
    return {"num": str(math.ceil(guarded * 2.0 ** frac_bits)), "frac_bits": frac_bits}


def width(sigma):
    return round_up(sigma, 16 if float(sigma) < 2.0 ** 32 else 0)


def theta_closed(kappa_tail, dim):
    """The closed-form tail factor 1 + sqrt(2 kappa ln 2 / N) ([46, Eqs.
    (64)-(66)]), in double precision."""
    return 1.0 + math.sqrt(2.0 * (kappa_tail * LN_2) / dim)


def ceil_lg(x):
    return (x - 1).bit_length()


# --- The LaTSEC width optimizer (Section 7.2; the parameter-selection notebook) ---------
#
# params.solve() evaluates the same optimum in 60-digit decimals.  The stored
# widths come from a double-precision solver, whose result can differ from the
# decimal one by more than the 2^-40 guard, so this port repeats its floating-
# point operations in the same order.

def smoothing_threshold(dim, eps_bits):
    """S(N) = sqrt(ln(2N) + eps_bits ln 2) / pi (sigma convention)."""
    return math.sqrt(math.log(2.0 * dim) + eps_bits * LN_2) / math.pi


def solve_widths(n, kappa, ell, d=512):
    """(sigma_s, sigma_R, sigma_r) minimizing B1 = Cx/sqrt(x) + Cy/sqrt(y) + Cz/sqrt(z)
    subject to x + a y + b z <= P (MQ), x + c z <= Q and y + c z <= Q (AK)."""
    geo = params.GEOMETRY[n]
    k, k_p, c_exp, beta_g = geo["k"], geo["k_p"], geo["c_exp"], 1
    q_h, q_sign = params.QUERY_BUDGETS[kappa]
    omega_c = next(w for w in range(1, d + 1) if 2 ** w * math.comb(d, w) >= q_h * 2 ** kappa)
    alpha, norm_g_hat, norm_g = float(geo["alpha_kp"]), float(geo["norm_g_hat"]), float(geo["norm_g"])
    beta_q = float(geo["beta_q"])
    dim = d * ell
    dimf = float(dim)
    sqrt_dim = math.sqrt(dimf)
    omega, qs = float(omega_c), float(q_sign)
    theta_s = theta_closed(kappa, dim)
    theta_big_r = theta_closed(kappa + ceil_lg((k - 1) * dim), 1)
    theta_r = theta_closed(kappa + ceil_lg(n * q_sign), dim)
    s_mq = smoothing_threshold(dim, kappa + 2) * 1.0
    s_ak = smoothing_threshold(k * dim, kappa + 2) * 1.0
    cx = omega * dimf * theta_s * float(beta_g)
    cy = omega * float(c_exp) * theta_big_r * sqrt_dim
    cz = theta_r * sqrt_dim
    a = float(k_p) / (alpha * alpha)
    t = math.sqrt(float(n)) + math.sqrt(float(k_p)) * norm_g_hat / alpha
    b = omega * omega * qs * (t * t)
    c = omega * omega * qs * norm_g * norm_g
    p = 1.0 / (2.0 * s_mq * s_mq)
    q = 1.0 / (2.0 * beta_q * beta_q * s_ak * s_ak)
    x, y, z = _minimize(cx, cy, cz, a, b, c, p, q)
    return 1.0 / math.sqrt(x), 1.0 / math.sqrt(y), 1.0 / math.sqrt(z)


def _inner(p0, q0, a, r):
    if p0 > (1.0 + a) * q0:
        return q0, q0, "both-caps"
    x0, y0 = r * p0 / (a + r), p0 / (a + r)
    if x0 <= q0 and y0 <= q0:
        return x0, y0, "interior"
    if x0 > q0:
        return q0, (p0 - q0) / a, "clamp-s"
    return p0 - a * q0, q0, "clamp-r"


def _two_term(a, b, kap, m):
    """argmin A/sqrt(u) + B/sqrt(w) subject to u + kap w = m; returns w."""
    rho = (kap * a / b) ** (2.0 / 3.0)
    return m / (rho + kap)


def _minimize(cx, cy, cz, a, b, c, p, q):
    """The notebook's minimize_Gxyz_fast: scan the pieces of the inner problem
    in z, with closed forms where they exist and golden section otherwise."""
    z_unit = min(p / b, q / c)
    cx, cy, cz = cx / math.sqrt(q), cy / math.sqrt(q), cz / math.sqrt(z_unit)
    b, c = b * z_unit / q, c * z_unit / q
    p_scaled, q_scaled = p / q, 1.0
    r = (a * cx / cy) ** (2.0 / 3.0)

    def xy(z):
        return _inner(p_scaled - b * z, q_scaled - c * z, a, r)

    def phi(z):
        x, y, _ = xy(z)
        return cx / math.sqrt(x) + cy / math.sqrt(y) + cz / math.sqrt(z)

    edge = 1e-12
    breakpoints = []
    for th in (a + r, (a + r) / r, 1.0 + a):
        if abs(th * c - b) > 1e-13 * max(th * c, b):
            t = (th * q_scaled - p_scaled) / (th * c - b)
            if edge < t < 1.0 - edge:
                breakpoints.append(t)
    breakpoints = sorted(set(breakpoints))
    edges = [edge] + breakpoints + [1.0 - edge]
    best = None

    def consider(z):
        nonlocal best
        try:
            v = phi(z)
        except ValueError:
            return
        if math.isfinite(v) and (best is None or v < best[1]):
            best = (z, v)

    for lo, hi in zip(edges, edges[1:]):
        if hi - lo <= 2.0 * edge:
            continue
        case = xy((lo + hi) / 2.0)[2]
        if case == "interior":
            c1 = math.sqrt(a + r) * (cx / math.sqrt(r) + cy)
            z = _two_term(c1, cz, b, p_scaled)
        elif case == "both-caps":
            z = _two_term(cx + cy, cz, c, q_scaled)
        else:
            z = golden(phi, lo, hi)
        if not lo <= z <= hi:
            z = lo if phi(lo) < phi(hi) else hi
        consider(z)
    for t in breakpoints:
        consider(t)
    z = best[0]
    x, y, _ = xy(z)
    return x * q, y * q, z * z_unit


# --- The SEC-only width selection ([46, Section 6]) -------------------------------------

def golden(f, lo, hi):
    """Golden-section minimum of a unimodal function: 200 steps."""
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, e = b - inv_phi * (b - a), a + inv_phi * (b - a)
    fc, fe = f(c), f(e)
    for _ in range(200):
        if fc < fe:
            b, e, fe = e, c, fc
            c = b - inv_phi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, e, fe
            e = a + inv_phi * (b - a)
            fe = f(e)
    return (a + b) / 2.0


def solve_sec_widths(n, k, k_p, c_exp, beta_g, ell, kappa, queries, omega_c, geo, d=512):
    """Minimize the share width times sigma_r subject to multi-query privacy
    x + a y + b z <= P and kernel smoothing y + c z <= K, where x = 1/sigma_s^2,
    y = 1/sigma_R^2 and z = 1/sigma_r^2.  eps = eps' = 2^-(kappa + 3).
    Returns (sigma_s, sigma_R, sigma_r)."""
    alpha, norm_g_hat = float(geo["alpha_kp"]), float(geo["norm_g_hat"])
    beta_kp, gamma_kp = float(geo["beta_kp"]), float(geo["gamma_kp"])
    eps_bits = kappa + 3
    dim = d * ell
    omega, q = float(omega_c), float(queries)
    beta_g_sq = float(beta_g) ** 2
    w_x, w_y = beta_g_sq, float(c_exp)
    a = beta_g_sq * k_p / (alpha * alpha)
    b = beta_g_sq * omega * omega * q * (math.sqrt(n) + math.sqrt(k_p) * norm_g_hat / alpha) ** 2
    c = omega * omega * q * norm_g_hat * norm_g_hat
    s_mq = math.sqrt(math.log(2.0 * dim) + eps_bits * LN_2) / math.pi
    geometry = math.sqrt(1.0 + min(float(k_p), beta_kp * beta_kp) / (gamma_kp * gamma_kp))
    log_term = math.log(2.0 * (k - 1) * dim) + eps_bits * LN_2 + math.log1p(2.0 * 2.0 ** -eps_bits)
    tau_ker = geometry * math.sqrt(log_term / (2.0 * math.pi * math.pi))
    p_budget, k_budget = 1.0 / (2.0 * s_mq * s_mq), 1.0 / (tau_ker * tau_ker)

    def inner(p0, k0):
        r = math.sqrt(a * w_x / w_y)
        y = p0 / (a + r)
        return (r * y, y) if y <= k0 else (p0 - a * k0, k0)

    hi = math.log(min(p_budget / b, k_budget / c))
    lo = hi - 64.0 * LN_2

    def phi(u):
        z = math.exp(u)
        x, y = inner(p_budget - b * z, k_budget - c * z)
        return math.log2(w_x / x + w_y / y) / 2.0 - math.log2(z) / 2.0

    z = math.exp(golden(phi, lo, hi))
    x, y = inner(p_budget - b * z, k_budget - c * z)
    return 1.0 / math.sqrt(x), 1.0 / math.sqrt(y), 1.0 / math.sqrt(z)


# --- Profile records ---------------------------------------------------------------------

def load_matrix(fixtures, n):
    with open(os.path.join(fixtures, "author", MATRIX_FIXTURE[n])) as f:
        return json.load(f)


def record(pid, category, kappa, d, q, p, ell, matrix, t_sig, k_c_det, q_h, q_sign,
           sigma_s, sigma_big_r, sigma_r, omega_c, admissibility):
    """The protocol-relevant part of a profile, with derived values filled in."""
    g = matrix["g"]
    n, k = len(g), len(g[0])
    dim = d * ell
    kappa_big_r = kappa + ceil_lg((k - 1) * dim)
    kappa_r = kappa + ceil_lg(n * q_sign)
    data = {
        "format": PROFILE_FORMAT,
        "id": pid,
        "category": category,
        "kappa": kappa,
        "ring": {"d": d, "q": str(q), "psi": str(canonical_root(d, q))},
        "p": p,
        "ell": ell,
        "sec": {
            "n": n, "k": k,
            "c_exp": sum(1 for x in g[0][1:] if x),
            "beta_g": max(abs(row[0]) for row in g),
            "t_uf": matrix["k_p"], "t_sig": t_sig, "k_c_det": k_c_det, "g": g,
        },
        "budgets": {"q_h": str(q_h), "q_sign_global": str(q_sign), "q_participate": str(q_sign // n)},
        "gaussian": {"convention": GAUSSIAN_CONVENTION, "sampler": SAMPLER_ID,
                     "sigma_s": width(sigma_s), "sigma_big_r": width(sigma_big_r), "sigma_r": width(sigma_r)},
        "challenge": {"omega_c": omega_c, "commitment_bytes": (kappa + 3) // 4},
        "tails": {
            "rule": "closed-form", "kappa_big_r": kappa_big_r, "kappa_r": kappa_r,
            "theta_s": round_up(theta_closed(kappa, dim), 32),
            "theta_big_r": round_up(theta_closed(kappa_big_r, 1), 32),
            "theta_r": round_up(theta_closed(kappa_r, dim), 32),
        },
        "admissibility": admissibility,
        "recon": {"algorithm": RECON_ALGORITHM_ID, "lll_delta_num": 99, "lll_delta_den": 100,
                  "passes": 2, "window": 2},
        "xof": XOF_ID,
        "codec_version": CODEC_VERSION,
    }
    data["encoding"] = derive_encoding(data)
    data["bounds"] = {key: str(v) for key, v in derive_bounds(data).items()}
    data["digest"] = compute_digest(data).hex()
    return data


def author_records(fixtures):
    out = []
    for row in params.PUBLISHED:
        n, kappa = row["n"], row["kappa"]
        geo = params.GEOMETRY[n]
        sol = params.solve(n, kappa, row["ell"])
        sigma_s, sigma_big_r, sigma_r = solve_widths(n, kappa, row["ell"])
        q_h, q_sign = params.QUERY_BUDGETS[kappa]
        k_c, k_c_det = THRESHOLDS[n]
        adm = {"predicate": "paper", "beta_lambda": geo["beta_lambda"], "omega_lambda": geo["omega_lambda"]}
        matrix = load_matrix(fixtures, n)
        for pid, category, q in (
            (f"author-n{n}-k{kappa}-nov2026", "author", row["q"]),
            (f"variant-n{n}-k{kappa}-q{row['b_q'] - 1}", "author-variant",
             largest_ntt_prime_below(row["b_q"] - 1, 512)),
        ):
            out.append(record(pid, category, kappa, 512, q, row["p"], row["ell"], matrix, k_c, k_c_det,
                              q_h, q_sign, sigma_s, sigma_big_r, sigma_r, sol["omega_c"], adm))
    return out


def functional_records(fixtures):
    out = []
    for (pid, n, d, bits, p, ell, t_sig, k_c_det, lg_qh, lg_qs, sigma_s, sigma_r, omega_c, cap) in FUNCTIONAL:
        out.append(record(pid, "functional", 128, d, largest_ntt_prime_below(bits, d), p, ell,
                          load_matrix(fixtures, n), t_sig, k_c_det, 1 << lg_qh, 1 << lg_qs,
                          sigma_s, sigma_s, sigma_r, omega_c, {"predicate": "l1", "l1_cap": cap}))
    return out


def sec_only_records(fixtures):
    """The sets of the SEC paper [46] with Q = 2^64 leakage queries, 32-bit modulus, kappa = 128."""
    out = []
    kappa, ell, queries = 128, 6, 1 << 64
    omega_c = next(w for w in range(1, 513) if 2 ** w * math.comb(512, w) >= 2 ** kappa)
    for n in (40, 60):
        geo = params.GEOMETRY[n]
        k_c, k_c_det = THRESHOLDS[n]
        sigma_s, sigma_big_r, sigma_r = solve_sec_widths(
            n, geo["k"], geo["k_p"], geo["c_exp"], 1, ell, kappa, queries, omega_c, geo)
        adm = {"predicate": "paper", "beta_lambda": geo["beta_lambda"], "omega_lambda": geo["omega_lambda"]}
        out.append(record(f"sec-only-n{n}-k{kappa}-q32", "sec-only", kappa, 512, largest_ntt_prime_below(32, 512),
                          ell // 2, ell, load_matrix(fixtures, n), k_c, k_c_det, 1, queries,
                          sigma_s, sigma_big_r, sigma_r, omega_c, adm))
    return out


PROTOCOL_FIELDS = ("format", "id", "category", "kappa", "ring", "p", "ell", "challenge",
                   "encoding", "tails", "admissibility", "bounds", "recon", "xof", "codec_version", "digest")


def compare(built, stored):
    """The protocol fields in which a rebuilt profile differs from the stored one."""
    diff = [key for key in PROTOCOL_FIELDS if built[key] != stored[key]]
    for key in ("q_h", "q_sign_global", "q_participate"):
        if built["budgets"][key] != stored["budgets"][key]:
            diff.append("budgets." + key)
    for key in ("convention", "sampler", "sigma_s", "sigma_big_r", "sigma_r"):
        if built["gaussian"][key] != stored["gaussian"][key]:
            diff.append("gaussian." + key)
    for key in ("n", "k", "c_exp", "beta_g", "t_uf", "t_sig", "k_c_det", "g"):
        if built["sec"][key] != stored["sec"][key]:
            diff.append("sec." + key)
    return diff


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--profiles", default=os.path.join(REPO, "profiles"))
    ap.add_argument("--fixtures", default=os.path.join(REPO, "fixtures"))
    args = ap.parse_args()
    built = functional_records(args.fixtures) + author_records(args.fixtures) + sec_only_records(args.fixtures)
    stored_ids = sorted(f[:-5] for f in os.listdir(args.profiles) if f.endswith(".json"))
    failures = 0
    if sorted(r["id"] for r in built) != stored_ids:
        print(f"FAIL the stored profiles are {stored_ids}")
        failures += 1
    for rec in built:
        with open(os.path.join(args.profiles, rec["id"] + ".json")) as f:
            stored = json.load(f)
        diff = compare(rec, stored)
        failures += bool(diff)
        print(f"{'OK  ' if not diff else 'FAIL'} {rec['id']}" + (f" differs in {diff}" if diff else ""))
    print("all profiles reproduced" if not failures else f"{failures} profile(s) not reproduced")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
