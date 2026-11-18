#!/usr/bin/env python3
"""Independent port of the LaTSEC width optimizer, for regression.

Sources
-------
* The paper, Sections 4.1 and 7 and Appendix N of the full version.
* parameter-search/LaTSEC-Parameter-Selection-v2.3-fixedtailbound+privacy.ipynb of this
  artifact (inputs: cell 56; expected outputs: cells 194-246).

What it does
------------
Recomputes the Gaussian widths, norm bounds, primes and ideal sizes of Tables 4 and 5
from the notebook's inputs, and compares them with the published values (the notebook
prints four decimals, the paper three).  It also reports three variants: exact
inversion of the tail inequality, the modulus floor under ``B_MSIS < q`` (Section
7.2), and the kernel-smoothing condition.  It does not estimate lattice hardness; see latsec_estimates.py.

Smoothing allocations (Section 7.1)
-----------------------------------
The paper uses two smoothing parameters: eps_AK = 2^-(kappa+2) for the abort keys
(item 3) and eps = eps' = 2^-(kappa+3) for the key-privacy requirements of SEC
(item 4).  The width solver follows the notebook, which uses kappa + 2 for both
thresholds; this decides nothing, because only the abort-key constraint binds.
solve() is therefore left exactly as the notebook has it, and the conditions at the
paper's allocations are checked separately in run_checks(): the bound
eps_AK <= 1/sqrt(2kD + 4) of Lemma 1, the multi-query smoothing condition and the
kernel-smoothing condition at kappa + 3.

Only the Python standard library is used.  Run ``python3 latsec-py/params.py``;
the exit status is non-zero if any published value is not reproduced.
build_profiles.py uses this module to re-derive the stored profiles.
"""

import argparse
import json
import sys
from decimal import Decimal, getcontext
from math import comb, floor

getcontext().prec = 60

PI = Decimal("3.14159265358979323846264338327950288419716939937510582097494")
LN2 = Decimal(2).ln()
HALF_LG_2PIE = (2 * PI * Decimal(1).exp()).ln() / LN2 / 2

RING_DEGREE = 512
SEED_BITS = 256
KIB_BITS = Decimal(8 * 1024)

# Matrix constants at the full float64 precision used by the v2.3 notebook (cell 56).
# beta_kp and gamma_kp are from SEC-Parameter-Selection-v1.ipynb (same matrices).
GEOMETRY = {
    40: dict(
        n=40, k=20, k_p=5, c_exp=3, beta_lambda=5, omega_lambda=25,
        alpha_kp=Decimal("0.464255900330341"),
        norm_g_hat=Decimal("4.4316793740317"),
        norm_g=Decimal("5.86070678667877"),
        beta_q=Decimal("20.6352940186031"),
        beta_kp=Decimal("2.9403658464661486"),
        gamma_kp=Decimal("0.19253138628300923"),
    ),
    60: dict(
        n=60, k=40, k_p=6, c_exp=3, beta_lambda=9, omega_lambda=45,
        alpha_kp=Decimal("0.517638090205041"),
        norm_g_hat=Decimal("3.93699368153548"),
        norm_g=Decimal("7.84160154378415"),
        beta_q=Decimal("26.7593053884503"),
        beta_kp=Decimal("2.893489471774352"),
        gamma_kp=Decimal("0.15210842969369098"),
    ),
}

# kappa -> (Q_H, Q_Sign)
QUERY_BUDGETS = {128: (2**63, 2**62), 192: (2**64, 2**64), 256: (2**64, 2**64)}

# Published values: Table 4 (four decimals from notebook cell 204), exact primes
# (cell 198), Table 5 (cell 246), and the estimator outputs of the notebook (cell 194).
PUBLISHED = [
    dict(n=40, kappa=128, ell=12, p=6, b_q=61, q=2**61 - 10239,
         lg_sigma_s=15.4641, lg_sigma_r=45.0134, lg_b1=51.5813, lg_b2=58.5470,
         sizes=(22.906, 22.875, 915.000, 13.883, 35.525, 40.750, 58.408),
         sec=dict(MQ=168.5, AK=152.4, MSIS=153.6)),
    dict(n=40, kappa=192, ell=16, p=8, b_q=63, q=2**63 - 17407,
         lg_sigma_s=15.9815, lg_sigma_r=46.8267, lg_b1=53.6024, lg_b2=60.5682,
         sizes=(31.531, 31.500, 1260.000, 19.029, 49.181, 56.147, 80.696),
         sec=dict(MQ=237.7, AK=216.1, MSIS=217.5)),
    dict(n=40, kappa=256, ell=20, p=10, b_q=64, q=2**64 - 1023,
         lg_sigma_s=16.1146, lg_sigma_r=47.4651, lg_b1=54.4011, lg_b2=61.3669,
         sizes=(40.031, 40.000, 1600.000, 23.952, 62.274, 70.981, 102.297),
         sec=dict(MQ=312.1, AK=284.7, MSIS=289.1)),
    dict(n=60, kappa=128, ell=12, p=6, b_q=64, q=2**64 - 1023,
         lg_sigma_s=15.9834, lg_sigma_r=45.8133, lg_b1=52.3811, lg_b2=61.0429,
         sizes=(24.031, 24.000, 1440.000, 14.273, 36.125, 42.622, 60.133),
         sec=dict(MQ=163.5, AK=142.2, MSIS=146.6)),
    dict(n=60, kappa=192, ell=16, p=8, b_q=66, q=2**66 - 16383,
         lg_sigma_s=16.4994, lg_sigma_r=47.6251, lg_b1=54.4008, lg_b2=63.0626,
         sizes=(33.031, 33.000, 1980.000, 19.547, 49.979, 58.641, 82.995),
         sec=dict(MQ=231.0, AK=202.6, MSIS=207.9)),
    dict(n=60, kappa=256, ell=20, p=10, b_q=66, q=2**66 - 16383,
         lg_sigma_s=16.6318, lg_sigma_r=48.2627, lg_b1=55.1987, lg_b2=63.8605,
         sizes=(41.281, 41.250, 2475.000, 24.599, 63.271, 74.098, 104.544),
         sec=dict(MQ=310.4, AK=273.3, MSIS=271.6)),
]

SIZE_NAMES = ("vk", "ak_i", "ak", "sk_i", "sig_i", "sig", "comm_i")

# Widths of an earlier draft, used only for the historical cross-check.
OCTOBER_WIDTHS = {(40, 128): (16.6855, 46.3392), (60, 128): (17.2046, 47.1391)}


def lg(x):
    """Base-two logarithm as a Decimal."""
    return Decimal(x).ln() / LN2


def ceil_lg(x):
    """ceil(log2(x)) for a positive integer x."""
    return (x - 1).bit_length()


def is_prime(n):
    """Deterministic Miller-Rabin; the base set is exact for n < 3.3e24."""
    bases = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
    if n < 2:
        return False
    for base in bases:
        if n % base == 0:
            return n == base
    odd, twos = n - 1, 0
    while odd % 2 == 0:
        odd //= 2
        twos += 1
    for base in bases:
        x = pow(base, odd, n)
        if x in (1, n - 1):
            continue
        for _ in range(twos - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def largest_ntt_prime_below(b_q, degree=RING_DEGREE):
    """Largest prime q < 2^b_q with q = 1 (mod 2 * degree)."""
    step = 2 * degree
    q = 2**b_q - 1
    q -= (q - 1) % step
    while not is_prime(q):
        q -= step
    return q


def challenge_weight(kappa, degree=RING_DEGREE):
    """Smallest w with 2^w * binom(degree, w) >= Q_H * 2^kappa."""
    q_h = QUERY_BUDGETS[kappa][0]
    for weight in range(1, degree + 1):
        if 2**weight * comb(degree, weight) >= q_h * 2**kappa:
            return weight
    raise ValueError("no challenge weight found")


def theta_closed(kappa_tail, dim):
    """[46, Eqs. (64)-(66)] (Section 4.1, item 3): 1 + sqrt(2 kappa ln2 / N)."""
    return 1 + (2 * Decimal(kappa_tail) * LN2 / Decimal(dim)).sqrt()


def theta_exact(kappa_tail, dim):
    """Smallest theta with N((theta^2 - 1)/2 - ln theta) >= kappa ln2, rounded up."""
    def slack(theta):
        return Decimal(dim) * ((theta * theta - 1) / 2 - theta.ln()) - Decimal(kappa_tail) * LN2

    low, high = Decimal(1), Decimal(64)
    for _ in range(200):
        mid = (low + high) / 2
        if slack(mid) >= 0:
            high = mid
        else:
            low = mid
    return high


def theta_october(kappa_tail, dim):
    """An earlier draft: min{1 + L/(3N), 1 + sqrt(4L/N)}; the linear branch is invalid."""
    budget = Decimal(kappa_tail) * LN2
    return min(1 + budget / (3 * Decimal(dim)), 1 + (4 * budget / Decimal(dim)).sqrt())


def tail_exponent_bits(theta, dim):
    """-log2 of C_tail(theta)^N, i.e. the tail security the factor actually certifies."""
    return Decimal(dim) * ((theta * theta - 1) / 2 - theta.ln()) / LN2


# eps = 2^-(kappa + extra): Section 7.1, items 3 and 4.
AK_EPS_EXTRA_BITS = 2
SEC_EPS_EXTRA_BITS = 3


def smoothing_threshold(dim, kappa, extra_bits=AK_EPS_EXTRA_BITS):
    """S(N) = sqrt(2) * eta_eps(Z^N) in the sigma convention, eps = 2^-(kappa + extra_bits)."""
    return (Decimal(2 * dim).ln() + Decimal(kappa + extra_bits) * LN2).sqrt() / PI


def eps_ak_within_lemma_bound(kappa, k, dim):
    """eps_AK = 2^-(kappa+2) <= 1/sqrt(2 k D + 4) (Lemma 1 with N = kD), exactly."""
    return 4 ** (kappa + AK_EPS_EXTRA_BITS) >= 2 * k * dim + 4


def solve(n, kappa, ell, theta=theta_closed, threshold_scale=Decimal(1)):
    """Minimize B1 under the MQ and AK constraints (solver case 4 of the notebook).

    Returns a dict of Decimals.  Raises AssertionError if case 4 does not apply.
    """
    geo = GEOMETRY[n]
    dim = RING_DEGREE * ell
    k, k_p, c_exp = geo["k"], geo["k_p"], geo["c_exp"]
    _, q_sign = QUERY_BUDGETS[kappa]
    omega_c = challenge_weight(kappa)
    sqrt_dim = Decimal(dim).sqrt()

    a_mq = k_p / geo["alpha_kp"] ** 2
    b_mq = omega_c**2 * q_sign * (
        Decimal(n).sqrt() + Decimal(k_p).sqrt() * geo["norm_g_hat"] / geo["alpha_kp"]) ** 2
    b_ak = omega_c**2 * q_sign * geo["norm_g"] ** 2
    s_mq = smoothing_threshold(dim, kappa) * threshold_scale
    s_ak = smoothing_threshold(k * dim, kappa) * threshold_scale
    budget_mq = 1 / (2 * s_mq**2)
    budget_ak = 1 / (2 * geo["beta_q"] ** 2 * s_ak**2)

    kappa_big_r = kappa + ceil_lg((k - 1) * dim)
    kappa_r = kappa + ceil_lg(n * q_sign)
    theta_s = theta(kappa, dim)
    theta_big_r = theta(kappa_big_r, 1)
    theta_r = theta(kappa_r, dim)

    coeff_s = omega_c * dim * theta_s
    coeff_big_r = omega_c * c_exp * theta_big_r * sqrt_dim
    coeff_r = theta_r * sqrt_dim

    # Case 4: both AK constraints bind, so x_s = x_R = rho * x_r.
    rho = (b_ak * (coeff_s + coeff_big_r) / coeff_r) ** (Decimal(2) / 3)
    x_r = budget_ak / (rho + b_ak)
    x_s = rho * x_r
    mq_load = x_s + a_mq * x_s + b_mq * x_r
    assert mq_load <= budget_mq, "MQ constraint is active; solver case 4 does not apply"

    sigma_s = 1 / x_s.sqrt()
    sigma_r = 1 / x_r.sqrt()
    beta_s = theta_s * sigma_s * sqrt_dim
    beta_big_r = theta_big_r * sigma_s
    beta_r = theta_r * sigma_r * sqrt_dim
    beta_sh = beta_s + c_exp * beta_big_r                 # beta_g = 1
    b1 = omega_c * sqrt_dim * beta_sh + beta_r
    b2 = geo["omega_lambda"] * geo["beta_lambda"] * b1
    b_msis = 2 * (b2 * b2 + omega_c).sqrt()

    sigma_eff = 1 / (x_s + omega_c**2 * q_sign * geo["norm_g_hat"] ** 2 * x_r).sqrt()
    return dict(
        n=n, kappa=kappa, ell=ell, dim=dim, k=k, omega_c=omega_c,
        sigma_s=sigma_s, sigma_r=sigma_r,
        theta_s=theta_s, theta_big_r=theta_big_r, theta_r=theta_r,
        kappa_big_r=kappa_big_r, kappa_r=kappa_r,
        beta_r=beta_r, beta_sh=beta_sh, b1=b1, b2=b2, b_msis=b_msis,
        sigma_mq=1 / (2 * mq_load).sqrt(),
        sigma_ak=1 / (2 * geo["beta_q"] ** 2 * (x_s + b_ak * x_r)).sqrt(),
        s_mq=s_mq, s_ak=s_ak, sigma_eff=sigma_eff,
    )


def kernel_smoothing_ratio(sol, extra_bits=SEC_EPS_EXTRA_BITS):
    """sigma_eff divided by the explicit kernel-smoothing threshold
    ([46, Eq. (56)]), eps' = 2^-(kappa + extra_bits)."""
    geo = GEOMETRY[sol["n"]]
    geometry_factor = (
        1 + min(Decimal(geo["k_p"]), geo["beta_kp"] ** 2) / geo["gamma_kp"] ** 2).sqrt()
    eps_prime = Decimal(2) ** (-(sol["kappa"] + extra_bits))
    log_term = (2 * (sol["k"] - 1) * sol["dim"] * (2 + 1 / eps_prime)).ln()
    return sol["sigma_eff"] / (geometry_factor * (log_term / (2 * PI * PI)).sqrt())


def sizes_kib(sol, p, b_q):
    """(vk, ak_i, ak, sk_i, sig_i, sig, comm_i) in KiB, Appendix N of the full version."""
    dim, kappa = sol["dim"], sol["kappa"]
    c_exp = GEOMETRY[sol["n"]]["c_exp"]
    challenge_bits = Decimal(sol["omega_c"]) + lg(comb(RING_DEGREE, sol["omega_c"]))
    challenge_bits = challenge_bits.to_integral_value(rounding="ROUND_CEILING")

    def response_bits(bound):
        return dim * (lg(bound / Decimal(dim).sqrt()) + HALF_LG_2PIE)

    key_bits = p * RING_DEGREE * b_q
    bits = (
        SEED_BITS + key_bits,
        key_bits,
        sol["n"] * key_bits,
        dim * (lg(sol["sigma_s"]) + lg(1 + c_exp) / 2 + HALF_LG_2PIE),
        challenge_bits + response_bits(sol["b1"]),
        challenge_bits + response_bits(sol["b2"]),
        2 * kappa + key_bits + response_bits(sol["b1"]),
    )
    return tuple(Decimal(b) / KIB_BITS for b in bits)


def estimator_instance(row, b_q=None, theta=theta_closed):
    """Inputs for the lattice estimator, mapped as in the notebook's build_params.

    MLWE_{p, ell', q, d, sigma} -> LWE(n = ell' d, m = p d), MSIS -> SIS(n = p d,
    m = (ell + 1) d, norm 2).  Widths and the bound are floats, q is an exact int.
    """
    b_q = row["b_q"] if b_q is None else b_q
    sol = solve(row["n"], row["kappa"], row["ell"], theta=theta)
    p = row["p"]
    ell_prime = row["ell"] - p
    geo = GEOMETRY[row["n"]]
    return dict(
        n=row["n"], kappa=row["kappa"], ell=row["ell"], p=p, b_q=b_q,
        q=largest_ntt_prime_below(b_q),
        lwe_n=ell_prime * RING_DEGREE, lwe_m=p * RING_DEGREE,
        sis_n=p * RING_DEGREE, sis_m=(row["ell"] + 1) * RING_DEGREE,
        sigma_mq=float(sol["sigma_mq"]), sigma_ak=float(sol["sigma_ak"]),
        b_msis=float(sol["b_msis"]), lg_b_msis=float(lg(sol["b_msis"])),
        targets=dict(MQ=row["kappa"], AK=row["kappa"] + 1 + ceil_lg(geo["k"]),
                     MSIS=row["kappa"]),
    )


class Report:
    """Collects pass/fail checks and prints them as they are made."""

    def __init__(self):
        self.failures = 0

    def check(self, label, ok, detail=""):
        if not ok:
            self.failures += 1
        print(f"  [{'ok' if ok else 'FAIL'}] {label}{(': ' + detail) if detail else ''}")

    def note(self, label, detail):
        """A recorded difference from a printed value that is not a failed requirement."""
        print(f"  [note] {label}: {detail}")


def run_checks():
    report = Report()
    solutions = {(r["n"], r["kappa"]): solve(r["n"], r["kappa"], r["ell"]) for r in PUBLISHED}

    print("Table 4: widths and bounds (computed; published values are checked below)")
    print("   n kappa ell | lg sigma_s lg sigma_r | theta_s theta_R theta_r"
          " |   lg B1    lg B2  lg B_MSIS")
    for row in PUBLISHED:
        sol = solutions[(row["n"], row["kappa"])]
        print(f"  {row['n']:2d}  {row['kappa']:3d}  {row['ell']:2d} |"
              f"  {lg(sol['sigma_s']):9.4f}  {lg(sol['sigma_r']):9.4f} |"
              f"  {sol['theta_s']:.4f} {sol['theta_big_r']:7.3f}  {sol['theta_r']:.4f} |"
              f" {lg(sol['b1']):8.4f} {lg(sol['b2']):8.4f} {lg(sol['b_msis']):9.4f}")

    print("\nRegression against the published Tables 4 and 5")
    worst_width = Decimal(0)
    worst_size = Decimal(0)
    for row in PUBLISHED:
        sol = solutions[(row["n"], row["kappa"])]
        for name, value in (("lg_sigma_s", lg(sol["sigma_s"])), ("lg_sigma_r", lg(sol["sigma_r"])),
                            ("lg_b1", lg(sol["b1"])), ("lg_b2", lg(sol["b2"]))):
            worst_width = max(worst_width, abs(value - Decimal(str(row[name]))))
        for value, expected in zip(sizes_kib(sol, row["p"], row["b_q"]), row["sizes"]):
            worst_size = max(worst_size, abs(value - Decimal(str(expected))))
    report.check("24 width/bound entries within 5e-5", worst_width < Decimal("5e-5"),
                 f"largest deviation {worst_width:.2e}")
    report.check("42 size entries within 6e-4 KiB", worst_size < Decimal("6e-4"),
                 f"largest deviation {worst_size:.2e} KiB")
    report.check("challenge weights 30/44/60",
                 [challenge_weight(k) for k in (128, 192, 256)] == [30, 44, 60])

    print("\nTail factors certify their targets")
    for row in PUBLISHED:
        sol = solutions[(row["n"], row["kappa"])]
        ok = (tail_exponent_bits(sol["theta_s"], sol["dim"]) >= row["kappa"]
              and tail_exponent_bits(sol["theta_big_r"], 1) >= sol["kappa_big_r"]
              and tail_exponent_bits(sol["theta_r"], sol["dim"]) >= sol["kappa_r"])
        report.check(f"n={row['n']} kappa={row['kappa']}", ok,
                     f"theta_r certifies 2^-{tail_exponent_bits(sol['theta_r'], sol['dim']):.1f}"
                     f" (needs 2^-{sol['kappa_r']}); beta_r/B1 = {sol['beta_r'] / sol['b1']:.6f}")
    october = theta_october(196, 6144)
    report.check("the tail formula of an earlier draft fails at N=6144, kappa_r=196",
                 tail_exponent_bits(october, 6144) < 1,
                 f"theta = {october:.7f} certifies only 2^-{tail_exponent_bits(october, 6144):.2f}")

    print("\nPrimes")
    for row in PUBLISHED:
        sol = solutions[(row["n"], row["kappa"])]
        q = row["q"]
        ok = (is_prime(q) and q % (2 * RING_DEGREE) == 1 and q.bit_length() == row["b_q"]
              and q == largest_ntt_prime_below(row["b_q"]) and q > 2 * sol["b_msis"] + 1)
        report.check(f"n={row['n']} kappa={row['kappa']}: q = 2^{row['b_q']} - {2**row['b_q'] - q}",
                     ok, "prime, 1 mod 1024, largest below 2^b_q, q > 2 B_MSIS + 1")

    print("\nModulus floor (closed-form tails | exact tail inversion)")
    print("   n kappa | lg B_MSIS  floor[(q-1)/2]  floor[B<q]  published |"
          " lg B_MSIS  floor[(q-1)/2]  floor[B<q]")
    at_floor = True
    for row in PUBLISHED:
        closed = lg(solutions[(row["n"], row["kappa"])]["b_msis"])
        exact = lg(solve(row["n"], row["kappa"], row["ell"], theta=theta_exact)["b_msis"])
        print(f"  {row['n']:2d}  {row['kappa']:3d}  | {closed:9.4f}  {floor(closed) + 2:14d}"
              f"  {floor(closed) + 1:10d}  {row['b_q']:9d} | {exact:9.4f}"
              f"  {floor(exact) + 2:14d}  {floor(exact) + 1:10d}")
        at_floor = at_floor and floor(closed) + 2 == row["b_q"]
    report.check("published b_q equals floor(lg B_MSIS) + 2 in all six rows", at_floor)

    print("\nConditions at the allocations of Section 7.1:"
          "\neps_AK = 2^-(kappa+2), eps = eps' = 2^-(kappa+3)")
    margins = {40: [], 60: []}
    for row in PUBLISHED:
        sol = solutions[(row["n"], row["kappa"])]
        mq = sol["sigma_mq"] / smoothing_threshold(sol["dim"], row["kappa"], SEC_EPS_EXTRA_BITS)
        kernel = kernel_smoothing_ratio(sol)
        kernel_at_ak = kernel_smoothing_ratio(sol, AK_EPS_EXTRA_BITS)
        margins[row["n"]].append((mq, kernel))
        report.check(f"n={row['n']} kappa={row['kappa']}",
                     eps_ak_within_lemma_bound(row["kappa"], sol["k"], sol["dim"])
                     and mq >= 1 and kernel >= 1,
                     f"eps_AK <= 1/sqrt(2kD+4); sigma_MQ / S_MQ = {mq:.3f}; kernel smoothing"
                     f" margin = {kernel:.3f} (at kappa+2: {kernel_at_ak:.3f})")

    def truncated(value, digits):
        scale = Decimal(10) ** digits
        return int(value * scale) / scale

    # Printed in Sections 7.1 and 7.2: factors 4.4 and 8.0 for sigma_MQ (compared
    # after rounding), and kernel margins "of at least" 4.68 and 6.58 (compared
    # after truncation, since they are stated as lower bounds).
    printed = {40: (Decimal("4.4"), Decimal("4.68")), 60: (Decimal("8.0"), Decimal("6.58"))}
    for n, (printed_mq, printed_kernel) in printed.items():
        least_mq = min(mq for mq, _ in margins[n])
        least_kernel = min(kernel for _, kernel in margins[n])
        report.check(f"n={n}: sigma_MQ exceeds S_MQ by the printed factor {printed_mq}",
                     least_mq.quantize(Decimal("0.1")) == printed_mq,
                     f"least factor {least_mq:.3f}")
        if truncated(least_kernel, 2) == printed_kernel:
            report.check(f"n={n}: kernel margin of at least {printed_kernel}", True,
                         f"least margin {least_kernel:.3f}")
        else:
            report.note(f"n={n}: the paper prints a kernel margin of at least {printed_kernel}",
                        f"the least margin at kappa+3 is {least_kernel:.3f}; the printed value is"
                        " the one at kappa+2")

    print("\nBinding constraints")
    for row in PUBLISHED:
        sol = solutions[(row["n"], row["kappa"])]
        report.check(f"n={row['n']} kappa={row['kappa']}",
                     abs(sol["sigma_ak"] / sol["s_ak"] - 1) < Decimal("1e-30"),
                     f"sigma_AK = S_AK = {sol['sigma_ak']:.4f}; sigma_MQ / S_MQ = "
                     f"{sol['sigma_mq'] / sol['s_mq']:.3f}")

    print("\nHistorical cross-check: the tail formula and sqrt(2 pi)-larger thresholds of an earlier draft")
    for (n, kappa), (old_s, old_r) in OCTOBER_WIDTHS.items():
        ell = next(r["ell"] for r in PUBLISHED if (r["n"], r["kappa"]) == (n, kappa))
        old = solve(n, kappa, ell, theta=theta_october, threshold_scale=(2 * PI).sqrt())
        deviation = max(abs(lg(old["sigma_s"]) - Decimal(str(old_s))),
                        abs(lg(old["sigma_r"]) - Decimal(str(old_r))))
        report.check(f"n={n} kappa={kappa} reproduces the widths of the earlier draft",
                     deviation < Decimal("2e-4"),
                     f"lg sigma_s = {lg(old['sigma_s']):.4f}, lg sigma_r = {lg(old['sigma_r']):.4f}")

    print(f"\n{'ALL CHECKS PASSED' if report.failures == 0 else str(report.failures) + ' CHECK(S) FAILED'}")
    return report.failures


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--instances", action="store_true",
                        help="print the estimator inputs of the published rows as JSON and exit")
    args = parser.parse_args()
    if args.instances:
        json.dump([estimator_instance(row) for row in PUBLISHED], sys.stdout, indent=2)
        print()
        return 0
    return 1 if run_checks() else 0


if __name__ == "__main__":
    sys.exit(main())
