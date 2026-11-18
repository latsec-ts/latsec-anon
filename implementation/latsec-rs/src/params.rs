//! Parameter formula engine: tail factors, smoothing thresholds, the width optimizer
//! of the parameter-selection notebook, and the ideal sizes of Appendix N of the full version.
//!
//! This is an `f64` port of `tools/latsec_params.py`, which is the 60-digit
//! reference.  It is analysis code: it produces real-valued widths that
//! [`crate::profile`] then rounds upward to exact dyadic values, from which all
//! integer bounds are derived in exact arithmetic.  Nothing here is used to decide
//! whether a signature verifies.
//!
//! All widths are in the `sigma-exp-half` convention, `Pr[x] ~ exp(-x^2 / (2 sigma^2))`.

use std::f64::consts::{E, LN_2, PI};

use num_bigint::BigUint;
use num_traits::One;

use crate::error::{Error, Result};

/// `ceil(log2(x))` for `x >= 1`.
pub fn ceil_lg(x: u128) -> u32 {
    assert!(x >= 1);
    128 - (x - 1).leading_zeros()
}

/// The binomial coefficient `binom(n, k)`.
pub fn binomial(n: u64, k: u64) -> BigUint {
    if k > n {
        return BigUint::from(0u8);
    }
    let k = k.min(n - k);
    let mut acc = BigUint::one();
    for i in 0..k {
        acc = acc * (n - i) / (i + 1);
    }
    acc
}

/// The size of the challenge set `{c : |c|_inf = 1, |c|_1 = w}` in degree `d`.
pub fn challenge_space_size(d: usize, weight: u32) -> BigUint {
    binomial(d as u64, u64::from(weight)) << weight
}

/// The smallest weight `w` with `2^w * binom(d, w) >= q_h * 2^kappa`, if any.
pub fn challenge_weight(d: usize, q_h: u128, kappa: u32) -> Option<u32> {
    let target = BigUint::from(q_h) << kappa;
    (1..=d as u32).find(|&w| challenge_space_size(d, w) >= target)
}

/// `ceil(w + lg binom(d, w))`, the ideal number of bits of a challenge.
pub fn challenge_bits(d: usize, weight: u32) -> u32 {
    let size = challenge_space_size(d, weight);
    // ceil(lg(size)) for a positive integer.
    (size - 1u32).bits() as u32
}

/// How the Gaussian tail factor `theta` is chosen for a tail budget of
/// `2^-kappa_tail` in dimension `N`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TailRule {
    /// The paper's closed form (equations 25-27, Lemma 49): `1 + sqrt(2 kappa ln2 / N)`.
    ClosedForm,
    /// The smallest `theta` satisfying the certified inequality.
    ExactInversion,
    /// An earlier draft, `min{1 + L/(3N), 1 + sqrt(4L/N)}` with `L = kappa ln2`.
    /// Its linear branch is invalid; kept only as a historical counterexample.
    October,
}

/// `-log2` of `C_tail(theta)^N`: the tail security that `theta` certifies.
pub fn tail_exponent_bits(theta: f64, dim: f64) -> f64 {
    dim * ((theta * theta - 1.0) / 2.0 - theta.ln()) / LN_2
}

/// The tail factor for a budget of `2^-kappa_tail` in dimension `dim`.
pub fn tail_theta(rule: TailRule, kappa_tail: u32, dim: usize) -> f64 {
    let budget = f64::from(kappa_tail) * LN_2;
    let n = dim as f64;
    match rule {
        TailRule::ClosedForm => 1.0 + (2.0 * budget / n).sqrt(),
        TailRule::October => (1.0 + budget / (3.0 * n)).min(1.0 + (4.0 * budget / n).sqrt()),
        TailRule::ExactInversion => {
            let (mut low, mut high) = (1.0f64, 64.0f64);
            for _ in 0..200 {
                let mid = (low + high) / 2.0;
                if tail_exponent_bits(mid, n) >= f64::from(kappa_tail) {
                    high = mid;
                } else {
                    low = mid;
                }
            }
            high
        }
    }
}

/// `S(N) = sqrt(2) * eta_eps(Z^N) = sqrt(ln(2N / eps)) / pi` with `eps = 2^-eps_bits`,
/// in the `sigma` convention (Section 2.1).  The underlying `eta` is the usual
/// approximation, not a certified bound.
pub fn smoothing_threshold(dim: usize, eps_bits: u32) -> f64 {
    ((2.0 * dim as f64).ln() + f64::from(eps_bits) * LN_2).sqrt() / PI
}

/// Spectral constants of an SEC matrix that enter the width selection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MatrixConstants {
    /// `min_{|V| = k_p} sigma_min(G_hat_V)`.
    pub alpha_kp: f64,
    /// `|G_hat|_2`.
    pub norm_g_hat: f64,
    /// `|G|_2`.
    pub norm_g: f64,
    /// `1 + sqrt(1 + delta_kp^2) / gamma_kp`.
    pub beta_q: f64,
    /// `max_{|U| = k_p} |G_hat_U|_2`.
    pub beta_kp: f64,
    /// `min sigma_min(M_U)` over the peeling minors.
    pub gamma_kp: f64,
}

/// Inputs of the width optimization.
#[derive(Clone, Debug, PartialEq)]
pub struct WidthInputs {
    pub d: usize,
    pub ell: usize,
    pub n: usize,
    pub k: usize,
    pub k_p: usize,
    pub c_exp: usize,
    pub beta_g: u32,
    pub kappa: u32,
    pub q_sign: u128,
    pub omega_c: u32,
    pub constants: MatrixConstants,
    pub tail: TailRule,
    /// `eps = 2^-eps_bits` in the two smoothing thresholds of the width solver.
    /// The notebook uses `kappa + 2` for both, and so does this port, so
    /// that the published widths are reproduced bit for bit.  The paper
    /// (Section 7.1) states `kappa + 2` for the abort keys only and
    /// `kappa + 3` for the SEC key-privacy requirements; those are not binding and
    /// are checked separately by [`manuscript_conditions`].
    pub eps_bits: u32,
    /// Multiplier on the smoothing thresholds; `1` for the paper and
    /// `sqrt(2 pi)` to reproduce the normalization of an earlier draft.
    pub threshold_scale: f64,
}

/// The tail budgets and factors of one parameter set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TailFactors {
    pub kappa_big_r: u32,
    pub kappa_r: u32,
    pub theta_s: f64,
    pub theta_big_r: f64,
    pub theta_r: f64,
}

impl WidthInputs {
    pub fn dim(&self) -> usize {
        self.d * self.ell
    }

    /// `kappa_R = kappa + ceil(lg((k-1) D))`, `kappa_r = kappa + ceil(lg(n Q_Sign))`,
    /// and the three tail factors.  `theta_R` is per entry of `R` (dimension one).
    pub fn tail_factors(&self) -> TailFactors {
        let dim = self.dim();
        let kappa_big_r = self.kappa + ceil_lg(((self.k - 1) * dim) as u128);
        let kappa_r = self.kappa + ceil_lg(self.n as u128 * self.q_sign);
        TailFactors {
            kappa_big_r,
            kappa_r,
            theta_s: tail_theta(self.tail, self.kappa, dim),
            theta_big_r: tail_theta(self.tail, kappa_big_r, 1),
            theta_r: tail_theta(self.tail, kappa_r, dim),
        }
    }
}

/// Which constraints determine the optimum, numbered as in the notebook.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolverCase {
    /// MQ binds; neither AK cap does.
    Interior = 1,
    /// MQ and the AK cap on `x_s` bind.
    ClampS = 2,
    /// MQ and the AK cap on `x_R` bind.
    ClampR = 3,
    /// Both AK caps bind and MQ is inactive.
    BothCaps = 4,
}

/// The result of the width optimization.
#[derive(Clone, Debug, PartialEq)]
pub struct WidthSolution {
    pub sigma_s: f64,
    pub sigma_big_r: f64,
    pub sigma_r: f64,
    pub tails: TailFactors,
    pub case: SolverCase,
    /// The minimized `B1 = C_s sigma_s + C_R sigma_R + C_r sigma_r`.
    pub b1: f64,
    pub sigma_mq: f64,
    pub sigma_ak: f64,
    pub s_mq: f64,
    pub s_ak: f64,
    /// The effective width of the kernel-smoothing condition.
    pub sigma_eff: f64,
}

/// The real-valued bounds that follow from a set of widths (Section 4.1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RealBounds {
    pub beta_s: f64,
    pub beta_big_r: f64,
    pub beta_r: f64,
    pub beta_sh: f64,
    pub b1: f64,
}

/// `beta_s`, `beta_R`, `beta_r`, `beta_sh` and `B1` for given widths.
pub fn real_bounds(inputs: &WidthInputs, sigma_s: f64, sigma_big_r: f64, sigma_r: f64) -> RealBounds {
    let tails = inputs.tail_factors();
    let sqrt_dim = (inputs.dim() as f64).sqrt();
    let beta_s = tails.theta_s * sigma_s * sqrt_dim;
    let beta_big_r = tails.theta_big_r * sigma_big_r;
    let beta_r = tails.theta_r * sigma_r * sqrt_dim;
    let beta_sh = f64::from(inputs.beta_g) * beta_s + inputs.c_exp as f64 * beta_big_r;
    RealBounds { beta_s, beta_big_r, beta_r, beta_sh, b1: f64::from(inputs.omega_c) * sqrt_dim * beta_sh + beta_r }
}

/// The coefficients of the optimization problem in the notebook's notation:
/// minimize `Cx/sqrt(x) + Cy/sqrt(y) + Cz/sqrt(z)` subject to
/// `x + a y + b z <= P`, `x + c z <= Q`, `y + c z <= Q`.
#[derive(Clone, Copy, Debug)]
struct Problem {
    cx: f64,
    cy: f64,
    cz: f64,
    a: f64,
    b: f64,
    c: f64,
    p: f64,
    q: f64,
}

fn problem_of(inputs: &WidthInputs) -> Problem {
    let g = &inputs.constants;
    let dim = inputs.dim() as f64;
    let sqrt_dim = dim.sqrt();
    let omega = f64::from(inputs.omega_c);
    let q_sign = inputs.q_sign as f64;
    let tails = inputs.tail_factors();
    let s_mq = smoothing_threshold(inputs.dim(), inputs.eps_bits) * inputs.threshold_scale;
    let s_ak = smoothing_threshold(inputs.k * inputs.dim(), inputs.eps_bits) * inputs.threshold_scale;
    Problem {
        cx: omega * dim * tails.theta_s * f64::from(inputs.beta_g),
        cy: omega * inputs.c_exp as f64 * tails.theta_big_r * sqrt_dim,
        cz: tails.theta_r * sqrt_dim,
        a: inputs.k_p as f64 / (g.alpha_kp * g.alpha_kp),
        b: omega
            * omega
            * q_sign
            * ((inputs.n as f64).sqrt() + (inputs.k_p as f64).sqrt() * g.norm_g_hat / g.alpha_kp).powi(2),
        c: omega * omega * q_sign * g.norm_g * g.norm_g,
        p: 1.0 / (2.0 * s_mq * s_mq),
        q: 1.0 / (2.0 * g.beta_q * g.beta_q * s_ak * s_ak),
    }
}

/// The inner problem at fixed `z`: minimize `Cx/sqrt(x) + Cy/sqrt(y)` subject to
/// `x + a y <= p0`, `x <= q0`, `y <= q0`.
fn inner(p0: f64, q0: f64, a: f64, r: f64) -> (f64, f64, SolverCase) {
    if p0 > (1.0 + a) * q0 {
        return (q0, q0, SolverCase::BothCaps);
    }
    let (x0, y0) = (r * p0 / (a + r), p0 / (a + r));
    if x0 <= q0 && y0 <= q0 {
        (x0, y0, SolverCase::Interior)
    } else if x0 > q0 {
        (q0, (p0 - q0) / a, SolverCase::ClampS)
    } else {
        (p0 - a * q0, q0, SolverCase::ClampR)
    }
}

/// Golden-section minimum of a unimodal function on `[lo, hi]`.
fn golden(f: impl Fn(f64) -> f64, lo: f64, hi: f64) -> f64 {
    let inv_phi = (5f64.sqrt() - 1.0) / 2.0;
    let (mut a, mut b) = (lo, hi);
    let mut c = b - inv_phi * (b - a);
    let mut e = a + inv_phi * (b - a);
    let (mut fc, mut fe) = (f(c), f(e));
    for _ in 0..200 {
        if fc < fe {
            b = e;
            e = c;
            fe = fc;
            c = b - inv_phi * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = e;
            fc = fe;
            e = a + inv_phi * (b - a);
            fe = f(e);
        }
    }
    (a + b) / 2.0
}

/// `argmin A/sqrt(u) + B/sqrt(w)` subject to `u + kap w = M`; returns `w`.
fn two_term(a: f64, b: f64, kap: f64, m: f64) -> f64 {
    let rho = (kap * a / b).powf(2.0 / 3.0);
    m / (rho + kap)
}

/// The notebook's `minimize_Gxyz_fast`: returns `(x, y, z, case)`.
fn minimize(problem: Problem) -> Result<(f64, f64, f64, SolverCase)> {
    let Problem { cx, cy, cz, a, b, c, p, q } = problem;
    let positive = [cx, cy, cz, a, b, c, p, q].iter().all(|v| v.is_finite() && *v > 0.0);
    if !positive {
        return Err(Error::invalid("width optimization needs positive, finite coefficients"));
    }
    // Non-dimensionalize: x, y in units of Q and z in units of Z = min(P/b, Q/c).
    let z_unit = (p / b).min(q / c);
    let (cx, cy, cz) = (cx / q.sqrt(), cy / q.sqrt(), cz / z_unit.sqrt());
    let (b, c) = (b * z_unit / q, c * z_unit / q);
    let (p_scaled, q_scaled) = (p / q, 1.0);
    let r = (a * cx / cy).powf(2.0 / 3.0);
    let xy = |z: f64| inner(p_scaled - b * z, q_scaled - c * z, a, r);
    let phi = |z: f64| {
        let (x, y, _) = xy(z);
        cx / x.sqrt() + cy / y.sqrt() + cz / z.sqrt()
    };
    const EDGE: f64 = 1e-12;
    let mut breakpoints: Vec<f64> = [a + r, (a + r) / r, 1.0 + a]
        .iter()
        .filter(|&&th| (th * c - b).abs() > 1e-13 * (th * c).max(b))
        .map(|&th| (th * q_scaled - p_scaled) / (th * c - b))
        .filter(|&t| t > EDGE && t < 1.0 - EDGE)
        .collect();
    breakpoints.sort_by(|u, v| u.partial_cmp(v).unwrap());
    breakpoints.dedup();
    let mut edges = vec![EDGE];
    edges.extend(&breakpoints);
    edges.push(1.0 - EDGE);

    let mut best: Option<(f64, f64)> = None;
    let consider = |z: f64, best: &mut Option<(f64, f64)>| {
        let v = phi(z);
        if v.is_finite() && best.is_none_or(|(_, bv)| v < bv) {
            *best = Some((z, v));
        }
    };
    for window in edges.windows(2) {
        let (lo, hi) = (window[0], window[1]);
        if hi - lo <= 2.0 * EDGE {
            continue;
        }
        let mut z = match xy((lo + hi) / 2.0).2 {
            SolverCase::Interior => {
                let c1 = (a + r).sqrt() * (cx / r.sqrt() + cy);
                two_term(c1, cz, b, p_scaled)
            }
            SolverCase::BothCaps => two_term(cx + cy, cz, c, q_scaled),
            _ => golden(phi, lo, hi),
        };
        if !(lo <= z && z <= hi) {
            z = if phi(lo) < phi(hi) { lo } else { hi };
        }
        consider(z, &mut best);
    }
    for &t in &breakpoints {
        consider(t, &mut best);
    }
    let (z, _) = best.ok_or_else(|| Error::invalid("width optimization found no admissible point"))?;
    let (x, y, case) = xy(z);
    Ok((x * q, y * q, z * z_unit, case))
}

/// Minimizes `B1` under the multi-query privacy (MQ) and abort-key (AK) precision
/// constraints, as the v2.3 notebook does.
pub fn solve_widths(inputs: &WidthInputs) -> Result<WidthSolution> {
    let problem = problem_of(inputs);
    let (x_s, x_big_r, x_r, case) = minimize(problem)?;
    let g = &inputs.constants;
    let omega = f64::from(inputs.omega_c);
    let q_sign = inputs.q_sign as f64;
    let mq_load = x_s + problem.a * x_big_r + problem.b * x_r;
    let ak_load = x_s.max(x_big_r) + problem.c * x_r;
    let (sigma_s, sigma_big_r, sigma_r) = (x_s.sqrt().recip(), x_big_r.sqrt().recip(), x_r.sqrt().recip());
    Ok(WidthSolution {
        sigma_s,
        sigma_big_r,
        sigma_r,
        tails: inputs.tail_factors(),
        case,
        b1: real_bounds(inputs, sigma_s, sigma_big_r, sigma_r).b1,
        sigma_mq: (2.0 * mq_load).sqrt().recip(),
        sigma_ak: (2.0 * g.beta_q * g.beta_q * ak_load).sqrt().recip(),
        s_mq: (2.0 * problem.p).sqrt().recip(),
        s_ak: (2.0 * g.beta_q * g.beta_q * problem.q).sqrt().recip(),
        sigma_eff: (x_big_r + omega * omega * q_sign * g.norm_g_hat * g.norm_g_hat * x_r).sqrt().recip(),
    })
}

/// The closed form of solver case 4 (both AK caps bind, MQ inactive), for comparison
/// with the general solver.  Returns `(sigma_s = sigma_R, sigma_r)`.
pub fn solve_widths_case4(inputs: &WidthInputs) -> (f64, f64) {
    let pr = problem_of(inputs);
    let rho = (pr.c * (pr.cx + pr.cy) / pr.cz).powf(2.0 / 3.0);
    let x_r = pr.q / (rho + pr.c);
    ((rho * x_r).sqrt().recip(), x_r.sqrt().recip())
}

/// `eps_AK = 2^-(kappa + 2)`: Section 7.1, item 3 of the paper.
pub const AK_EPS_EXTRA_BITS: u32 = 2;
/// `eps = eps' = 2^-(kappa + 3)` for the key-privacy requirements of SEC: item 4.
pub const SEC_EPS_EXTRA_BITS: u32 = 3;

/// The requirements of Section 7.1 of the paper that depend on a
/// smoothing parameter, evaluated at the allocations the manuscript states.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ManuscriptConditions {
    /// `eps_AK <= 1 / sqrt(2 k D + 4)` (Lemma 1 with `N = k D`), checked exactly.
    pub eps_ak_within_lemma_bound: bool,
    /// `sigma_AK / (sqrt(2) eta_{eps_AK}(Z^{kD}))`; one when the constraint binds.
    pub abort_key_margin: f64,
    /// `sigma_MQ / (sqrt(2) eta_eps(Z^D))` at `eps = 2^-(kappa + 3)`.
    pub multi_query_margin: f64,
    /// The margin of the kernel-smoothing condition at `eps' = 2^-(kappa + 3)`.
    pub kernel_smoothing_margin: f64,
}

impl ManuscriptConditions {
    /// Whether all four hold, with a relative tolerance of `1e-9` on the binding
    /// abort-key constraint.
    pub fn hold(&self) -> bool {
        self.eps_ak_within_lemma_bound
            && self.abort_key_margin >= 1.0 - 1e-9
            && self.multi_query_margin >= 1.0
            && self.kernel_smoothing_margin >= 1.0
    }
}

/// Evaluates [`ManuscriptConditions`] for a solved parameter set.  Independent of
/// `inputs.eps_bits` and of `inputs.threshold_scale`.
pub fn manuscript_conditions(inputs: &WidthInputs, sol: &WidthSolution) -> ManuscriptConditions {
    let ak_bits = inputs.kappa + AK_EPS_EXTRA_BITS;
    let sec_bits = inputs.kappa + SEC_EPS_EXTRA_BITS;
    // 1/eps_AK^2 = 4^ak_bits >= 2 k D + 4.
    let n_bound = 2 * (inputs.k * inputs.dim()) as u128 + 4;
    let eps_ak_within_lemma_bound = 2 * ak_bits >= 128 || (1u128 << (2 * ak_bits)) >= n_bound;
    ManuscriptConditions {
        eps_ak_within_lemma_bound,
        abort_key_margin: sol.sigma_ak / smoothing_threshold(inputs.k * inputs.dim(), ak_bits),
        multi_query_margin: sol.sigma_mq / smoothing_threshold(inputs.dim(), sec_bits),
        kernel_smoothing_margin: kernel_smoothing_ratio_at(inputs, sol.sigma_eff, sec_bits),
    }
}

/// `sigma_eff` divided by the explicit kernel-smoothing threshold ([46, Eq. (56)]),
/// with `eps' = 2^-eps_bits` for the solver's `inputs.eps_bits`.  The condition
/// holds when the ratio is at least one.
pub fn kernel_smoothing_ratio(inputs: &WidthInputs, sigma_eff: f64) -> f64 {
    kernel_smoothing_ratio_at(inputs, sigma_eff, inputs.eps_bits)
}

/// [`kernel_smoothing_ratio`] with `eps' = 2^-eps_bits` given explicitly.
pub fn kernel_smoothing_ratio_at(inputs: &WidthInputs, sigma_eff: f64, eps_bits: u32) -> f64 {
    let g = &inputs.constants;
    let geometry = (1.0 + (inputs.k_p as f64).min(g.beta_kp * g.beta_kp) / (g.gamma_kp * g.gamma_kp)).sqrt();
    // ln(2 (k-1) D (2 + 1/eps')), with 1/eps' = 2^eps_bits dominating the sum.
    let log_term = (2.0 * (inputs.k - 1) as f64 * inputs.dim() as f64).ln()
        + f64::from(eps_bits) * LN_2
        + (2.0 * (-f64::from(eps_bits)).exp2()).ln_1p();
    sigma_eff / (geometry * (log_term / (2.0 * PI * PI)).sqrt())
}

/// The ideal (entropy-coded) sizes of Appendix N of the full version, in bits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IdealSizes {
    pub vk: f64,
    pub ak_i: f64,
    pub ak: f64,
    pub sk_i: f64,
    pub sig_i: f64,
    pub sig: f64,
    pub comm_i: f64,
}

impl IdealSizes {
    /// The seven sizes in KiB, in the column order of Table 5.
    pub fn kib(&self) -> [f64; 7] {
        [self.vk, self.ak_i, self.ak, self.sk_i, self.sig_i, self.sig, self.comm_i].map(|bits| bits / 8192.0)
    }
}

/// Appendix N of the full version for the given dimensions, widths and bounds.
#[allow(clippy::too_many_arguments)]
pub fn ideal_sizes(
    d: usize,
    ell: usize,
    p: usize,
    n: usize,
    c_exp: usize,
    kappa: u32,
    omega_c: u32,
    modulus_bits: u32,
    sigma_s: f64,
    b1: f64,
    b2: f64,
) -> IdealSizes {
    const SEED_BITS: f64 = 256.0;
    let dim = (d * ell) as f64;
    let half_lg_2pie = (2.0 * PI * E).log2() / 2.0;
    let response = |bound: f64| dim * ((bound / dim.sqrt()).log2() + half_lg_2pie);
    let key_bits = (p * d) as f64 * f64::from(modulus_bits);
    let challenge = f64::from(challenge_bits(d, omega_c));
    IdealSizes {
        vk: SEED_BITS + key_bits,
        ak_i: key_bits,
        ak: n as f64 * key_bits,
        sk_i: dim * (sigma_s.log2() + (1.0 + c_exp as f64).log2() / 2.0 + half_lg_2pie),
        sig_i: challenge + response(b1),
        sig: challenge + response(b2),
        comm_i: 2.0 * f64::from(kappa) + key_bits + response(b1),
    }
}

/// Inputs of the SEC-only width selection of the SEC paper [46] (no signature scheme,
/// hence no abort-key constraint).
#[derive(Clone, Debug, PartialEq)]
pub struct SecWidthInputs {
    pub d: usize,
    pub ell: usize,
    pub n: usize,
    pub k: usize,
    pub k_p: usize,
    pub c_exp: usize,
    pub beta_g: u32,
    pub kappa: u32,
    /// Leakage-query budget `Q`; zero for standalone sharing.
    pub queries: u128,
    pub omega_c: u32,
    pub constants: MatrixConstants,
    /// `eps = eps' = 2^-eps_bits`; the SEC paper uses `kappa + 3`.
    pub eps_bits: u32,
}

/// The result of [`solve_sec_widths`].
#[derive(Clone, Debug, PartialEq)]
pub struct SecWidthSolution {
    pub sigma_s: f64,
    pub sigma_big_r: f64,
    /// Absent when there are no leakage queries.
    pub sigma_r: Option<f64>,
    /// The share width `sqrt(beta_g^2 sigma_s^2 + c_exp sigma_R^2)`.
    pub sigma_sh: f64,
    /// Whether the kernel-smoothing constraint binds (the notebook's "clamp" case).
    pub kernel_binding: bool,
    pub sigma_mq: f64,
    pub sigma_eff: f64,
    pub s_mq: f64,
    pub tau_ker: f64,
}

impl SecWidthSolution {
    /// Ideal secret and share sizes in KiB: `D (lg sigma + 1/2 lg(2 pi e))` bits.
    pub fn sizes_kib(&self, dim: usize) -> (f64, f64) {
        let half_lg_2pie = (2.0 * PI * E).log2() / 2.0;
        let size = |sigma: f64| dim as f64 * (sigma.log2() + half_lg_2pie) / 8192.0;
        (size(self.sigma_s), size(self.sigma_sh))
    }
}

/// The SEC-only width selection ([46, Section 6]; `SEC-Parameter-Selection-v1`).
///
/// With `x = 1/sigma_s^2`, `y = 1/sigma_R^2`, `z = 1/sigma_r^2`, the constraints are
/// multi-query privacy `x + a y + b z <= P` and kernel smoothing `y + c z <= K`.  The
/// objective is the share width for `Q = 0` and the product of the share width and
/// `sigma_r` otherwise.
pub fn solve_sec_widths(inputs: &SecWidthInputs) -> Result<SecWidthSolution> {
    let g = &inputs.constants;
    let dim = inputs.d * inputs.ell;
    let omega = f64::from(inputs.omega_c);
    let queries = inputs.queries as f64;
    let beta_g_sq = f64::from(inputs.beta_g).powi(2);
    let (w_x, w_y) = (beta_g_sq, inputs.c_exp as f64);
    let a = beta_g_sq * inputs.k_p as f64 / (g.alpha_kp * g.alpha_kp);
    let b = beta_g_sq
        * omega
        * omega
        * queries
        * ((inputs.n as f64).sqrt() + (inputs.k_p as f64).sqrt() * g.norm_g_hat / g.alpha_kp).powi(2);
    let c = omega * omega * queries * g.norm_g_hat * g.norm_g_hat;
    let s_mq = smoothing_threshold(dim, inputs.eps_bits);
    let geometry = (1.0 + (inputs.k_p as f64).min(g.beta_kp * g.beta_kp) / (g.gamma_kp * g.gamma_kp)).sqrt();
    let log_term = (2.0 * (inputs.k - 1) as f64 * dim as f64).ln()
        + f64::from(inputs.eps_bits) * LN_2
        + (2.0 * (-f64::from(inputs.eps_bits)).exp2()).ln_1p();
    let tau_ker = geometry * (log_term / (2.0 * PI * PI)).sqrt();
    let (p, k) = (1.0 / (2.0 * s_mq * s_mq), 1.0 / (tau_ker * tau_ker));
    if ![a, p, k].iter().all(|v| v.is_finite() && *v > 0.0) {
        return Err(Error::invalid("SEC width selection needs positive, finite coefficients"));
    }
    // Inner problem at fixed z: minimize Wx/x + Wy/y with x + a y <= p0 and y <= k0.
    let inner = |p0: f64, k0: f64| -> (f64, f64, bool) {
        let r = (a * w_x / w_y).sqrt();
        let y = p0 / (a + r);
        if y <= k0 {
            (r * y, y, false)
        } else {
            (p0 - a * k0, k0, true)
        }
    };
    let (x, y, z, kernel_binding) = if inputs.queries == 0 {
        let (x, y, clamp) = inner(p, k);
        (x, y, None, clamp)
    } else {
        // Convex in u = ln z; search up to 64 bits below the largest feasible z.
        let hi = (p / b).min(k / c).ln();
        let lo = hi - 64.0 * LN_2;
        let phi = |u: f64| {
            let z = u.exp();
            let (x, y, _) = inner(p - b * z, k - c * z);
            (w_x / x + w_y / y).log2() / 2.0 - z.log2() / 2.0
        };
        let z = golden(phi, lo, hi).exp();
        let (x, y, clamp) = inner(p - b * z, k - c * z);
        (x, y, Some(z), clamp)
    };
    if !(x > 0.0 && y > 0.0) {
        return Err(Error::invalid("SEC width selection is infeasible"));
    }
    let zz = z.unwrap_or(0.0);
    let (sigma_s, sigma_big_r) = (x.sqrt().recip(), y.sqrt().recip());
    Ok(SecWidthSolution {
        sigma_s,
        sigma_big_r,
        sigma_r: z.map(|z| z.sqrt().recip()),
        sigma_sh: (w_x * sigma_s * sigma_s + w_y * sigma_big_r * sigma_big_r).sqrt(),
        kernel_binding,
        sigma_mq: (2.0 * (x + a * y + b * zz)).sqrt().recip(),
        sigma_eff: (y + c * zz).sqrt().recip(),
        s_mq,
        tau_ker,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::published::{self, PUBLISHED_ROWS};

    #[test]
    fn integer_helpers() {
        assert_eq!(ceil_lg(1), 0);
        assert_eq!(ceil_lg(2), 1);
        assert_eq!(ceil_lg(3), 2);
        assert_eq!(ceil_lg(1 << 70), 70);
        assert_eq!(ceil_lg((1 << 70) + 1), 71);
        assert_eq!(binomial(512, 30), binomial(512, 482));
        assert_eq!(binomial(10, 3), BigUint::from(120u32));
        assert_eq!(binomial(3, 5), BigUint::from(0u32));
    }

    #[test]
    fn challenge_weights_match_the_paper() {
        assert_eq!(challenge_weight(512, 1 << 63, 128), Some(30));
        assert_eq!(challenge_weight(512, 1 << 64, 192), Some(44));
        assert_eq!(challenge_weight(512, 1 << 64, 256), Some(60));
        // SEC paper: |C| >= 2^kappa.
        assert_eq!(challenge_weight(512, 1, 128), Some(19));
        assert_eq!(challenge_weight(512, 1, 192), Some(31));
        assert_eq!(challenge_weight(512, 1, 256), Some(44));
        assert_eq!(challenge_weight(16, 1 << 63, 128), None);
        assert_eq!(challenge_bits(512, 30), 192);
        assert_eq!(challenge_bits(512, 60), 323);
    }

    #[test]
    fn closed_form_tail_factor_certifies_its_budget() {
        for (kappa, dim) in [(128u32, 6144usize), (196, 6144), (326, 10240), (145, 1), (275, 1)] {
            let theta = tail_theta(TailRule::ClosedForm, kappa, dim);
            assert!(tail_exponent_bits(theta, dim as f64) >= f64::from(kappa));
            let exact = tail_theta(TailRule::ExactInversion, kappa, dim);
            assert!(exact <= theta);
            let certified = tail_exponent_bits(exact, dim as f64);
            assert!(certified >= f64::from(kappa) && certified < f64::from(kappa) + 1e-6);
        }
        // Published values (Table 4 and the notebook).
        assert!((tail_theta(TailRule::ClosedForm, 196, 6144) - 1.2103).abs() < 5e-5);
        assert!((tail_theta(TailRule::ClosedForm, 145, 1) - 15.178).abs() < 5e-4);
        assert!((tail_theta(TailRule::ExactInversion, 196, 6144) - 1.1522).abs() < 5e-5);
    }

    #[test]
    fn october_formula_is_a_counterexample() {
        let theta = tail_theta(TailRule::October, 196, 6144);
        let certified = tail_exponent_bits(theta, 6144.0);
        assert!((certified - 0.48).abs() < 0.01, "certifies 2^-{certified}");
    }

    #[test]
    fn reproduces_table_4() {
        for row in PUBLISHED_ROWS {
            let inputs = published::width_inputs(row);
            let sol = solve_widths(&inputs).unwrap();
            assert_eq!(sol.case, SolverCase::BothCaps, "n = {}, kappa = {}", row.n, row.kappa);
            let (case4_s, case4_r) = solve_widths_case4(&inputs);
            assert!((sol.sigma_s / case4_s - 1.0).abs() < 1e-9);
            assert!((sol.sigma_r / case4_r - 1.0).abs() < 1e-9);
            assert!((sol.sigma_s / sol.sigma_big_r - 1.0).abs() < 1e-9);
            let geo = published::geometry(row.n);
            let multiplier = f64::from(geo.beta_lambda * geo.omega_lambda);
            let b2 = multiplier * sol.b1;
            let b_msis = 2.0 * (b2 * b2 + f64::from(inputs.omega_c)).sqrt();
            for (name, value, expected) in [
                ("lg sigma_s", sol.sigma_s.log2(), row.lg_sigma_s),
                ("lg sigma_r", sol.sigma_r.log2(), row.lg_sigma_r),
                ("lg B1", sol.b1.log2(), row.lg_b1),
                ("lg B2", b2.log2(), row.lg_b2),
                ("lg B_MSIS", b_msis.log2(), row.lg_b2 + 1.0),
            ] {
                assert!((value - expected).abs() < 6e-5, "n = {}, kappa = {}: {name} = {value}", row.n, row.kappa);
            }
            // The published modulus is the smallest allowed by q > 2 B_MSIS + 1.
            assert_eq!(b_msis.log2().floor() as u32 + 2, row.modulus_bits);
            // The abort-key constraint binds exactly and MQ has slack.
            assert!((sol.sigma_ak / sol.s_ak - 1.0).abs() < 1e-9);
            assert!(sol.sigma_mq / sol.s_mq > 4.0);
            // Kernel smoothing is not evaluated by the LaTSEC notebook; it holds.
            let ratio = kernel_smoothing_ratio(&inputs, sol.sigma_eff);
            let expected_ratio = if row.n == 40 { 4.68 } else { 6.61 };
            assert!((ratio - expected_ratio).abs() < 0.02, "kernel smoothing ratio {ratio}");
            // The conditions at the allocations of the paper:
            // eps_AK = 2^-(kappa+2), eps = eps' = 2^-(kappa+3).  Section 7.2 prints
            // the factors 4.4 and 8.0 for sigma_MQ; Section 7.1 prints kernel
            // margins of at least 4.68 and 6.58.  The 6.58 is reproduced.  The
            // 4.68 is the margin at kappa + 2; at kappa + 3 the least margin for
            // n = 40 is 4.667.
            let conditions = manuscript_conditions(&inputs, &sol);
            assert!(conditions.hold(), "n = {}, kappa = {}: {conditions:?}", row.n, row.kappa);
            assert!((conditions.abort_key_margin - 1.0).abs() < 1e-9);
            let (printed_mq, least_kernel) = if row.n == 40 { (4.4, 4.66) } else { (8.0, 6.58) };
            assert!((conditions.multi_query_margin - printed_mq).abs() < 0.08, "{conditions:?}");
            let kernel = conditions.kernel_smoothing_margin;
            assert!(kernel >= least_kernel && kernel < least_kernel + 0.02, "{conditions:?}");
            assert!(kernel < ratio, "the margin at kappa + 3 is below the one at kappa + 2");
            // Tail factors certify their budgets.
            let dim = inputs.dim() as f64;
            assert!(tail_exponent_bits(sol.tails.theta_s, dim) >= f64::from(row.kappa));
            assert!(tail_exponent_bits(sol.tails.theta_big_r, 1.0) >= f64::from(sol.tails.kappa_big_r));
            assert!(tail_exponent_bits(sol.tails.theta_r, dim) >= f64::from(sol.tails.kappa_r));
        }
    }

    #[test]
    fn reproduces_table_5() {
        let mut worst: f64 = 0.0;
        for row in PUBLISHED_ROWS {
            let inputs = published::width_inputs(row);
            let sol = solve_widths(&inputs).unwrap();
            let geo = published::geometry(row.n);
            let b2 = f64::from(geo.beta_lambda * geo.omega_lambda) * sol.b1;
            let sizes = ideal_sizes(
                inputs.d,
                row.ell,
                row.p,
                row.n,
                geo.c_exp,
                row.kappa,
                inputs.omega_c,
                row.modulus_bits,
                sol.sigma_s,
                sol.b1,
                b2,
            );
            for (value, expected) in sizes.kib().iter().zip(row.sizes_kib) {
                worst = worst.max((value - expected).abs());
            }
        }
        assert!(worst < 6e-4, "largest deviation {worst} KiB");
    }

    #[test]
    fn october_widths_follow_from_the_old_formulas() {
        for (row, old_s, old_r) in [(&PUBLISHED_ROWS[0], 16.6855, 46.3392), (&PUBLISHED_ROWS[3], 17.2046, 47.1391)] {
            let mut inputs = published::width_inputs(row);
            inputs.tail = TailRule::October;
            inputs.threshold_scale = (2.0 * PI).sqrt();
            let sol = solve_widths(&inputs).unwrap();
            assert!((sol.sigma_s.log2() - old_s).abs() < 2e-4);
            assert!((sol.sigma_r.log2() - old_r).abs() < 2e-4);
        }
    }

    #[test]
    fn exact_tails_lower_the_bound() {
        let mut inputs = published::width_inputs(&PUBLISHED_ROWS[3]);
        let closed = solve_widths(&inputs).unwrap().b1.log2();
        inputs.tail = TailRule::ExactInversion;
        let exact = solve_widths(&inputs).unwrap().b1.log2();
        assert!((closed - exact - 0.071).abs() < 2e-3, "difference {}", closed - exact);
    }

    #[test]
    fn reproduces_the_sec_only_tables() {
        for row in published::SEC_ONLY_ROWS {
            let geo = published::geometry(row.n);
            let inputs = SecWidthInputs {
                d: 512,
                ell: row.ell,
                n: geo.n,
                k: geo.k,
                k_p: geo.k_p,
                c_exp: geo.c_exp,
                beta_g: geo.beta_g,
                kappa: row.kappa,
                queries: if row.with_queries { 1 << 64 } else { 0 },
                omega_c: challenge_weight(512, 1, row.kappa).unwrap(),
                constants: geo.constants,
                eps_bits: row.kappa + 3,
            };
            let sol = solve_sec_widths(&inputs).unwrap();
            let label = format!("n = {}, kappa = {}, ell = {}, Q {}", row.n, row.kappa, row.ell, row.with_queries);
            assert!((sol.sigma_s.log2() - row.lg_sigma_s).abs() < 6e-5, "{label}: lg sigma_s = {}", sol.sigma_s.log2());
            assert!((sol.sigma_big_r.log2() - row.lg_sigma_big_r).abs() < 6e-5, "{label}: lg sigma_R");
            match (sol.sigma_r, row.lg_sigma_r) {
                (Some(sigma_r), Some(expected)) => {
                    assert!((sigma_r.log2() - expected).abs() < 6e-5, "{label}: lg sigma_r")
                }
                (None, None) => {}
                _ => panic!("{label}: sigma_r presence differs"),
            }
            // Every retained row has the kernel-smoothing constraint binding, and
            // multi-query privacy is always active.
            assert!(sol.kernel_binding, "{label}");
            assert!((sol.sigma_mq / sol.s_mq - 1.0).abs() < 1e-9, "{label}");
            assert!(sol.sigma_eff / sol.tau_ker >= 1.0 - 1e-9, "{label}");
            if let Some((secret, share)) = row.sizes_kib {
                let (got_secret, got_share) = sol.sizes_kib(512 * row.ell);
                assert!((got_secret - secret).abs() < 6e-4 && (got_share - share).abs() < 6e-4, "{label}: sizes");
            }
        }
    }

    /// Brute-force check of the general solver on problems where MQ is active.
    #[test]
    fn general_solver_matches_a_grid_search() {
        let problems = [
            Problem { cx: 3.0, cy: 2.0, cz: 5.0, a: 1.5, b: 2.0, c: 1.0, p: 1.0, q: 1.0 },
            Problem { cx: 1.0, cy: 8.0, cz: 2.0, a: 0.5, b: 4.0, c: 3.0, p: 0.8, q: 1.0 },
            Problem { cx: 9.0, cy: 1.0, cz: 1.0, a: 2.0, b: 0.7, c: 5.0, p: 1.2, q: 0.9 },
            Problem { cx: 1.0, cy: 1.0, cz: 1.0, a: 1.0, b: 1.0, c: 1.0, p: 5.0, q: 1.0 },
            Problem { cx: 2.0, cy: 3.0, cz: 40.0, a: 6.0, b: 9.0, c: 0.3, p: 2.0, q: 1.0 },
        ];
        let mut cases = std::collections::BTreeSet::new();
        for pr in problems {
            let (x, y, z, case) = minimize(pr).unwrap();
            cases.insert(case as u8);
            let value = pr.cx / x.sqrt() + pr.cy / y.sqrt() + pr.cz / z.sqrt();
            assert!(x + pr.a * y + pr.b * z <= pr.p * (1.0 + 1e-9));
            assert!(x + pr.c * z <= pr.q * (1.0 + 1e-9) && y + pr.c * z <= pr.q * (1.0 + 1e-9));
            // Grid over z; for each z the inner optimum over (x, y) by a second grid.
            let mut best = f64::INFINITY;
            let steps = 400;
            let zmax = (pr.p / pr.b).min(pr.q / pr.c);
            for iz in 1..steps {
                let zz = zmax * f64::from(iz) / f64::from(steps);
                let cap = pr.q - pr.c * zz;
                let budget = pr.p - pr.b * zz;
                for iy in 1..=steps {
                    let yy = cap * f64::from(iy) / f64::from(steps);
                    let xx = cap.min(budget - pr.a * yy);
                    if xx <= 0.0 {
                        continue;
                    }
                    best = best.min(pr.cx / xx.sqrt() + pr.cy / yy.sqrt() + pr.cz / zz.sqrt());
                }
            }
            assert!(value <= best * (1.0 + 1e-9), "solver {value} is worse than the grid {best}");
            assert!(value >= best * (1.0 - 2e-2), "solver {value} is implausibly below the grid {best}");
        }
        assert!(cases.len() >= 3, "test problems should reach several solver cases: {cases:?}");
    }
}
