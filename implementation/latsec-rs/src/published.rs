//! Values published in the paper and its notebooks, used for regression tests and
//! to build the as-published author profiles.  Nothing here is derived by this
//! crate.
//!
//! Sources:
//!
//! * The paper, Tables 4 and 5 (three decimals).
//! * `parameter-search/LaTSEC-Parameter-Selection-v2.3-fixedtailbound+privacy.ipynb` of
//!   this artifact: matrix constants (cell 56), exact primes (cell 198), widths to
//!   four decimals (cell 204), sizes (cell 246) and estimator outputs (cell 194).
//! * The SEC parameter-selection notebook of the SEC paper [46]
//!   (`SEC-Parameter-Selection-v1.ipynb`, not part of this artifact) for `beta_kp`
//!   and `gamma_kp` of the same matrices; `parameter-search/SEC-geometry/` of this
//!   artifact recomputes `gamma_kp`.
//!
//! The matrix constants are float64 SVD results.  They carry no certified error
//! bound.

use crate::params::{challenge_weight, MatrixConstants, TailRule, WidthInputs};

/// The ring degree of every published parameter set.
pub const RING_DEGREE: usize = 512;

/// Per-matrix data of the two published SEC instances.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geometry {
    pub n: usize,
    pub k: usize,
    pub k_p: usize,
    pub k_c: usize,
    /// Deterministic authorization threshold from the exhaustive checks of [46].
    pub k_c_det: usize,
    pub c_exp: usize,
    pub beta_g: u32,
    /// Enforced bounds of the admissibility predicate.
    pub beta_lambda: u32,
    pub omega_lambda: u32,
    /// Largest values observed with the SageMath pipeline of [46].
    pub beta_lambda_observed: u32,
    pub omega_lambda_observed: u32,
    pub constants: MatrixConstants,
    /// Name of the fixture under `fixtures/author`.
    pub fixture: &'static str,
}

const GEOMETRY_40: Geometry = Geometry {
    n: 40,
    k: 20,
    k_p: 5,
    k_c: 26,
    k_c_det: 35,
    c_exp: 3,
    beta_g: 1,
    beta_lambda: 5,
    omega_lambda: 25,
    beta_lambda_observed: 4,
    omega_lambda_observed: 24,
    constants: MatrixConstants {
        alpha_kp: 0.464255900330341,
        norm_g_hat: 4.4316793740317,
        norm_g: 5.86070678667877,
        beta_q: 20.6352940186031,
        beta_kp: 2.9403658464661486,
        gamma_kp: 0.19253138628300923,
    },
    fixture: "G-40-19-3.json",
};

const GEOMETRY_60: Geometry = Geometry {
    n: 60,
    k: 40,
    k_p: 6,
    k_c: 46,
    k_c_det: 56,
    c_exp: 3,
    beta_g: 1,
    beta_lambda: 9,
    omega_lambda: 45,
    beta_lambda_observed: 8,
    omega_lambda_observed: 42,
    constants: MatrixConstants {
        alpha_kp: 0.517638090205041,
        norm_g_hat: 3.93699368153548,
        norm_g: 7.84160154378415,
        beta_q: 26.7593053884503,
        beta_kp: 2.893489471774352,
        gamma_kp: 0.15210842969369098,
    },
    fixture: "G-60-39-3.json",
};

/// The published instance with `n` parties (40 or 60).
pub fn geometry(n: usize) -> &'static Geometry {
    match n {
        40 => &GEOMETRY_40,
        60 => &GEOMETRY_60,
        _ => panic!("no published SEC instance with n = {n}"),
    }
}

/// `(Q_H, Q_Sign)` for a security level.
pub fn query_budgets(kappa: u32) -> (u128, u128) {
    match kappa {
        128 => (1 << 63, 1 << 62),
        192 | 256 => (1 << 64, 1 << 64),
        _ => panic!("no published query budgets for kappa = {kappa}"),
    }
}

/// The lattice-estimator outputs recorded in the notebook, in bits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublishedSecurity {
    pub mq: f64,
    pub ak: f64,
    pub msis: f64,
}

/// One row of Tables 4 and 5.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublishedRow {
    pub n: usize,
    pub kappa: u32,
    pub ell: usize,
    pub p: usize,
    /// `ceil(lg q)`.
    pub modulus_bits: u32,
    pub q: u128,
    pub lg_sigma_s: f64,
    pub lg_sigma_r: f64,
    pub lg_b1: f64,
    pub lg_b2: f64,
    /// `|vk|, |ak_i|, |ak|, |sk_i|, |sig_i|, |sig|, Comm_i` in KiB.
    pub sizes_kib: [f64; 7],
    pub security: PublishedSecurity,
}

/// Tables 4 and 5 of the paper.
pub const PUBLISHED_ROWS: &[PublishedRow] = &[
    PublishedRow {
        n: 40,
        kappa: 128,
        ell: 12,
        p: 6,
        modulus_bits: 61,
        q: (1 << 61) - 10239,
        lg_sigma_s: 15.4641,
        lg_sigma_r: 45.0134,
        lg_b1: 51.5813,
        lg_b2: 58.5470,
        sizes_kib: [22.906, 22.875, 915.000, 13.883, 35.525, 40.750, 58.408],
        security: PublishedSecurity { mq: 168.5, ak: 152.4, msis: 153.6 },
    },
    PublishedRow {
        n: 40,
        kappa: 192,
        ell: 16,
        p: 8,
        modulus_bits: 63,
        q: (1 << 63) - 17407,
        lg_sigma_s: 15.9815,
        lg_sigma_r: 46.8267,
        lg_b1: 53.6024,
        lg_b2: 60.5682,
        sizes_kib: [31.531, 31.500, 1260.000, 19.029, 49.181, 56.147, 80.696],
        security: PublishedSecurity { mq: 237.7, ak: 216.1, msis: 217.5 },
    },
    PublishedRow {
        n: 40,
        kappa: 256,
        ell: 20,
        p: 10,
        modulus_bits: 64,
        q: (1 << 64) - 1023,
        lg_sigma_s: 16.1146,
        lg_sigma_r: 47.4651,
        lg_b1: 54.4011,
        lg_b2: 61.3669,
        sizes_kib: [40.031, 40.000, 1600.000, 23.952, 62.274, 70.981, 102.297],
        security: PublishedSecurity { mq: 312.1, ak: 284.7, msis: 289.1 },
    },
    PublishedRow {
        n: 60,
        kappa: 128,
        ell: 12,
        p: 6,
        modulus_bits: 64,
        q: (1 << 64) - 1023,
        lg_sigma_s: 15.9834,
        lg_sigma_r: 45.8133,
        lg_b1: 52.3811,
        lg_b2: 61.0429,
        sizes_kib: [24.031, 24.000, 1440.000, 14.273, 36.125, 42.622, 60.133],
        security: PublishedSecurity { mq: 163.5, ak: 142.2, msis: 146.6 },
    },
    PublishedRow {
        n: 60,
        kappa: 192,
        ell: 16,
        p: 8,
        modulus_bits: 66,
        q: (1 << 66) - 16383,
        lg_sigma_s: 16.4994,
        lg_sigma_r: 47.6251,
        lg_b1: 54.4008,
        lg_b2: 63.0626,
        sizes_kib: [33.031, 33.000, 1980.000, 19.547, 49.979, 58.641, 82.995],
        security: PublishedSecurity { mq: 231.0, ak: 202.6, msis: 207.9 },
    },
    PublishedRow {
        n: 60,
        kappa: 256,
        ell: 20,
        p: 10,
        modulus_bits: 66,
        q: (1 << 66) - 16383,
        lg_sigma_s: 16.6318,
        lg_sigma_r: 48.2627,
        lg_b1: 55.1987,
        lg_b2: 63.8605,
        sizes_kib: [41.281, 41.250, 2475.000, 24.599, 63.271, 74.098, 104.544],
        security: PublishedSecurity { mq: 310.4, ak: 273.3, msis: 271.6 },
    },
];

/// The published row for `(n, kappa)`, if there is one.
pub fn published_row(n: usize, kappa: u32) -> Option<&'static PublishedRow> {
    PUBLISHED_ROWS.iter().find(|r| r.n == n && r.kappa == kappa)
}

/// The inputs the v2.3 notebook feeds its optimizer for a published row.
pub fn width_inputs(row: &PublishedRow) -> WidthInputs {
    let geo = geometry(row.n);
    let (q_h, q_sign) = query_budgets(row.kappa);
    WidthInputs {
        d: RING_DEGREE,
        ell: row.ell,
        n: geo.n,
        k: geo.k,
        k_p: geo.k_p,
        c_exp: geo.c_exp,
        beta_g: geo.beta_g,
        kappa: row.kappa,
        q_sign,
        omega_c: challenge_weight(RING_DEGREE, q_h, row.kappa).expect("published challenge weight"),
        constants: geo.constants,
        tail: TailRule::ClosedForm,
        eps_bits: row.kappa + 2,
        threshold_scale: 1.0,
    }
}

/// One SEC-only parameter set of the SEC paper [46].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SecOnlyRow {
    pub n: usize,
    pub kappa: u32,
    /// `false` for `Q = 0`, `true` for `Q = 2^64` leakage queries.
    pub with_queries: bool,
    pub ell: usize,
    /// `ceil(lg q)` of the hard-core modulus the row was reported with.
    pub modulus_bits: u32,
    pub lg_sigma_s: f64,
    pub lg_sigma_big_r: f64,
    pub lg_sigma_r: Option<f64>,
    /// Ideal `(secret, share)` sizes in KiB, where recorded here.
    pub sizes_kib: Option<(f64, f64)>,
}

#[allow(clippy::too_many_arguments)]
const fn sec_only_row(
    n: usize,
    kappa: u32,
    with_queries: bool,
    ell: usize,
    modulus_bits: u32,
    lg_sigma_s: f64,
    lg_sigma_big_r: f64,
    lg_sigma_r: Option<f64>,
    sizes_kib: Option<(f64, f64)>,
) -> SecOnlyRow {
    SecOnlyRow { n, kappa, with_queries, ell, modulus_bits, lg_sigma_s, lg_sigma_big_r, lg_sigma_r, sizes_kib }
}

/// Tables 6 and 7 of the SEC paper [46]: widths to four decimals from the outputs
/// of its notebooks `SEC-Parameter-Selection-v1.ipynb` (moduli of the LaTSEC
/// sets) and `SEC-Parameter-Selection-v1-q32.ipynb` (`q = 2^32 - 10239`), cells
/// 210 and 221.  The notebooks are not part of this artifact.
pub const SEC_ONLY_ROWS: &[SecOnlyRow] = &[
    sec_only_row(40, 128, false, 12, 61, 2.9574, 4.7360, None, Some((3.753, 5.697))),
    sec_only_row(40, 128, true, 12, 61, 3.7298, 5.2219, Some(43.6462), Some((4.333, 6.068))),
    sec_only_row(40, 192, false, 16, 63, 3.2359, 4.9954, None, Some((5.283, 7.856))),
    sec_only_row(40, 192, true, 16, 63, 4.0394, 5.4781, Some(44.6152), Some((6.087, 8.350))),
    sec_only_row(40, 256, false, 20, 64, 3.4364, 5.1856, None, Some((6.854, 10.058))),
    sec_only_row(40, 256, true, 20, 64, 4.2569, 5.6663, Some(45.3127), Some((7.880, 10.674))),
    sec_only_row(60, 128, false, 12, 64, 2.4630, 5.2100, None, Some((3.383, 6.041))),
    sec_only_row(60, 128, true, 12, 64, 2.6704, 5.7112, Some(43.9338), Some((3.538, 6.416))),
    sec_only_row(60, 192, false, 16, 66, 2.7327, 5.4679, None, Some((4.780, 8.313))),
    sec_only_row(60, 192, true, 16, 66, 2.9442, 5.9691, Some(44.8980), Some((4.991, 8.812))),
    sec_only_row(60, 256, false, 20, 66, 2.9284, 5.6573, None, Some((6.219, 10.628))),
    sec_only_row(60, 256, true, 20, 66, 3.1420, 6.1585, Some(45.5926), Some((6.486, 11.252))),
    sec_only_row(40, 128, false, 6, 32, 2.9521, 4.7312, None, Some((1.875, 2.847))),
    sec_only_row(40, 128, true, 6, 32, 3.7238, 5.2172, Some(43.6412), Some((2.164, 3.032))),
    sec_only_row(40, 192, false, 8, 32, 3.2323, 4.9921, None, Some((2.640, 3.926))),
    sec_only_row(40, 192, true, 8, 32, 4.0355, 5.4748, Some(44.6118), Some((3.041, 4.173))),
    sec_only_row(40, 256, false, 10, 32, 3.4337, 5.1830, None, Some((3.425, 5.027))),
    sec_only_row(40, 256, true, 10, 32, 4.2540, 5.6637, Some(45.3101), Some((3.938, 5.335))),
    sec_only_row(60, 128, false, 6, 32, 2.4579, 5.2052, None, Some((1.689, 3.019))),
    sec_only_row(60, 128, true, 6, 32, 2.6652, 5.7063, Some(43.9290), None),
    sec_only_row(60, 192, false, 8, 32, 2.7292, 5.4645, None, None),
    sec_only_row(60, 192, true, 8, 32, 2.9406, 5.9657, Some(44.8946), None),
    sec_only_row(60, 256, false, 10, 32, 2.9257, 5.6547, None, None),
    sec_only_row(60, 256, true, 10, 32, 3.1393, 6.1559, Some(45.5900), None),
];
