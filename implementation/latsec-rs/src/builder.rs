//! Construction of the standard profiles.
//!
//! * **Author profiles** reproduce the six parameter sets of Tables 4 and 5 of the
//!   paper from its matrices and the ported optimizer.
//! * **Functional profiles** are small or deliberately unbalanced parameter sets
//!   for tests.  They carry no security claim.

use std::path::Path;

use crate::error::{Error, Result};
use crate::fixture::AuthorMatrix;
use crate::intmat::ReconParams;
use crate::params::{solve_widths, SolverCase, TailRule};
use crate::profile::{
    Admissibility, Category, EstimateRecord, Evidence, MatrixConstantsRecord, Profile, ProfileInputs, ReconStatistics,
};
use crate::published::{self, PublishedRow};

const NOTEBOOK: &str = "parameter-search/LaTSEC-Parameter-Selection-v2.3-fixedtailbound+privacy.ipynb";

/// The identifier of the author profile for `(n, kappa)`.
pub fn author_profile_id(n: usize, kappa: u32) -> String {
    format!("author-n{n}-k{kappa}-nov2026")
}

/// The as-published profile of one row of Table 4, on the paper's matrix.
pub fn author_profile(matrix: &AuthorMatrix, row: &PublishedRow) -> Result<Profile> {
    let geo = published::geometry(row.n);
    if matrix.n != geo.n || matrix.k != geo.k || matrix.k_p != geo.k_p || matrix.c_exp != geo.c_exp {
        return Err(Error::invalid("matrix fixture does not match the published instance"));
    }
    let inputs = published::width_inputs(row);
    let widths = solve_widths(&inputs)?;
    if widths.case != SolverCase::BothCaps {
        return Err(Error::invalid("published rows are expected in solver case 4"));
    }
    let (q_h, q_sign) = published::query_budgets(row.kappa);
    let c = &geo.constants;
    let target_ak = f64::from(row.kappa + 1 + crate::params::ceil_lg(geo.k as u128));
    let estimate = |instance: &str, bits: f64, target: f64| EstimateRecord {
        instance: instance.to_string(),
        bits,
        target_bits: target,
        source: "the parameter-selection notebook (lattice-estimator 27a581b, ADPS16, SageMath 10.7); reproduced with \
                 tools/latsec_estimates.py, see tools/results"
            .to_string(),
    };
    let evidence = Evidence {
        claim: "As-published parameters of the paper. The security levels are lattice-estimator \
                results for the underlying MLWE/MSIS problems; the forking loss of the reduction is not charged. \
                Running this prototype adds no security evidence."
            .to_string(),
        matrix_constants: Some(MatrixConstantsRecord {
            alpha_kp: c.alpha_kp,
            norm_g_hat: c.norm_g_hat,
            norm_g: c.norm_g,
            beta_q: c.beta_q,
            beta_kp: c.beta_kp,
            gamma_kp: c.gamma_kp,
            method: "float64 SVD over all subsets (MatrixResult notebooks); no error bound".to_string(),
            certified: false,
        }),
        recon_statistics: Some(ReconStatistics {
            observed_linf_max: geo.beta_lambda_observed,
            observed_weight_max: geo.omega_lambda_observed,
            pipeline: "the SageMath pipeline of [46] (Sage LLL, passes = window = 2)".to_string(),
            note: "Sampled maxima over the sets tested in [46]. The enforced bounds are beta_lambda = observed + 1 \
                   and omega_lambda slightly above the observed weight. Rejection rates under this crate's \
                   reconstruction algorithm must be measured separately."
                .to_string(),
        }),
        estimates: vec![
            estimate("MQ", row.security.mq, f64::from(row.kappa)),
            estimate("AK", row.security.ak, target_ak),
            estimate("MSIS", row.security.msis, f64::from(row.kappa)),
        ],
        notes: vec![
            "Widths are the optimizer's values rounded upward to dyadic rationals; bounds are derived exactly from \
             them and exceed the paper's real-valued bounds by a relative 2^-30 at most."
                .to_string(),
            "H_c is SHAKE256 feeding an unbiased sample-in-ball shuffle. Codec version 3: the signature carries \
             the challenge ranked in b_c bits; shares, partial responses and signature responses are Golomb-Rice \
             coded."
                .to_string(),
            "q_participate = floor(Q_Sign / n), so local caps imply at most Q_Sign sessions with an honest signer."
                .to_string(),
        ],
    };
    Profile::derive(ProfileInputs {
        id: author_profile_id(row.n, row.kappa),
        category: Category::Author,
        description: format!("LaTSEC, n = {}, kappa = {}: Table 4 of the paper on its SEC matrix", row.n, row.kappa),
        provenance: vec![
            "parameters: Tables 4 and 5 of the paper".to_string(),
            format!("optimizer inputs and exact prime: {NOTEBOOK}"),
            format!("matrix: {} (fixtures/author/{})", matrix.source.notebook, geo.fixture),
            "widths recomputed by src/params.rs (port of the notebook's optimizer)".to_string(),
        ],
        kappa: row.kappa,
        d: published::RING_DEGREE,
        q: row.q,
        p: row.p,
        ell: row.ell,
        matrix_name: matrix.name.clone(),
        matrix_source: Some(matrix.source.clone()),
        g: matrix.g.clone(),
        t_uf: geo.k_p,
        t_sig: geo.k_c,
        k_c_det: Some(geo.k_c_det),
        q_h,
        q_sign_global: q_sign,
        q_participate: q_sign / geo.n as u128,
        accounting: "global Q_Sign as in the paper; each signer enforces q_participate locally".to_string(),
        sigma_s: widths.sigma_s,
        sigma_big_r: widths.sigma_big_r,
        sigma_r: widths.sigma_r,
        omega_c: inputs.omega_c,
        tail_rule: TailRule::ClosedForm,
        admissibility: Admissibility::Paper { beta_lambda: geo.beta_lambda, omega_lambda: geo.omega_lambda },
        recon: ReconParams::DEFAULT,
        evidence,
    })
}

/// Lattice estimates for the published widths at the largest NTT-friendly prime
/// below `2^(b_q - 1)`, from `tools/results/20261106-estimator-d857637-all-variants.json`
/// (variant `bq-minus-1`; lattice-estimator `d857637`, ADPS16, SageMath 10.9):
/// `(n, kappa, MQ, AK, MSIS)` in bits.
const VARIANT_ESTIMATES: [(usize, u32, f64, f64, f64); 6] = [
    (40, 128, 172.9, 155.9, 150.1),
    (40, 192, 243.2, 220.8, 212.9),
    (40, 256, 319.2, 290.8, 283.2),
    (60, 128, 167.6, 145.4, 143.4),
    (60, 192, 236.2, 207.0, 203.8),
    (60, 256, 317.1, 278.9, 266.0),
];

/// The identifier of the one-bit-smaller-modulus variant for `(n, kappa)`.
pub fn variant_profile_id(n: usize, kappa: u32, modulus_bits: u32) -> String {
    format!("variant-n{n}-k{kappa}-q{modulus_bits}")
}

/// A variant of an author profile with the modulus one bit shorter: the same
/// widths and bounds, and `q` the largest prime below `2^(b_q - 1)` with
/// `q = 1 mod 2d`.  The Euclidean requirement `B_MSIS < q` holds, but the paper's
/// rule `q > 2 B_MSIS + 1` (imposed by its pinned estimator commit) does not.
/// This is our variant, not a parameter set of the paper.
pub fn author_variant_profile(matrix: &AuthorMatrix, row: &PublishedRow) -> Result<Profile> {
    let mut profile = author_profile(matrix, row)?;
    let bits = row.modulus_bits - 1;
    let q = crate::ring::largest_ntt_prime_below(bits, published::RING_DEGREE)?;
    let &(_, _, mq, ak, msis) = VARIANT_ESTIMATES
        .iter()
        .find(|e| e.0 == row.n && e.1 == row.kappa)
        .ok_or_else(|| Error::invalid("no recorded estimate for this variant"))?;
    let ring = crate::ring::Ring::new(published::RING_DEGREE, q)?;
    profile.id = variant_profile_id(row.n, row.kappa, bits);
    profile.category = Category::AuthorVariant;
    profile.description = format!(
        "LaTSEC, n = {}, kappa = {}: the widths and bounds of the paper with a {bits}-bit modulus (one bit below \
         Table 4)",
        row.n, row.kappa
    );
    profile
        .provenance
        .push(format!("modulus: largest prime below 2^{bits} that is 1 mod 1024 (Section 7.2 of the paper)"));
    profile.ring.q = q;
    profile.ring.psi = ring.psi();
    let source = "tools/results/20261106-estimator-d857637-all-variants.json (variant bq-minus-1; \
                  lattice-estimator d857637, ADPS16, SageMath 10.9)";
    for estimate in profile.evidence.estimates.iter_mut() {
        estimate.bits = match estimate.instance.as_str() {
            "MQ" => mq,
            "AK" => ak,
            _ => msis,
        };
        estimate.source = source.to_string();
    }
    profile.evidence.claim = "Our variant of the published parameters, not a parameter set of the paper. The \
                              estimates were run with an estimator commit that accepts Euclidean SIS bounds up to q; \
                              B_MSIS is between 0.52 q and 0.91 q. The forking loss is not charged."
        .to_string();
    profile.evidence.notes.push(
        "The paper's rule q > 2 B_MSIS + 1 fails by design; the Euclidean requirement B_MSIS < q holds.".to_string(),
    );
    profile.refresh();
    profile.validate()?;
    Ok(profile)
}

/// The six one-bit-smaller-modulus variants.
pub fn author_variant_profiles(fixture_dir: impl AsRef<Path>) -> Result<Vec<Profile>> {
    let dir = fixture_dir.as_ref().join("author");
    published::PUBLISHED_ROWS
        .iter()
        .map(|row| {
            let matrix = AuthorMatrix::load(dir.join(published::geometry(row.n).fixture))?;
            author_variant_profile(&matrix, row)
        })
        .collect()
}

/// Explicit parameters of a functional profile.
struct FunctionalSpec {
    id: &'static str,
    description: &'static str,
    d: usize,
    /// The modulus is the largest prime below `2^modulus_bits` that is `1 mod 2d`.
    modulus_bits: u32,
    p: usize,
    ell: usize,
    t_sig: usize,
    k_c_det: Option<usize>,
    lg_q_h: u32,
    lg_q_sign: u32,
    sigma_s: f64,
    sigma_r: f64,
    omega_c: u32,
    l1_cap: u32,
    notes: &'static [&'static str],
}

fn functional_profile(matrix: &AuthorMatrix, spec: &FunctionalSpec) -> Result<Profile> {
    let q_sign = 1u128 << spec.lg_q_sign;
    Profile::derive(ProfileInputs {
        id: spec.id.to_string(),
        category: Category::Functional,
        description: spec.description.to_string(),
        provenance: vec![
            format!("matrix: {} (fixtures/author)", matrix.source.notebook),
            "all other values chosen by hand for testing in src/builder.rs".to_string(),
        ],
        kappa: 128,
        d: spec.d,
        q: crate::ring::largest_ntt_prime_below(spec.modulus_bits, spec.d)?,
        p: spec.p,
        ell: spec.ell,
        matrix_name: matrix.name.clone(),
        matrix_source: Some(matrix.source.clone()),
        g: matrix.g.clone(),
        t_uf: matrix.k_p,
        t_sig: spec.t_sig,
        k_c_det: spec.k_c_det,
        q_h: 1u128 << spec.lg_q_h,
        q_sign_global: q_sign,
        q_participate: q_sign / matrix.n as u128,
        accounting: "functional profile; the budgets only size the tail factors".to_string(),
        sigma_s: spec.sigma_s,
        sigma_big_r: spec.sigma_s,
        sigma_r: spec.sigma_r,
        omega_c: spec.omega_c,
        tail_rule: TailRule::ClosedForm,
        admissibility: Admissibility::L1 { l1_cap: spec.l1_cap },
        recon: ReconParams::DEFAULT,
        evidence: Evidence {
            claim: "No security claim. Functional test profile.".to_string(),
            notes: spec.notes.iter().map(|s| s.to_string()).collect(),
            ..Evidence::default()
        },
    })
}

/// Reconstruction facts used below were obtained exhaustively with
/// `latsec recon-stats` (all subsets of both matrices):
///
/// * 10 x 6 matrix: 20% of the 7-sets and 6.7% of the 8-sets are unauthorized; every
///   set of 9 or 10 is authorized; every authorized set of size >= 7 has
///   `|lambda|_1 = 4`.
/// * 20 x 10 matrix: 0.76% of the 14-sets and 0.12% of the 15-sets are unauthorized;
///   every set of 16 or more is authorized; the largest `|lambda|_1` is 14 at size
///   14, 10 at size 15 and 8 from size 16.
const FUNCTIONAL_N10: FunctionalSpec = FunctionalSpec {
    id: "functional-n10-d16",
    description: "Small functional fixture: a 10 x 6 SEC matrix of [46], ring degree 16",
    d: 16,
    modulus_bits: 30,
    p: 2,
    ell: 4,
    t_sig: 7,
    k_c_det: Some(9),
    lg_q_h: 20,
    lg_q_sign: 20,
    sigma_s: 8.0,
    sigma_r: 1048576.0,
    omega_c: 4,
    l1_cap: 4,
    notes: &["t_sig = 7 is below the size (9) from which every set is authorized, so some sets of size 7 and 8 are \
         rejected as unauthorized. Every authorized set of size >= 7 meets the l1 cap."],
};

const FUNCTIONAL_N20: FunctionalSpec = FunctionalSpec {
    id: "functional-n20-d16",
    description: "Small functional fixture: a 20 x 10 SEC matrix of [46], ring degree 16",
    d: 16,
    modulus_bits: 31,
    p: 2,
    ell: 4,
    t_sig: 14,
    k_c_det: Some(16),
    lg_q_h: 20,
    lg_q_sign: 20,
    sigma_s: 8.0,
    sigma_r: 1048576.0,
    omega_c: 4,
    l1_cap: 8,
    notes: &["The l1 cap 8 admits every set of 16 or more parties but rejects some authorized sets of 14 and 15, so \
         both rejection paths (unauthorized, and authorized but outside the bounds) are exercised."],
};

const FUNCTIONAL_FULL_DEGREE: FunctionalSpec = FunctionalSpec {
    id: "functional-n20-d512",
    description: "Full-degree functional fixture: a 20 x 10 SEC matrix of [46], d = 512, 66-bit prime, wide masks",
    d: 512,
    modulus_bits: 66,
    p: 2,
    ell: 4,
    t_sig: 14,
    k_c_det: Some(16),
    lg_q_h: 63,
    lg_q_sign: 62,
    sigma_s: 46000.0,
    sigma_r: 72057594037927936.0, // 2^56
    omega_c: 30,
    l1_cap: 8,
    notes: &["Exercises d = 512 challenge sampling, the 66-bit modulus 2^66 - 16383, masks of width 2^56 and a \
         signature bound above 2^64 (66-bit response coefficients), with only 20 parties."],
};

/// The functional profiles, built from two small SEC matrices of [46].
pub fn functional_profiles(fixture_dir: impl AsRef<Path>) -> Result<Vec<Profile>> {
    let dir = fixture_dir.as_ref().join("author");
    let small = AuthorMatrix::load(dir.join("G-10-5-3.json"))?;
    let medium = AuthorMatrix::load(dir.join("G-20-9-3.json"))?;
    Ok(vec![
        functional_profile(&small, &FUNCTIONAL_N10)?,
        functional_profile(&medium, &FUNCTIONAL_N20)?,
        functional_profile(&medium, &FUNCTIONAL_FULL_DEGREE)?,
    ])
}

/// The six author profiles.
pub fn author_profiles(fixture_dir: impl AsRef<Path>) -> Result<Vec<Profile>> {
    let dir = fixture_dir.as_ref().join("author");
    published::PUBLISHED_ROWS
        .iter()
        .map(|row| {
            let matrix = AuthorMatrix::load(dir.join(published::geometry(row.n).fixture))?;
            author_profile(&matrix, row)
        })
        .collect()
}

/// The identifier of the SEC-only profile for `(n, kappa)` at the 32-bit modulus.
pub fn sec_only_profile_id(n: usize, kappa: u32) -> String {
    format!("sec-only-n{n}-k{kappa}-q32")
}

/// A standalone SEC parameter set of the SEC paper [46] (Table 7 there): leakage budget
/// `Q = 2^64`, hard-core modulus `q = 2^32 - 10239`, `p = ell/2`.
///
/// The profile has the shape of a LaTSEC profile so that the same code paths run,
/// but it is privacy-only: `sigma_r` is the SEC paper's leakage-mask width, `B1`
/// bounds a leakage response `c s_i + r_i`, and the signature bound `B2` is far
/// above `q`.
pub fn sec_only_profile(matrix: &AuthorMatrix, row: &published::SecOnlyRow) -> Result<Profile> {
    let geo = published::geometry(row.n);
    if !row.with_queries || row.modulus_bits != 32 {
        return Err(Error::invalid("SEC-only profiles are built for the Q = 2^64, 32-bit rows"));
    }
    let queries = 1u128 << 64;
    let omega_c = crate::params::challenge_weight(published::RING_DEGREE, 1, row.kappa)
        .ok_or_else(|| Error::invalid("no challenge weight"))?;
    let widths = crate::params::solve_sec_widths(&crate::params::SecWidthInputs {
        d: published::RING_DEGREE,
        ell: row.ell,
        n: geo.n,
        k: geo.k,
        k_p: geo.k_p,
        c_exp: geo.c_exp,
        beta_g: geo.beta_g,
        kappa: row.kappa,
        queries,
        omega_c,
        constants: geo.constants,
        eps_bits: row.kappa + 3,
    })?;
    let sigma_r = widths.sigma_r.ok_or_else(|| Error::invalid("missing leakage width"))?;
    let c = &geo.constants;
    Profile::derive(ProfileInputs {
        id: sec_only_profile_id(row.n, row.kappa),
        category: Category::SecOnly,
        description: format!(
            "SEC alone, n = {}, kappa = {}, Q = 2^64, q = 2^32 - 10239: Table 7 of the SEC paper [46]",
            row.n, row.kappa
        ),
        provenance: vec![
            "parameters: Tables 6 and 7 of the SEC paper [46]".to_string(),
            "optimizer: the SEC parameter-selection notebook of [46] (SEC-Parameter-Selection-v1-q32.ipynb, not part \
             of this artifact)"
                .to_string(),
            format!("matrix: {} (fixtures/author/{})", matrix.source.notebook, geo.fixture),
            "widths recomputed by src/params.rs (solve_sec_widths)".to_string(),
        ],
        kappa: row.kappa,
        d: published::RING_DEGREE,
        q: crate::ring::largest_ntt_prime_below(32, published::RING_DEGREE)?,
        p: row.ell / 2,
        ell: row.ell,
        matrix_name: matrix.name.clone(),
        matrix_source: Some(matrix.source.clone()),
        g: matrix.g.clone(),
        t_uf: geo.k_p,
        t_sig: geo.k_c,
        k_c_det: Some(geo.k_c_det),
        q_h: 1,
        q_sign_global: queries,
        q_participate: queries / geo.n as u128,
        accounting: "Q = 2^64 leakage queries in the multi-query privacy model of [46]; there is no random oracle"
            .to_string(),
        sigma_s: widths.sigma_s,
        sigma_big_r: widths.sigma_big_r,
        sigma_r,
        omega_c,
        tail_rule: TailRule::ClosedForm,
        admissibility: Admissibility::Paper { beta_lambda: geo.beta_lambda, omega_lambda: geo.omega_lambda },
        recon: ReconParams::DEFAULT,
        evidence: Evidence {
            claim: "Privacy-only parameters of the SEC paper [46] (multi-query privacy and kernel smoothing, with \
                    kappa_MLWE = kappa + 1). No signature claim: the modulus is not sized for an MSIS bound and \
                    signing is disabled for this profile."
                .to_string(),
            matrix_constants: Some(MatrixConstantsRecord {
                alpha_kp: c.alpha_kp,
                norm_g_hat: c.norm_g_hat,
                norm_g: c.norm_g,
                beta_q: c.beta_q,
                beta_kp: c.beta_kp,
                gamma_kp: c.gamma_kp,
                method: "float64 SVD over all subsets (MatrixResult notebooks); no error bound".to_string(),
                certified: false,
            }),
            recon_statistics: None,
            estimates: Vec::new(),
            notes: vec![
                "eps = eps' = 2^-(kappa + 3); the kernel-smoothing constraint binds and multi-query privacy is \
                 active (sigma_MQ = S_MQ)."
                    .to_string(),
                "The admissibility bounds and B2 are carried over from the LaTSEC instance for uniformity; SEC \
                 reconstruction itself only needs an authorized set of at least k_c parties."
                    .to_string(),
            ],
        },
    })
}

/// The SEC-only profiles that are kept as full-degree test profiles: both
/// matrices at `kappa = 128`.
pub fn sec_only_profiles(fixture_dir: impl AsRef<Path>) -> Result<Vec<Profile>> {
    let dir = fixture_dir.as_ref().join("author");
    published::SEC_ONLY_ROWS
        .iter()
        .filter(|row| row.with_queries && row.modulus_bits == 32 && row.kappa == 128)
        .map(|row| {
            let matrix = AuthorMatrix::load(dir.join(published::geometry(row.n).fixture))?;
            sec_only_profile(&matrix, row)
        })
        .collect()
}

/// All profiles written by `latsec profiles build`.
pub fn standard_profiles(fixture_dir: impl AsRef<Path>) -> Result<Vec<Profile>> {
    let mut profiles = functional_profiles(&fixture_dir)?;
    profiles.extend(author_profiles(&fixture_dir)?);
    profiles.extend(author_variant_profiles(&fixture_dir)?);
    profiles.extend(sec_only_profiles(&fixture_dir)?);
    Ok(profiles)
}
