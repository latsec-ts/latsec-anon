//! Command-line front end: profile generation and checking, reconstruction
//! statistics, a signing demonstration and measurements.

use std::collections::BTreeMap;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

use latsec::builder;
use latsec::fixture::AuthorMatrix;
use latsec::intmat::ReconParams;
use latsec::latsec::{verify, SessionId, SessionMessage};
use latsec::profile::{Admissibility, Profile};
use latsec::sampling::{OsRandom, RandomSource, SeededRandom};
use latsec::scheme::{ActiveSet, PartyId, Scheme};
use latsec::sim::{Deployment, SessionOutcome, SessionTiming};
use latsec::stats::{random_subset, size_statistics, SizeStatistics};

const USAGE: &str = "\
usage: latsec <command> [options]

commands:
  profiles build [--fixtures DIR] [--out DIR]
      Write the standard profiles (six author profiles and the functional ones).
      The directories default to fixtures/ and profiles/ of the repository.
  profiles check FILE...
      Validate profiles and print their bounds and security-side checks.
  recon-stats --matrix FILE [--sizes A..B] [--samples N] [--seed TEXT]
              [--threads N] [--beta B --omega W | --l1 L] [--json FILE]
      Reconstruction-vector statistics per set size with this crate's algorithm.
  recon-compare --matrix FILE --reference FILE
      Compare this crate's reconstruction vectors with a reference file written
      by tools/gen_recon_fixtures.py (the SageMath pipeline of [46]).
  demo --profile FILE [--size T | --set I,J,...] [--seed TEXT]
      Trusted key generation, one signing session and verification.
  measure --profile FILE [--sessions N] [--size T] [--seed TEXT] [--json FILE]
          [--warmup N] [--keygen-reps N]
      Honest sessions on random signer sets: admissibility, partial rejections,
      idle signers, sizes and timings (medians and quartiles, one thread, after
      N warm-up sessions; the record includes the CPU, its frequency policy and
      the build settings).  Pin the process to one core for stable timings.
  kat [--profiles DIR] [--out DIR] [--seed TEXT]
      Write one deterministic test vector per profile (defaults: profiles/ and
      fixtures/kat/ of the repository).
  matrix-check --matrix FILE [--spectral] [--threads N]
      Recheck an author matrix: rank, expansion, peeling, planted witnesses, and
      optionally the float64 matrix constants (exhaustive; slow for n = 60).

Party indices are zero-based.  With --seed, all randomness is derived from the
text (reproducible, not secret); without it the operating system supplies it.";

type CliResult<T> = Result<T, String>;

/// Options of the form `--name value`, plus positional arguments.
struct Args {
    options: BTreeMap<String, String>,
    positional: Vec<String>,
}

impl Args {
    fn parse(raw: &[String]) -> CliResult<Self> {
        let mut options = BTreeMap::new();
        let mut positional = Vec::new();
        let mut iter = raw.iter();
        while let Some(arg) = iter.next() {
            match arg.strip_prefix("--") {
                Some(name) => {
                    let value = iter.next().ok_or_else(|| format!("option --{name} needs a value"))?;
                    options.insert(name.to_string(), value.clone());
                }
                None => positional.push(arg.clone()),
            }
        }
        Ok(Args { options, positional })
    }

    fn get(&self, name: &str) -> Option<&str> {
        self.options.get(name).map(String::as_str)
    }

    fn required(&self, name: &str) -> CliResult<&str> {
        self.get(name).ok_or_else(|| format!("missing option --{name}"))
    }

    fn number<T: std::str::FromStr>(&self, name: &str, default: T) -> CliResult<T> {
        match self.get(name) {
            Some(text) => text.parse().map_err(|_| format!("option --{name}: cannot parse {text:?}")),
            None => Ok(default),
        }
    }

    fn reject_unknown(&self, known: &[&str]) -> CliResult<()> {
        match self.options.keys().find(|k| !known.contains(&k.as_str())) {
            Some(name) => Err(format!("unknown option --{name}")),
            None => Ok(()),
        }
    }
}

fn random_source(args: &Args, label: &str) -> Box<dyn RandomSource> {
    match args.get("seed") {
        Some(seed) => Box::new(SeededRandom::new(seed.as_bytes(), label)),
        None => Box::new(OsRandom::new()),
    }
}

fn default_threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

fn profiles_build(args: &Args) -> CliResult<()> {
    args.reject_unknown(&["fixtures", "out"])?;
    let fixtures = args.get("fixtures").unwrap_or(concat!(env!("CARGO_MANIFEST_DIR"), "/../fixtures"));
    let out = args.get("out").unwrap_or(concat!(env!("CARGO_MANIFEST_DIR"), "/../profiles"));
    std::fs::create_dir_all(out).map_err(err)?;
    for profile in builder::standard_profiles(fixtures).map_err(err)? {
        let path = format!("{out}/{}.json", profile.id);
        profile.save(&path).map_err(err)?;
        println!("wrote {path}");
    }
    Ok(())
}

fn print_profile(profile: &Profile) {
    let [lg_s, lg_big_r, lg_r, lg_b1, lg_b2, lg_msis] = profile.display_logs();
    println!("profile   {}  [{:?}]", profile.id, profile.category);
    println!(
        "  ring    d = {}, q = {} ({} bits), psi = {}",
        profile.ring.d,
        profile.ring.q,
        128 - profile.ring.q.leading_zeros(),
        profile.ring.psi
    );
    println!(
        "  shape   p = {}, ell = {}, n = {}, k = {}, t_uf = {}, t_sig = {}, kappa = {}",
        profile.p, profile.ell, profile.sec.n, profile.sec.k, profile.sec.t_uf, profile.sec.t_sig, profile.kappa
    );
    println!("  widths  lg sigma_s = {lg_s:.4}, lg sigma_R = {lg_big_r:.4}, lg sigma_r = {lg_r:.4}");
    println!(
        "  bounds  lg B1 = {lg_b1:.4}, lg B2 = {lg_b2:.4}, lg B_MSIS = {lg_msis:.4}, omega_c = {}",
        profile.challenge.omega_c
    );
    println!("  admit   {:?}", profile.admissibility);
    println!("  claim   {}", profile.evidence.claim);
    for check in profile.checks() {
        let detail = if check.detail.is_empty() { String::new() } else { format!(" ({})", check.detail) };
        println!("  [{}] {}{detail}", if check.ok { "ok" } else { "NO" }, check.name);
    }
}

fn profiles_check(args: &Args) -> CliResult<()> {
    args.reject_unknown(&[])?;
    if args.positional.is_empty() {
        return Err("profiles check needs at least one file".into());
    }
    for path in &args.positional {
        let profile = Profile::load(path).map_err(err)?;
        print_profile(&profile);
        println!("  digest  {}", profile.digest);
        println!();
    }
    Ok(())
}

fn parse_range(text: &str) -> CliResult<(usize, usize)> {
    let (a, b) = text.split_once("..").ok_or("expected A..B")?;
    let lo = a.parse().map_err(|_| "bad range start")?;
    let hi = b.parse().map_err(|_| "bad range end")?;
    if lo > hi {
        return Err("empty range".into());
    }
    Ok((lo, hi))
}

fn histogram_json(hist: &BTreeMap<u64, u64>) -> String {
    let items: Vec<String> = hist.iter().map(|(k, v)| format!("\"{k}\": {v}")).collect();
    format!("{{{}}}", items.join(", "))
}

fn recon_stats(args: &Args) -> CliResult<()> {
    args.reject_unknown(&["matrix", "sizes", "samples", "seed", "threads", "beta", "omega", "l1", "json"])?;
    let matrix = AuthorMatrix::load(args.required("matrix")?).map_err(err)?;
    let g = matrix.matrix();
    let (lo, hi) = match args.get("sizes") {
        Some(text) => parse_range(text)?,
        None => (1, matrix.n),
    };
    if lo == 0 || hi > matrix.n {
        return Err(format!("sizes must lie in 1..{}", matrix.n));
    }
    let samples: u64 = args.number("samples", 10_000)?;
    let seed = args.get("seed").unwrap_or("recon-stats");
    let threads: usize = args.number("threads", default_threads())?;
    let predicate = match (args.get("beta"), args.get("omega"), args.get("l1")) {
        (Some(_), Some(_), None) => {
            Some(Admissibility::Paper { beta_lambda: args.number("beta", 0)?, omega_lambda: args.number("omega", 0)? })
        }
        (None, None, Some(_)) => Some(Admissibility::L1 { l1_cap: args.number("l1", 0)? }),
        (None, None, None) => None,
        _ => return Err("give either --beta and --omega, or --l1".into()),
    };
    let params = ReconParams::DEFAULT;
    println!(
        "matrix {} ({} x {}), algorithm {}, delta = {}/{}, passes = {}, window = {}, seed {seed:?}",
        matrix.name,
        matrix.n,
        matrix.k,
        latsec::profile::RECON_ALGORITHM_ID,
        params.delta.num,
        params.delta.den,
        params.passes,
        params.window
    );
    if let Some(p) = &predicate {
        println!("predicate {p:?}");
    }
    println!("size  mode    tested  unauth  unauth%  rejected | max_inf mean_inf P[inf<=2] | min_wt mean_wt max_wt | max_l1 mean_l1 |  us/set");
    let mut all: Vec<SizeStatistics> = Vec::new();
    for size in lo..=hi {
        let s = size_statistics(&g, params, predicate.as_ref(), size, samples, seed.as_bytes(), threads);
        println!(
            "{:4}  {:6} {:7} {:7} {:7.3}% {:9} | {:7} {:8.3} {:9.4} | {:6} {:7.2} {:6} | {:6} {:7.2} | {:7.0}",
            s.size,
            if s.exhaustive { "exact" } else { "sample" },
            s.tested,
            s.unauthorized(),
            100.0 * s.unauthorized() as f64 / s.tested as f64,
            s.rejected_by_bounds,
            SizeStatistics::max_of(&s.linf),
            SizeStatistics::mean_of(&s.linf),
            SizeStatistics::fraction_at_most(&s.linf, 2),
            s.weight.keys().next().copied().unwrap_or(0),
            SizeStatistics::mean_of(&s.weight),
            SizeStatistics::max_of(&s.weight),
            SizeStatistics::max_of(&s.l1),
            SizeStatistics::mean_of(&s.l1),
            s.micros_per_set,
        );
        for (set, lambda) in &s.rejected_sets {
            println!("      rejected: set {set:?}");
            println!("                lambda {lambda:?}");
        }
        all.push(s);
    }
    let tested: u64 = all.iter().map(|s| s.tested).sum();
    let authorized: u64 = all.iter().map(|s| s.authorized).sum();
    let rejected: u64 = all.iter().map(|s| s.rejected_by_bounds).sum();
    println!(
        "total: {tested} sets, {authorized} authorized, {} unauthorized, {rejected} authorized sets rejected by the bounds; \
         max |lambda|_inf = {}, max weight = {}, max |lambda|_1 = {}",
        tested - authorized,
        all.iter().map(|s| SizeStatistics::max_of(&s.linf)).max().unwrap_or(0),
        all.iter().map(|s| SizeStatistics::max_of(&s.weight)).max().unwrap_or(0),
        all.iter().map(|s| SizeStatistics::max_of(&s.l1)).max().unwrap_or(0),
    );
    if let Some(path) = args.get("json") {
        let rows: Vec<String> = all
            .iter()
            .map(|s| {
                let rejected: Vec<String> = s
                    .rejected_sets
                    .iter()
                    .map(|(set, lambda)| format!("{{\"set\": {set:?}, \"lambda\": {lambda:?}}}"))
                    .collect();
                format!(
                    "    {{\"size\": {}, \"exhaustive\": {}, \"tested\": {}, \"authorized\": {}, \"rejected_by_bounds\": {}, \
                     \"linf\": {}, \"weight\": {}, \"l1\": {}, \"rejected_sets\": [{}]}}",
                    s.size, s.exhaustive, s.tested, s.authorized, s.rejected_by_bounds,
                    histogram_json(&s.linf), histogram_json(&s.weight), histogram_json(&s.l1), rejected.join(", ")
                )
            })
            .collect();
        let predicate_text = predicate.map_or("null".to_string(), |p| format!("\"{p:?}\""));
        let text = format!(
            "{{\n  \"matrix\": \"{}\",\n  \"matrix_source_sha256\": \"{}\",\n  \"algorithm\": \"{}\",\n  \"delta\": \"{}/{}\",\n  \
             \"passes\": {},\n  \"window\": {},\n  \"seed\": \"{seed}\",\n  \"samples_per_size\": {samples},\n  \
             \"predicate\": {predicate_text},\n  \"sizes\": [\n{}\n  ]\n}}\n",
            matrix.name, matrix.source.cell_sources_sha256, latsec::profile::RECON_ALGORITHM_ID, params.delta.num, params.delta.den,
            params.passes, params.window, rows.join(",\n")
        );
        std::fs::write(path, text).map_err(err)?;
        println!("wrote {path}");
    }
    Ok(())
}

fn recon_compare(args: &Args) -> CliResult<()> {
    args.reject_unknown(&["matrix", "reference"])?;
    let matrix = AuthorMatrix::load(args.required("matrix")?).map_err(err)?;
    let g = matrix.matrix();
    let text = std::fs::read_to_string(args.required("reference")?).map_err(err)?;
    let reference: serde_json::Value = serde_json::from_str(&text).map_err(err)?;
    if reference["matrix"].as_str() != Some(matrix.name.as_str()) {
        return Err("the reference file is for another matrix".into());
    }
    let cases = reference["cases"].as_array().ok_or("reference file has no cases")?;
    let norms =
        |v: &[i64]| (v.iter().map(|x| x.unsigned_abs()).max().unwrap_or(0), v.iter().filter(|&&x| x != 0).count());
    let (mut authorized, mut auth_mismatch, mut identical, mut same_norms) = (0u64, 0u64, 0u64, 0u64);
    let (mut ours_shorter, mut theirs_shorter) = (0u64, 0u64);
    // size -> (authorized, identical)
    let mut by_size: BTreeMap<usize, (u64, u64)> = BTreeMap::new();
    for case in cases {
        let set: Vec<usize> = case["set"]
            .as_array()
            .ok_or("case without a set")?
            .iter()
            .map(|x| x.as_u64().map(|v| v as usize).ok_or("bad party index"))
            .collect::<Result<_, _>>()?;
        let ours = latsec::intmat::recon_vector(&g.select_rows(&set), ReconParams::DEFAULT);
        let expected_auth = case["authorized"].as_bool().ok_or("case without authorization flag")?;
        if ours.is_some() != expected_auth {
            auth_mismatch += 1;
            continue;
        }
        let Some(ours) = ours else { continue };
        authorized += 1;
        let ours: Vec<i64> = ours.iter().map(|x| i64::try_from(x).unwrap_or(i64::MAX)).collect();
        let theirs: Vec<i64> = case["lambda"]
            .as_array()
            .ok_or("authorized case without a vector")?
            .iter()
            .map(|x| x.as_i64().ok_or("bad coefficient"))
            .collect::<Result<_, _>>()?;
        let entry = by_size.entry(set.len()).or_insert((0, 0));
        entry.0 += 1;
        if ours == theirs {
            identical += 1;
            entry.1 += 1;
        }
        let (a, b) = (norms(&ours), norms(&theirs));
        if a == b {
            same_norms += 1;
        } else if a < b {
            ours_shorter += 1;
        } else {
            theirs_shorter += 1;
        }
    }
    println!(
        "matrix {}, reference by {} (SageMath {}), {} sets",
        matrix.name,
        reference["generator"].as_str().unwrap_or("?"),
        reference["sage_version"].as_str().unwrap_or("?"),
        cases.len()
    );
    println!("  authorization agrees on {} sets, differs on {auth_mismatch}", cases.len() as u64 - auth_mismatch);
    println!(
        "  {authorized} authorized: identical vector in {identical} ({:.2}%), different in {}",
        100.0 * identical as f64 / authorized.max(1) as f64,
        authorized - identical
    );
    println!("  (linf, weight) equal in {same_norms}; lexicographically smaller here in {ours_shorter}, in the reference in {theirs_shorter}");
    println!("  size: authorized, identical");
    for (size, (auth, same)) in &by_size {
        println!("  {size:4}: {auth:6} {same:6}  ({:.1}%)", 100.0 * *same as f64 / (*auth).max(1) as f64);
    }
    if auth_mismatch > 0 {
        return Err("authorization differs from the reference".into());
    }
    Ok(())
}

fn kib(bytes: usize) -> f64 {
    bytes as f64 / 1024.0
}

fn session_id(rng: &mut dyn RandomSource) -> SessionId {
    let mut sid = [0u8; 32];
    rng.fill(&mut sid);
    SessionId(sid)
}

/// Loads the profile named by `--profile` for a signing command.
fn load_scheme(args: &Args) -> CliResult<Arc<Scheme>> {
    let profile = Profile::load(args.required("profile")?).map_err(err)?;
    let scheme = Scheme::new(profile).map_err(err)?;
    if !scheme.supports_signing() {
        return Err(format!("{} is a SEC-only profile and does not support signing", scheme.profile().id));
    }
    Ok(Arc::new(scheme))
}

fn parse_set(text: &str) -> CliResult<ActiveSet> {
    let parties: Vec<PartyId> = text
        .split(',')
        .map(|t| t.trim().parse().map_err(|_| format!("bad party index {t:?}")))
        .collect::<CliResult<_>>()?;
    ActiveSet::new(&parties).map_err(err)
}

fn random_set(rng: &mut dyn RandomSource, n: usize, size: usize) -> ActiveSet {
    let parties: Vec<PartyId> = random_subset(rng, n, size).into_iter().map(|p| p as PartyId).collect();
    ActiveSet::new(&parties).expect("distinct parties")
}

fn demo(args: &Args) -> CliResult<()> {
    args.reject_unknown(&["profile", "size", "set", "seed"])?;
    let scheme = load_scheme(args)?;
    let profile = scheme.profile().clone();
    print_profile(&profile);
    let mut rng = random_source(args, "demo");

    let started = Instant::now();
    let mut deployment = Deployment::keygen(Arc::clone(&scheme), rng.as_mut());
    println!("\nkey generation (trusted dealer): {:.2} s", started.elapsed().as_secs_f64());
    let vk_bytes = deployment.vk.to_bytes(&scheme);
    let ak_bytes = deployment.abort_keys[0].to_bytes(&scheme);
    println!(
        "  |vk| = {:.3} KiB, |ak_i| = {:.3} KiB, |ak| = {:.1} KiB",
        kib(vk_bytes.len()),
        kib(ak_bytes.len()),
        kib(ak_bytes.len() * profile.sec.n)
    );

    let act = match (args.get("set"), args.get("size")) {
        (Some(text), None) => parse_set(text)?,
        (None, size) => {
            let size = match size {
                Some(_) => args.number("size", 0)?,
                None => profile.sec.t_sig,
            };
            if size == 0 || size > profile.sec.n {
                return Err("--size out of range".into());
            }
            // Draw sets until one is admissible, as a coordinator would.
            let mut tries = 0;
            loop {
                let candidate = random_set(rng.as_mut(), profile.sec.n, size);
                tries += 1;
                match scheme.lambda(&candidate) {
                    Ok(_) => break candidate,
                    Err(reason) if tries < 1000 => println!("  set {:?} rejected: {reason}", candidate.parties()),
                    Err(reason) => return Err(format!("no admissible set of size {size} found: {reason}")),
                }
            }
        }
        _ => return Err("give either --set or --size".into()),
    };
    let lambda = scheme.lambda(&act).map_err(|reason| format!("signer set rejected before Round 1: {reason}"))?;
    let active: usize = lambda.iter().filter(|&&x| x != 0).count();
    println!("\nsigner set ({} parties): {:?}", act.len(), act.parties());
    println!("lambda: {:?}", lambda.as_slice());
    println!(
        "  |lambda|_inf = {}, weight = {active} ({} idle signers), |lambda|_1 = {}",
        lambda.iter().map(|x| x.unsigned_abs()).max().unwrap_or(0),
        act.len() - active,
        lambda.iter().map(|x| x.unsigned_abs()).sum::<u64>()
    );

    let sid = session_id(rng.as_mut());
    let msg = b"LaTSEC demonstration message";
    let (outcome, timing) = deployment.sign(sid, &act, msg, rng.as_mut()).map_err(err)?;
    print_timing(&timing, act.len());
    match outcome {
        SessionOutcome::Aborted { blamed, stage } => {
            println!("session aborted at stage {stage}, blaming {blamed:?}");
            Err("honest session aborted".into())
        }
        SessionOutcome::Signed { signature, transcript } => {
            let sig_bytes = signature.to_bytes(&scheme).map_err(err)?;
            let resp_bytes = transcript.responses[0].to_bytes(&scheme).map_err(err)?;
            let open_bytes = transcript.openings[0].to_bytes(&scheme);
            println!("signature verifies: {}", verify(&scheme, &deployment.vk, &transcript.msg, &signature));
            println!(
                "  |sig| = {:.3} KiB, |z_i| = {:.3} KiB, opening = {:.3} KiB, commitment = {} bytes",
                kib(sig_bytes.len()),
                kib(resp_bytes.len()),
                kib(open_bytes.len()),
                transcript.commitments[0].0.len()
            );
            let mut other = SessionMessage::from_bytes(&scheme, &transcript.msg).map_err(err)?;
            other.payload = b"another message".to_vec();
            let other = other.to_bytes(&scheme).map_err(err)?;
            println!("  another message verifies: {}", verify(&scheme, &deployment.vk, &other, &signature));
            Ok(())
        }
    }
}

fn print_timing(t: &SessionTiming, signers: usize) {
    let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
    println!(
        "timing: reconstruction vector {:.2} ms; per signer: round 1 {:.2} ms, round 2 {:.3} ms, round 3 {:.2} ms; \
              combine {:.1} ms; verify {:.2} ms",
        ms(t.recon),
        ms(t.round1) / signers as f64,
        ms(t.round2) / signers as f64,
        ms(t.round3) / signers as f64,
        ms(t.combine),
        ms(t.verify)
    );
}

/// Mean and maximum of a list of sizes or ratios.
#[derive(Default)]
struct Spread {
    sum: f64,
    max: f64,
    count: usize,
}

impl Spread {
    fn add(&mut self, value: f64) {
        self.sum += value;
        self.max = self.max.max(value);
        self.count += 1;
    }

    fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }

    fn json(&self) -> serde_json::Value {
        serde_json::json!({"mean": round_to(self.mean(), 3), "max": round_to(self.max, 3)})
    }
}

fn round_to(value: f64, digits: i32) -> f64 {
    let scale = 10f64.powi(digits);
    (value * scale).round() / scale
}

/// Timing samples, in milliseconds.
#[derive(Default)]
struct Samples(Vec<f64>);

impl Samples {
    fn add(&mut self, duration: Duration) {
        self.0.push(duration.as_secs_f64() * 1e3);
    }

    /// One sample for a duration that covers `parts` equal calls.
    fn add_share(&mut self, duration: Duration, parts: usize) {
        self.0.push(duration.as_secs_f64() * 1e3 / parts.max(1) as f64);
    }

    fn mean(&self) -> f64 {
        if self.0.is_empty() {
            0.0
        } else {
            self.0.iter().sum::<f64>() / self.0.len() as f64
        }
    }

    /// The `p`-quantile with linear interpolation between order statistics.
    fn quantile(&self, p: f64) -> f64 {
        if self.0.is_empty() {
            return 0.0;
        }
        let mut sorted = self.0.clone();
        sorted.sort_by(f64::total_cmp);
        let position = p * (sorted.len() - 1) as f64;
        let (low, high) = (position.floor() as usize, position.ceil() as usize);
        sorted[low] + (sorted[high] - sorted[low]) * (position - low as f64)
    }

    fn median(&self) -> f64 {
        self.quantile(0.5)
    }

    fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "samples": self.0.len(),
            "median": round_to(self.median(), 4),
            "p25": round_to(self.quantile(0.25), 4),
            "p75": round_to(self.quantile(0.75), 4),
            "min": round_to(self.quantile(0.0), 4),
            "max": round_to(self.quantile(1.0), 4),
            "mean": round_to(self.mean(), 4),
        })
    }

    /// `median [p25, p75]`.
    fn show(&self) -> String {
        let digits = if self.median() < 0.1 {
            4
        } else if self.median() < 100.0 {
            2
        } else {
            1
        };
        format!("{:.digits$} [{:.digits$}, {:.digits$}]", self.median(), self.quantile(0.25), self.quantile(0.75))
    }
}

/// The first line of a file under `/proc` or `/sys`, if it can be read.
fn first_line(path: &str) -> Option<String> {
    std::fs::read_to_string(path).ok().and_then(|text| text.lines().next().map(|line| line.trim().to_string()))
}

/// The machine and build a measurement ran on, as far as the process can see.
/// Fields that cannot be read (other systems than Linux, restricted `/sys`) are
/// null.
fn environment() -> serde_json::Value {
    let field = |path: &str, key: &str| {
        std::fs::read_to_string(path).ok().and_then(|text| {
            text.lines()
                .find(|line| line.starts_with(key))
                .and_then(|line| line.split_once(':'))
                .map(|(_, value)| value.trim().to_string())
        })
    };
    let affinity = field("/proc/self/status", "Cpus_allowed_list");
    // Frequency settings of the first CPU the process may run on.
    let cpu = affinity
        .as_deref()
        .and_then(|list| list.split([',', '-']).next())
        .and_then(|first| first.parse::<usize>().ok())
        .unwrap_or(0);
    let cpufreq = |name: &str| first_line(&format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq/{name}"));
    serde_json::json!({
        "cpu_model": field("/proc/cpuinfo", "model name"),
        "kernel": first_line("/proc/sys/kernel/osrelease"),
        "cpus_allowed": affinity,
        "cpufreq_cpu": cpu,
        "cpufreq_driver": cpufreq("scaling_driver"),
        "cpufreq_governor": cpufreq("scaling_governor"),
        "energy_performance_preference": cpufreq("energy_performance_preference"),
        "cpuinfo_max_khz": cpufreq("cpuinfo_max_freq").and_then(|v| v.parse::<u64>().ok()),
        "threads": 1,
        "rustc": env!("LATSEC_BUILD_RUSTC"),
        "cargo_profile": env!("LATSEC_BUILD_PROFILE"),
        "opt_level": env!("LATSEC_BUILD_OPT_LEVEL"),
        "target": env!("LATSEC_BUILD_TARGET"),
        "target_features": env!("LATSEC_BUILD_TARGET_FEATURES"),
        "rustflags": env!("LATSEC_BUILD_RUSTFLAGS"),
        "debug_assertions": cfg!(debug_assertions),
    })
}

/// Honest sessions on random signer sets, with sizes and timings.
///
/// Method.  One thread.  Key generation is run `--keygen-reps` times and the last
/// key is used.  `--warmup` signed sessions are run first and not recorded.  Then
/// signer sets are drawn until `--sessions` sessions have completed.  Each phase
/// is timed with the monotonic clock around the library call, so the time of a
/// signing round includes what the call does: Round 1 parses the session message
/// and evaluates the predicate `P`; `PartialVerify` and `Combine` evaluate it
/// again.  The rounds exchange values in memory, so encoding and decoding are not
/// in the phase times; they are timed separately, per object.  A per-signer round
/// time is the session's total for the round divided by the number of signers.
/// Medians and quartiles are over sessions (over calls for `PartialVerify`, the
/// predicate and the codecs).
fn measure(args: &Args) -> CliResult<()> {
    args.reject_unknown(&["profile", "sessions", "size", "seed", "json", "warmup", "keygen-reps"])?;
    let scheme = load_scheme(args)?;
    let profile = scheme.profile().clone();
    let sessions: usize = args.number("sessions", 20)?;
    let warmup: usize = args.number("warmup", 3)?;
    let keygen_reps: usize = args.number::<usize>("keygen-reps", 5)?.max(1);
    let size: usize = args.number("size", profile.sec.t_sig)?;
    if size == 0 || size > profile.sec.n {
        return Err("--size out of range".into());
    }
    let mut rng = random_source(args, "measure");
    print_profile(&profile);

    let mut t_keygen = Samples::default();
    let mut deployment = None;
    for _ in 0..keygen_reps {
        // The new key is built into a temporary and the clock is read before the
        // previous key is replaced, so that dropping it (which erases its shares)
        // is not timed.
        let started = Instant::now();
        let fresh = Deployment::keygen(Arc::clone(&scheme), rng.as_mut());
        t_keygen.add(started.elapsed());
        deployment = Some(fresh);
    }
    let mut deployment = deployment.expect("at least one key generation");
    let vk_len = deployment.vk.to_bytes(&scheme).len();
    let ak_len = deployment.abort_keys[0].to_bytes(&scheme).len();
    let mut share_len = Spread::default();
    for signer in &deployment.signers {
        share_len.add(signer.share().to_bytes(&scheme).map_err(err)?.len() as f64);
    }

    // Warm-up: caches, branch predictors and the frequency governor settle.
    let (mut warmed, mut warm_attempts) = (0usize, 0usize);
    while warmed < warmup && warm_attempts < 50 * warmup.max(1) {
        warm_attempts += 1;
        let act = random_set(rng.as_mut(), profile.sec.n, size);
        let sid = session_id(rng.as_mut());
        if let Ok((SessionOutcome::Signed { .. }, _)) = deployment.sign(sid, &act, b"warm-up", rng.as_mut()) {
            warmed += 1;
        }
    }

    let (mut signed, mut aborted, mut not_authorized, mut rejected_bounds, mut too_small) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    let (mut t_recon, mut t_round1, mut t_round2, mut t_round3) =
        (Samples::default(), Samples::default(), Samples::default(), Samples::default());
    let (mut t_combine, mut t_verify, mut t_partial, mut t_predicate) =
        (Samples::default(), Samples::default(), Samples::default(), Samples::default());
    let (mut t_enc_response, mut t_dec_response, mut t_enc_sig, mut t_dec_sig, mut t_enc_opening, mut t_dec_opening) = (
        Samples::default(),
        Samples::default(),
        Samples::default(),
        Samples::default(),
        Samples::default(),
        Samples::default(),
    );
    let (mut weight, mut l1, mut l2, mut linf) =
        (Spread::default(), Spread::default(), Spread::default(), Spread::default());
    let (mut sig_len, mut resp_len, mut msg_len) = (Spread::default(), Spread::default(), Spread::default());
    let (mut open_len, mut cmt_len) = (0usize, 0usize);
    let (mut norm_ratio, mut partial_ratio) = (Spread::default(), Spread::default());
    let mut attempts = 0usize;
    while signed + aborted < sessions && attempts < 50 * sessions.max(1) {
        attempts += 1;
        let act = random_set(rng.as_mut(), profile.sec.n, size);
        let sid = session_id(rng.as_mut());
        let payload = format!("measurement session {attempts}");
        match deployment.sign(sid, &act, payload.as_bytes(), rng.as_mut()) {
            Err(latsec::latsec::InvocationError::Inadmissible(reason)) => match reason {
                latsec::scheme::Inadmissible::NotAuthorized => not_authorized += 1,
                latsec::scheme::Inadmissible::TooFewSigners { .. } => too_small += 1,
                _ => rejected_bounds += 1,
            },
            Err(other) => return Err(other.to_string()),
            Ok((SessionOutcome::Aborted { .. }, _)) => aborted += 1,
            Ok((SessionOutcome::Signed { signature, transcript }, timing)) => {
                signed += 1;
                t_recon.add(timing.recon);
                t_round1.add_share(timing.round1, size);
                t_round2.add_share(timing.round2, size);
                t_round3.add_share(timing.round3, size);
                t_combine.add(timing.combine);
                t_verify.add(timing.verify);
                for &party in act.parties() {
                    let started = Instant::now();
                    let ok = latsec::latsec::partial_verify(
                        &scheme,
                        &deployment.vk,
                        &deployment.abort_keys,
                        &transcript,
                        party,
                    )
                    .map_err(err)?;
                    t_partial.add(started.elapsed());
                    assert!(ok, "an honest response must pass partial verification");
                }
                let started = Instant::now();
                let parsed = latsec::latsec::admissible(&scheme, &sid, &act, &transcript.msg);
                t_predicate.add(started.elapsed());
                let lambda = parsed.map_err(|reason| reason.to_string())?.lambda;
                weight.add(lambda.iter().filter(|&&x| x != 0).count() as f64);
                l1.add(lambda.iter().map(|x| x.unsigned_abs()).sum::<u64>() as f64);
                l2.add(lambda.iter().map(|&x| (x * x) as f64).sum::<f64>().sqrt());
                linf.add(lambda.iter().map(|x| x.unsigned_abs()).max().unwrap_or(0) as f64);
                msg_len.add(transcript.msg.len() as f64);

                let started = Instant::now();
                let sig_bytes = signature.to_bytes(&scheme).map_err(err)?;
                t_enc_sig.add(started.elapsed());
                let started = Instant::now();
                let decoded = latsec::latsec::Signature::from_bytes(&scheme, &sig_bytes).map_err(err)?;
                t_dec_sig.add(started.elapsed());
                assert_eq!(decoded, signature);
                sig_len.add(sig_bytes.len() as f64);
                for response in &transcript.responses {
                    let started = Instant::now();
                    let bytes = response.to_bytes(&scheme).map_err(err)?;
                    t_enc_response.add(started.elapsed());
                    let started = Instant::now();
                    let decoded = latsec::latsec::PartialResponse::from_bytes(&scheme, &bytes).map_err(err)?;
                    t_dec_response.add(started.elapsed());
                    assert_eq!(&decoded, response);
                    resp_len.add(bytes.len() as f64);
                }
                for opening in &transcript.openings {
                    let started = Instant::now();
                    let bytes = opening.to_bytes(&scheme);
                    t_enc_opening.add(started.elapsed());
                    let started = Instant::now();
                    let decoded = latsec::latsec::Opening::from_bytes(&scheme, &bytes).map_err(err)?;
                    t_dec_opening.add(started.elapsed());
                    assert_eq!(&decoded, opening);
                    open_len = bytes.len();
                }
                cmt_len = transcript.commitments[0].0.len();
                let norm = |z: &[Vec<i128>]| z.iter().flatten().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
                norm_ratio.add(norm(&signature.z) / scheme.b2() as f64);
                for response in &transcript.responses {
                    partial_ratio.add(norm(&response.z) / scheme.b1() as f64);
                }
            }
        }
    }
    let environment = environment();
    println!(
        "\nkey generation (trusted dealer), {} parties: median {} ms over {keygen_reps} runs (mean {:.1} ms)",
        profile.sec.n,
        t_keygen.show(),
        t_keygen.mean()
    );
    println!(
        "signer sets of size {size}: {attempts} drawn, {not_authorized} not authorized, \
              {rejected_bounds} authorized but outside the bounds, {too_small} below t_sig"
    );
    println!(
        "sessions: {signed} signed, {aborted} aborted (honest partial rejections); {warmed} warm-up sessions before"
    );
    if signed > 0 {
        println!(
            "reconstruction vectors: mean weight {:.2} of {size} signers ({:.2} idle), mean l1 {:.2}, \
                  mean l2 {:.2}, max linf {}",
            weight.mean(),
            size as f64 - weight.mean(),
            l1.mean(),
            l2.mean(),
            linf.max
        );
        println!("norms: max |z_i| / B1 = {:.4}, max |z| / B2 = {:.5}", partial_ratio.max, norm_ratio.max);
        println!(
            "mean time: recon {:.2} ms; per signer: round 1 {:.2} ms, round 2 {:.3} ms, round 3 {:.2} ms; \
                  partial verify {:.2} ms per call; combine {:.1} ms; verify {:.2} ms",
            t_recon.mean(),
            t_round1.mean(),
            t_round2.mean(),
            t_round3.mean(),
            t_partial.mean(),
            t_combine.mean(),
            t_verify.mean()
        );
        println!("median time in ms [quartiles]:");
        for (name, samples) in [
            ("reconstruction vector (initiator)", &t_recon),
            ("round 1, per signer", &t_round1),
            ("round 2, per signer", &t_round2),
            ("round 3, per signer", &t_round3),
            ("partial verify, per call", &t_partial),
            ("combine", &t_combine),
            ("verify", &t_verify),
            ("predicate P, per call", &t_predicate),
            ("encode partial response", &t_enc_response),
            ("decode partial response", &t_dec_response),
            ("encode signature", &t_enc_sig),
            ("decode signature", &t_dec_sig),
            ("encode opening", &t_enc_opening),
            ("decode opening", &t_dec_opening),
        ] {
            println!("  {name:<34} {}", samples.show());
        }
        println!(
            "encoded sizes (KiB, mean / max): vk {:.3}, ak_i {:.3}, ak {:.1}, sk_i {:.3} / {:.3}, \
                  z_i {:.3} / {:.3}, sig {:.3} / {:.3}, opening {:.3}, msg {:.0} bytes; commitment {cmt_len} bytes",
            kib(vk_len),
            kib(ak_len),
            kib(ak_len * profile.sec.n),
            share_len.mean() / 1024.0,
            share_len.max / 1024.0,
            resp_len.mean() / 1024.0,
            resp_len.max / 1024.0,
            sig_len.mean() / 1024.0,
            sig_len.max / 1024.0,
            kib(open_len),
            msg_len.mean()
        );
        println!(
            "  per-signer communication (commitment + opening + response), mean: {:.3} KiB",
            (cmt_len as f64 + open_len as f64 + resp_len.mean()) / 1024.0
        );
    }
    let text = |key: &str| environment[key].as_str().unwrap_or("unknown").to_string();
    println!(
        "environment: {}; CPUs allowed {}; governor {} ({}, EPP {}); {}, {} profile, opt-level {}; one thread",
        text("cpu_model"),
        text("cpus_allowed"),
        text("cpufreq_governor"),
        text("cpufreq_driver"),
        text("energy_performance_preference"),
        text("rustc"),
        text("cargo_profile"),
        text("opt_level"),
    );
    if let Some(path) = args.get("json") {
        let record = serde_json::json!({
            "profile": profile.id,
            "profile_digest": profile.digest,
            "codec_version": profile.codec_version,
            "seed": args.get("seed"),
            "set_size": size,
            "sets_drawn": attempts,
            "not_authorized": not_authorized,
            "rejected_by_bounds": rejected_bounds,
            "signed": signed,
            "aborted": aborted,
            "warmup_sessions": warmed,
            "keygen_runs": keygen_reps,
            "lambda": {"weight": weight.json(), "l1": l1.json(), "l2": l2.json(), "linf": linf.json()},
            "partial_norm_over_b1": {"mean": round_to(partial_ratio.mean(), 6), "max": round_to(partial_ratio.max, 6)},
            "norm_over_b2": {"mean": round_to(norm_ratio.mean(), 6), "max": round_to(norm_ratio.max, 6)},
            // Means, as in the earlier records (results/20261108-*).
            "ms": {
                "keygen": round_to(t_keygen.mean(), 3),
                "recon": round_to(t_recon.mean(), 4),
                "round1_per_signer": round_to(t_round1.mean(), 4),
                "round2_per_signer": round_to(t_round2.mean(), 4),
                "round3_per_signer": round_to(t_round3.mean(), 4),
                "partial_verify_per_call": round_to(t_partial.mean(), 4),
                "combine": round_to(t_combine.mean(), 4),
                "verify": round_to(t_verify.mean(), 4),
            },
            "timing_ms": {
                "keygen": t_keygen.json(),
                "recon": t_recon.json(),
                "round1_per_signer": t_round1.json(),
                "round2_per_signer": t_round2.json(),
                "round3_per_signer": t_round3.json(),
                "partial_verify_per_call": t_partial.json(),
                "combine": t_combine.json(),
                "verify": t_verify.json(),
                "predicate_per_call": t_predicate.json(),
                "encode_partial_response": t_enc_response.json(),
                "decode_partial_response": t_dec_response.json(),
                "encode_signature": t_enc_sig.json(),
                "decode_signature": t_dec_sig.json(),
                "encode_opening": t_enc_opening.json(),
                "decode_opening": t_dec_opening.json(),
            },
            "method": "one thread; monotonic clock around each library call; warm-up sessions not recorded; \
                       round times are the session total divided by the number of signers; rounds exchange \
                       values in memory, codecs are timed separately; quantiles over sessions (over calls for \
                       partial verify, the predicate and the codecs)",
            "environment": environment,
            "bytes": {
                "vk": vk_len,
                "ak_i": ak_len,
                "sk_i": share_len.json(),
                "z_i": resp_len.json(),
                "sig": sig_len.json(),
                "msg": msg_len.json(),
                "opening": open_len,
                "commitment": cmt_len,
            },
        });
        let text = serde_json::to_string_pretty(&record).map_err(err)? + "\n";
        std::fs::write(path, text).map_err(err)?;
        println!("wrote {path}");
    }
    if aborted > 0 {
        return Err(format!("{aborted} honest sessions aborted"));
    }
    Ok(())
}

fn kat(args: &Args) -> CliResult<()> {
    args.reject_unknown(&["profiles", "out", "seed"])?;
    let dir = args.get("profiles").unwrap_or(concat!(env!("CARGO_MANIFEST_DIR"), "/../profiles"));
    let out = args.get("out").unwrap_or(concat!(env!("CARGO_MANIFEST_DIR"), "/../fixtures/kat"));
    let seed = args.get("seed").unwrap_or("latsec-kat-1");
    std::fs::create_dir_all(out).map_err(err)?;
    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .map_err(err)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|ext| ext == "json"))
        .collect();
    paths.sort();
    for path in paths {
        let scheme = Arc::new(Scheme::new(Profile::load(&path).map_err(err)?).map_err(err)?);
        if !scheme.supports_signing() {
            println!("skipped {} (SEC-only profile)", scheme.profile().id);
            continue;
        }
        let started = Instant::now();
        let vector = latsec::kat::TestVector::generate(Arc::clone(&scheme), seed).map_err(err)?;
        let target = format!("{out}/{}.json", vector.profile);
        vector.save(&target).map_err(err)?;
        println!(
            "wrote {target} ({:.1} s; signature {} bytes)",
            started.elapsed().as_secs_f64(),
            vector.bytes_signature
        );
    }
    Ok(())
}

fn matrix_check(args: &Args) -> CliResult<()> {
    use latsec::expander::{certify_expansion, certify_peeling, privacy_threshold, spectral_statistics, Graph, Ratio};
    args.reject_unknown(&["matrix", "spectral", "threads"])?;
    let matrix = AuthorMatrix::load(args.required("matrix")?).map_err(err)?;
    let threads: usize = args.number("threads", default_threads())?;
    let (gamma, eps) = (Ratio::parse(&matrix.gamma).map_err(err)?, Ratio::parse(&matrix.eps_exp).map_err(err)?);
    let g = matrix.matrix();
    println!(
        "matrix {} ({} x {}), c_exp = {}, gamma = {}, eps = {}",
        matrix.name, matrix.n, matrix.k, matrix.c_exp, matrix.gamma, matrix.eps_exp
    );
    println!("  rank over Z: {} (k = {})", latsec::intmat::row_hnf(&g).rank(), matrix.k);
    println!("  k_p from the paper's formula: {} (fixture: {})", privacy_threshold(matrix.n, gamma, eps), matrix.k_p);
    let graph = Graph::from_matrix(&matrix.g).map_err(err)?;
    let started = Instant::now();
    let expansion = certify_expansion(&graph, matrix.c_exp, gamma, eps);
    println!(
        "  expansion: {} subsets of size <= {} in {:.2} s: {}",
        expansion.subsets_checked,
        expansion.max_size,
        started.elapsed().as_secs_f64(),
        expansion.violation.as_ref().map_or("certified".to_string(), |v| format!("FAILS on {v:?}"))
    );
    let started = Instant::now();
    let peeling = certify_peeling(&graph, matrix.k_p);
    println!(
        "  peeling ([46, Algorithm 3]): {} subsets of size <= {} in {:.2} s: {}",
        peeling.subsets_checked,
        peeling.max_size,
        started.elapsed().as_secs_f64(),
        peeling.failure.as_ref().map_or("all succeed".to_string(), |v| format!("STUCK on {v:?}"))
    );
    for (support, witness) in matrix.planted_supports.iter().zip(&matrix.planted_witnesses) {
        let lambda: Vec<num_bigint::BigInt> = witness.iter().map(|&x| x.into()).collect();
        let image = g.select_rows(support).left_mul_vec(&lambda);
        let ok = image.iter().enumerate().all(|(j, x)| *x == num_bigint::BigInt::from(i64::from(j == 0)));
        println!(
            "  planted support of {} parties, witness weight {}: {}",
            support.len(),
            witness.iter().filter(|&&x| x != 0).count(),
            if ok { "reconstructs" } else { "DOES NOT reconstruct" }
        );
    }
    if args.get("spectral").is_some_and(|v| v != "no") {
        let started = Instant::now();
        let s = spectral_statistics(&matrix.g, matrix.k_p, threads).map_err(err)?;
        println!(
            "  float64 constants over {} subsets in {:.1} s (not certified):",
            s.subsets,
            started.elapsed().as_secs_f64()
        );
        println!("    alpha_kp = {:.15}  beta_kp = {:.15}  delta_kp = {:.15}", s.alpha_kp, s.beta_kp, s.delta_kp);
        println!(
            "    gamma_kp = {:.15}  |G_hat|_2 = {:.15}  |G|_2 = {:.15}  beta_Q = {:.13}",
            s.gamma_kp, s.norm_g_hat, s.norm_g, s.beta_q
        );
    }
    if expansion.violation.is_some() || peeling.failure.is_some() {
        return Err("matrix fails a structural check".into());
    }
    Ok(())
}

fn run() -> CliResult<()> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let Some(command) = raw.first() else { return Err(USAGE.to_string()) };
    match command.as_str() {
        "profiles" => {
            let sub = raw.get(1).map(String::as_str);
            let args = Args::parse(&raw[2.min(raw.len())..])?;
            match sub {
                Some("build") => profiles_build(&args),
                Some("check") => profiles_check(&args),
                _ => Err(USAGE.to_string()),
            }
        }
        "recon-stats" => recon_stats(&Args::parse(&raw[1..])?),
        "recon-compare" => recon_compare(&Args::parse(&raw[1..])?),
        "demo" => demo(&Args::parse(&raw[1..])?),
        "measure" => measure(&Args::parse(&raw[1..])?),
        "kat" => kat(&Args::parse(&raw[1..])?),
        "matrix-check" => matrix_check(&Args::parse(&raw[1..])?),
        "help" | "--help" | "-h" => {
            println!("{USAGE}");
            Ok(())
        }
        _ => Err(USAGE.to_string()),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}
