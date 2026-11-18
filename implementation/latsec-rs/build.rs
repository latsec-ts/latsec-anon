//! Records the compiler and build settings, so that `latsec measure` can write
//! them next to its timings.

use std::env;
use std::process::Command;

fn main() {
    let rustc = env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let version = Command::new(rustc)
        .arg("--version")
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|text| text.trim().to_string())
        .unwrap_or_else(|| "unknown".into());
    let var = |name: &str| env::var(name).unwrap_or_default();
    println!("cargo:rustc-env=LATSEC_BUILD_RUSTC={version}");
    println!("cargo:rustc-env=LATSEC_BUILD_PROFILE={}", var("PROFILE"));
    println!("cargo:rustc-env=LATSEC_BUILD_OPT_LEVEL={}", var("OPT_LEVEL"));
    println!("cargo:rustc-env=LATSEC_BUILD_TARGET={}", var("TARGET"));
    println!("cargo:rustc-env=LATSEC_BUILD_TARGET_FEATURES={}", var("CARGO_CFG_TARGET_FEATURE"));
    // Separated by the unit separator; empty when no flags are set.
    println!("cargo:rustc-env=LATSEC_BUILD_RUSTFLAGS={}", var("CARGO_ENCODED_RUSTFLAGS").replace('\u{1f}', " "));
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=RUSTFLAGS");
}
