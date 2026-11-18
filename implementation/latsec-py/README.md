# latsec-py: the LaTSEC test-vector generator

A Python implementation of SEC and LaTSEC written to be read next to the paper;
`[46]` is the SEC paper, by its number in the paper's bibliography. It
favours readability over speed. It was written independently of the Rust
implementation in [`../latsec-rs`](../latsec-rs) and agrees with it byte for byte:
it reproduces every profile in [`../profiles`](../profiles) and every test vector
in [`../fixtures/kat`](../fixtures/kat).

| Module | Manuscript | Content |
|---|---|---|
| `xof.py` | §3.1 | SHAKE256 with domain separation and framed inputs; seeded randomness; uniform integers |
| `sampler.py` | §2.1 | The discrete Gaussian sampler of Canonne, Kamath and Steinke, in exact integer arithmetic with bounded statistical error |
| `ring.py` | §2.1, §3.1 | `R_q` and its NTT, the public matrix `A = [I_p \| A']`, the challenge set `C` and sample-in-ball |
| `sec.py` | §2.4 | `SEC.Deal`, `SEC.Rec`, the coefficient checks, and Algorithm 1 of [46] in exact integer arithmetic (HNF, integral LLL, Babai, greedy) |
| `codec.py` | §7.3 | Canonical encodings: residues, Golomb–Rice coded Gaussians, the ranked challenge (`b_c` bits) |
| `profiles.py` | Table 2, §4.1 | Profiles: derived bounds, Golomb–Rice parameters, digest |
| `latsec.py` | Figure 1 | `Keygen`, the predicate `P`, `ShareSign_1..3`, `PartialVerify`, `Combine`, `Verify` |
| `params.py` | §7.1–7.2 | The width optimizer of the parameter-selection notebook at 60 digits, with regression against Tables 4 and 5 |
| `build_profiles.py` | §7 | Re-derives every stored profile from the published inputs |
| `vectors.py` | — | The test-vector generator and checker |
| `matrix_check.py` | §2.4 | Rank, `k_p`, expansion, peeling ([46, Algorithm 3]), planted witnesses, float64 matrix constants |
| `recon_stats.py` | §7.1 | Reconstruction-vector statistics per signer-set size |
| `test_latsec.py` | — | Unit tests, with expected values from hand computation, the paper, or the SageMath pipeline of [46] |

Party indices are zero-based: party `i` of the manuscript is index `i - 1`.

## Use

Python 3.10 or later with `pycryptodome` (from PyPI, or the Debian package
`python3-pycryptodome`; both namespaces, `Crypto` and `Cryptodome`, are
accepted); `numpy` is needed only for `matrix_check.py --spectral`. Run the scripts from this directory:

```sh
python3 -m pip install -r requirements.txt
python3 -m unittest test_latsec                     # about 80 s
python3 build_profiles.py                           # every profile re-derived and compared
python3 vectors.py --verify ../fixtures/kat/*.json  # every test vector, about 6 min
python3 vectors.py ../profiles/functional-n10-d16.json   # print one vector
python3 params.py                                   # Tables 4 and 5 of the paper
python3 matrix_check.py ../fixtures/author/G-40-19-3.json --spectral
python3 recon_stats.py ../fixtures/author/G-40-19-3.json --sizes 26..27 \
    --samples 2000 --seed demo --beta 5 --omega 25 --jobs 8
```

`make check` in the repository root runs these together with the Rust checks.

## How the two implementations are tested against each other

- **Test vectors.** `vectors.py` runs key generation and one signing session per
  profile from a seeded SHAKE256 stream and records every encoding (or its digest).
  The randomness is consumed in a fixed order, documented in `vectors.py`, so the
  output is a function of the seed. `latsec kat` in `latsec-rs` writes the same files.
- **Profiles.** `build_profiles.py` rebuilds each profile from the published inputs
  and compares every protocol field and the 64-byte digest. The width optimizer is
  evaluated as the Rust code does it, in double precision, because rounding to the
  stored dyadic widths can depend on the last bits.
- **Reconstruction.** `recon_stats.py` draws signer sets exactly as
  `latsec recon-stats` does, so both report the same counts. Both reproduce the
  SageMath pipeline of [46] on the reference sets of `../fixtures/sage`.

This is a reference implementation: it is variable-time and keeps secrets in
ordinary Python integers.
