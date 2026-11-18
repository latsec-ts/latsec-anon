# LaTSEC

Artifact for the paper "LaTSEC: Lattice-Based Threshold Signatures with Native Non-Interactive Identifiable Abort and Compact Per-Party Storage".

## Contents

- `parameter-search/`: the notebook that selects the parameter sets of Section 7 and estimates their security, the notebooks of the two SEC instances, and the estimator calibration scripts. See [`parameter-search/readme.md`](parameter-search/readme.md).
- `implementation/`: the Rust and Python implementations of Section 8, their parameter profiles and test vectors, the recorded measurements, and the scripts that produce the tables of Section 8, of Appendix F, and of the extended benchmarks (Appendix P of the full version). See [`implementation/README.md`](implementation/README.md).
- `LaTSEC-full-version.pdf`: the full version of the paper, with the deferred proofs, the complete specification, and the detailed size and storage analyses.

## Software prerequisites

| Part | Software |
|---|---|
| `implementation/`: build, checks and measurements | Rust with Cargo 1.85 or later (measurements: 1.100.0); Python 3.10 or later with PyCryptodome 3.19 or later; GNU Make, a POSIX shell and `taskset`. Linux on x86-64 was used. The first Rust build downloads the pinned crates from crates.io |
| `implementation/`: optional tools | NumPy 1.24 or later (one spectral check); SageMath 10.7 or later and the lattice estimator below (regenerating reference vectors and security estimates) |
| `parameter-search/` | SageMath 10.7 with a Jupyter kernel; the lattice estimator at commit `27a581b` of https://github.com/malb/lattice-estimator |

No special hardware is needed. The implementation takes 2 MB of disk space before building and about 1 GB with the build output.

## Reproducing the results

1. **Functional check (about 3 minutes).** In `implementation/`, run `make check`. It builds the Rust implementation, runs the tests of both implementations, re-derives every parameter profile in Python, and checks that the two implementations agree on the test vectors. `make check-full` (about 15 minutes) adds every test vector and the slow checks.
2. **Parameter sets (Section 7, Tables 3 to 5).** `python3 implementation/latsec-py/params.py` recomputes the selected widths, bounds, primes and sizes and the smoothing margins from the published inputs, with the standard library only. The selection itself and the lattice estimates are in `parameter-search/LaTSEC-Parameter-Selection-v2.3-fixedtailbound+privacy.ipynb`, whose recorded outputs can be read without running it; rerunning it needs SageMath and the lattice estimator.
3. **Measurements (Table 6 in Section 8, Table 7 in Appendix F, and the extended benchmarks in Appendix P of the full version; about 15 minutes).** In `implementation/`, run `tools/run_benchmarks.sh myrun` and then `python3 tools/bench_tables.py --prefix myrun`. The script pins the runs to one core that it chooses and names; a second argument selects another. The runs are seeded: the signer sets, the encoded sizes and the norms equal those of the recorded runs in `implementation/results/`, while the running times depend on the machine.
4. **Everything else.** [`implementation/README.md`](implementation/README.md) has a table that maps each result of the paper to the command that reproduces it, its running time, and the recorded output to compare with.

Both implementations are research prototypes: they are not constant-time, key generation uses a trusted dealer, and all parties run in one process. Running them adds no security evidence.
