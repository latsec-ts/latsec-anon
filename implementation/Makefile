#	Makefile -- checks for both implementations, and cleanup.
#
#	make check        Rust tests, lints and formatting; Python unit tests; the
#	                  Python re-derivation of every profile; the Python generator
#	                  against the small and one author test vector  (~3 min)
#	make check-full   adds every test vector, the ignored Rust test (n = 60 matrix
#	                  constants) and the Python matrix checks  (~15 min)
#	make vectors      regenerate profiles/ and fixtures/kat/ with the Rust CLI
#	make clean        remove build output and caches of both implementations

PYTHON	?= python3
CARGO	?= cargo
LATSEC	= latsec-rs/target/release/latsec

.PHONY: check check-full check-rs check-py vectors clean

check: check-rs check-py
	cd latsec-py && $(PYTHON) build_profiles.py
	cd latsec-py && $(PYTHON) vectors.py --verify ../fixtures/kat/functional-*.json \
		../fixtures/kat/author-n40-k128-nov2026.json

check-rs:
	cd latsec-rs && $(CARGO) test --release
	cd latsec-rs && $(CARGO) clippy --release --all-targets -- -D warnings
	cd latsec-rs && $(CARGO) fmt --check

check-py:
	cd latsec-py && $(PYTHON) params.py > /dev/null
	cd latsec-py && $(PYTHON) -m unittest test_latsec

check-full: check
	cd latsec-rs && $(CARGO) test --release -- --ignored
	cd latsec-py && $(PYTHON) vectors.py --verify ../fixtures/kat/*.json
	cd latsec-py && for m in 10-5-3 20-9-3 40-19-3; do \
		$(PYTHON) matrix_check.py ../fixtures/author/G-$$m.json || exit 1; done

vectors:
	cd latsec-rs && $(CARGO) build --release
	$(LATSEC) profiles build
	$(LATSEC) kat

clean:
	cd latsec-rs && $(CARGO) clean
	$(MAKE) -C latsec-py clean
	$(RM) -r tools/__pycache__
