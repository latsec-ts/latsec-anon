#	latsec-py/Makefile

PYTHON	?= python3

.PHONY: test vectors clean

test:
	$(PYTHON) params.py > /dev/null
	$(PYTHON) -m unittest test_latsec
	$(PYTHON) build_profiles.py

vectors:
	$(PYTHON) vectors.py --verify ../fixtures/kat/*.json

clean:
	$(RM) -r __pycache__
	$(RM) *.pyc *.prof *.log
