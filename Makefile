PYTHON ?= python3
.PHONY: check
check:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m pytest -q -m "not live"
