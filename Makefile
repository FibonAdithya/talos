PYTHON ?= python3
.PHONY: check
check:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m pytest -q -m "not live"
	$(PYTHON) -m agentify check .

.PHONY: mirror-images
mirror-images:
	PYTHON=$(PYTHON) ./scripts/mirror_images.sh
