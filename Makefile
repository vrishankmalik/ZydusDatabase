.PHONY: test test-live test-all coverage refresh-fixtures reconcile enrich-patents enrich-labeling build-workbook

# Offline unit suite (default — fast, no network, must pass in CI)
test:
	python3 -m pytest tests/ -v --tb=short

# Integration suite — hits live government sites (run nightly or on demand)
test-live:
	python3 -m pytest tests/ -v --tb=short -m integration

# Run everything: offline + integration
test-all:
	python3 -m pytest tests/ -v --tb=short -m "unit or integration"

# Coverage report (offline suite only)
coverage:
	python3 -m pytest tests/ --cov=app --cov-report=term-missing --cov-report=html --tb=short
	@echo "HTML report: htmlcov/index.html"

# Re-record all HTTP fixtures from live sources.
# Requires network access; writes fixture JSON/HTML to tests/fixtures/.
refresh-fixtures:
	python3 tests/scripts/refresh_fixtures.py

# Completeness reconciliation against the Health Canada DPD bulk extract.
# Downloads allfiles.zip (~1.4 MB), caches locally, then asserts that the
# pipeline DIN set covers the extract DIN set within 0.5% for sample ingredients.
# Run nightly or after any change that touches source pagination or result caps.
# Requires live network.  DPD_MAX_RESULTS is overridden to 9999 by the test itself.
reconcile:
	python3 -m pytest tests/reconciliation/ -v --tb=short -m integration -s

# Enrichment pipeline — chain these three for a fully enriched workbook.
# Usage: make enrich-patents DINS="02498014 02498022"
enrich-patents:
	python3 -m app.enrichment.patents --dins $(DINS)

# Usage: make enrich-labeling DRUG_CODE=12345 DIN=02498014 STRENGTH="50 mg"
enrich-labeling:
	python3 -m app.enrichment.labeling --drug-code $(DRUG_CODE) --din $(DIN) --strength "$(STRENGTH)"

# Usage: make build-workbook Q="alpelisib" FIELD=ingredient
build-workbook:
	python3 -m app.enrichment.workbook --q "$(Q)" --field $(FIELD)
