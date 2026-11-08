# Project 015 — Technical Audit

Audit date: 2026-11-08. Scope: supplied `aml-financial-crime-intelligence.zip`.

## Verdict
**Conditionally ready for GitHub publication as a synthetic-data research/demo portfolio project; not certified as production-ready or fully tested.** The source archive contains a modular Python package, seven Streamlit page modules, synthetic demo artifacts, model outputs, tests, CI configuration, documentation, and visual assets.

## Verification performed
- Original ZIP integrity: `unzip -t` passed.
- Python syntax: `python -m compileall -q src app tests` passed.
- `python -m pytest -q`: stopped at collection with two import errors because `duckdb` is absent from the audit runtime. No full test-pass claim can be made. Other runtime dependencies (including Streamlit, Polars and PyArrow) are also unavailable in this runtime.
- Inspected `requirements.txt`, `requirements-dev.txt`, `pyproject.toml`, `.github/workflows/ci.yml`, README, provenance documentation, Streamlit entry point, smoke tests, and demo manifest.
- The source archive includes an internal `.git/` directory (~36 MB), inappropriate for a portable GitHub upload. Removed it from the distribution ZIP only; source project code was preserved.

## Data and metrics provenance
- Synthetic demo manifest: 6,000 accounts, 528,250 total transactions including two history months, 498,019 transactions in the analysis window, 126,540 account-months, 16 organized rings.
- README reports out-of-time evaluation on 46,153 account-months with 642 positives, 50 investigations per month, 85.5% precision at capacity and 55.5% recall for deployed priority scoring. These are **author-reported metrics from supplied artifacts**, not independently reproduced in this audit.
- The full-profile claim of 100,000 accounts and 5.7 million transactions is not the bundled demo dataset; its runtime/performance claim has not been verified.
- All data and ground-truth patterns are generated. These results must not be described as validated real-world AML detection, production savings, or regulatory compliance.

## Important issues and recommendations
1. **CI not verified**: Run GitHub Actions for Python 3.11 and 3.12 and inspect lint, unit tests, dashboard smoke tests and reproducibility checks. Fix any failures before claiming green CI.
2. **Deployment and package pins**: `requirements.txt` uses exact versions, while `pyproject.toml` uses broad minimums. Verify all pinned versions resolve on the chosen Streamlit Cloud Python runtime. Reconcile runtime and development installation paths if necessary.
3. **Demo identity is not authentication**: `app/streamlit_app.py` allows editing the `Signed in as` text field; this is only a demonstration actor label, not verified identity. Do not expose the case-management workflow as an authenticated production system.
4. **Demo case persistence**: The dashboard uses a seeded SQLite case database and a mutable `cases_live.db`. Treat it as a single-user demonstration; do not imply concurrent multi-user consistency or durable hosted persistence.
5. **Model evaluation**: Verify training/validation/test time boundaries and all features for label leakage. Run independent regeneration and compare metrics before publishing performance as reproduced.
6. **Synthetic typology generalization**: Pattern injection and matching detectors can make evaluation optimistic; clearly describe this risk. Real investigations require human review and legal/compliance controls.
7. **Scalability**: Independently benchmark the full-profile build before repeating the README's time-to-build claim.

## Publication checklist
- [ ] GitHub Actions green on supported Python versions
- [ ] All seven Streamlit pages load with committed demo artifacts
- [ ] Case audit trail mutation test passes
- [ ] Regenerated demo matches published manifest and metric tolerances
- [ ] Public deployment contains no secrets, customer data, or writable sensitive database
- [ ] README clearly distinguishes synthetic demonstration metrics from production outcomes

## Audit package changes
- Added this audit report.
- Removed `.git/`, Python bytecode/cache, and other local transient metadata from the distributed archive.
- Did **not** modify models, dataset, source algorithms, reported metrics, or dashboard behavior without validated reproduction.
