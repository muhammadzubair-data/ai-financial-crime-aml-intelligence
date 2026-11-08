# Deployment guide

## Streamlit Community Cloud (public demo)

The repository ships with the built demo artifacts in `data/demo/` (about 21 MB), so the cloud app starts without
training anything.

1. Push the repository to GitHub (public, or private with Streamlit Cloud access granted).
2. Go to [share.streamlit.io](https://share.streamlit.io), choose **Create app → Deploy a public app from GitHub**.
3. Select the repository and branch, and set **Main file path** to `app/streamlit_app.py`.
4. Under **Advanced settings**, choose Python 3.12. No secrets are needed.
5. Deploy. Dependencies install from `requirements.txt` (the last line, `-e .`, installs the `amlintel` package).

Notes:

* The theme comes from `.streamlit/config.toml`.
* Case edits made in the app (assignments, notes, status changes) are written to `data/demo/cases_live.db`, a copy
  of the seeded database. On Streamlit Cloud this file resets whenever the app restarts, which is the right
  behaviour for a public demo.
* If `data/demo/` is missing, the app builds it on first start (about a minute).
* Memory use is well inside the free tier: the demo loads about 500k transactions and 127k scored rows.

## Local

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
streamlit run app/streamlit_app.py
```

Open http://localhost:8501. The name in **Signed in as** (sidebar) is recorded on every case action.

## Rebuilding data and models

```bash
python scripts/build_demo.py --profile demo        # ~1 minute, overwrites data/demo/
python scripts/build_demo.py --profile full        # 100k accounts, see technical documentation for timings
```

To reset the case queue to its seeded state, delete `data/demo/cases_live.db`.

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request, on Python 3.11 and 3.12:

1. `ruff check` on source, app and tests
2. `pytest` (36 tests, including a render test of every dashboard page and a scripted case status change)
3. Two full demo rebuilds with different hash seeds, which must match byte for byte, and a comparison with the
   committed artifacts (identical transactions, headline metrics within 0.02 to allow for CPU floating-point
   differences between machines), via `scripts/check_reproducible.py`

If you intentionally change the generator or models, rebuild `data/demo/` and commit the new artifacts in the same
pull request so the reproducibility step passes.

## Moving towards production

This is a demonstration build. For a bank deployment, the main changes would be:

* **Data:** replace the generator with a loader from the core banking or transaction data warehouse into the same
  Parquet schema; run the build on a schedule (Airflow, Dagster or a cron job).
* **Case store:** move from SQLite to the bank's case-management system or a managed Postgres, with single sign-on
  so the audit log records real identities.
* **Access control:** restrict who can see customer data, change case status, file reports and change policy.
* **Model governance:** register each trained model with its metrics and data snapshot; promote only after
  independent validation; keep the simulator's policy change records with the change requests.
