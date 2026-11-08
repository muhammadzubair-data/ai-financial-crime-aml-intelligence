# Technical documentation

## 1. Architecture

The platform is a batch pipeline (`python -m amlintel.build`) that writes Parquet, CSV, JSON, a LightGBM model and
a SQLite case database into `data/<profile>/`, and a Streamlit application that reads those artifacts.

| Stage | Module | Output |
|---|---|---|
| Generate data | `data/generator.py` | `accounts.parquet`, `transactions.parquet` |
| Account-month features | `data/pipeline.py` | in-memory feature table |
| Monthly graph features | `graph/network.py` | merged into feature table |
| Pattern discovery | `detection/typologies.py` | `indicators.parquet`, `indicator_eval.csv` |
| Scenario rules | `detection/rules.py` | rule flags, `rules.csv` |
| Models and evaluation | `detection/models.py` | `scored.parquet`, `metrics.json`, `lgbm_model.txt` |
| Explanations | `detection/explain.py` | `shap_alerts.parquet`, `shap_global.csv` |
| Queue and cases | `cases/queue.py`, `cases/store.py` | `cases.db` |
| Governance views | `simulation.py` | `capacity_curve.csv` |
| Provenance | `build.py` | `manifest.json` |

Profiles (`config.py`): `demo` (6,000 accounts, ships with the repo), `test` (900 accounts, 12 months, used by the
test suite) and `full` (100,000 accounts).

## 2. Data model

**transactions.parquet**

| Column | Meaning |
|---|---|
| `txn_id` | Unique ID, in time order |
| `timestamp`, `month`, `month_idx` | Time; `month_idx` is 0 for the first analysis month, negative for look-back history |
| `from_account`, `to_account` | Account IDs; `CASH` for cash deposits (sender) and withdrawals (receiver) |
| `amount` | Currency-neutral amount, 2 decimals |
| `channel` | `transfer`, `card`, `wire`, `cash_deposit`, `cash_withdrawal` |
| `from_country`, `to_country`, `is_cross_border` | Jurisdictions of each side |
| `is_laundering`, `typology`, `instance_id`, `network_id` | Synthetic ground truth (evaluation only) |

**accounts.parquet**: `account_id`, `customer_type`, `business_kind`, `country`, `opened_on`, `kyc_risk`, and
ground-truth `network_id` / `network_role`.

**scored.parquet**: one row per account-month with all features, rule flags, every model score, within-month
percentile ranks, the 0–100 `risk_score`, the split and the ground-truth label.

## 3. Features

Computed per account-month by `build_account_month_features`, one calendar month at a time with a 4-day look-back
so that time-window features at the start of a month see the end of the previous month. The result is identical to
a single pass, and memory stays flat as data grows.

* **DuckDB SQL** over a long "legs" table (one row per account per transaction side): counts, values, distinct
  counterparties, cash deposits, deposits between 90% and 100% of the 10,000 threshold, withdrawals, cross-border
  and high-risk-jurisdiction value, round amounts, transactions of 10,000+, largest transaction, active days.
* **Polars rolling windows:** the largest number of distinct counterparties (payments of 1,000+) within any 72 hours,
  outgoing and incoming.
* **Pass-through:** for every outgoing payment, the money received in the previous 24 hours, capped at the payment
  amount, summed per month. Implemented with per-account running totals and binary search, which is exact even when
  several transactions share a timestamp. A test checks it against brute force.
* **Static KYC data:** account age, KYC level, customer and business type, high-risk home jurisdiction.
* **Graph features** (section 4) and **pattern-indicator counts** (section 5).

No feature reads the label columns; a test asserts this.

## 4. Graph analytics

For every month, `monthly_graph_features` builds a directed graph of accounts weighted by amount (cash excluded):

| Feature | Graph | Why |
|---|---|---|
| Weighted PageRank × node count | all flows | how central the account is in where money ends up |
| Strongly connected component size | flows ≥ 1,000 | money that can circulate and return |
| Short-cycle count (length ≤ 6) | flows ≥ 1,000 | direct evidence of circular movement |
| Max in-degree of the accounts it pays | flows ≥ 2,500 | "pays into a collector" (fan-in member) |
| Max out-degree of the accounts paying it | flows ≥ 2,500 | "paid by a distributor" (fan-out member) |
| Share of neighbours with high KYC risk | all flows | exposure to riskier counterparties |
| Number of neighbours | all flows | breadth of relationships |

Cycle enumeration runs only inside strongly connected components and is capped per component. Subgraphs are built
with sorted nodes and edges, so results never depend on Python's hash seed.

**Alert clustering** (`alert_networks`): queued accounts in the same month are linked when money moved directly
between them, or when both dealt with the same low-degree counterparty (15 or fewer counterparties). Shared large
merchants or employers do not link accounts. Linked accounts become one "network cluster" that an investigator
can review together. On the demo data these clusters line up closely with the injected rings.

**Ego networks** (`ego_network`) power the Network Explorer: the 1- or 2-hop neighbourhood of an account in a month,
above an amount threshold, trimmed to the strongest flows.

## 5. Pattern discovery

| Detector | Logic |
|---|---|
| Structuring | 3+ cash deposits at 90–100% of the threshold, gaps of 10 days or less; notes any onward payment of half the total within 7 days |
| Fan-out / fan-in | 6+ distinct non-routine counterparties (payments of 2,000+) within 72 hours; a relationship is routine once the pair has transacted in 2+ earlier months |
| Rapid pass-through | inflow of 15,000+ with 85%+ sent on within 24 hours |
| Circular flow | time-respecting cycles of 3–6 accounts on payments of 5,000+: each hop within 7 days of the last, keeping 80–105% of the amount; allows a 2-week spill into the next month |
| Scatter-gather | a source pays 3+ intermediaries, each forwarding 85–100% to the same sink within 72 hours (DuckDB range join) |

Each indicator stores the accounts and transactions involved plus an evidence sentence. `evaluate_indicators`
measures precision (share of indicators where most flagged transactions are laundering) and scheme recall.

## 6. Detection models

* **Rules** (`rules.py`): eight scenarios with weights, in one table for compliance review. Rule score = weighted
  sum of rules fired; ties broken by anomaly rank.
* **Isolation Forest:** 300 trees, signed log transform of features, fitted on training months only.
* **LightGBM:** early stopping on validation months, then refit on train + validation with the chosen number of
  trees. Training labels cover 70% of true positives (`label_coverage`).
* **Blend:** within-month percentile ranks of LightGBM, network heuristic, rule score and anomaly score. A grid
  search on validation months starts from pure LightGBM and accepts a blend only if it improves validation
  precision at capacity by at least one point (`blend_min_gain`).
* **risk_score:** within-month percentile (0–100) of the blended score. Risk bands in the queue: Critical ≥ 99.7,
  High ≥ 99.3, otherwise Elevated.

## 7. Evaluation

* Split by month: train 0–13, validation 14–15, test 16–23 (`ModelConfig`).
* **Precision and recall at capacity:** each test month, the top K account-months by score (K = 50 for the demo).
* PR-AUC and ROC-AUC on the pooled test set; PR curves stored for the dashboard.
* **Ring-month coverage:** (ring, month) pairs where at least one genuinely involved member is in the queue.
* **Recall by typology** for the deployed queue.
* **Raw rule alert stream:** volume, precision and recall if every rule alert were worked.

## 8. Case management

`CaseStore` (SQLite) has three tables: `cases`, `notes`, and an append-only `audit_log`. Allowed status changes:

```
New ──> In review ──> Awaiting information ──> In review
 │          │                │
 │          ├──> Escalated to MLRO ──> SAR filed (locked)
 │          │                └──────> Closed - no further action
 └──────────┴──> Closed - no further action ──> In review (re-open)
```

Closing a case or recording a SAR requires a written rationale. Every create, assign, status change and note is
logged with actor and UTC time. The demo queue's earlier months carry a simulated investigation history marked
`[demo-seed]`; the latest month is a live mix of new and in-progress cases.

## 9. Reproducibility

* Fixed seeds everywhere (generator, Isolation Forest, LightGBM with `deterministic=True`, label subsampling).
* Row order fixed after DuckDB/Polars group-bys; money aggregates rounded to cents so parallel summation cannot
  change the last bits.
* Graph steps independent of Python's string hash seed.
* `scripts/check_reproducible.py` compares two builds. CI runs two builds with different hash seeds and requires
  an exact match, then checks the committed demo artifacts (identical transactions, headline metrics within 0.02,
  since LightGBM's floating-point results can differ slightly between CPU types).

## 10. Scaling

The `full` profile was built end to end on a 2-vCPU, 8 GB machine:

| Item | Value |
|---|---|
| Accounts | 100,000 |
| Transactions (with 2-month history) | 5,691,643 (5,355,008 in the analysis window) |
| Account-months scored | 1,978,384 |
| Laundering schemes / rings | 2,719 / 180 |
| Pattern indicators | 10,026 |
| Cases seeded (capacity 833 per month) | 6,664 |
| Total build time | 9 min 42 s (generate 41 s, features + graphs 2 min 11 s, pattern detection 3 min 58 s, models 1 min 18 s, cases 1 min 22 s) |
| Peak memory | 5.1 GB |

Out-of-time results at that scale (capacity 833 investigations a month, 6,772 suspicious account-months in test):
the deployed score reaches 73.2% precision and 73.1% recall at capacity (PR-AUC 0.725), against 20.6% / 20.5% for
rules ranked by score. Organised-ring coverage is 666 of 686 ring-months (rules: 540). Early stopping kept only
14 trees at this scale, so there is headroom from tuning the learning rate and tree count.

What keeps it inside 8 GB:

* features are computed one month at a time (section 3);
* graph features are computed one monthly graph at a time;
* scatter-gather uses a DuckDB range join instead of a pandas self-merge, and pass-through uses binary search
  instead of per-transaction scans;
* the transaction table is written to disk and released before modelling; features are stored as float32;
* LightGBM is fitted on all positives and 20% of training negatives (`neg_sample_rate`), and Isolation Forest on at
  most 200,000 training rows; scoring runs in chunks of 250,000 rows.

Beyond this size, the natural next steps are to run the monthly stages in parallel (they are independent), store
the account-month table in DuckDB or Parquet partitions instead of memory, and replace the Python loops in the fan
and cycle detectors with SQL or a graph database.

## 11. Tests

36 pytest tests, about 25 seconds:

* data: determinism, well-formed transactions, every typology injected inside the window, features match hand
  computation, pass-through matches brute force, no label leakage, labels mark every participant
* detectors: each typology positive and negative cases, routine payroll not flagged, time ordering for cycles
* graph: cycle counts, ego networks, alert clustering ignores shared hubs
* models: precision at K, strict time ordering of splits, model beats rules on PR-AUC, blend weights sum to 1
* cases: full audited workflow, invalid transitions, missing rationale
* simulator: capacity trade-off, no-rules policy
* app: every page renders without errors; a scripted status change on the case page lands in the audit log
