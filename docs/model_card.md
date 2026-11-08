# Model card: AML priority score

## Purpose

Rank account-months for human review so that a fixed monthly investigation capacity is spent on the activity most
likely to be suspicious. The score decides **what to look at first**. It never decides that an account is
laundering money, never files a report, and never triggers automatic action against a customer. A trained
investigator reviews every case and records the rationale for the decision.

## What is scored

One row per active account per calendar month (the account-month). The deployed priority score is a blend of
within-month percentile ranks:

| Component | Weight (chosen on validation) |
|---|---|
| LightGBM risk model | 95% |
| Isolation Forest anomaly score | 5% |
| Network-exposure heuristic | 0% |
| Scenario rule score | 0% |

The blend search starts from the pure LightGBM ranking and only accepts a blend that beats it on the validation
months by at least one percentage point of precision at capacity. Weights can be changed on the Impact Simulator
page; any change should go through change control (see below).

## Model details

* **Algorithm:** LightGBM binary classifier, 225 trees (early stopping on validation months), learning rate 0.04,
  31 leaves, bagging and feature subsampling, deterministic training.
* **Class imbalance:** `scale_pos_weight` set to the square root of the negative/positive ratio.
* **Features (47):** flow volumes and counterparties; cash deposits and near-threshold deposits; cross-border and
  high-risk-jurisdiction exposure; 72-hour fan-in/fan-out bursts; 24-hour pass-through; account age, KYC level and
  customer type; monthly graph features (weighted PageRank, circular-flow component size, short cycles, links to
  collector and distributor accounts, high-KYC neighbour share); counts of pattern indicators by typology.
* **Explanations:** SHAP TreeExplainer. Each queued case stores its top six drivers with the account's actual
  values; any account can be explained live on the AI Detection page.

## Training and evaluation data

* Fully synthetic data from `amlintel.data.generator` (see [data_provenance.md](data_provenance.md)).
* Time-based split: months 1–14 train, 15–16 validation, 17–24 test (March to October 2026). No shuffling.
* Training labels are deliberately incomplete: only 70% of suspicious account-months carry a label, mimicking
  labels that come from past investigations. Evaluation uses the full synthetic truth.

## Performance (test months)

Capacity 50 investigations per month; 46,153 account-months; 642 suspicious (1.4%).

| Approach | Precision at capacity | Recall at capacity | PR-AUC | ROC-AUC |
|---|---|---|---|---|
| Deployed priority score | 0.855 | 0.555 | 0.780 | 0.981 |
| LightGBM alone | 0.865 | 0.561 | 0.808 | 0.981 |
| Isolation Forest | 0.360 | 0.219 | 0.229 | 0.906 |
| Rules (ranked) | 0.275 | 0.170 | 0.121 | 0.884 |

Recall at capacity by typology (deployed queue): circular flow 82%, rapid pass-through 73%, scatter-gather 64%,
structuring 53%, fan-in 40%, fan-out 21%.

Organised rings: in 64 of 71 ring-months, at least one genuinely involved member was in the queue (rules: 47).

## Limitations

* **Synthetic data is cleaner than reality.** Launderers adapt, data quality is worse, and real labels are noisier.
  Expect much lower hit rates in production. Use these results to compare approaches and to size the opportunity,
  not to promise outcomes.
* **Label feedback loop.** In production the model learns from cases investigators already chose to work. The
  anomaly score and pattern detectors stay in the pipeline to surface behaviour the model has not learned.
* **Monthly windows.** Schemes that straddle month boundaries are split. The cycle detector allows a two-week
  spill-over; other features do not.
* **Low-signal mules.** Accounts that each send or receive a single modest payment for a fan-in/fan-out scheme are
  the main recall gap.
* **Fairness.** Customer type, business type, KYC level and home jurisdiction are features. Outcomes should be
  monitored by segment so that no group is investigated more often because of who they are rather than what their
  accounts do. Nationality and other protected attributes are not used.

## Monitoring and change control

* **Monthly:** precision at capacity on closed cases, alert volume, backlog, share of cases by customer segment,
  feature drift (population stability on the top 15 SHAP features).
* **Quarterly, or on drift:** retrain on the latest labelled cases with the same time-based protocol; compare to the
  incumbent on the most recent untouched months before promotion.
* **Any threshold, weight or rule change:** model it on the Impact Simulator page, export the policy change record
  (JSON), attach it to a change request, and apply only after compliance approval. Every case action is written to
  the append-only audit log.
