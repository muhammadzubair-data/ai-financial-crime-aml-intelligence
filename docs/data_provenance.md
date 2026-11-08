# Data provenance

## Source

All data in this project is **synthetic**. It is produced by `src/amlintel/data/generator.py` from a fixed random
seed and contains no real people, accounts, banks or transactions. Account IDs (`ACC000001`…) are sequential
placeholders. High-risk jurisdictions are the placeholder codes `HR1`, `HR2` and `HR3`; the platform reads the real
list from policy configuration in production and does not label any real country.

## Why synthetic

Real transaction data cannot be published. Public AML benchmarks, notably the IBM AMLSim family and the
"IBM Transactions for Anti Money Laundering" dataset, were considered. They are a good fit for research, but their
licence terms, size and lack of customer attributes make them awkward for a public, self-contained demo. This
generator follows the same design idea as AMLSim (simulate ordinary behaviour, then inject known laundering
patterns so ground truth exists) and adds the customer context and benign look-alikes the platform needs.
The pipeline reads plain Parquet with documented columns, so a licensed benchmark or real bank extract can be
dropped in by writing a small loader to the same schema.

## How it is built

**Customers.** Individuals (84%) and businesses (16%: merchants, employers, trading companies), across 12 home
jurisdictions. Each customer gets an account-opening date (about one in five open during the observed period), a KYC risk
level, an activity rate and a spending-power factor so that affluent customers and larger businesses routinely
move large sums.

**Ordinary behaviour.** Transfers to a personal list of counterparties chosen by preferential attachment and
mostly domestic; card payments to merchants; monthly payroll from employers to a fixed staff list; cash deposits
and withdrawals; cross-border wires.

**Benign look-alikes (on purpose).** Behaviour that resembles laundering but is legitimate, so that rules and
detectors are tested honestly:

* merchants collecting many payments (looks like fan-in)
* employers paying many staff at once (looks like fan-out)
* "salary sweepers" who move most of their pay on within hours (looks like pass-through)
* intercompany treasury loops among related companies (looks like circular flow)
* life events such as a house sale received and paid on within a day (large pass-through)
* cash-heavy small businesses with occasional deposits just under the threshold

**Laundering.** 16 organised rings (10–23 members plus distributor "hubs" and "exit" accounts, often in
higher-risk or offshore-style jurisdictions) run 10–17 schemes each over 6–18 months, plus 30 one-off schemes among
otherwise ordinary customers. Six typologies:

| Typology | What is injected |
|---|---|
| Fan-out | A hub pays 6–12 ring members 1,500–9,600 each within about 2.5 days |
| Fan-in | 6–12 members pay one collector 1,500–9,600 each within about 3 days |
| Circular flow | Funds pass through 3–6 accounts and return to the start, losing 1–4% a hop, 2–48 hours apart |
| Structuring | 4–9 cash deposits at 90–99.5% of the 10,000 threshold, then 90–97% wired to an exit account |
| Scatter-gather | A hub splits 40k–200k across 3–6 intermediaries that forward 95–99% to one sink |
| Rapid pass-through | 20k–150k received and 90–99% sent on within 2–20 hours, repeated 2–4 times |

**History.** Two months of ordinary activity before the analysis window give detectors context (for example which
relationships are routine). History is excluded from features, models and evaluation, and contains no injected
laundering.

## Demo dataset (data/demo)

| Item | Value |
|---|---|
| Accounts | 6,000 |
| Transactions (analysis window) | 498,019 |
| Transactions including 2-month history | 528,250 |
| Window | November 2024 to October 2026 (24 months) |
| Account-months | 126,540 |
| Laundering transactions | 1,747 in 238 schemes |
| Organised rings | 16 |

## Labels

Every injected transaction carries `is_laundering`, `typology`, `instance_id` (scheme) and `network_id` (ring or
`LONE`). An account-month is labelled suspicious when the account sent or received any laundering transaction that
month. Labels are used only for training (70% subsample) and evaluation, never as features.

## Reproducibility

The same seed produces byte-identical transactions, indicators and scores, independent of Python's hash seed.
CI rebuilds the demo and checks this with `scripts/check_reproducible.py`.

## What it does not represent

Real criminal behaviour is adaptive and far more varied than six templates; real data has missing fields, merged
customers and channel quirks; real labels are biased by what was investigated. Treat results as a fair comparison
of methods on the same data, not as expected production performance.
