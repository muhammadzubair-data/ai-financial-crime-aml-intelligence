# Case study: finding money-laundering networks, not just suspicious transactions

## The problem

A digital bank's financial crime team was drowning in alerts. Its rule-based monitoring flagged hundreds of
accounts every month, around 95% of them for perfectly ordinary reasons: a shop collecting card payments, a company
running payroll, a customer selling a house. Analysts could open about 50 investigations a month, so most alerts
were skimmed or never reviewed.

The rules also missed the point. Money laundering is rarely one strange transaction. It is a pattern spread across
many accounts: money split into small amounts, passed through mule accounts within hours, sent round in a circle, or
deposited in cash just under the reporting limit. Looking at one account at a time hides all of that.

## What I built

An end-to-end platform that looks at the **network of money flows**, ranks suspicious activity to fit the team's
real capacity, and explains every case to the investigator who picks it up.

1. **Realistic test data.** Real bank data can't be shared, so I built a generator that simulates 6,000 customers
   and half a million transactions over two years, then plants 238 laundering schemes run by 16 organised rings.
   It also plants the legitimate behaviour that fools rules (payroll, merchants, intercompany transfers, house
   sales), so the comparison is fair.
2. **Network analysis.** Every month the platform builds a graph of who paid whom and measures things a
   transaction-level system can't see: whether an account sits in a loop where money comes back, whether it pays
   into an account that collects from many others, how central it is in where money ends up.
3. **Pattern detection.** Six detectors look for structuring, fan-in, fan-out, rapid pass-through, circular flows
   and layering through intermediaries. Each finding comes with a plain-English sentence an analyst can paste
   into a case note.
4. **Risk scoring.** Rules, an anomaly detector (Isolation Forest) and a supervised model (LightGBM) are compared
   on exactly the same data and judged on the question the team actually asks: *if we can open 50 cases this
   month, how many will be worth it?*
5. **Investigation workflow.** A ranked queue, the evidence behind every case (model drivers, rules fired,
   patterns found, linked accounts, transactions), notes, a controlled status workflow and an audit log of every
   action.
6. **Governance.** A model card, an out-of-time evaluation, and a simulator where compliance can test changes to
   capacity, thresholds and rules and export the result as a change record before anything goes live.

## Results

Tested on eight months of data the models had never seen:

* **85.5% of the cases in the queue were genuinely suspicious**, compared with 27.5% when the same number of cases
  were picked by rule score, and 5.5% across the full rule alert stream.
* **Three times as many suspicious accounts found** for the same 400 investigations (342 vs 110).
* **Organised rings were spotted in 64 of the 71 months they were active**, against 47 for rules.
* Effort per suspicious account found fell from about 27 analyst hours (triaging every rule alert) to 3.5.

## What I'd flag honestly

* The data is synthetic and cleaner than reality, so real hit rates will be lower. The value is in the comparison:
  the same data, the same capacity, very different results.
* Mule accounts that each make a single modest payment are still hard to catch (21–40% of fan-in/fan-out members).
* A blend of model and anomaly score was chosen on validation data and turned out one point worse on test than
  the model alone. I reported that rather than tuning it away, because that is how a model risk reviewer would
  want to see it.

## Stack

Python, DuckDB, Polars, NetworkX, scikit-learn, LightGBM, SHAP, SQLite, Streamlit, Plotly, pytest, GitHub Actions.
Fully reproducible: the same seed rebuilds byte-identical data, scores and metrics, checked automatically on every
push.
