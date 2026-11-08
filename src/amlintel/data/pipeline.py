"""Feature pipeline: transactions -> account-month feature table.

The unit of monitoring is the **account-month**: one row per account per calendar
month in which it was active.  This mirrors how many AML teams run periodic
monitoring and makes "investigation capacity per month" a natural constraint.

* DuckDB handles the heavy SQL aggregations (volumes, counterparties, cash,
  cross-border and high-risk-jurisdiction exposure).
* Polars handles the time-window features that SQL expresses poorly
  (72-hour fan-in/fan-out bursts, 24-hour pass-through of funds).

Ground-truth labels are attached at the very end and kept in clearly named
``label*`` columns.  No feature reads them.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import polars as pl

from amlintel.config import CASH_THRESHOLD, HIGH_RISK_JURISDICTIONS

KYC_ORD = {"low": 0, "medium": 1, "high": 2}
KIND_ORD = {"retail": 0, "merchant": 1, "employer": 2, "trading": 3}


def add_month(tx: pd.DataFrame, start_date: str) -> pd.DataFrame:
    tx = tx.copy()
    tx["month"] = tx["timestamp"].dt.strftime("%Y-%m")
    s = pd.Timestamp(start_date)
    tx["month_idx"] = (tx["timestamp"].dt.year - s.year) * 12 + (tx["timestamp"].dt.month - s.month)
    return tx


def _legs(tx: pd.DataFrame) -> pl.DataFrame:
    """Long format: one row per (account, transaction, direction).

    Built in Polars (Arrow memory) rather than pandas: at production scale the
    long table has twice as many rows as the transaction table.
    """
    t = pl.from_pandas(tx[["txn_id", "timestamp", "month", "month_idx", "from_account", "to_account",
                           "from_country", "to_country", "amount", "channel", "is_cross_border"]])
    out = t.filter(pl.col("from_account") != "CASH").select(
        "txn_id", "timestamp", "month", "month_idx",
        pl.col("from_account").alias("account_id"), pl.col("to_account").alias("counterparty"),
        pl.col("to_country").alias("cp_country"), pl.lit("out").alias("direction"),
        "amount", "channel", "is_cross_border")
    inn = t.filter(pl.col("to_account") != "CASH").select(
        "txn_id", "timestamp", "month", "month_idx",
        pl.col("to_account").alias("account_id"), pl.col("from_account").alias("counterparty"),
        pl.col("from_country").alias("cp_country"), pl.lit("in").alias("direction"),
        "amount", "channel", "is_cross_border")
    return pl.concat([out, inn])


SQL_AGG = f"""
SELECT
  account_id, month, month_idx,
  SUM(CASE WHEN direction='out' THEN 1 ELSE 0 END)                         AS out_cnt,
  SUM(CASE WHEN direction='out' THEN amount ELSE 0 END)                    AS out_amt,
  SUM(CASE WHEN direction='in'  THEN 1 ELSE 0 END)                         AS in_cnt,
  SUM(CASE WHEN direction='in'  THEN amount ELSE 0 END)                    AS in_amt,
  COUNT(DISTINCT CASE WHEN direction='out' AND counterparty<>'CASH' THEN counterparty END) AS uniq_out_cp,
  COUNT(DISTINCT CASE WHEN direction='in'  AND counterparty<>'CASH' THEN counterparty END) AS uniq_in_cp,
  SUM(CASE WHEN channel='cash_deposit' THEN 1 ELSE 0 END)                  AS cash_dep_cnt,
  SUM(CASE WHEN channel='cash_deposit' THEN amount ELSE 0 END)             AS cash_dep_amt,
  SUM(CASE WHEN channel='cash_deposit' AND amount >= {0.9 * CASH_THRESHOLD}
            AND amount < {CASH_THRESHOLD} THEN 1 ELSE 0 END)               AS near_thr_cnt,
  SUM(CASE WHEN channel='cash_withdrawal' THEN amount ELSE 0 END)          AS cash_wd_amt,
  SUM(CASE WHEN direction='out' AND is_cross_border=1 THEN amount ELSE 0 END) AS xb_out_amt,
  SUM(CASE WHEN direction='in'  AND is_cross_border=1 THEN amount ELSE 0 END) AS xb_in_amt,
  SUM(CASE WHEN cp_country IN {tuple(HIGH_RISK_JURISDICTIONS)} AND counterparty<>'CASH'
           THEN amount ELSE 0 END)                                          AS hrj_amt,
  SUM(CASE WHEN amount >= 1000 AND amount % 100 = 0 THEN 1 ELSE 0 END)      AS round_cnt,
  SUM(CASE WHEN amount >= 10000 THEN 1 ELSE 0 END)                          AS large_cnt,
  MAX(amount)                                                               AS max_amt,
  COUNT(DISTINCT CAST(timestamp AS DATE))                                   AS active_days
FROM legs
GROUP BY account_id, month, month_idx
"""


def _sql_aggregates(legs: pl.DataFrame) -> pd.DataFrame:
    con = duckdb.connect()
    con.register("legs", legs.to_arrow())
    df = con.execute(SQL_AGG).df()
    con.close()
    return df


def _pass_through(lf: pl.DataFrame, window_h: int = 24) -> pl.DataFrame:
    acct = lf["account_id"].to_numpy()
    codes = {a: i for i, a in enumerate(np.unique(acct))}
    code = np.array([codes[a] for a in acct], dtype=np.int64)
    secs = lf["timestamp"].cast(pl.Datetime("us")).dt.epoch("s").to_numpy().astype(np.int64)
    key = code * 10**11 + secs
    is_in = (lf["direction"] == "in").to_numpy()
    amt = lf["amount"].to_numpy()
    ki = key[is_in]
    order = np.argsort(ki, kind="stable")
    ki = ki[order]
    cum = np.concatenate([[0.0], np.cumsum(amt[is_in][order])])
    ko = key[~is_in]
    in_24h = cum[np.searchsorted(ki, ko, side="right")] - cum[np.searchsorted(ki, ko - window_h * 3600, side="right")]
    passed = np.minimum(amt[~is_in], in_24h)
    out = pl.DataFrame({"account_id": acct[~is_in], "month": lf["month"].to_numpy()[~is_in], "passed": passed})
    return out.group_by(["account_id", "month"]).agg(pl.col("passed").sum().alias("pass_through_amt"))


def _window_features(legs: pl.DataFrame) -> pd.DataFrame:
    """72h fan bursts and 24h pass-through, computed per account with Polars."""
    lf = legs.select(["account_id", "timestamp", "month", "direction",
                      "counterparty", "amount"]).sort(["account_id", "timestamp"])

    def fan(direction: str, name: str) -> pl.DataFrame:
        d = (lf.filter((pl.col("direction") == direction) & (pl.col("counterparty") != "CASH")
                       & (pl.col("amount") >= 1000))
             .with_columns(pl.col("counterparty").cast(pl.Categorical).to_physical().alias("cp")))
        if d.height == 0:
            return pl.DataFrame(schema={"account_id": pl.String, "month": pl.String, name: pl.Int64})
        r = (d.rolling(index_column="timestamp", period="72h", group_by="account_id")
              .agg(pl.col("cp").n_unique().alias(name), pl.col("month").last()))
        return r.group_by(["account_id", "month"]).agg(pl.col(name).max())

    fo = fan("out", "fan_out_72h")
    fi = fan("in", "fan_in_72h")

    # pass-through: for each outflow, how much of it could have come from money
    # received in the previous 24h.  Binary search on per-account running inflow
    # totals: exact, order-independent with tied timestamps, and fast.
    ptr = _pass_through(lf)

    out = fo.join(fi, on=["account_id", "month"], how="full", coalesce=True)
    out = out.join(ptr, on=["account_id", "month"], how="full", coalesce=True)
    return out.to_pandas()


def build_account_month_features(tx: pd.DataFrame, accounts: pd.DataFrame,
                                 start_date: str) -> pd.DataFrame:
    if "month" not in tx:
        tx = add_month(tx, start_date)
    # Process one month at a time (plus a 4-day look-back so 72h/24h windows at the
    # start of a month see the end of the previous one). Results are identical to a
    # single pass, and memory stays flat as the customer base grows.
    parts = []
    lookback = pd.Timedelta(days=4)
    for month in sorted(tx["month"].unique()):
        m_start = pd.Timestamp(month + "-01")
        m_end = m_start + pd.offsets.MonthBegin(1)
        chunk = tx[(tx.timestamp >= m_start - lookback) & (tx.timestamp < m_end)]
        legs = _legs(chunk)
        base = _sql_aggregates(legs.filter(pl.col("month") == month))
        win = _window_features(legs)
        win = win[win.month == month]
        parts.append(base.merge(win, on=["account_id", "month"], how="left"))
        del legs
    f = pd.concat(parts, ignore_index=True)
    # parallel float sums can differ in the last bits between runs; money is cents
    money = [c for c in f.columns if c.endswith("_amt")]
    f[money] = f[money].round(2)
    for c in ("fan_out_72h", "fan_in_72h", "pass_through_amt"):
        f[c] = f[c].fillna(0)
    f["pass_through_amt"] = f["pass_through_amt"].round(2)

    # Ratios
    f["net_flow_ratio"] = (f.out_amt / (f.in_amt + 1.0)).clip(0, 20)
    f["in_out_balance"] = np.minimum(f.in_amt, f.out_amt) / np.maximum(np.maximum(f.in_amt, f.out_amt), 1.0)
    f["xb_out_share"] = f.xb_out_amt / np.maximum(f.out_amt, 1.0)
    f["xb_in_share"] = f.xb_in_amt / np.maximum(f.in_amt, 1.0)
    f["pass_through_ratio"] = (f.pass_through_amt / np.maximum(f.out_amt, 1.0)).clip(0, 1)
    f["round_share"] = f.round_cnt / np.maximum(f.out_cnt + f.in_cnt, 1)
    f["avg_txn_amt"] = (f.out_amt + f.in_amt) / np.maximum(f.out_cnt + f.in_cnt, 1)
    f["log_total_flow"] = np.log1p(f.out_amt + f.in_amt)

    # Static customer attributes (KYC data the bank already holds)
    a = accounts[["account_id", "customer_type", "business_kind", "country", "opened_on", "kyc_risk"]]
    f = f.merge(a, on="account_id", how="left")
    month_start = pd.to_datetime(f["month"] + "-01")
    f["account_age_days"] = (month_start - f["opened_on"]).dt.days.clip(lower=0)
    f["kyc_level"] = f["kyc_risk"].map(KYC_ORD).astype(int)
    f["is_business"] = (f["customer_type"] == "business").astype(int)
    f["business_kind_code"] = f["business_kind"].map(KIND_ORD).astype(int)
    f["home_hrj"] = f["country"].isin(HIGH_RISK_JURISDICTIONS).astype(int)
    f = f.drop(columns=["opened_on"])
    return f


def attach_labels(features: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    """Ground truth for evaluation only (synthetic data gives us complete labels)."""
    L = tx[tx.is_laundering == 1]
    parts = []
    for col in ("from_account", "to_account"):
        parts.append(L[[col, "month", "typology", "network_id"]].rename(columns={col: "account_id"}))
    lab = pd.concat(parts)
    lab = lab[lab.account_id != "CASH"]
    agg = lab.groupby(["account_id", "month"]).agg(
        label_typology=("typology", lambda s: s.value_counts().index[0]),
        label_network=("network_id", "first"),
    ).reset_index()
    agg["label"] = 1
    out = features.merge(agg, on=["account_id", "month"], how="left")
    out["label"] = out["label"].fillna(0).astype(int)
    out["label_typology"] = out["label_typology"].fillna("none")
    out["label_network"] = out["label_network"].fillna("")
    return out
