"""Risk-ranked investigation queue and demo case history.

Each month the top-K account-months by hybrid priority become cases (K = team
capacity).  Linked alerted accounts are grouped into network clusters so one
investigator can work a ring together instead of as unrelated alerts.

For the public demo, earlier months are pre-populated with a *simulated*
investigation history (assignments, notes, decisions) so the workflow pages have
something realistic to show.  That simulation uses the synthetic ground truth
and is clearly labelled as seed data in the notes and audit log.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from amlintel.cases.store import CaseStore, drivers_to_json
from amlintel.graph.network import alert_networks

ANALYSTS = ["A. Rahman", "S. Patel", "L. Moreau", "J. Okafor", "M. Weber"]
SEED = "demo-seed"

TYP_NAMES = {"structuring": "structuring", "fan_out": "fan-out", "fan_in": "fan-in",
             "pass_through": "rapid pass-through", "cycle": "circular flow",
             "scatter_gather": "scatter-gather layering"}


def risk_band(score: float) -> str:
    if score >= 99.7:
        return "Critical"
    if score >= 99.3:
        return "High"
    return "Elevated"


def build_queue(scored: pd.DataFrame, tx: pd.DataFrame, indicators: pd.DataFrame,
                capacity: int, months: list[str]) -> pd.DataFrame:
    rows = []
    ind_acc = indicators.assign(account_id=indicators.accounts.str.split(",")).explode("account_id")
    for m in months:
        g = scored[scored.month == m].nlargest(capacity, "score_hybrid").copy()
        g["queue_rank"] = np.arange(1, len(g) + 1)
        clusters = alert_networks(tx[tx.month == m], g.account_id.tolist())
        sizes = pd.Series(clusters).value_counts()
        g["network_cluster"] = [
            f"{m}-N{clusters[a]:02d}" if a in clusters and sizes[clusters[a]] > 1 else ""
            for a in g.account_id]
        ia = ind_acc[ind_acc.month == m]
        ind_map = ia.groupby("account_id").agg(ids=("indicator_id", lambda s: ",".join(sorted(set(s)))),
                                               typs=("typology", lambda s: ",".join(sorted(set(s)))))
        g = g.merge(ind_map, left_on="account_id", right_index=True, how="left")
        g[["ids", "typs"]] = g[["ids", "typs"]].fillna("")
        rows.append(g)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _summary(r) -> str:
    bits = []
    if r.typs:
        bits.append("patterns: " + ", ".join(TYP_NAMES.get(t, t) for t in r.typs.split(",")))
    if r.rules_fired:
        bits.append(f"rules fired: {r.rules_fired}")
    if r.network_cluster:
        bits.append("linked to other alerted accounts this month")
    flow = f"in {r.in_amt:,.0f} / out {r.out_amt:,.0f} across {int(r.uniq_in_cp + r.uniq_out_cp)} counterparties"
    return "; ".join([flow] + bits)


def seed_case_db(path: Path, queue: pd.DataFrame, drivers: pd.DataFrame, live_month: str,
                 seed: int = 42) -> CaseStore:
    if Path(path).exists():
        Path(path).unlink()
    store = CaseStore(path)
    rng = np.random.default_rng(seed)
    drv = {k: g for k, g in drivers.groupby(["account_id", "month"])}
    for r in queue.itertuples(index=False):
        case_id = f"CASE-{r.month.replace('-', '')}-{r.queue_rank:03d}"
        created = pd.Timestamp(r.month + "-01") + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=7)
        d = drv.get((r.account_id, r.month))
        store.create({
            "case_id": case_id, "account_id": r.account_id, "month": r.month,
            "risk_score": float(r.risk_score), "risk_band": risk_band(r.risk_score),
            "queue_rank": int(r.queue_rank), "status": "New", "assigned_to": None,
            "network_cluster": r.network_cluster, "typologies": r.typs,
            "rules_fired": r.rules_fired, "indicator_ids": r.ids,
            "drivers_json": drivers_to_json(d) if d is not None else "[]",
            "summary": _summary(r),
        }, actor="monitoring-engine", ts=str(created))

        analyst = ANALYSTS[int(rng.integers(len(ANALYSTS)))]
        t = created + pd.Timedelta(hours=int(rng.integers(1, 30)))
        if r.month == live_month:
            # current month: a realistic mix of untouched and in-progress work
            roll = rng.random()
            if roll < 0.45:
                continue
            store.assign(case_id, analyst, "team-lead", str(t))
            if roll < 0.7:
                continue
            store.set_status(case_id, "In review", analyst, ts=str(t + pd.Timedelta(hours=2)))
            if roll > 0.9:
                store.set_status(case_id, "Awaiting information", analyst,
                                 "Requested source-of-funds evidence from the customer",
                                 ts=str(t + pd.Timedelta(days=1)))
            continue

        # closed months: simulated outcomes from synthetic ground truth
        store.assign(case_id, analyst, "team-lead", str(t))
        t = t + pd.Timedelta(hours=int(rng.integers(2, 48)))
        store.set_status(case_id, "In review", analyst, ts=str(t))
        t = t + pd.Timedelta(days=int(rng.integers(1, 6)))
        if r.label == 1:
            store.add_note(case_id, analyst,
                           f"[{SEED}] Flows not consistent with the customer's profile. {_summary(r)}. "
                           "Escalating for MLRO review.", ts=str(t))
            store.set_status(case_id, "Escalated to MLRO", analyst, ts=str(t))
            t = t + pd.Timedelta(days=int(rng.integers(2, 10)))
            if rng.random() < 0.85:
                store.set_status(case_id, "SAR filed", "MLRO",
                                 f"[{SEED}] Reasonable grounds for suspicion; report submitted", ts=str(t))
            else:
                store.set_status(case_id, "Closed - no further action", "MLRO",
                                 f"[{SEED}] Insufficient grounds after review", ts=str(t))
        else:
            reason = (f"[{SEED}] Activity explained by customer profile "
                      f"({'business' if r.is_business else 'personal'} account); no further action")
            store.add_note(case_id, analyst, reason, ts=str(t))
            store.set_status(case_id, "Closed - no further action", analyst, reason, ts=str(t))
    return store
