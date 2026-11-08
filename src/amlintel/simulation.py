"""Scenario simulation for compliance and model governance.

Lets a user change policy levers and see the modelled trade-offs on the
out-of-time test months:

* investigation capacity (cases the team can open per month)
* minimum risk score before a case is opened
* hybrid blend weights (model / network / rules / anomaly)
* which scenario rules are switched on
* effort and cost assumptions

All outputs are **modelled on synthetic data**.  They show the shape of the
trade-off, not a forecast for a real institution.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from amlintel.detection.rules import RULES


@dataclass
class Policy:
    capacity: int = 50
    min_risk_score: float = 99.0         # 0-100 percentile within month: only the top 1% become alerts
    w_lgbm: float = 0.7
    w_network: float = 0.15
    w_rules: float = 0.10
    w_anomaly: float = 0.05
    active_rules: tuple[str, ...] = tuple(r.rule_id for r in RULES)
    hours_per_case: float = 3.0
    hours_per_rule_alert: float = 1.5    # triage time for a raw rule alert
    cost_per_hour: float = 45.0


def _rule_score(df: pd.DataFrame, active: tuple[str, ...]) -> pd.Series:
    w = {r.rule_id: r.weight for r in RULES}
    s = pd.Series(0.0, index=df.index)
    for rid in active:
        s += w[rid] * df[rid]
    return s


def simulate(test: pd.DataFrame, policy: Policy) -> tuple[pd.DataFrame, dict]:
    d = test.copy()
    rs = _rule_score(d, policy.active_rules)
    d["r_rules_p"] = (rs + 1e-3 * d.r_anomaly).groupby(d.month).rank(pct=True)
    tot = policy.w_lgbm + policy.w_network + policy.w_rules + policy.w_anomaly
    tot = tot if tot > 0 else 1.0
    d["prio"] = (policy.w_lgbm * d.r_lgbm + policy.w_network * d.r_network
                 + policy.w_rules * d.r_rules_p + policy.w_anomaly * d.r_anomaly) / tot
    d["prio_pct"] = 100 * d.groupby("month").prio.rank(pct=True)
    d["rule_alert_p"] = (rs > 0).astype(int)

    rows = []
    for m, g in d.groupby("month"):
        eligible = g[g.prio_pct >= policy.min_risk_score]
        worked = eligible.nlargest(policy.capacity, "prio")
        pos = int(g.label.sum())
        tp = int(worked.label.sum())
        ra = g[g.rule_alert_p == 1]
        ra_worked = ra.nlargest(policy.capacity, "r_rules_p")
        rows.append({
            "month": m, "positives": pos,
            "eligible_alerts": len(eligible), "investigated": len(worked),
            "backlog": max(len(eligible) - len(worked), 0),
            "true_positives": tp, "precision": tp / max(len(worked), 1),
            "recall": tp / pos if pos else np.nan,
            "rule_alerts": len(ra), "rule_tp_within_capacity": int(ra_worked.label.sum()),
            "rule_tp_all": int(ra.label.sum()),
        })
    mo = pd.DataFrame(rows)
    hours = mo.investigated.sum() * policy.hours_per_case
    rule_hours = mo.rule_alerts.sum() * policy.hours_per_rule_alert
    summary = {
        "precision": float(mo.true_positives.sum() / max(mo.investigated.sum(), 1)),
        "recall": float(mo.true_positives.sum() / max(mo.positives.sum(), 1)),
        "true_positives": int(mo.true_positives.sum()),
        "investigated": int(mo.investigated.sum()),
        "missed": int(mo.positives.sum() - mo.true_positives.sum()),
        "avg_backlog": float(mo.backlog.mean()),
        "hours": float(hours), "cost": float(hours * policy.cost_per_hour),
        "rule_alerts": int(mo.rule_alerts.sum()),
        "rule_hours_all_alerts": float(rule_hours),
        "rule_cost_all_alerts": float(rule_hours * policy.cost_per_hour),
        "rule_recall_all_alerts": float(mo.rule_tp_all.sum() / max(mo.positives.sum(), 1)),
        "rule_recall_within_capacity": float(mo.rule_tp_within_capacity.sum() / max(mo.positives.sum(), 1)),
        "cost_per_true_positive": float(hours * policy.cost_per_hour / max(mo.true_positives.sum(), 1)),
    }
    return mo, summary


def capacity_curve(test: pd.DataFrame, policy: Policy, capacities=(10, 20, 30, 50, 75, 100, 150, 200)) -> pd.DataFrame:
    out = []
    for k in capacities:
        # threshold removed so the curve shows the pure effect of capacity
        _, s = simulate(test, Policy(**{**policy.__dict__, "capacity": k, "min_risk_score": 0.0}))
        out.append({"capacity": k, "precision": s["precision"], "recall": s["recall"],
                    "true_positives": s["true_positives"], "cost": s["cost"]})
    return pd.DataFrame(out)
