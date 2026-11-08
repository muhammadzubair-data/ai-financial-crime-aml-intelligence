"""Rule-based transaction monitoring baseline.

These are the kind of scenario rules most banks run today.  They are easy to
explain and audit, which is why the platform keeps them, but they fire on lots
of legitimate behaviour.  Thresholds live in one table so compliance can review
and tune them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from amlintel.config import CASH_THRESHOLD


@dataclass(frozen=True)
class Rule:
    rule_id: str
    name: str
    description: str
    weight: float
    fn: Callable[[pd.DataFrame], pd.Series]


RULES: tuple[Rule, ...] = (
    Rule("R01", "Possible structuring",
         f"3+ cash deposits between 90% and 100% of the {CASH_THRESHOLD:,.0f} threshold in a month",
         3.0, lambda d: d.near_thr_cnt >= 3),
    Rule("R02", "High cash volume", "Cash deposits of 30,000+ in a month",
         1.0, lambda d: d.cash_dep_amt >= 30_000),
    Rule("R03", "Rapid movement of funds",
         "80%+ of outgoing value was received within the previous 24h, and 15,000+ sent",
         2.0, lambda d: (d.pass_through_ratio >= 0.8) & (d.out_amt >= 15_000)),
    Rule("R04", "Fan-out burst", "Payments of 1,000+ to 8+ counterparties within 72 hours",
         1.5, lambda d: d.fan_out_72h >= 8),
    Rule("R05", "Fan-in burst", "Receipts of 1,000+ from 8+ counterparties within 72 hours",
         1.5, lambda d: d.fan_in_72h >= 8),
    Rule("R06", "High-risk jurisdiction exposure",
         "10,000+ sent to or received from a high-risk jurisdiction",
         2.0, lambda d: d.hrj_amt >= 10_000),
    Rule("R07", "Large flows on a new account", "Account under 180 days old with 25,000+ incoming",
         1.5, lambda d: (d.account_age_days < 180) & (d.in_amt >= 25_000)),
    Rule("R08", "Round-amount pattern", "Half or more of 5+ transactions are round amounts of 1,000+",
         0.5, lambda d: (d.round_share >= 0.5) & ((d.out_cnt + d.in_cnt) >= 5)),
)

RULE_INFO = pd.DataFrame([(r.rule_id, r.name, r.description, r.weight) for r in RULES],
                         columns=["rule_id", "name", "description", "weight"])


def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    flags = pd.DataFrame({r.rule_id: r.fn(df).astype(int).to_numpy() for r in RULES}, index=df.index)
    weights = np.array([r.weight for r in RULES])
    flags["rule_score"] = flags[[r.rule_id for r in RULES]].to_numpy() @ weights
    flags["rule_alert"] = (flags["rule_score"] > 0).astype(int)
    fired = pd.Series("", index=df.index)
    for r in RULES:
        fired = fired + np.where(flags[r.rule_id] == 1, r.rule_id + ",", "")
    flags["rules_fired"] = fired.str.rstrip(",")
    return flags
