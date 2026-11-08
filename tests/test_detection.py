import networkx as nx
import pandas as pd

from amlintel.detection import typologies as T
from amlintel.detection.models import precision_at_k, top_k_per_month
from amlintel.detection.rules import RULES, apply_rules
from amlintel.graph.network import alert_networks, cycle_counts, ego_network
from conftest import make_tx


# --- pattern detectors --------------------------------------------------------------
def test_structuring_detected_with_onward_transfer():
    rows = [("CASH", "A", f"2026-01-0{d} 10:00", 9500, "cash_deposit") for d in (1, 2, 4, 5)]
    rows.append(("A", "X", "2026-01-07 12:00", 36000, "wire"))
    ind = T.detect_structuring(make_tx(rows))
    assert len(ind) == 1
    assert "4 cash deposits" in ind[0][-1] and "sent on to X" in ind[0][-1]


def test_deposits_above_threshold_are_not_structuring():
    rows = [("CASH", "A", f"2026-01-0{d} 10:00", 10500, "cash_deposit") for d in (1, 2, 3, 4)]
    assert T.detect_structuring(make_tx(rows)) == []


def test_time_respecting_cycle():
    rows = [("A", "B", "2026-01-01 09:00", 50000, "transfer"),
            ("B", "C", "2026-01-02 09:00", 49000, "transfer"),
            ("C", "A", "2026-01-03 09:00", 48000, "transfer")]
    ind = T.detect_cycles(make_tx(rows))
    assert len(ind) == 1 and set(ind[0][2].split(",")) == {"A", "B", "C"}


def test_cycle_out_of_time_order_is_ignored():
    rows = [("C", "A", "2026-01-01 09:00", 48000, "transfer"),   # returns before it left
            ("A", "B", "2026-01-02 09:00", 50000, "transfer"),
            ("B", "C", "2026-01-03 09:00", 49000, "transfer")]
    # each rotation must also be time ordered: C->A->B->C is, so this IS a cycle starting at C
    assert len(T.detect_cycles(make_tx(rows))) == 1
    rows[0] = ("C", "A", "2026-01-01 09:00", 4800, "transfer")   # amount collapse breaks the chain
    assert T.detect_cycles(make_tx(rows)) == []


def test_fan_out_flags_new_counterparties_but_not_routine_payroll():
    burst = [("H", f"M{i}", f"2026-03-01 {10 + i}:00", 5000, "transfer") for i in range(7)]
    assert len(T.detect_fan(make_tx(burst), "out")) == 1
    payroll = []
    for m in ("01", "02", "03"):
        payroll += [("E", f"S{i}", f"2026-{m}-25 09:00", 3000, "transfer") for i in range(10)]
    ind = T.detect_fan(make_tx(payroll), "out")
    months = {r[4].strftime("%Y-%m") for r in ind}
    assert "2026-03" not in months          # third month is routine, no longer flagged


def test_pass_through():
    rows = [("S", "A", "2026-01-01 08:00", 50000, "transfer"),
            ("A", "X", "2026-01-01 15:00", 30000, "transfer"),
            ("A", "Y", "2026-01-01 18:00", 17000, "transfer")]
    ind = T.detect_pass_through(make_tx(rows))
    assert len(ind) == 1 and "94%" in ind[0][-1]
    slow = [rows[0], ("A", "X", "2026-01-05 15:00", 47000, "transfer")]
    assert T.detect_pass_through(make_tx(slow)) == []


def test_scatter_gather():
    rows = []
    for i in range(4):
        rows.append(("S", f"M{i}", f"2026-01-01 0{i}:00", 20000, "transfer"))
        rows.append((f"M{i}", "K", f"2026-01-02 0{i}:00", 19500, "transfer"))
    ind = T.detect_scatter_gather(make_tx(rows))
    assert len(ind) == 1 and "4 intermediaries" in ind[0][-1]


def test_detect_all_ids_and_columns():
    rows = [("CASH", "A", f"2026-01-0{d} 10:00", 9500, "cash_deposit") for d in (1, 2, 4)]
    ind = T.detect_all(make_tx(rows))
    assert list(ind.columns[:2]) == ["indicator_id", "typology"] and ind.indicator_id.is_unique


# --- graph -----------------------------------------------------------------------------
def test_cycle_counts_and_ego_network():
    G = nx.DiGraph([("A", "B"), ("B", "C"), ("C", "A"), ("C", "D")])
    c = cycle_counts(G)
    assert c["A"] == c["B"] == c["C"] == 1 and c["D"] == 0
    tx = make_tx([("A", "B", "2026-01-01 09:00", 2000, "transfer"),
                  ("B", "C", "2026-01-01 10:00", 2000, "transfer"),
                  ("C", "D", "2026-01-01 11:00", 2000, "transfer")])
    assert set(ego_network(tx, "A", radius=1).nodes) == {"A", "B"}
    assert set(ego_network(tx, "A", radius=2).nodes) == {"A", "B", "C"}


def test_alert_networks_ignore_shared_hubs():
    rows = [("A", "MULE", "2026-01-01 09:00", 5000, "transfer"),
            ("B", "MULE", "2026-01-01 10:00", 5000, "transfer")]
    rows += [(f"C{i}", "SHOP", "2026-01-02 09:00", 1500, "transfer") for i in range(30)]
    rows += [("A", "SHOP", "2026-01-03 09:00", 1500, "transfer"), ("Z", "SHOP", "2026-01-03 09:00", 1500, "transfer")]
    cl = alert_networks(make_tx(rows), ["A", "B", "Z"])
    assert cl["A"] == cl["B"]           # linked through a low-degree mule account
    assert cl["Z"] != cl["A"]           # sharing a big merchant is not a link


# --- rules & ranking ------------------------------------------------------------------
def test_rules_fire_on_crafted_rows():
    base = dict(near_thr_cnt=0, cash_dep_amt=0, pass_through_ratio=0, out_amt=0, fan_out_72h=0,
                fan_in_72h=0, hrj_amt=0, account_age_days=1000, in_amt=0, round_share=0, out_cnt=0, in_cnt=0)
    df = pd.DataFrame([base, {**base, "near_thr_cnt": 4}, {**base, "pass_through_ratio": 0.9, "out_amt": 20000}])
    f = apply_rules(df)
    assert f.rule_alert.tolist() == [0, 1, 1]
    assert f.rules_fired.tolist() == ["", "R01", "R03"]
    assert set(f.columns) >= {r.rule_id for r in RULES}


def test_precision_at_k_and_top_k():
    df = pd.DataFrame({"month": ["m1"] * 4 + ["m2"] * 4, "label": [1, 0, 1, 0, 0, 0, 0, 1],
                       "s": [0.9, 0.8, 0.1, 0.0, 0.9, 0.8, 0.7, 0.1]})
    p = precision_at_k(df, "s", 2)
    assert p.precision.tolist() == [0.5, 0.0]
    assert p.recall.tolist() == [0.5, 0.0]
    assert len(top_k_per_month(df, "s", 2)) == 4
