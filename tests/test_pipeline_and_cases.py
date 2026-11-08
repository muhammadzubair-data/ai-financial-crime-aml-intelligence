import json

import pandas as pd
import pytest

from amlintel.cases.store import CaseStore
from amlintel.simulation import Policy, capacity_curve, simulate


# --- end-to-end build ----------------------------------------------------------------------
def test_build_writes_all_artifacts(built):
    out, manifest = built
    for f in ["accounts.parquet", "transactions.parquet", "scored.parquet", "indicators.parquet", "metrics.json",
              "manifest.json", "cases.db", "shap_alerts.parquet", "shap_global.csv", "lgbm_model.txt", "rules.csv"]:
        assert (out / f).exists(), f
    assert manifest["counts"]["laundering_instances"] > 0
    assert "synthetic" in manifest["data_provenance"].lower()


def test_time_split_is_strictly_ordered(built):
    out, _ = built
    sc = pd.read_parquet(out / "scored.parquet")
    last_train = sc.loc[sc.split == "train", "month_idx"].max()
    first_valid = sc.loc[sc.split == "valid", "month_idx"].min()
    first_test = sc.loc[sc.split == "test", "month_idx"].min()
    assert last_train < first_valid < first_test


def test_metrics_shape_and_model_beats_rules(built):
    out, _ = built
    m = json.loads((out / "metrics.json").read_text())
    assert set(m["models"]) == {"rules", "anomaly", "lgbm", "hybrid"}
    for v in m["models"].values():
        assert 0 <= v["precision_at_capacity"] <= 1 and 0 <= v["pr_auc"] <= 1
    assert m["models"]["lgbm"]["pr_auc"] > m["models"]["rules"]["pr_auc"]
    assert abs(sum(m["hybrid_weights"].values()) - 1) < 1e-9


def test_risk_score_is_monthly_percentile(built):
    out, _ = built
    sc = pd.read_parquet(out / "scored.parquet")
    g = sc.groupby("month").risk_score
    assert (g.max() == 100).all() and (sc.risk_score > 0).all()


def test_every_queued_case_has_evidence(built):
    out, _ = built
    cases = CaseStore(out / "cases.db").cases()
    assert len(cases) and cases.case_id.is_unique
    assert (cases.drivers_json.str.len() > 2).all()
    assert cases.summary.str.len().gt(10).all()


# --- case store ----------------------------------------------------------------------------
@pytest.fixture
def store(tmp_path):
    s = CaseStore(tmp_path / "c.db")
    s.create({"case_id": "C1", "account_id": "A", "month": "2026-01", "risk_score": 99.9,
              "risk_band": "Critical", "queue_rank": 1})
    return s


def test_case_workflow_is_audited(store):
    store.assign("C1", "S. Patel", "lead")
    store.set_status("C1", "In review", "S. Patel")
    store.add_note("C1", "S. Patel", "Checked source of funds.")
    store.set_status("C1", "Escalated to MLRO", "S. Patel")
    store.set_status("C1", "SAR filed", "MLRO", "Grounds for suspicion")
    actions = store.audit("C1").action.tolist()
    assert actions == ["case_created", "assigned", "status_changed", "note_added", "status_changed", "status_changed"]
    assert store.case("C1")["status"] == "SAR filed"


def test_invalid_transitions_and_missing_rationale(store):
    with pytest.raises(ValueError):
        store.set_status("C1", "SAR filed", "x", "skip the review")      # not allowed from New
    store.set_status("C1", "In review", "x")
    with pytest.raises(ValueError):
        store.set_status("C1", "Closed - no further action", "x", "  ")  # rationale required
    with pytest.raises(ValueError):
        store.add_note("C1", "x", "")
    with pytest.raises(KeyError):
        store.assign("NOPE", "x", "y")


# --- simulator -----------------------------------------------------------------------------
def test_simulator_capacity_tradeoff(built):
    out, _ = built
    sc = pd.read_parquet(out / "scored.parquet")
    test = sc[sc.split == "test"]
    cc = capacity_curve(test, Policy(), capacities=(5, 10, 20, 40))
    assert cc.recall.is_monotonic_increasing
    mo, s = simulate(test, Policy(capacity=10, active_rules=()))
    assert s["rule_alerts"] == 0 and (mo.investigated <= 10).all()
    assert s["cost"] == pytest.approx(s["hours"] * Policy().cost_per_hour)
