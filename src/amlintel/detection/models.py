"""Suspicious-activity detection models and evaluation.

Four scoring approaches are compared on the same account-month table:

1. **Rules**           - the scenario rules in :mod:`amlintel.detection.rules`
2. **Isolation Forest** - unsupervised anomaly detection (no labels needed)
3. **LightGBM**        - supervised risk model, trained only where labels exist
4. **Hybrid priority** - blend of the above plus a network-exposure score,
                         with weights chosen on validation months only

Evaluation is strictly out-of-time: fit on the first months, choose settings on
the next two, and report on later months the models never saw.  The headline
metric is **precision at investigation capacity**: if the team can open K cases a
month, how many of those K turn out to be genuinely suspicious?
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from amlintel.config import ModelConfig

BASE_FEATURES = [
    "out_cnt", "out_amt", "in_cnt", "in_amt", "uniq_out_cp", "uniq_in_cp",
    "cash_dep_cnt", "cash_dep_amt", "near_thr_cnt", "cash_wd_amt", "xb_out_amt", "xb_in_amt",
    "hrj_amt", "round_cnt", "large_cnt", "max_amt", "active_days", "fan_out_72h", "fan_in_72h",
    "pass_through_amt", "net_flow_ratio", "in_out_balance", "xb_out_share", "xb_in_share",
    "pass_through_ratio", "round_share", "avg_txn_amt", "log_total_flow",
    "account_age_days", "kyc_level", "is_business", "business_kind_code", "home_hrj",
]
GRAPH_FEATURES = [
    "pagerank_scaled", "scc_size", "cycle_count", "succ_max_in_deg", "pred_max_out_deg",
    "nbr_high_kyc_share", "n_neighbors",
]
INDICATOR_FEATURES = [
    "ind_structuring", "ind_fan_out", "ind_fan_in", "ind_pass_through", "ind_cycle",
    "ind_scatter_gather", "ind_total",
]
FEATURES = BASE_FEATURES + GRAPH_FEATURES + INDICATOR_FEATURES

FEATURE_LABELS = {
    "out_cnt": "Outgoing transactions", "out_amt": "Outgoing value", "in_cnt": "Incoming transactions",
    "in_amt": "Incoming value", "uniq_out_cp": "Distinct payees", "uniq_in_cp": "Distinct payers",
    "cash_dep_cnt": "Cash deposits (count)", "cash_dep_amt": "Cash deposits (value)",
    "near_thr_cnt": "Deposits just under threshold", "cash_wd_amt": "Cash withdrawals",
    "xb_out_amt": "Cross-border outgoing", "xb_in_amt": "Cross-border incoming",
    "hrj_amt": "High-risk jurisdiction value", "round_cnt": "Round-amount transactions",
    "large_cnt": "Transactions of 10,000+", "max_amt": "Largest transaction",
    "active_days": "Active days", "fan_out_72h": "Max payees in 72h", "fan_in_72h": "Max payers in 72h",
    "pass_through_amt": "Value passed through in 24h", "net_flow_ratio": "Out/in value ratio",
    "in_out_balance": "In/out balance", "xb_out_share": "Cross-border share (out)",
    "xb_in_share": "Cross-border share (in)", "pass_through_ratio": "Pass-through ratio (24h)",
    "round_share": "Round-amount share", "avg_txn_amt": "Average transaction",
    "log_total_flow": "Total flow (log)", "account_age_days": "Account age (days)",
    "kyc_level": "KYC risk level", "is_business": "Business customer",
    "business_kind_code": "Business type (0 retail, 1 merchant, 2 employer, 3 trading)", "home_hrj": "Domiciled in high-risk jurisdiction",
    "pagerank_scaled": "Network centrality (PageRank)", "scc_size": "Circular-flow component size",
    "cycle_count": "Short cycles involved in", "succ_max_in_deg": "Pays into a collector account",
    "pred_max_out_deg": "Paid by a distributor account", "nbr_high_kyc_share": "High-KYC neighbour share",
    "n_neighbors": "Counterparties (network)", "ind_structuring": "Structuring indicators",
    "ind_fan_out": "Fan-out indicators", "ind_fan_in": "Fan-in indicators",
    "ind_pass_through": "Pass-through indicators", "ind_cycle": "Cycle indicators",
    "ind_scatter_gather": "Scatter-gather indicators", "ind_total": "All pattern indicators",
}

SCORE_COLS = {"rules": "score_rules", "anomaly": "score_iforest",
              "lgbm": "score_lgbm", "hybrid": "score_hybrid"}
MODEL_NAMES = {"rules": "Rule-based monitoring", "anomaly": "Isolation Forest",
               "lgbm": "LightGBM risk model", "hybrid": "Deployed priority score"}


def split_masks(df: pd.DataFrame, cfg: ModelConfig):
    tr = df.month_idx < cfg.train_months
    va = (df.month_idx >= cfg.train_months) & (df.month_idx < cfg.train_months + cfg.valid_months)
    te = df.month_idx >= cfg.train_months + cfg.valid_months
    return tr, va, te


def network_score(df: pd.DataFrame) -> pd.Series:
    """Transparent network-exposure heuristic (no labels involved)."""
    return (2.0 * (df.cycle_count > 0) + np.log1p(df.scc_size - 1)
            + 1.0 * (df.succ_max_in_deg >= 5) + 1.0 * (df.pred_max_out_deg >= 5)
            + 1.5 * df.ind_total.clip(upper=4) + 2.0 * df.nbr_high_kyc_share)


def _month_pct_rank(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("month")[col].rank(pct=True, method="average")


@dataclass
class TrainedModels:
    lgbm: lgb.Booster
    iforest: IsolationForest
    hybrid_weights: dict
    best_iteration: int
    valid_ap: float


def _chunked(fn, X: pd.DataFrame, size: int = 250_000) -> np.ndarray:
    """Apply a scoring function in row chunks to bound temporary memory."""
    return np.concatenate([fn(X.iloc[i:i + size]) for i in range(0, len(X), size)]) if len(X) else np.array([])


def _log_features(X: pd.DataFrame) -> np.ndarray:
    return np.sign(X.to_numpy(float)) * np.log1p(np.abs(X.to_numpy(float)))


def fit_and_score(df: pd.DataFrame, cfg: ModelConfig) -> tuple[pd.DataFrame, TrainedModels]:
    # Scores are added to ``df`` in place (no copy): at production scale the
    # account-month table is the largest object in memory.
    tr, va, te = split_masks(df, cfg)
    X = df[FEATURES]
    # Training labels are incomplete on purpose (see ModelConfig.label_coverage).
    rng = np.random.default_rng(cfg.random_state)
    known = rng.random(len(df)) < cfg.label_coverage
    df["label_known"] = (df["label"].to_numpy() * known).astype(int)
    y = df["label_known"]

    # Negative down-sampling for the supervised fit (all positives kept). Rankings
    # within a month are unaffected; it keeps memory flat at production scale.
    fit_tr = tr.to_numpy() & ((y.to_numpy() == 1) | (rng.random(len(df)) < cfg.neg_sample_rate))
    fit_tr = pd.Series(fit_tr, index=df.index)

    # --- Isolation Forest (unsupervised, fit on training months only) ----------
    iso = IsolationForest(n_estimators=300, max_samples=4096, random_state=cfg.random_state, n_jobs=-1)
    tr_idx = np.flatnonzero(tr.to_numpy())
    if len(tr_idx) > cfg.iforest_fit_rows:
        tr_idx = np.sort(rng.choice(tr_idx, cfg.iforest_fit_rows, replace=False))
    iso.fit(_log_features(X.iloc[tr_idx]))
    df["score_iforest"] = _chunked(lambda a: -iso.score_samples(_log_features(a)), X)

    # --- LightGBM: early stopping on validation months ------------------------
    params = dict(cfg.lgbm_params, seed=cfg.random_state,
                  scale_pos_weight=float((y[fit_tr] == 0).sum() / max((y[fit_tr] == 1).sum(), 1)) ** 0.5)
    dtr = lgb.Dataset(X[fit_tr], y[fit_tr])
    dva = lgb.Dataset(X[va], y[va], reference=dtr)
    booster = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
    best_it = booster.best_iteration or 200
    valid_pred = booster.predict(X[va], num_iteration=best_it)
    valid_ap = float(average_precision_score(df.loc[va, "label"], valid_pred))

    # score train/valid with the validation-stage model (honest for weight tuning)
    df["score_lgbm"] = _chunked(lambda a: booster.predict(a, num_iteration=best_it), X)

    df["score_rules"] = df["rule_score"] + 1e-3 * df["score_iforest"].rank(pct=True)  # tie-break
    df["score_network"] = network_score(df)

    # --- Hybrid weights: grid search on validation months only ---------------
    for key, col in (("lgbm", "score_lgbm"), ("network", "score_network"),
                     ("rules", "score_rules"), ("anomaly", "score_iforest")):
        df[f"r_{key}"] = _month_pct_rank(df, col)
    def _valid_precision(wl, wn, wr, wa):
        s = wl * df.r_lgbm + wn * df.r_network + wr * df.r_rules + wa * df.r_anomaly
        v = df[va].assign(_s=s[va], label=df.loc[va, "label_known"])
        return precision_at_k(v, "_s", cfg.capacity_per_month)["precision"].mean()

    # Start from the pure model; a blend must beat it on validation by a clear
    # margin, otherwise we are just fitting noise in two months of data.
    best_w = {"lgbm": 1.0, "network": 0.0, "rules": 0.0, "anomaly": 0.0}
    base_p = best_p = _valid_precision(1.0, 0.0, 0.0, 0.0)
    grid = [0.0, 0.05, 0.1, 0.2, 0.3]
    for wn, wr, wa in itertools.product(grid, grid, grid):
        wl = 1.0 - wn - wr - wa
        if wl < 0.4 or wl == 1.0:
            continue
        p = _valid_precision(wl, wn, wr, wa)
        if p > best_p and p >= base_p + cfg.blend_min_gain:
            best_p, best_w = p, {"lgbm": round(wl, 2), "network": wn, "rules": wr, "anomaly": wa}

    # --- Refit LightGBM on train+valid with the chosen number of rounds -------
    trva = fit_tr | va
    final = lgb.train(params, lgb.Dataset(X[trva], y[trva]), num_boost_round=best_it)
    df.loc[te, "score_lgbm"] = _chunked(final.predict, X[te])
    df["r_lgbm"] = _month_pct_rank(df, "score_lgbm")
    df["score_hybrid"] = (best_w["lgbm"] * df.r_lgbm + best_w["network"] * df.r_network
                          + best_w["rules"] * df.r_rules + best_w["anomaly"] * df.r_anomaly)
    df["risk_score"] = (100 * _month_pct_rank(df, "score_hybrid")).round(1)
    df["split"] = np.select([tr, va, te], ["train", "valid", "test"], default="other")
    return df, TrainedModels(final, iso, best_w, best_it, valid_ap)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def top_k_per_month(df: pd.DataFrame, score_col: str, k: int) -> pd.DataFrame:
    """The K highest-scoring rows in each month (the investigation queue)."""
    return df.sort_values(score_col, ascending=False, kind="stable").groupby("month").head(k)


def precision_at_k(df: pd.DataFrame, score_col: str, k: int) -> pd.DataFrame:
    rows = []
    for month, g in df.groupby("month"):
        top = g.nlargest(k, score_col)
        pos = int(g.label.sum())
        tp = int(top.label.sum())
        rows.append({"month": month, "k": len(top), "true_positives": tp, "positives": pos,
                     "precision": tp / max(len(top), 1), "recall": tp / pos if pos else np.nan})
    return pd.DataFrame(rows)


def evaluate(scored: pd.DataFrame, cfg: ModelConfig) -> dict:
    te = scored[scored.split == "test"]
    k = cfg.capacity_per_month
    out = {"capacity_per_month": k, "test_months": sorted(te.month.unique().tolist()),
           "test_rows": int(len(te)), "test_positives": int(te.label.sum()),
           "base_rate": float(te.label.mean()), "models": {}, "monthly": {}, "pr_curves": {}}
    for key, col in SCORE_COLS.items():
        pk = precision_at_k(te, col, k)
        y, s = te.label.to_numpy(), te[col].to_numpy()
        prec, rec, _ = precision_recall_curve(y, s)
        idx = np.unique(np.linspace(0, len(prec) - 1, 250).astype(int))
        out["pr_curves"][key] = {"precision": prec[idx].round(4).tolist(), "recall": rec[idx].round(4).tolist()}
        out["monthly"][key] = pk.round(4).to_dict(orient="records")
        out["models"][key] = {
            "name": MODEL_NAMES[key],
            "precision_at_capacity": float(pk.precision.mean()),
            "recall_at_capacity": float(pk.recall.mean()),
            "pr_auc": float(average_precision_score(y, s)),
            "roc_auc": float(roc_auc_score(y, s)),
            "true_positives_found": int(pk.true_positives.sum()),
            "investigations": int(pk.k.sum()),
        }
    # Raw rule alert stream (what the team would face without prioritisation)
    alerts = te[te.rule_alert == 1]
    out["rules_alert_stream"] = {
        "alerts": int(len(alerts)),
        "alerts_per_month": float(len(alerts) / te.month.nunique()),
        "precision": float(alerts.label.mean()) if len(alerts) else 0.0,
        "recall": float(alerts.label.sum() / max(te.label.sum(), 1)),
        "false_positive_rate": float(1 - alerts.label.mean()) if len(alerts) else 0.0,
    }
    # Network coverage: ring-months (an organised ring active in a month) with at
    # least one genuinely involved member inside that month's queue
    nets = te[(te.label == 1) & te.label_network.str.startswith("NET")]
    active = set(zip(nets.label_network, nets.month))
    covered = {}
    for key, col in SCORE_COLS.items():
        q = top_k_per_month(te, col, k)
        q = q[(q.label == 1) & q.label_network.str.startswith("NET")]
        hit = set(zip(q.label_network, q.month))
        covered[key] = {"ring_months_active": len(active), "ring_months_detected": len(active & hit),
                        "coverage": len(active & hit) / max(len(active), 1)}
    out["ring_coverage"] = covered
    # Per-typology recall at capacity for the deployed hybrid
    q = top_k_per_month(te, "score_hybrid", k)
    typ = te[te.label == 1].groupby("label_typology").size().rename("positives").to_frame()
    typ["found"] = q[q.label == 1].groupby("label_typology").size()
    typ = typ.fillna(0)
    typ["recall_at_capacity"] = typ.found / typ.positives
    out["typology_recall"] = typ.reset_index().to_dict(orient="records")
    return out
