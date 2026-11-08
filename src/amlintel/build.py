"""End-to-end build: data -> features -> graph -> patterns -> models -> cases.

Usage::

    python -m amlintel.build --profile demo

Writes every artifact the dashboard needs into ``data/<profile>/``.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import pandas as pd

from amlintel import __version__
from amlintel.cases.queue import build_queue, seed_case_db
from amlintel.config import ModelConfig, Profile, get_profile
from amlintel.data.generator import generate
from amlintel.data.pipeline import add_month, attach_labels, build_account_month_features
from amlintel.detection import typologies as T
from amlintel.detection.explain import explain_rows, global_importance
from amlintel.detection.models import FEATURES, evaluate, fit_and_score, top_k_per_month
from amlintel.detection.rules import RULE_INFO, RULES, apply_rules
from amlintel.graph.network import monthly_graph_features
from amlintel.simulation import Policy, capacity_curve

SCORED_KEEP = [
    "account_id", "month", "month_idx", "split", "label", "label_typology", "label_network",
    "customer_type", "business_kind", "country", "kyc_risk", "is_business",
    "score_rules", "score_iforest", "score_lgbm", "score_network", "score_hybrid", "risk_score",
    "r_lgbm", "r_network", "r_rules", "r_anomaly", "rule_score", "rule_alert", "rules_fired",
    *[r.rule_id for r in RULES],
]


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def build(profile: Profile, cfg: ModelConfig | None = None, out_dir: Path | None = None) -> dict:
    # Investigation capacity scales with the customer base (50 cases/month per 6,000 accounts).
    # Large profiles fit LightGBM on 20% of training negatives (all positives kept).
    cfg = cfg or ModelConfig(capacity_per_month=max(10, round(50 * profile.n_accounts / 6_000)),
                             neg_sample_rate=1.0 if profile.n_accounts <= 20_000 else 0.2)
    out = Path(out_dir or profile.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1. data ----------------------------------------------------------------
    accounts, tx = generate(profile)
    tx = add_month(tx, profile.start_date)
    log(f"generated {len(accounts):,} accounts, {len(tx):,} transactions", t0)

    # 2. features (analysis window only; history gives context) --------------
    feats = build_account_month_features(tx, accounts, profile.start_date)
    feats = feats[feats.month_idx >= 0]
    win = tx[tx.month_idx >= 0]
    kyc = dict(zip(accounts.account_id, accounts.kyc_risk.map({"low": 0, "medium": 1, "high": 2})))
    gfeat = monthly_graph_features(win, kyc)
    log(f"graph features for {gfeat.month.nunique()} monthly graphs", t0)

    # 3. pattern discovery -----------------------------------------------------
    ind = T.detect_all(tx)
    ind = ind[ind.start >= pd.Timestamp(profile.start_date)].reset_index(drop=True)
    ind_eval = T.evaluate_indicators(ind, win)
    ifeat = T.indicator_account_features(ind)
    log(f"{len(ind):,} pattern indicators", t0)

    df = (feats.merge(gfeat, on=["account_id", "month"], how="left")
               .merge(ifeat, on=["account_id", "month"], how="left"))
    for c in FEATURES:
        if c not in df:
            df[c] = 0
    df[FEATURES] = df[FEATURES].fillna(0)
    df["scc_size"] = df["scc_size"].clip(lower=1)
    # DuckDB/Polars group-bys do not guarantee row order; fix it for reproducibility
    df = df.sort_values(["month", "account_id"], kind="stable").reset_index(drop=True)
    df = attach_labels(df, win)
    df = pd.concat([df, apply_rules(df)], axis=1)
    df[FEATURES] = df[FEATURES].astype("float32")      # halves memory at production scale

    # Write raw data now and keep only what later steps need: at 100k accounts the
    # transaction table is the largest object in memory.
    counts = {
        "accounts": int(len(accounts)), "transactions_total": int(len(tx)),
        "transactions_analysis_window": int(len(win)),
        "history_months": profile.history_months,
        "laundering_transactions": int(win.is_laundering.sum()),
        "laundering_instances": int(win.loc[win.is_laundering == 1, "instance_id"].nunique()),
        "organised_rings": int(win.loc[win.network_id.str.startswith("NET"), "network_id"].nunique()),
    }
    window = {"start": str(win.timestamp.min()), "end": str(win.timestamp.max()),
              "months": sorted(win.month.unique().tolist())}
    accounts.to_parquet(out / "accounts.parquet", index=False)
    tx_cols = ["txn_id", "timestamp", "month", "month_idx", "from_account", "to_account", "amount",
               "channel", "from_country", "to_country", "is_cross_border",
               "is_laundering", "typology", "instance_id", "network_id"]
    tx[tx_cols].to_parquet(out / "transactions.parquet", index=False, compression="zstd")
    first_test = cfg.train_months + cfg.valid_months
    tx_queue = win.loc[win.month_idx >= first_test,
                       ["from_account", "to_account", "amount", "month", "timestamp"]].copy()
    del tx, win, feats, gfeat, ifeat

    # 4. models -----------------------------------------------------------------
    scored, models = fit_and_score(df, cfg)
    metrics = evaluate(scored, cfg)
    metrics["hybrid_weights"] = models.hybrid_weights
    metrics["lgbm_best_iteration"] = models.best_iteration
    metrics["lgbm_valid_pr_auc"] = models.valid_ap
    log(f"models trained (best_it={models.best_iteration}, weights={models.hybrid_weights})", t0)

    # 5. explainability ----------------------------------------------------------
    test = scored[scored.split == "test"]
    queue_rows = top_k_per_month(test, "score_hybrid", cfg.capacity_per_month * 2)
    drivers = explain_rows(models.lgbm, queue_rows)
    sample = test.sample(min(6000, len(test)), random_state=cfg.random_state)
    gimp = global_importance(models.lgbm, sample)
    log("SHAP explanations computed", t0)

    # 6. case queue + seeded history ------------------------------------------
    months = sorted(test.month.unique())
    queue = build_queue(test, tx_queue, ind, cfg.capacity_per_month, months)
    seed_case_db(out / "cases.db", queue, drivers, live_month=months[-1], seed=cfg.random_state)
    log(f"case database seeded with {len(queue):,} cases", t0)

    # 7. governance views --------------------------------------------------------
    w = models.hybrid_weights
    pol = Policy(capacity=cfg.capacity_per_month, w_lgbm=w["lgbm"], w_network=w["network"],
                 w_rules=w["rules"], w_anomaly=w["anomaly"])
    cap = capacity_curve(test, pol)

    # 8. write artifacts -----------------------------------------------------------
    keep = [c for c in SCORED_KEEP if c in scored] + [c for c in FEATURES if c not in SCORED_KEEP]
    sc = scored[keep].copy()
    fcols = sc.select_dtypes("float64").columns
    sc[fcols] = sc[fcols].astype("float32")
    sc.to_parquet(out / "scored.parquet", index=False, compression="zstd")
    ind.to_parquet(out / "indicators.parquet", index=False, compression="zstd")
    ind_eval.to_csv(out / "indicator_eval.csv", index=False)
    drivers.to_parquet(out / "shap_alerts.parquet", index=False)
    gimp.to_csv(out / "shap_global.csv", index=False)
    cap.to_csv(out / "capacity_curve.csv", index=False)
    RULE_INFO.to_csv(out / "rules.csv", index=False)
    models.lgbm.save_model(str(out / "lgbm_model.txt"))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    manifest = {
        "project": "AML Financial Crime Intelligence Platform",
        "package_version": __version__,
        "profile": profile.__dict__ | {"ring_size": list(profile.ring_size),
                                       "instances_per_ring": list(profile.instances_per_ring)},
        "model_config": cfg.to_dict(),
        "data_provenance": (
            "Fully synthetic, generated by amlintel.data.generator (AMLSim-style: simulated "
            "background behaviour plus injected laundering typologies with ground-truth labels). "
            "No real customer or bank data is used."),
        "counts": counts | {"account_months": int(len(scored)), "indicators": int(len(ind)),
                            "cases": int(len(queue))},
        "window": window,
        "build_seconds": round(time.time() - t0, 1),
        "python": platform.python_version(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    log(f"artifacts written to {out}", t0)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="demo")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    m = build(get_profile(a.profile), out_dir=Path(a.out) if a.out else None)
    print(json.dumps(m["counts"], indent=2))


if __name__ == "__main__":
    main()
