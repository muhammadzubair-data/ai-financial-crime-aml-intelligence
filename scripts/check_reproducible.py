"""Compare two builds of the same profile (used in CI).

    python scripts/check_reproducible.py <dir_a> <dir_b>                # must match exactly
    python scripts/check_reproducible.py <dir_a> <dir_b> --tolerance 0.02
        # headline metrics within tolerance (for comparing builds from different machines,
        # where floating-point differences in the CPU's maths libraries are expected)
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("a")
ap.add_argument("b")
ap.add_argument("--tolerance", type=float, default=None)
args = ap.parse_args()
a, b = Path(args.a), Path(args.b)
ma, mb = (json.loads((d / "metrics.json").read_text()) for d in (a, b))
problems = []

if args.tolerance is None:
    if ma["models"] != mb["models"]:
        problems.append("metrics.json model results differ")
    if ma["hybrid_weights"] != mb["hybrid_weights"]:
        problems.append("hybrid weights differ")
    for f in ("transactions.parquet", "indicators.parquet", "scored.parquet"):
        if not pd.read_parquet(a / f).equals(pd.read_parquet(b / f)):
            problems.append(f"{f} differs")
    ok = "Reproducible: metrics, transactions, indicators and scores match exactly."
else:
    if not pd.read_parquet(a / "transactions.parquet").equals(pd.read_parquet(b / "transactions.parquet")):
        problems.append("transactions differ (the generator must be exact on any machine)")
    for model, va in ma["models"].items():
        vb = mb["models"][model]
        for k in ("precision_at_capacity", "recall_at_capacity", "pr_auc"):
            if abs(va[k] - vb[k]) > args.tolerance:
                problems.append(f"{model}.{k}: {va[k]:.4f} vs {vb[k]:.4f}")
    ok = f"Consistent: identical transactions, headline metrics within {args.tolerance}."

if problems:
    raise SystemExit("NOT reproducible: " + "; ".join(problems))
print(ok)
