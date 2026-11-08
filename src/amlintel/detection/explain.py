"""SHAP explanations for the LightGBM risk model.

Investigators need to know *why* an account was flagged.  For each queued alert
we store the top drivers with their actual values, phrased so they can be pasted
into a case note.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from amlintel.detection.models import FEATURE_LABELS, FEATURES


def explain_rows(model: lgb.Booster, rows: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=["account_id", "month", "rank", "feature", "label",
                                     "value", "shap", "direction"])
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(rows[FEATURES])
    if isinstance(sv, list):  # older shap returns [neg, pos]
        sv = sv[1]
    out = []
    vals = rows[FEATURES].to_numpy(float)
    for i, (acct, month) in enumerate(zip(rows.account_id, rows.month)):
        order = np.argsort(-np.abs(sv[i]))[:top_n]
        for r, j in enumerate(order, 1):
            out.append((acct, month, r, FEATURES[j], FEATURE_LABELS.get(FEATURES[j], FEATURES[j]),
                        float(vals[i, j]), float(sv[i, j]),
                        "raises risk" if sv[i, j] > 0 else "lowers risk"))
    return pd.DataFrame(out, columns=["account_id", "month", "rank", "feature", "label",
                                      "value", "shap", "direction"])


def global_importance(model: lgb.Booster, sample: pd.DataFrame) -> pd.DataFrame:
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(sample[FEATURES])
    if isinstance(sv, list):
        sv = sv[1]
    imp = pd.DataFrame({
        "feature": FEATURES,
        "label": [FEATURE_LABELS.get(f, f) for f in FEATURES],
        "mean_abs_shap": np.abs(sv).mean(axis=0),
        # sign of correlation between feature value and its SHAP effect
        "direction": [float(np.corrcoef(sample[f].rank(), sv[:, i])[0, 1])
                      if sample[f].nunique() > 1 and np.std(sv[:, i]) > 0 else 0.0
                      for i, f in enumerate(FEATURES)],
    })
    return imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
