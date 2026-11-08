"""06 · Model Performance & Explainability."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import MODEL_COLORS, MODEL_SHORT, SERIES, TYP_LABEL, load, load_json, pct, setup, synthetic_badge

setup("Model Performance & Explainability",
      "Out-of-time evaluation, explainability and the model card that governs how the scores may be used.")
synthetic_badge()

m = load_json("metrics.json")
man = load_json("manifest.json")
K = m["capacity_per_month"]
cfg = man["model_config"]
order = ["hybrid", "lgbm", "anomaly", "rules"]

st.markdown(
    f"Trained on months 1–{cfg['train_months']}, tuned on the next {cfg['valid_months']}, and tested on "
    f"**{len(m['test_months'])} later months the models never saw** ({m['test_months'][0]} to {m['test_months'][-1]}). "
    f"Test set: {m['test_rows']:,} account-months, {m['test_positives']:,} suspicious "
    f"({pct(m['base_rate'], 1)}). Training labels cover only {pct(cfg['label_coverage'])} of true cases, "
    "as real investigation labels would.")

tbl = pd.DataFrame([{
    "Approach": MODEL_SHORT[k] if k != "hybrid" else "Deployed priority score",
    f"Precision @ {K}/month": m["models"][k]["precision_at_capacity"],
    f"Recall @ {K}/month": m["models"][k]["recall_at_capacity"],
    "PR-AUC": m["models"][k]["pr_auc"], "ROC-AUC": m["models"][k]["roc_auc"],
    "Suspicious found": m["models"][k]["true_positives_found"],
    "Ring-months caught": f"{m['ring_coverage'][k]['ring_months_detected']} / {m['ring_coverage'][k]['ring_months_active']}",
} for k in order])
st.dataframe(tbl, hide_index=True, width="stretch",
             column_config={c: st.column_config.NumberColumn(format="%.3f") for c in tbl.columns[1:5]})
ras = m["rules_alert_stream"]
st.caption(f"Working every rule alert instead would mean {ras['alerts']:,} reviews "
           f"({ras['alerts_per_month']:.0f}/month) at {pct(ras['precision'], 1)} precision, still catching only "
           f"{pct(ras['recall'])} of suspicious account-months.")

left, right = st.columns(2)
with left:
    st.subheader("Precision–recall curves")
    fig = go.Figure()
    for k in order:
        c = m["pr_curves"][k]
        fig.add_trace(go.Scatter(x=c["recall"], y=c["precision"], mode="lines", name=MODEL_SHORT[k],
                                 line=dict(color=MODEL_COLORS[k], width=2),
                                 hovertemplate=MODEL_SHORT[k] + "<br>recall %{x:.2f} · precision %{y:.2f}<extra></extra>"))
    fig.add_hline(y=m["base_rate"], line=dict(color="#898781", dash="dot", width=1),
                  annotation_text="base rate", annotation_font_color="#52514e")
    fig.update_layout(height=360, xaxis=dict(title="Recall", range=[0, 1]), yaxis=dict(title="Precision", range=[0, 1.02]))
    st.plotly_chart(fig, width="stretch")

with right:
    st.subheader("Precision at capacity, month by month")
    fig = go.Figure()
    for k in order:
        mo = pd.DataFrame(m["monthly"][k])
        fig.add_trace(go.Scatter(x=mo.month, y=mo.precision, mode="lines+markers", name=MODEL_SHORT[k],
                                 line=dict(color=MODEL_COLORS[k], width=2), marker=dict(size=8),
                                 hovertemplate=MODEL_SHORT[k] + " · %{x}<br>%{y:.0%}<extra></extra>"))
    best = pd.DataFrame(m["monthly"]["hybrid"])
    fig.add_trace(go.Scatter(x=best.month, y=(best.positives.clip(upper=K) / K), mode="lines", name="Best possible",
                             line=dict(color="#898781", width=2, dash="dot"),
                             hovertemplate="%{x}<br>best possible %{y:.0%}<extra></extra>"))
    fig.update_layout(height=360, yaxis=dict(tickformat=".0%", range=[0, 1.02], title=f"Share of top {K} that are suspicious"))
    st.plotly_chart(fig, width="stretch")
    st.caption("In months with fewer than K suspicious accounts, even a perfect ranking cannot fill the queue "
               "(dotted line).")

left, right = st.columns(2)
with left:
    st.subheader("What drives the risk model")
    g = load("shap_global.csv").head(15).iloc[::-1]
    fig = go.Figure(go.Bar(x=g.mean_abs_shap, y=g.label, orientation="h", marker_color=SERIES[0],
                           customdata=g.direction,
                           hovertemplate="%{y}<br>mean |SHAP| %{x:.3f}<br>direction %{customdata:+.2f}<extra></extra>"))
    fig.update_layout(height=470, xaxis_title="Mean |SHAP| on test months")
    st.plotly_chart(fig, width="stretch")
    st.caption("Network features (centrality, links to collector/distributor accounts, circular flows) sit beside "
               "classic flow features. That is the point of moving from transaction-level to network-level monitoring.")

with right:
    st.subheader("Pattern detector quality")
    ie = load("indicator_eval.csv")
    ie["Pattern"] = ie.typology.map(TYP_LABEL)
    st.dataframe(ie[["Pattern", "indicators", "precision", "instances_recovered", "instances_injected", "instance_recall"]],
                 hide_index=True, width="stretch",
                 column_config={"indicators": "Indicators raised",
                                "precision": st.column_config.ProgressColumn("Precision", min_value=0, max_value=1, format="%.2f"),
                                "instances_recovered": "Schemes recovered", "instances_injected": "Schemes injected",
                                "instance_recall": st.column_config.ProgressColumn("Recall", min_value=0, max_value=1, format="%.2f")})
    st.caption("Circular-flow and fan-out detectors fire on legitimate intercompany treasury loops and payroll-like "
               "payments, which is exactly why indicators feed the model as evidence instead of creating cases on their own.")
    st.subheader("Recall by pattern (deployed queue)")
    tr = pd.DataFrame(m["typology_recall"])
    tr["Pattern"] = tr.label_typology.map(TYP_LABEL)
    st.dataframe(tr[["Pattern", "positives", "found", "recall_at_capacity"]], hide_index=True, width="stretch",
                 column_config={"positives": "Suspicious", "found": "In queue",
                                "recall_at_capacity": st.column_config.ProgressColumn("Recall", min_value=0, max_value=1, format="%.2f")})

st.divider()
st.subheader("Model card")
w = m["hybrid_weights"]
a, b = st.columns(2)
with a:
    st.markdown(f"""
**Intended use.** Rank account-months for human review within a fixed monthly capacity. The score decides
*what to look at first*; it never decides that an account is laundering money, and never triggers
automatic action against a customer.

**Model.** LightGBM gradient boosting ({m['lgbm_best_iteration']} trees, early stopping on validation),
{len(load('shap_global.csv'))} features: flow volumes, cash and threshold behaviour, cross-border and high-risk
jurisdiction exposure, 72-hour fan bursts, 24-hour pass-through, monthly graph features (PageRank, circular-flow
components, short cycles, links to collector/distributor accounts) and pattern-indicator counts.
Deployed blend: LightGBM {w['lgbm']:.0%}, network {w['network']:.0%}, rules {w['rules']:.0%}, anomaly {w['anomaly']:.0%}.

**Data.** {man['data_provenance']}

**Validation.** Time-based split (no shuffling), precision and recall at fixed investigation capacity,
PR-AUC, ring-level coverage, per-pattern recall. Validation PR-AUC {m['lgbm_valid_pr_auc']:.3f}.
""")
with b:
    st.markdown("""
**Known limitations**
- Synthetic data is cleaner than reality. Real hit rates will be much lower; use these numbers to compare
  approaches, not to promise outcomes.
- Labels in production come from past investigations, so the model learns what investigators already
  recognise. Isolation Forest and the pattern detectors remain in the pipeline to surface new behaviour.
- Monthly windows miss schemes that straddle month boundaries (the cycle detector allows a two-week spill-over).
- Fan-in/fan-out mules who each make one modest payment are the main recall gap.
- Business type and KYC level are features; monitor outcomes by segment so that no customer group is
  over-investigated for who they are rather than what they do.

**Monitoring and change control.** Re-validate monthly on closed cases (precision at capacity, alert
volume, segment mix); retrain quarterly or on drift; any threshold or weight change is made on the Impact
Simulator page, exported as a policy record and approved before release.
""")
