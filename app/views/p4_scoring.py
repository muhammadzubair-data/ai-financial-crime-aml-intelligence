"""04 · AI Detection & Risk Scoring."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from amlintel.detection.models import FEATURE_LABELS, FEATURES
from common import MODEL_COLORS, MODEL_SHORT, SERIES, STATUS, lgbm_model, load_json, scored, setup, synthetic_badge

setup("AI Detection & Risk Scoring",
      "How each detection approach ranks the same accounts, and why the model scores an account the way it does.")
synthetic_badge()

sc = scored()
m = load_json("metrics.json")
K = m["capacity_per_month"]
months = sorted(sc.month.unique())
w = m["hybrid_weights"]

st.markdown(
    f"**Deployed priority score** = {w['lgbm']:.0%} LightGBM risk"
    + "".join(f" + {w[k]:.0%} {n}" for k, n in (("network", "network exposure"), ("rules", "rule score"),
                                                 ("anomaly", "Isolation Forest anomaly")) if w[k] > 0)
    + ", each converted to a within-month percentile. Weights were chosen on validation months only; "
      "a blend had to beat the pure model there by at least one point to be used.")

c1, c2 = st.columns([1, 3])
month = c1.selectbox("Month", months, index=len(months) - 1)
sm = sc[sc.month == month].copy()
split = sm.split.iloc[0]
c2.markdown(f'<br><span class="pill">{split} period</span> '
            + ("<span class='note'>Scores in training months are in-sample; judge models on test months.</span>"
               if split == "train" else ""), unsafe_allow_html=True)

cols = {"hybrid": "score_hybrid", "lgbm": "score_lgbm", "anomaly": "score_iforest", "rules": "score_rules"}
tops = {k: set(sm.nlargest(K, c).account_id) for k, c in cols.items()}
c = st.columns(4)
for i, (k, col) in enumerate(cols.items()):
    hit = sm[sm.account_id.isin(tops[k])].label.sum()
    c[i].markdown(f'<div class="kpi"><div class="lbl"><span style="color:{MODEL_COLORS[k]}">■</span> '
                  f'{MODEL_SHORT[k]} · top {K}</div><div class="val">{hit} / {K}</div>'
                  f'<div class="sub">genuinely suspicious (synthetic truth)</div></div>', unsafe_allow_html=True)

st.write("")
left, right = st.columns([1, 1])
with left:
    st.subheader("Do the approaches agree?")
    keys = list(cols)
    mat = np.array([[len(tops[a] & tops[b]) for b in keys] for a in keys])
    fig = go.Figure(go.Heatmap(z=mat, x=[MODEL_SHORT[k] for k in keys], y=[MODEL_SHORT[k] for k in keys],
                               colorscale=[[0, "#f0efec"], [1, "#256abf"]], showscale=False,
                               text=mat, texttemplate="%{text}", textfont=dict(size=14),
                               hovertemplate="%{y} ∩ %{x}: %{z} shared accounts<extra></extra>", xgap=2, ygap=2))
    fig.update_layout(height=330, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Accounts shared between each pair of top-{K} lists. Rules and the model mostly disagree: "
               "rules key on single thresholds, the model on combinations of flow and network behaviour.")

with right:
    st.subheader("Hit rate by priority band")
    bands = [(99.5, 100.01, "Top 0.5%"), (99, 99.5, "0.5–1%"), (98, 99, "1–2%"), (95, 98, "2–5%"),
             (90, 95, "5–10%"), (0, 90, "Rest")]
    rows = []
    for lo, hi, name in bands:
        g = sm[(sm.risk_score >= lo) & (sm.risk_score < hi)]
        rows.append((name, len(g), g.label.mean() if len(g) else 0))
    bd = pd.DataFrame(rows, columns=["band", "n", "rate"])
    fig = go.Figure(go.Bar(x=bd.band, y=bd.rate, marker_color=SERIES[0],
                           text=[f"{r:.0%}" for r in bd.rate], textposition="outside",
                           customdata=bd.n,
                           hovertemplate="%{x}<br>%{y:.1%} suspicious of %{customdata:,} accounts<extra></extra>"))
    fig.update_layout(height=330, yaxis=dict(tickformat=".0%", range=[0, 1.1], title="Share suspicious (synthetic truth)"),
                      xaxis_title="Priority score band (within month)")
    st.plotly_chart(fig, width="stretch")
    st.caption("Suspicious activity concentrates in the top bands; below the top 2% it is close to the base rate.")

st.divider()
st.subheader("Top-ranked accounts this month")
view = sm.nlargest(150, "score_hybrid")[["account_id", "risk_score", "score_lgbm", "r_network", "r_anomaly",
                                         "rule_score", "rules_fired", "customer_type", "business_kind",
                                         "in_amt", "out_amt", "label", "label_typology"]]
sel = st.dataframe(
    view, hide_index=True, width="stretch", height=360, on_select="rerun", selection_mode="single-row",
    column_config={
        "risk_score": st.column_config.ProgressColumn("Priority (pct)", min_value=0, max_value=100, format="%.1f"),
        "score_lgbm": st.column_config.NumberColumn("LightGBM", format="%.3f"),
        "r_network": st.column_config.NumberColumn("Network pct", format="%.2f"),
        "r_anomaly": st.column_config.NumberColumn("Anomaly pct", format="%.2f"),
        "rule_score": st.column_config.NumberColumn("Rule score", format="%.1f"),
        "in_amt": st.column_config.NumberColumn("In", format="localized"),
        "out_amt": st.column_config.NumberColumn("Out", format="localized"),
        "label": st.column_config.CheckboxColumn("Truth*"),
        "label_typology": "Truth pattern*",
    })
st.caption("*Synthetic ground truth, shown for evaluation only.")

acct = view.iloc[sel.selection.rows[0]].account_id if sel.selection.rows else view.iloc[0].account_id
acct = st.text_input("Explain account", value=acct).strip().upper()
row = sm[sm.account_id == acct]
if row.empty:
    st.info("That account has no activity in the selected month.")
    st.stop()

left, right = st.columns([1.3, 1])
with left:
    st.subheader(f"Why {acct} scored {row.risk_score.iloc[0]:.1f}")
    import shap  # local import: heavy
    ex = shap.TreeExplainer(lgbm_model())
    sv = ex.shap_values(row[FEATURES])
    sv = sv[1] if isinstance(sv, list) else sv
    contrib = pd.DataFrame({"feature": FEATURES, "shap": sv[0], "value": row[FEATURES].to_numpy()[0]})
    contrib["label"] = contrib.feature.map(FEATURE_LABELS)
    top = contrib.reindex(contrib.shap.abs().sort_values(ascending=False).index).head(10).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top.shap, y=[f"{l} = {v:,.2f}" if abs(v) < 100 else f"{l} = {v:,.0f}" for l, v in zip(top.label, top.value)],
        orientation="h", marker_color=[STATUS["critical"] if s > 0 else SERIES[0] for s in top.shap],
        hovertemplate="%{y}<br>effect on log-odds: %{x:+.3f}<extra></extra>"))
    fig.update_layout(height=380, xaxis_title="Contribution to risk (log-odds)")
    st.plotly_chart(fig, width="stretch")
    st.caption("Red bars push risk up, blue bars pull it down. Values are the account's actual figures for the month.")

with right:
    st.subheader("Risk over time")
    hist = sc[sc.account_id == acct].sort_values("month")
    fig = go.Figure(go.Scatter(x=hist.month, y=hist.risk_score, mode="lines+markers",
                               line=dict(color=SERIES[0], width=2), marker=dict(size=8),
                               hovertemplate="%{x}: %{y:.1f}<extra></extra>"))
    fig.add_hline(y=99, line=dict(color="#898781", dash="dot", width=1), annotation_text="top 1%",
                  annotation_font_color="#52514e")
    fig.update_layout(height=260, yaxis=dict(range=[0, 101], title="Priority percentile"))
    st.plotly_chart(fig, width="stretch")
    r = row.iloc[0]
    st.markdown(f"**Profile:** {r.customer_type}, {r.business_kind}, {r.country}, KYC {r.kyc_risk}")
    st.markdown(f"**Rules fired:** {r.rules_fired or 'none'}")
