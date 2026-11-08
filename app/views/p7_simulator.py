"""07 · Compliance & Business Impact Simulator."""
import json
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from amlintel.simulation import Policy, capacity_curve, simulate
from common import SERIES, current_user, kpi, load, load_json, money, pct, scored, setup, synthetic_badge

setup("Compliance & Business Impact Simulator",
      "Change capacity, thresholds, model weights and rules, and see the modelled trade-offs before approving a change.")
synthetic_badge()

m = load_json("metrics.json")
sc = scored()
test = sc[sc.split == "test"]
rules = load("rules.csv")
w0 = m["hybrid_weights"]

with st.sidebar:
    st.markdown("### Policy levers")
    cap = st.slider("Investigations per month", 10, 300, m["capacity_per_month"], step=5)
    thr = st.slider("Alert threshold (priority percentile)", 90.0, 99.9, 99.0, step=0.1,
                    help="Account-months at or above this percentile become alerts. Alerts beyond capacity "
                         "wait in the backlog; below the threshold nothing is opened, even if capacity is free.")
    st.markdown("**Blend weights**")
    wl = st.slider("LightGBM risk", 0.0, 1.0, float(w0["lgbm"]), 0.05)
    wn = st.slider("Network exposure", 0.0, 1.0, float(w0["network"]), 0.05)
    wr = st.slider("Rule score", 0.0, 1.0, float(w0["rules"]), 0.05)
    wa = st.slider("Isolation Forest anomaly", 0.0, 1.0, float(w0["anomaly"]), 0.05)
    st.markdown("**Active scenario rules**")
    active = tuple(r.rule_id for r in rules.itertuples()
                   if st.checkbox(f"{r.rule_id} · {r.name}", value=True, key=f"rule_{r.rule_id}"))
    st.markdown("**Effort and cost**")
    hpc = st.number_input("Hours per investigation", 0.5, 20.0, 3.0, 0.5)
    hpa = st.number_input("Hours to triage one rule alert", 0.25, 10.0, 1.5, 0.25)
    cph = st.number_input("Loaded cost per analyst hour", 10.0, 300.0, 45.0, 5.0)

if wl + wn + wr + wa == 0:
    st.warning("All blend weights are zero. Set at least one weight above zero.")
    st.stop()

pol = Policy(capacity=cap, min_risk_score=thr, w_lgbm=wl, w_network=wn, w_rules=wr, w_anomaly=wa,
             active_rules=active, hours_per_case=hpc, hours_per_rule_alert=hpa, cost_per_hour=cph)
base = Policy(capacity=m["capacity_per_month"], min_risk_score=99.0, w_lgbm=w0["lgbm"], w_network=w0["network"],
              w_rules=w0["rules"], w_anomaly=w0["anomaly"], hours_per_case=hpc,
              hours_per_rule_alert=hpa, cost_per_hour=cph)


@st.cache_data(show_spinner=False)
def run(p: Policy):
    return simulate(test, p)


mo, s = run(pol)
_, b = run(base)


def delta(x, y, fmt):
    d = x - y
    return "no change vs approved policy" if abs(d) < 1e-9 else f"{'+' if d > 0 else '−'}{fmt(abs(d))} vs approved policy"


c = st.columns(4)
kpi(c[0], "Hit rate of opened cases", pct(s["precision"]), delta(s["precision"], b["precision"], lambda v: f"{100 * v:.1f} pts"))
kpi(c[1], "Share of suspicious activity caught", pct(s["recall"]), delta(s["recall"], b["recall"], lambda v: f"{100 * v:.1f} pts"))
kpi(c[2], "Suspicious accounts found", f"{s['true_positives']}", delta(s["true_positives"], b["true_positives"], lambda v: f"{v:.0f}"))
kpi(c[3], "Suspicious left unreviewed", f"{s['missed']}", f"of {s['true_positives'] + s['missed']} in {len(mo)} months")
st.write("")
c = st.columns(4)
kpi(c[0], "Investigations opened", f"{s['investigated']:,}", f"alerts left unworked: {s['avg_backlog']:,.0f}/month on average")
kpi(c[1], "Analyst hours", f"{s['hours']:,.0f} h", f"{s['hours'] / len(mo):,.0f} h per month")
kpi(c[2], "Modelled cost", money(s["cost"]), f"{len(mo)} months")
kpi(c[3], "Cost per suspicious account found", money(s["cost_per_true_positive"]))

st.divider()
left, right = st.columns(2)
with left:
    st.subheader("Capacity trade-off")
    cc = capacity_curve(test, pol)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cc.capacity, y=cc.precision, mode="lines+markers", name="Hit rate (precision)",
                             line=dict(color=SERIES[0], width=2), marker=dict(size=8),
                             hovertemplate="%{x} cases/month<br>precision %{y:.0%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=cc.capacity, y=cc.recall, mode="lines+markers", name="Share caught (recall)",
                             line=dict(color=SERIES[1], width=2), marker=dict(size=8),
                             hovertemplate="%{x} cases/month<br>recall %{y:.0%}<extra></extra>"))
    fig.add_vline(x=cap, line=dict(color="#898781", dash="dot", width=1))
    fig.update_layout(height=340, xaxis_title="Investigations per month", yaxis=dict(tickformat=".0%", range=[0, 1.02]))
    st.plotly_chart(fig, width="stretch")
    st.caption("More capacity catches more, with a falling hit rate per case (alert threshold ignored here so only "
               "capacity changes). The dotted line is your current setting.")

with right:
    st.subheader("Against rule-only monitoring")
    comp = pd.DataFrame([
        {"Approach": "This policy (ranked queue)", "Reviews": s["investigated"], "Hours": s["hours"],
         "Cost": s["cost"], "Share caught": s["recall"]},
        {"Approach": "Rules, work every alert", "Reviews": s["rule_alerts"], "Hours": s["rule_hours_all_alerts"],
         "Cost": s["rule_cost_all_alerts"], "Share caught": s["rule_recall_all_alerts"]},
        {"Approach": f"Rules, top {cap} alerts by rule score", "Reviews": s["investigated"], "Hours": s["hours"],
         "Cost": s["cost"], "Share caught": s["rule_recall_within_capacity"]},
    ])
    st.dataframe(comp, hide_index=True, width="stretch",
                 column_config={"Hours": st.column_config.NumberColumn(format="localized"),
                                "Cost": st.column_config.NumberColumn(format="localized"),
                                "Share caught": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
    st.caption("Rule alerts here use only the rules switched on in the sidebar.")

st.subheader("Month by month")
st.dataframe(mo, hide_index=True, width="stretch",
             column_config={"month": "Month", "positives": "Suspicious (truth)", "eligible_alerts": "Alerts",
                            "investigated": "Opened", "backlog": "Unworked", "true_positives": "Suspicious found",
                            "precision": st.column_config.NumberColumn("Hit rate", format="%.2f"),
                            "recall": st.column_config.NumberColumn("Share caught", format="%.2f"),
                            "rule_alerts": "Rule alerts", "rule_tp_within_capacity": "Rules: found in top K",
                            "rule_tp_all": "Rules: found in all alerts"})

st.divider()
st.subheader("Governance record")
st.caption("Changes are never applied from this page. Export the proposed policy with its modelled impact, "
           "attach it to a change request, and apply it only after approval.")
record = {
    "proposed_by": current_user(), "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "policy": {**pol.__dict__, "active_rules": list(pol.active_rules)},
    "approved_policy_baseline": {**base.__dict__, "active_rules": list(base.active_rules)},
    "modelled_impact_on_test_months": s, "baseline_impact": b,
    "evaluation_months": m["test_months"],
    "caveat": "Modelled on synthetic data with complete ground truth; real-world performance will differ.",
}
st.download_button("Export policy change record (JSON)", json.dumps(record, indent=2, default=float),
                   "aml_policy_change.json", "application/json")
