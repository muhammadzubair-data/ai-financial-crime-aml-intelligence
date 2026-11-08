"""01 · Executive Financial Crime Overview."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (
    MODEL_COLORS,
    SERIES,
    TYP_LABEL,
    kpi,
    load_json,
    pct,
    scored,
    setup,
    store,
    synthetic_badge,
    test_months,
)

setup("Executive Financial Crime Overview",
      "Alert volumes, detection quality and investigator workload for the out-of-time evaluation period.")
synthetic_badge()

m = load_json("metrics.json")
man = load_json("manifest.json")
months = test_months()
sc = scored()
te = sc[sc.month.isin(months)]
K = m["capacity_per_month"]
dep, rules = m["models"]["hybrid"], m["models"]["rules"]
ras = m["rules_alert_stream"]
cases = store().cases()

# --- headline KPIs -----------------------------------------------------------------------
c = st.columns(5)
kpi(c[0], "Account-months monitored", f"{len(te):,}", f"{months[0]} to {months[-1]}")
kpi(c[1], "Rule alerts per month", f"{ras['alerts_per_month']:,.0f}",
    f"{pct(ras['false_positive_rate'])} were false positives")
kpi(c[2], "Cases opened per month", f"{K}", "team capacity, ranked by priority score")
kpi(c[3], "Suspicious hit rate in queue", pct(dep["precision_at_capacity"], 1),
    f"vs {pct(rules['precision_at_capacity'], 1)} for rules at the same capacity")
cov = m["ring_coverage"]["hybrid"]
kpi(c[4], "Organised-ring months caught", pct(cov["coverage"]),
    f"{cov['ring_months_detected']} of {cov['ring_months_active']} active ring-months")

st.write("")
c = st.columns(5)
open_cases = cases[cases.status.isin(["New", "In review", "Awaiting information", "Escalated to MLRO"])]
kpi(c[0], "Open cases", f"{len(open_cases)}", f"{(open_cases.assigned_to.isna()).sum()} unassigned")
kpi(c[1], "SARs filed", f"{(cases.status == 'SAR filed').sum()}", "simulated history, earlier months")
kpi(c[2], "Closed, no further action", f"{(cases.status == 'Closed - no further action').sum()}",
    "documented rationale on every closure")
tp_r, tp_d = rules["true_positives_found"], dep["true_positives_found"]
kpi(c[3], "Suspicious accounts found", f"{tp_d}", f"vs {tp_r} with rules, same {dep['investigations']} reviews")
kpi(c[4], "Rule alerts needing triage", f"{ras['alerts']:,}",
    f"captures only {pct(ras['recall'])} of suspicious activity")

st.divider()

# --- trends --------------------------------------------------------------------------------
left, right = st.columns(2)
with left:
    st.subheader("Alert volume vs investigation capacity")
    vol = te.groupby("month").agg(rule_alerts=("rule_alert", "sum")).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=vol.month, y=vol.rule_alerts, name="Rule alerts raised",
                         marker_color=SERIES[3], marker_line_width=0,
                         hovertemplate="%{x}<br>%{y} rule alerts<extra></extra>"))
    fig.add_trace(go.Scatter(x=vol.month, y=[K] * len(vol), name="Cases the team can open",
                             mode="lines", line=dict(color=SERIES[0], width=2, dash="dash"),
                             hovertemplate="Capacity: %{y}<extra></extra>"))
    fig.update_layout(height=330, yaxis_title="Alerts per month", bargap=0.35)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Rules raise about {ras['alerts_per_month'] / K:.0f}× more alerts than the team can work, "
               "so most are either rushed or never reviewed.")

with right:
    st.subheader("Suspicious accounts found per month")
    fig = go.Figure()
    for key in ("hybrid", "rules"):
        mo = pd.DataFrame(m["monthly"][key])
        fig.add_trace(go.Scatter(x=mo.month, y=mo.true_positives, mode="lines+markers",
                                 name=f"{'Deployed priority' if key == 'hybrid' else 'Rules'} (top {K})",
                                 line=dict(color=MODEL_COLORS[key], width=2), marker=dict(size=8),
                                 hovertemplate="%{x}<br>%{y} suspicious of " + str(K) + "<extra></extra>"))
    mo = pd.DataFrame(m["monthly"]["hybrid"])
    fig.add_trace(go.Scatter(x=mo.month, y=mo.positives, mode="lines", name="All suspicious account-months",
                             line=dict(color="#898781", width=2, dash="dot"),
                             hovertemplate="%{x}<br>%{y} suspicious in total<extra></extra>"))
    fig.update_layout(height=330, yaxis_title="Account-months")
    st.plotly_chart(fig, width="stretch")
    st.caption("Same number of investigations each month; the priority score fills the queue with "
               "far more genuinely suspicious activity.")

left, right = st.columns(2)
with left:
    st.subheader("What the queue is catching")
    tr = pd.DataFrame(m["typology_recall"])
    tr["typ"] = tr.label_typology.map(TYP_LABEL)
    tr = tr.sort_values("recall_at_capacity")
    fig = go.Figure(go.Bar(
        x=tr.recall_at_capacity, y=tr.typ, orientation="h", marker_color=SERIES[0],
        text=[f"{int(f)} of {int(p)}" for f, p in zip(tr.found, tr.positives)], textposition="outside",
        hovertemplate="%{y}: %{x:.0%} caught<extra></extra>"))
    fig.update_layout(height=320, xaxis=dict(tickformat=".0%", range=[0, 1.15], title="Share caught within capacity"))
    st.plotly_chart(fig, width="stretch")
    st.caption("Fan-out and fan-in members are the hardest: each mule moves one modest payment. "
               "Network context helps, but these remain the main recall gap.")

with right:
    st.subheader("Investigator workload (current month)")
    live = cases[cases.month == months[-1]].copy()
    live["assigned_to"] = live.assigned_to.fillna("Unassigned")
    wl = live.groupby(["assigned_to", "status"]).size().unstack(fill_value=0)
    order = [s for s in ["New", "In review", "Awaiting information"] if s in wl.columns]
    colors = {"New": SERIES[0], "In review": SERIES[1], "Awaiting information": SERIES[2]}
    fig = go.Figure()
    for s in order:
        fig.add_trace(go.Bar(y=wl.index, x=wl[s], name=s, orientation="h", marker_color=colors[s],
                             marker_line=dict(color="#fcfcfb", width=2),
                             hovertemplate="%{y}: %{x} " + s.lower() + "<extra></extra>"))
    fig.update_layout(barmode="stack", height=320, xaxis_title="Cases", legend_traceorder="normal")
    st.plotly_chart(fig, width="stretch")

st.divider()
st.subheader("Modelled operational efficiency")
hrs_case, hrs_alert = 3.0, 1.5
rule_hours = ras["alerts"] * hrs_alert
dep_hours = dep["investigations"] * hrs_case
c = st.columns(3)
kpi(c[0], "Analyst hours, triaging every rule alert", f"{rule_hours:,.0f} h",
    f"{ras['alerts']:,} alerts × {hrs_alert} h · finds {round(ras['recall'] * m['test_positives'])} suspicious")
kpi(c[1], "Analyst hours, priority queue", f"{dep_hours:,.0f} h",
    f"{dep['investigations']} cases × {hrs_case} h · finds {tp_d} suspicious")
kpi(c[2], "Hours per suspicious account found",
    f"{dep_hours / max(tp_d, 1):.1f} h",
    f"vs {rule_hours / max(round(ras['recall'] * m['test_positives']), 1):.1f} h reviewing all rule alerts")
st.caption("Effort assumptions are illustrative and adjustable on the Impact Simulator page. "
           "Results come from synthetic data with known ground truth; real-world hit rates are much lower "
           "because labels are incomplete and launderers adapt.")
