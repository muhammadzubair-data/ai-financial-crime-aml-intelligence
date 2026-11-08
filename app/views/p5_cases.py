"""05 · Investigation Case Management."""
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from amlintel.cases.queue import ANALYSTS
from amlintel.cases.store import OPEN_STATUSES, STATUSES, TRANSITIONS
from common import (
    SERIES,
    STATUS,
    TYP_LABEL,
    band_pill,
    current_user,
    load,
    money,
    setup,
    store,
    synthetic_badge,
    test_months,
    window_tx,
)

setup("Investigation Case Management",
      "The risk-ranked queue, the evidence behind each case, and an audit trail of every decision.")
synthetic_badge()

cs = store()
cases = cs.cases()
rules = load("rules.csv").set_index("rule_id")
ind = load("indicators.parquet").set_index("indicator_id")
tx = window_tx()
months = test_months()

f1, f2, f3, f4 = st.columns([1, 1.6, 1, 1])
month = f1.selectbox("Queue month", months[::-1])
status_f = f2.multiselect("Status", STATUSES, default=[s for s in OPEN_STATUSES] if month == months[-1] else [],
                          placeholder="All statuses")
band_f = f3.multiselect("Risk band", ["Critical", "High", "Elevated"], placeholder="All bands")
owner_f = f4.selectbox("Assigned to", ["Anyone", "Unassigned", *ANALYSTS])

q = cases[cases.month == month]
if status_f:
    q = q[q.status.isin(status_f)]
if band_f:
    q = q[q.risk_band.isin(band_f)]
if owner_f == "Unassigned":
    q = q[q.assigned_to.isna()]
elif owner_f != "Anyone":
    q = q[q.assigned_to == owner_f]

st.caption(f"{len(q)} of {len(cases[cases.month == month])} cases in the {month} queue. "
           "Select a row to open the case.")
tbl = q.assign(patterns=q.typologies.apply(lambda s: ", ".join(TYP_LABEL.get(t, t) for t in s.split(",")) if s else ""),
               linked=q.network_cluster != "")[
    ["case_id", "queue_rank", "account_id", "risk_score", "risk_band", "status", "assigned_to", "patterns",
     "rules_fired", "linked"]]
sel = st.dataframe(tbl, hide_index=True, width="stretch", height=330, on_select="rerun",
                   selection_mode="single-row",
                   column_config={"queue_rank": st.column_config.NumberColumn("Rank"),
                                  "risk_score": st.column_config.ProgressColumn("Risk", min_value=95, max_value=100, format="%.1f"),
                                  "risk_band": "Band", "assigned_to": "Owner", "patterns": "Patterns",
                                  "rules_fired": "Rules", "linked": st.column_config.CheckboxColumn("Linked")})
if not len(q):
    st.stop()
if sel.selection.rows:
    st.session_state["case_id"] = q.iloc[sel.selection.rows[0]].case_id
case_id = st.session_state.get("case_id")
if case_id not in set(q.case_id):
    case_id = q.iloc[0].case_id
case = cs.case(case_id)

st.divider()
h1, h2 = st.columns([3, 1.2])
h1.markdown(f"### {case['case_id']} · {case['account_id']}")
h1.markdown(band_pill(case["risk_band"]) + f'<span class="pill">{case["status"]}</span>'
            f'<span class="pill">Owner: {case["assigned_to"] or "unassigned"}</span>'
            f'<span class="pill">Queue rank {case["queue_rank"]} · risk {case["risk_score"]:.1f}</span>'
            + (f'<span class="pill">Linked cluster {case["network_cluster"]}</span>' if case["network_cluster"] else ""),
            unsafe_allow_html=True)
h1.markdown(f'<div class="evidence">{case["summary"]}</div>', unsafe_allow_html=True)

with h2:
    with st.container(border=True):
        st.markdown("**Actions**")
        st.caption(f"Acting as {current_user()}")
        with st.form("assign", clear_on_submit=True, border=False):
            who = st.selectbox("Assign to", ANALYSTS)
            if st.form_submit_button("Assign", width="stretch"):
                cs.assign(case_id, who, current_user())
                st.rerun()
        nxt = TRANSITIONS[case["status"]]
        if nxt:
            with st.form("status", clear_on_submit=True, border=False):
                new = st.selectbox("Move to", nxt)
                why = st.text_area("Rationale", height=70, placeholder="Required to close or record a SAR")
                if st.form_submit_button("Update status", width="stretch"):
                    try:
                        cs.set_status(case_id, new, current_user(), why)
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
        else:
            st.caption("SAR filed: case is locked.")

tab1, tab2, tab3, tab4 = st.tabs(["Evidence", "Transactions", "Notes", "Audit trail"])
with tab1:
    a, b = st.columns([1.2, 1])
    with a:
        st.markdown("**Model drivers** (SHAP, top contributions)")
        drv = pd.DataFrame(json.loads(case["drivers_json"] or "[]"))
        if drv.empty:
            st.caption("No stored explanation for this case.")
        else:
            drv = drv.iloc[::-1]
            fig = go.Figure(go.Bar(
                x=drv.shap, y=[f"{l} = {v:,.2f}" if abs(v) < 100 else f"{l} = {v:,.0f}" for l, v in zip(drv.label, drv.value)],
                orientation="h", marker_color=[STATUS["critical"] if s > 0 else SERIES[0] for s in drv.shap],
                hovertemplate="%{y}<br>%{x:+.3f} log-odds<extra></extra>"))
            fig.update_layout(height=280, xaxis_title="Contribution to risk")
            st.plotly_chart(fig, width="stretch")
    with b:
        st.markdown("**Rules fired**")
        if case["rules_fired"]:
            for rid in case["rules_fired"].split(","):
                st.markdown(f"- **{rid} · {rules.loc[rid, 'name']}**: {rules.loc[rid, 'description']}")
        else:
            st.caption("No scenario rule fired. This case was surfaced by the model alone.")
        st.markdown("**Pattern indicators**")
        if case["indicator_ids"]:
            for iid in case["indicator_ids"].split(","):
                r = ind.loc[iid]
                st.markdown(f'<div class="evidence"><b>{TYP_LABEL.get(r.typology)}</b> · {iid}<br>{r.evidence}</div>',
                            unsafe_allow_html=True)
        else:
            st.caption("None for this account this month.")
        if case["network_cluster"]:
            linked = cases[(cases.network_cluster == case["network_cluster"]) & (cases.case_id != case_id)]
            st.markdown("**Linked cases** (alerted accounts connected by money flows)")
            st.dataframe(linked[["case_id", "account_id", "risk_band", "status"]], hide_index=True,
                         width="stretch")

with tab2:
    t = tx[(tx.month == case["month"]) & ((tx.from_account == case["account_id"]) | (tx.to_account == case["account_id"]))]
    t = t.assign(direction=(t.from_account == case["account_id"]).map({True: "out", False: "in"}),
                 counterparty=lambda d: d.to_account.where(d.direction == "out", d.from_account))
    a, b, c = st.columns(3)
    a.metric("Money in", money(t.loc[t.direction == "in", "amount"].sum()))
    b.metric("Money out", money(t.loc[t.direction == "out", "amount"].sum()))
    c.metric("Counterparties", t.counterparty.nunique())
    st.dataframe(t.sort_values("timestamp")[["txn_id", "timestamp", "direction", "counterparty", "amount",
                                             "channel", "from_country", "to_country"]],
                 hide_index=True, width="stretch", height=320,
                 column_config={"amount": st.column_config.NumberColumn(format="localized")})

with tab3:
    notes = cs.notes(case_id)
    for n in notes.itertuples():
        st.markdown(f"**{n.author}** · {n.created_at}  \n{n.text}")
    if notes.empty:
        st.caption("No notes yet.")
    with st.form("note", clear_on_submit=True):
        text = st.text_area("Add a note", placeholder="What did you check, what did you find, what is next?")
        if st.form_submit_button("Save note"):
            try:
                cs.add_note(case_id, current_user(), text)
                st.rerun()
            except ValueError as e:
                st.error(str(e))

with tab4:
    au = cs.audit(case_id)
    st.dataframe(au[["ts", "actor", "action", "details"]], hide_index=True, width="stretch",
                 column_config={"ts": "Time (UTC)"})
    st.caption("Append-only log: every creation, assignment, status change and note is recorded with actor and time.")
    st.download_button("Download full audit log (CSV)", cs.audit().to_csv(index=False), "audit_log.csv", "text/csv")
