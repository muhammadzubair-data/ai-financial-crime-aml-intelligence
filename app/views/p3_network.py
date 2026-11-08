"""03 · Suspicious Network Explorer."""
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from amlintel.graph.network import ego_network
from common import (
    INK,
    SEQ,
    TYP_LABEL,
    kpi,
    load,
    money,
    scored,
    setup,
    store,
    synthetic_badge,
    test_months,
    window_tx,
)

setup("Suspicious Network Explorer",
      "Follow the money around an account: who pays whom, how much, and whether alerted accounts are linked.")
synthetic_badge()

tx = window_tx()
sc = scored()
ind = load("indicators.parquet")
cases = store().cases()
months = sorted(tx.month.unique())
tmonths = test_months()

c1, c2, c3, c4 = st.columns([1, 1.6, 0.8, 1])
month = c1.selectbox("Month", months, index=len(months) - 1)
mc = cases[cases.month == month]
start_from = c2.radio("Start from", ["Linked alert cluster", "Queued case", "Any account"], horizontal=True,
                      index=0 if (mc.network_cluster != "").any() else (1 if len(mc) else 2))
radius = c3.selectbox("Hops", [1, 2], index=1)
min_amt = c4.select_slider("Hide flows under", [0, 500, 1000, 2500, 5000, 10000], value=1000)

focus_nodes: list[str] = []
if start_from == "Linked alert cluster" and len(mc):
    cl = mc[mc.network_cluster != ""].groupby("network_cluster").agg(
        accounts=("account_id", list), n=("account_id", "size"), top=("risk_score", "max")).reset_index()
    if cl.empty:
        st.info("No linked alert clusters this month. Pick a queued case or any account instead.")
        st.stop()
    cl = cl.sort_values(["n", "top"], ascending=False)
    lab = {r.network_cluster: f"{r.network_cluster} · {r.n} linked alerted accounts" for r in cl.itertuples()}
    pick = st.selectbox("Cluster", cl.network_cluster, format_func=lab.get)
    focus_nodes = cl.set_index("network_cluster").loc[pick, "accounts"]
elif start_from == "Queued case" and len(mc):
    lab = {r.account_id: f"#{r.queue_rank} · {r.account_id} · {r.risk_band}" for r in mc.itertuples()}
    focus_nodes = [st.selectbox("Case account", mc.account_id, format_func=lab.get)]
else:
    active = sc[sc.month == month].sort_values("risk_score", ascending=False).account_id
    focus_nodes = [st.selectbox("Account", active.tolist())]

show_truth = st.toggle("Show synthetic ground truth (laundering involvement)", value=False,
                       help="Only possible because the data is synthetic. Investigators never see this.")

# --- build the neighbourhood -------------------------------------------------------------------
tm = tx[tx.month == month]
G = nx.DiGraph()
for a in focus_nodes:
    G = nx.compose(G, ego_network(tm, a, radius=radius if len(focus_nodes) == 1 else 1,
                                  min_amount=min_amt, max_nodes=90 if len(focus_nodes) == 1 else 40))
if G.number_of_nodes() == 0:
    st.warning("No flows above the threshold for this account in the selected month.")
    st.stop()

risk = sc[sc.month == month].set_index("account_id")["risk_score"]
truth = set(tm.loc[tm.is_laundering == 1, "from_account"]) | set(tm.loc[tm.is_laundering == 1, "to_account"])
queued = set(mc.account_id)
pos = nx.spring_layout(G, seed=7, k=1.6 / np.sqrt(max(G.number_of_nodes(), 1)), weight=None, iterations=200)

edge_traces, annotations = [], []
amts = np.array([d["amount"] for _, _, d in G.edges(data=True)])
amax = amts.max() if len(amts) else 1
mid_x, mid_y, mid_txt = [], [], []
for u, v, d in G.edges(data=True):
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    w = 0.8 + 3.2 * np.sqrt(d["amount"] / amax)
    annotations.append(dict(ax=x0, ay=y0, x=x1, y=y1, xref="x", yref="y", axref="x", ayref="y",
                            showarrow=True, arrowhead=2, arrowsize=0.9, arrowwidth=w,
                            arrowcolor="rgba(82,81,78,0.45)", standoff=9, startstandoff=6))
    mid_x.append((x0 + x1) / 2)
    mid_y.append((y0 + y1) / 2)
    mid_txt.append(f"{u} → {v}<br>{d['amount']:,.0f} in {d['n']} transaction(s)")

nodes = list(G.nodes)
rs = np.array([risk.get(n, np.nan) for n in nodes], dtype=float)
flow = np.array([sum(d["amount"] for *_, d in G.in_edges(n, data=True))
                 + sum(d["amount"] for *_, d in G.out_edges(n, data=True)) for n in nodes])
size = 10 + 22 * np.sqrt(flow / max(flow.max(), 1))
line_w = [4 if n in focus_nodes else (2 if n in queued else 1) for n in nodes]
line_c = [INK if n in focus_nodes else ("#52514e" if n in queued else "#fcfcfb") for n in nodes]
symbols = ["diamond" if (show_truth and n in truth) else "circle" for n in nodes]
hover = [f"<b>{n}</b><br>Risk score: {('%.1f' % r) if not np.isnan(r) else 'n/a'}"
         f"<br>Flow in view: {f:,.0f}" + ("<br>In this month's case queue" if n in queued else "")
         + (("<br><i>Synthetic truth: involved in laundering</i>") if (show_truth and n in truth) else "")
         for n, r, f in zip(nodes, rs, flow)]

fig = go.Figure()
fig.add_trace(go.Scatter(x=mid_x, y=mid_y, mode="markers", marker=dict(size=10, color="rgba(0,0,0,0)"),
                         hovertext=mid_txt, hoverinfo="text", showlegend=False))
fig.add_trace(go.Scatter(
    x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers+text",
    text=[n if (n in focus_nodes and len(focus_nodes) == 1) else "" for n in nodes], textposition="top center",
    textfont=dict(size=10, color=INK),
    marker=dict(size=size, symbol=symbols, color=np.nan_to_num(rs, nan=0),
                colorscale=[[0, "#f0efec"], [0.5, SEQ[1]], [0.8, SEQ[3]], [0.95, SEQ[5]], [1, SEQ[6]]],
                cmin=0, cmax=100, line=dict(width=line_w, color=line_c),
                colorbar=dict(title="Risk score", thickness=10, len=0.6)),
    hovertext=hover, hoverinfo="text", showlegend=False))
fig.update_layout(height=620, annotations=annotations, plot_bgcolor="#fcfcfb",
                  xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=0, r=0, t=10, b=0))

left, right = st.columns([2.3, 1])
with left:
    st.plotly_chart(fig, width="stretch")
    st.caption("Node colour = monthly risk score (darker is riskier) · size = money moved in view · "
               "arrow width = amount · thick black ring = account(s) you started from · grey ring = other queued case · "
               "hover any node or arrow for details"
               + (" · diamond = synthetic laundering truth" if show_truth else ""))

with right:
    st.subheader("Network summary")
    Gu = G.to_undirected()
    sub_tx = tm[tm.from_account.isin(G.nodes) & tm.to_account.isin(G.nodes) & (tm.amount >= min_amt)]
    cyc = [c for c in nx.simple_cycles(G, length_bound=6) if len(c) >= 2]
    a, b = st.columns(2)
    kpi(a, "Accounts", f"{G.number_of_nodes()}")
    kpi(b, "Money flows", f"{G.number_of_edges()}")
    a, b = st.columns(2)
    kpi(a, "Value in view", money(sub_tx.amount.sum()))
    kpi(b, "Circular paths", f"{len(cyc)}", "money returning to its origin")
    a, b = st.columns(2)
    kpi(a, "Queued cases in view", f"{len(queued & set(G.nodes))}")
    hi = int((rs >= 99).sum())
    kpi(b, "Top-1% risk accounts", f"{hi}")

    st.write("")
    st.markdown("**Pattern indicators touching these accounts**")
    acc_set = set(G.nodes)
    mi = ind[(ind.month == month)]
    hits = mi[mi.accounts.str.split(",").apply(lambda s: bool(acc_set & set(s)))]
    if hits.empty:
        st.caption("None this month.")
    for r in hits.head(8).itertuples():
        st.markdown(f'<div class="evidence"><b>{TYP_LABEL.get(r.typology)}</b> · {r.indicator_id}<br>'
                    f'{r.evidence}</div>', unsafe_allow_html=True)
    if len(hits) > 8:
        st.caption(f"+{len(hits) - 8} more on the Transaction Monitoring page.")
    if cyc:
        st.markdown("**Circular paths**")
        for c_ in cyc[:5]:
            st.markdown("`" + " → ".join(c_ + [c_[0]]) + "`")

st.divider()
st.subheader("Accounts in this view")
tbl = pd.DataFrame({"account_id": nodes, "risk_score": rs, "flow_in_view": flow,
                    "queued": [n in queued for n in nodes]})
prof = load("accounts.parquet").set_index("account_id")[["customer_type", "business_kind", "country", "kyc_risk", "opened_on"]]
tbl = tbl.join(prof, on="account_id").sort_values("risk_score", ascending=False)
if show_truth:
    tbl["synthetic_truth"] = tbl.account_id.isin(truth)
st.dataframe(tbl, hide_index=True, width="stretch", height=300,
             column_config={"risk_score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=100, format="%.1f"),
                            "flow_in_view": st.column_config.NumberColumn("Flow in view", format="localized"),
                            "queued": st.column_config.CheckboxColumn("Queued case"),
                            "opened_on": st.column_config.DateColumn("Opened")})
