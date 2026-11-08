"""02 · Transaction Monitoring Center."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import SERIES, TYP_LABEL, kpi, load, money, pct, scored, setup, synthetic_badge, window_tx

setup("Transaction Monitoring Center",
      "Search transactions, see which scenario rules fired and review pattern indicators with their evidence.")
synthetic_badge()

tx = window_tx()
sc = scored()
ind = load("indicators.parquet")
rules = load("rules.csv")
months = sorted(tx.month.unique())

f1, f2, f3, f4 = st.columns([1, 1.4, 1, 1.2])
month = f1.selectbox("Month", months, index=len(months) - 1)
channels = f2.multiselect("Channel", sorted(tx.channel.unique()), placeholder="All channels")
min_amt = f3.number_input("Minimum amount", 0, 500_000, 0, step=1000)
acct = f4.text_input("Account ID", placeholder="e.g. ACC001234").strip().upper()

t = tx[tx.month == month]
if channels:
    t = t[t.channel.isin(channels)]
if min_amt:
    t = t[t.amount >= min_amt]
if acct:
    t = t[(t.from_account == acct) | (t.to_account == acct)]

c = st.columns(5)
kpi(c[0], "Transactions", f"{len(t):,}")
kpi(c[1], "Value moved", money(t.amount.sum()))
kpi(c[2], "Cross-border share", pct(t.is_cross_border.mean() if len(t) else 0))
near = t[(t.channel == "cash_deposit") & (t.amount >= 9000) & (t.amount < 10000)]
kpi(c[3], "Cash deposits just under 10k", f"{len(near):,}")
mi = ind[ind.month == month]
kpi(c[4], "Pattern indicators", f"{len(mi):,}", "this month, all accounts")

st.write("")
left, right = st.columns([1.3, 1])
with left:
    st.subheader("Daily activity")
    d = t.assign(day=t.timestamp.dt.date).groupby("day").agg(n=("txn_id", "size"), v=("amount", "sum")).reset_index()
    fig = go.Figure(go.Scatter(x=d.day, y=d.v, mode="lines", line=dict(color=SERIES[0], width=2),
                               fill="tozeroy", fillcolor="rgba(42,120,214,0.08)",
                               customdata=d.n,
                               hovertemplate="%{x}<br>Value %{y:,.0f}<br>%{customdata:,} transactions<extra></extra>"))
    fig.update_layout(height=300, yaxis_title="Value moved")
    st.plotly_chart(fig, width="stretch")

with right:
    st.subheader("Scenario rules this month")
    sm = sc[sc.month == month]
    rows = []
    for r in rules.itertuples():
        hit = sm[sm[r.rule_id] == 1]
        rows.append({"Rule": f"{r.rule_id} · {r.name}", "Alerts": len(hit),
                     "Confirmed suspicious*": f"{hit.label.mean():.0%}" if len(hit) else "–",
                     "Logic": r.description})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=320)
    st.caption("*Share later confirmed against the synthetic ground truth. In production this would come "
               "from investigation outcomes.")

st.divider()
st.subheader("Pattern indicators")
st.caption("Investigative indicators, not findings. Each line explains what was seen so an analyst can "
           "decide whether it fits the customer's known business.")
typ = st.multiselect("Pattern type", sorted(mi.typology.unique()), format_func=lambda x: TYP_LABEL.get(x, x),
                     placeholder="All patterns")
show = mi if not typ else mi[mi.typology.isin(typ)]
if acct:
    show = show[show.accounts.str.contains(acct)]
view = show.assign(Pattern=show.typology.map(TYP_LABEL), Accounts=show.accounts.str.count(",") + 1)[
    ["indicator_id", "Pattern", "primary_account", "Accounts", "total_amount", "start", "evidence"]
].rename(columns={"indicator_id": "Indicator", "primary_account": "Primary account",
                  "total_amount": "Amount", "start": "Started", "evidence": "Evidence"})
sel = st.dataframe(view, hide_index=True, width="stretch", height=300,
                   on_select="rerun", selection_mode="single-row",
                   column_config={"Amount": st.column_config.NumberColumn(format="localized"),
                                  "Started": st.column_config.DatetimeColumn(format="D MMM YYYY, HH:mm")})
if sel.selection.rows:
    r = show.iloc[sel.selection.rows[0]]
    st.markdown(f'<div class="evidence"><b>{r.indicator_id} · {TYP_LABEL.get(r.typology)}</b><br>{r.evidence}</div>',
                unsafe_allow_html=True)
    ids = r.txn_ids.split(",")
    tt = tx[tx.txn_id.isin(ids)].sort_values("timestamp")
    st.dataframe(tt[["txn_id", "timestamp", "from_account", "to_account", "amount", "channel",
                     "from_country", "to_country"]], hide_index=True, width="stretch",
                 column_config={"amount": st.column_config.NumberColumn(format="localized")})
    st.caption("Open the Suspicious Network Explorer to see these accounts' wider money flows.")

st.divider()
st.subheader("Transactions")
st.caption(f"Showing the largest 500 of {len(t):,} matching transactions.")
st.dataframe(t.nlargest(500, "amount")[["txn_id", "timestamp", "from_account", "to_account", "amount", "channel",
                                        "from_country", "to_country", "is_cross_border"]],
             hide_index=True, width="stretch", height=360,
             column_config={"amount": st.column_config.NumberColumn(format="localized"),
                            "is_cross_border": st.column_config.CheckboxColumn("Cross-border")})
