"""Money-laundering pattern discovery.

Each detector scans raw transactions for a pattern that AML guidance associates
with laundering and returns *investigative indicators*: a pattern, the accounts
and transactions involved, and a plain-English evidence line.

An indicator is a reason to look, not a finding of wrongdoing.  Legitimate
behaviour (payroll runs, merchants collecting card payments, a family moving
money around) can and does trigger these detectors; the evaluation in this
module measures exactly how often.
"""
from __future__ import annotations

import duckdb
import networkx as nx
import numpy as np
import pandas as pd

from amlintel.config import CASH_THRESHOLD
from amlintel.graph.network import ordered_subgraph

COLUMNS = ["typology", "primary_account", "accounts", "txn_ids", "start", "end",
           "total_amount", "n_txns", "evidence"]


def _row(typ, primary, accounts, txns, start, end, total, evidence):
    return (typ, primary, ",".join(sorted(set(accounts))), ",".join(txns), start, end,
            float(total), len(txns), evidence)


# ---------------------------------------------------------------------------
def detect_structuring(tx: pd.DataFrame, min_deposits: int = 3, gap_days: float = 10) -> list:
    d = tx[(tx.channel == "cash_deposit") & (tx.amount >= 0.9 * CASH_THRESHOLD)
           & (tx.amount < CASH_THRESHOLD)].sort_values(["to_account", "timestamp"])
    out = []
    outflows = tx[tx.from_account != "CASH"].set_index("from_account")
    for acct, g in d.groupby("to_account"):
        ts = g.timestamp.to_numpy()
        breaks = np.flatnonzero(np.diff(ts) > np.timedelta64(int(gap_days * 86400), "s")) + 1
        for ep in np.split(np.arange(len(g)), breaks):
            if len(ep) < min_deposits:
                continue
            e = g.iloc[ep]
            total = e.amount.sum()
            start, end = e.timestamp.min(), e.timestamp.max()
            ev = (f"{len(e)} cash deposits between {e.amount.min():,.0f} and {e.amount.max():,.0f} "
                  f"(just under the {CASH_THRESHOLD:,.0f} reporting threshold) within "
                  f"{(end - start).days + 1} days, total {total:,.0f}")
            txns = list(e.txn_id)
            accts = [acct]
            if acct in outflows.index:
                o = outflows.loc[[acct]]
                o = o[(o.timestamp > end) & (o.timestamp <= end + pd.Timedelta(days=7))]
                big = o[o.amount >= 0.5 * total]
                if len(big):
                    b = big.iloc[0]
                    days = (b.timestamp - end).days
                    when = "the same day" if days == 0 else f"{days} day{'s' if days != 1 else ''} later"
                    ev += f"; {b.amount:,.0f} sent on to {b.to_account} {when}"
                    txns.append(b.txn_id)
                    accts.append(b.to_account)
            out.append(_row("structuring", acct, accts, txns, start, end, total, ev))
    return out


# ---------------------------------------------------------------------------
def _non_routine_mask(tx: pd.DataFrame) -> pd.Series:
    """True when the pair has NOT transacted in 2+ earlier months.

    Payroll runs and supplier payments repeat month after month; excluding those
    routine relationships removes the biggest source of fan-out false positives.
    """
    pm = tx[["from_account", "to_account", "month"]].drop_duplicates().sort_values("month")
    pm["prior_months"] = pm.groupby(["from_account", "to_account"]).cumcount()
    prior = tx.merge(pm, on=["from_account", "to_account", "month"], how="left")["prior_months"]
    return pd.Series(prior.to_numpy() < 2, index=tx.index)


def detect_fan(tx: pd.DataFrame, direction: str = "out", min_cp: int = 6,
               min_amount: float = 2_000, window_h: int = 72) -> list:
    t = tx[(tx.from_account != "CASH") & (tx.to_account != "CASH") & (tx.amount >= min_amount)]
    t = t[_non_routine_mask(t)]
    key, cp = ("from_account", "to_account") if direction == "out" else ("to_account", "from_account")
    typ = "fan_out" if direction == "out" else "fan_in"
    counts = t.groupby(key).size()
    t = t[t[key].isin(counts[counts >= min_cp].index)].sort_values([key, "timestamp"])
    out = []
    W = np.timedelta64(window_h * 3600, "s")
    for acct, g in t.groupby(key):
        ts = g.timestamp.to_numpy()
        cps = g[cp].to_numpy()
        i, used_until = 0, -1
        for j in range(len(g)):
            while ts[j] - ts[i] > W:
                i += 1
            if j <= used_until:
                continue
            uniq = set(cps[i:j + 1])
            if len(uniq) >= min_cp:
                # extend the episode while it stays inside the window from i
                k = j
                while k + 1 < len(g) and ts[k + 1] - ts[i] <= W:
                    k += 1
                e = g.iloc[i:k + 1]
                total = e.amount.sum()
                hours = (e.timestamp.max() - e.timestamp.min()).total_seconds() / 3600
                verb = "sent to" if direction == "out" else "received from"
                ev = (f"{total:,.0f} {verb} {e[cp].nunique()} non-routine counterparties within "
                      f"{hours:.0f} hours (each {e.amount.min():,.0f}-{e.amount.max():,.0f})")
                out.append(_row(typ, acct, [acct, *e[cp]], list(e.txn_id),
                                e.timestamp.min(), e.timestamp.max(), total, ev))
                used_until = k
                i = k + 1 if k + 1 < len(g) else k
    return out


# ---------------------------------------------------------------------------
def detect_pass_through(tx: pd.DataFrame, min_in: float = 15_000, ratio: float = 0.85,
                        window_h: int = 24) -> list:
    """Large inflow followed by outflows of most of it within ``window_h`` hours.

    Vectorised: outflows are sorted by (account, time) and summed with a running
    total, so each inflow's 24h outflow is two binary searches, not a scan.
    """
    inflows = tx[(tx.to_account != "CASH") & (tx.amount >= min_in)]
    outs = tx[(tx.from_account != "CASH") & tx.from_account.isin(inflows.to_account.unique())]
    if inflows.empty or outs.empty:
        return []
    codes = {a: i for i, a in enumerate(sorted(set(outs.from_account) | set(inflows.to_account)))}
    BIG = 10**11
    secs_o = outs.timestamp.to_numpy().astype("datetime64[s]").astype(np.int64)
    key_o = outs.from_account.map(codes).to_numpy(np.int64) * BIG + secs_o
    order = np.argsort(key_o, kind="stable")
    key_o = key_o[order]
    outs = outs.iloc[order]
    cum = np.concatenate([[0.0], np.cumsum(outs.amount.to_numpy())])
    secs_i = inflows.timestamp.to_numpy().astype("datetime64[s]").astype(np.int64)
    key_i = inflows.to_account.map(codes).to_numpy(np.int64) * BIG + secs_i
    lo = np.searchsorted(key_o, key_i, side="right")
    hi = np.searchsorted(key_o, key_i + window_h * 3600, side="right")
    sent = cum[hi] - cum[lo]
    hit = np.flatnonzero((hi > lo) & (sent >= ratio * inflows.amount.to_numpy()))
    out = []
    for j in hit:
        r = inflows.iloc[j]
        o = outs.iloc[lo[j]:hi[j]]
        hrs = (o.timestamp.max() - r.timestamp).total_seconds() / 3600
        src = "a cash deposit" if r.from_account == "CASH" else r.from_account
        ev = (f"received {r.amount:,.0f} from {src} and moved {sent[j]:,.0f} "
              f"({sent[j] / r.amount:.0%}) out to {o.to_account.nunique()} account(s) within {hrs:.0f} hours")
        out.append(_row("pass_through", r.to_account, [r.to_account, r.from_account, *o.to_account],
                        [r.txn_id, *o.txn_id], r.timestamp, o.timestamp.max(), r.amount, ev))
    return out


# ---------------------------------------------------------------------------
def detect_cycles(tx: pd.DataFrame, min_amount: float = 5_000, max_len: int = 6,
                  hop_days: float = 7, min_ratio: float = 0.80, max_cycles: int = 20_000) -> list:
    """Time-respecting cycles: money leaves an account and returns to it, hop by hop."""
    t = tx[(tx.from_account != "CASH") & (tx.to_account != "CASH") & (tx.amount >= min_amount)]
    out = []
    hop = pd.Timedelta(days=hop_days)
    for month, tm in t.groupby("month"):
        # allow cycles that spill into the next 2 weeks
        nxt = t[(t.timestamp > tm.timestamp.max()) & (t.timestamp <= tm.timestamp.max() + pd.Timedelta(days=14))]
        pool = pd.concat([tm, nxt])
        G = nx.DiGraph()
        G.add_edges_from(zip(tm.from_account, tm.to_account))
        G.add_edges_from(zip(nxt.from_account, nxt.to_account))
        edges = {k: g.sort_values("timestamp") for k, g in pool.groupby(["from_account", "to_account"])}
        seen = set()
        for comp in sorted(nx.strongly_connected_components(G), key=min):
            if len(comp) < 3:
                continue
            cands = []
            for k, cyc in enumerate(nx.simple_cycles(ordered_subgraph(G, comp), length_bound=max_len)):
                if k >= max_cycles:          # guard against huge dense components at scale
                    break
                if len(cyc) >= 3:
                    i0 = cyc.index(min(cyc))           # canonical rotation
                    cands.append(cyc[i0:] + cyc[:i0])
            for cyc in sorted(cands):
                if frozenset(cyc) in seen:
                    continue
                found = _time_ordered(cyc, edges, hop, min_ratio, month)
                if found is not None:
                    seen.add(frozenset(cyc))
                    rows = found
                    total = rows[0].amount
                    days = (rows[-1].timestamp - rows[0].timestamp).total_seconds() / 86400
                    ev = (f"{total:,.0f} moved through {len(cyc)} accounts and returned to "
                          f"{rows[0].from_account} as {rows[-1].amount:,.0f} within {days:.1f} days")
                    out.append(_row("cycle", rows[0].from_account, list(cyc),
                                    [r.txn_id for r in rows], rows[0].timestamp,
                                    rows[-1].timestamp, total, ev))
    return out


def _time_ordered(cyc, edges, hop, min_ratio, month):
    n = len(cyc)
    for rot in range(n):
        order = cyc[rot:] + cyc[:rot]
        first = edges.get((order[0], order[1]))
        if first is None:
            continue
        for start in first[first.month == month].itertuples(index=False):
            chain, prev = [start], start
            for i in range(1, n):
                e = edges.get((order[i], order[(i + 1) % n]))
                if e is None:
                    break
                cand = e[(e.timestamp > prev.timestamp) & (e.timestamp <= prev.timestamp + hop)
                         & (e.amount >= min_ratio * prev.amount) & (e.amount <= 1.05 * prev.amount)]
                if cand.empty:
                    break
                prev = next(cand.itertuples(index=False))
                chain.append(prev)
            if len(chain) == n:
                return chain
    return None


# ---------------------------------------------------------------------------
def detect_scatter_gather(tx: pd.DataFrame, min_amount: float = 5_000, min_mid: int = 3,
                          forward_h: int = 72, min_ratio: float = 0.85) -> list:
    """Source splits funds across intermediaries that forward them to one sink."""
    t = tx[(tx.from_account != "CASH") & (tx.to_account != "CASH") & (tx.amount >= min_amount)]
    a = t[["txn_id", "timestamp", "from_account", "to_account", "amount"]]
    # Range join in DuckDB: a self-merge in pandas explodes for busy business accounts.
    con = duckdb.connect()
    con.register("a", a)
    m = con.execute(f"""
        SELECT x.txn_id AS txn_id_1, x.timestamp AS timestamp_1, x.from_account AS from_account_1,
               x.to_account AS to_account_1, x.amount AS amount_1,
               y.txn_id AS txn_id_2, y.timestamp AS timestamp_2, y.to_account AS to_account_2,
               y.amount AS amount_2
        FROM a x JOIN a y
          ON x.to_account = y.from_account
         AND y.timestamp > x.timestamp
         AND y.timestamp <= x.timestamp + INTERVAL {int(forward_h)} HOUR
         AND y.amount >= {min_ratio} * x.amount AND y.amount <= x.amount
         AND x.from_account <> y.to_account
        ORDER BY timestamp_1, txn_id_1, txn_id_2""").df()
    con.close()
    out = []
    for (src, sink), g in m.groupby(["from_account_1", "to_account_2"]):
        g = g.sort_values("timestamp_1")
        ts = g.timestamp_1.to_numpy()
        breaks = np.flatnonzero(np.diff(ts) > np.timedelta64(48 * 3600, "s")) + 1
        for ep in np.split(np.arange(len(g)), breaks):
            e = g.iloc[ep].drop_duplicates("to_account_1")
            if len(e) < min_mid:
                continue
            total = e.amount_1.sum()
            ev = (f"{src} split {total:,.0f} across {len(e)} intermediaries, which forwarded "
                  f"{e.amount_2.sum():,.0f} to {sink} within {forward_h} hours")
            out.append(_row("scatter_gather", src, [src, sink, *e.to_account_1],
                            [*e.txn_id_1, *e.txn_id_2], e.timestamp_1.min(), e.timestamp_2.max(),
                            total, ev))
    return out


# ---------------------------------------------------------------------------
def detect_all(tx: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows += detect_structuring(tx)
    rows += detect_fan(tx, "out")
    rows += detect_fan(tx, "in")
    rows += detect_pass_through(tx)
    rows += detect_cycles(tx)
    rows += detect_scatter_gather(tx)
    ind = pd.DataFrame(rows, columns=COLUMNS)
    ind = ind.sort_values(["start", "typology", "primary_account", "txn_ids"], kind="stable").reset_index(drop=True)
    ind.insert(0, "indicator_id", [f"IND-{i:05d}" for i in range(1, len(ind) + 1)])
    ind["month"] = pd.to_datetime(ind["start"]).dt.strftime("%Y-%m")
    return ind


def indicator_account_features(ind: pd.DataFrame) -> pd.DataFrame:
    """Per account-month counts of indicators, by typology (model features)."""
    if ind.empty:
        return pd.DataFrame(columns=["account_id", "month"])
    e = ind.assign(account_id=ind.accounts.str.split(",")).explode("account_id")
    piv = (e.groupby(["account_id", "month", "typology"]).size()
             .unstack(fill_value=0).add_prefix("ind_").reset_index())
    for t in ("structuring", "fan_out", "fan_in", "pass_through", "cycle", "scatter_gather"):
        if f"ind_{t}" not in piv:
            piv[f"ind_{t}"] = 0
    piv["ind_total"] = piv.filter(like="ind_").sum(axis=1)
    return piv


def evaluate_indicators(ind: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    """How well do the pattern detectors line up with the injected ground truth?

    * precision: share of indicators where most flagged transactions are laundering
    * instance recall: share of injected instances with any transaction inside an
      indicator of the *same* typology
    """
    lab = tx.set_index("txn_id")[["is_laundering", "instance_id", "typology"]]
    rows = []
    for typ, g in ind.groupby("typology"):
        txns = g.txn_ids.str.split(",")
        frac = txns.apply(lambda ids: lab.loc[ids, "is_laundering"].mean())
        hit_inst = set(lab.loc[[x for ids in txns for x in ids]].query("typology == @typ").instance_id)
        n_inst = tx.loc[tx.typology == typ, "instance_id"].nunique()
        rows.append({"typology": typ, "indicators": len(g),
                     "precision": float((frac >= 0.5).mean()),
                     "instances_injected": n_inst,
                     "instances_recovered": len(hit_inst),
                     "instance_recall": len(hit_inst) / n_inst if n_inst else np.nan})
    return pd.DataFrame(rows)
