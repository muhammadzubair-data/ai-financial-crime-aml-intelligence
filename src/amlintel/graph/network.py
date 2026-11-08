"""Transaction network intelligence (NetworkX).

Builds one directed, amount-weighted account graph per month and derives
network features that row-level fraud models cannot see:

* weighted PageRank            - how central an account is in money flows
* strongly-connected component - size of the circular-flow structure it sits in
* cycle participation          - short directed cycles (length <= 6) on material edges
* successor max in-degree      - "do I pay into an account that collects from many?"
* predecessor max out-degree   - "am I paid by an account that pays out to many?"
* high-KYC neighbour share     - exposure to higher-risk counterparties

Cash is an external node and is excluded from the graph.
"""
from __future__ import annotations

from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd

MATERIAL_EDGE = 1_000.0   # edges below this amount are ignored for structural features
FAN_EDGE = 2_500.0
MAX_CYCLE_LEN = 6


def month_graph(tx_month: pd.DataFrame, min_amount: float = 0.0) -> nx.DiGraph:
    t = tx_month[(tx_month.from_account != "CASH") & (tx_month.to_account != "CASH")]
    if min_amount:
        t = t[t.amount >= min_amount]
    e = (t.groupby(["from_account", "to_account"])
          .agg(amount=("amount", "sum"), n=("amount", "size")).reset_index())
    G = nx.DiGraph()
    G.add_weighted_edges_from(e[["from_account", "to_account", "amount"]].itertuples(index=False),
                              weight="amount")
    nx.set_edge_attributes(G, dict(zip(zip(e.from_account, e.to_account), e.n)), "n")
    return G


def ordered_subgraph(G: nx.DiGraph, nodes) -> nx.DiGraph:
    """Induced subgraph with sorted node/edge order.

    ``G.subgraph(set)`` iterates in set order, which depends on Python's string
    hash seed; building the subgraph explicitly keeps results reproducible.
    """
    keep = set(nodes)
    H = nx.DiGraph()
    H.add_nodes_from(sorted(keep))
    H.add_edges_from(sorted((u, v) for u, v in G.edges(keep) if v in keep))
    return H


def cycle_counts(G: nx.DiGraph, max_len: int = MAX_CYCLE_LEN, cap: int = 50_000) -> Counter:
    """Count how many short directed cycles each node participates in."""
    c: Counter = Counter()
    # Only SCCs with >1 node can contain cycles; this prunes the search hugely.
    for comp in nx.strongly_connected_components(G):
        if len(comp) < 2:
            continue
        sub = ordered_subgraph(G, comp)
        for i, cyc in enumerate(nx.simple_cycles(sub, length_bound=max_len)):
            if len(cyc) < 2 or i >= cap:
                if i >= cap:
                    break
                continue
            c.update(cyc)
    return c


def monthly_graph_features(tx: pd.DataFrame, kyc_level: dict[str, int]) -> pd.DataFrame:
    rows = []
    for month, tm in tx.groupby("month", sort=True):
        G = month_graph(tm)
        if G.number_of_nodes() == 0:
            continue
        pr = nx.pagerank(G, weight="amount", max_iter=200, tol=1e-8)
        Gm = month_graph(tm, MATERIAL_EDGE)
        scc_size = {}
        for comp in nx.strongly_connected_components(Gm):
            for n_ in comp:
                scc_size[n_] = len(comp)
        cyc = cycle_counts(Gm)
        Gf = month_graph(tm, FAN_EDGE)
        indeg = dict(Gf.in_degree())
        outdeg = dict(Gf.out_degree())
        for n_ in G.nodes:
            succ = list(Gf.successors(n_)) if n_ in Gf else []
            pred = list(Gf.predecessors(n_)) if n_ in Gf else []
            nbrs = set(G.successors(n_)) | set(G.predecessors(n_))
            rows.append((
                n_, month,
                pr.get(n_, 0.0) * G.number_of_nodes(),           # scale-free pagerank
                scc_size.get(n_, 1),
                cyc.get(n_, 0),
                max((indeg[s] for s in succ), default=0),
                max((outdeg[p] for p in pred), default=0),
                float(np.mean([kyc_level.get(x, 0) == 2 for x in nbrs])) if nbrs else 0.0,
                len(nbrs),
            ))
    return pd.DataFrame(rows, columns=[
        "account_id", "month", "pagerank_scaled", "scc_size", "cycle_count",
        "succ_max_in_deg", "pred_max_out_deg", "nbr_high_kyc_share", "n_neighbors",
    ])


# ---------------------------------------------------------------------------
# Investigator-facing helpers
# ---------------------------------------------------------------------------
def ego_network(tx: pd.DataFrame, account: str, months: list[str] | None = None,
                radius: int = 2, min_amount: float = 500.0, max_nodes: int = 120) -> nx.DiGraph:
    """Money-flow neighbourhood around an account (for the Network Explorer)."""
    t = tx if months is None else tx[tx.month.isin(months)]
    G = month_graph(t, min_amount)
    if account not in G:
        return nx.DiGraph()
    U = G.to_undirected(as_view=True)
    keep = nx.single_source_shortest_path_length(U, account, cutoff=radius)
    if len(keep) > max_nodes:
        # keep the strongest flows first
        ranked = sorted(keep, key=lambda n_: (keep[n_],
                        -(G.get_edge_data(account, n_, {}).get("amount", 0)
                          + G.get_edge_data(n_, account, {}).get("amount", 0))))
        keep = {n_: keep[n_] for n_ in ranked[:max_nodes]}
    H = G.subgraph(keep).copy()
    nx.set_node_attributes(H, keep, "hops")
    return H


def alert_networks(tx_month: pd.DataFrame, alerted: list[str], max_shared_degree: int = 15) -> dict[str, int]:
    """Group alerted accounts that are linked by material money flows.

    Two alerted accounts are linked when money moved directly between them, or
    when both dealt with the same low-degree counterparty (a shared mule or
    collection account).  Shared high-degree counterparties such as large
    merchants or employers do not link accounts, otherwise half the customer
    base would end up in one "network".

    Returns account -> cluster number (0 = largest).  Unlinked accounts get
    singleton clusters.
    """
    G = month_graph(tx_month, MATERIAL_EDGE).to_undirected()
    alerted = sorted(a for a in set(alerted) if a in G)
    aset = set(alerted)
    A = nx.Graph()
    A.add_nodes_from(alerted)
    for a in alerted:
        for c in G.neighbors(a):
            if c in aset:
                A.add_edge(a, c)
            elif G.degree(c) <= max_shared_degree:
                for b in G.neighbors(c):
                    if b != a and b in aset:
                        A.add_edge(a, b)
    out = {}
    comps = sorted(nx.connected_components(A), key=lambda c: (-len(c), min(c)))
    for i, comp in enumerate(comps):
        for a in comp:
            out[a] = i
    return out
