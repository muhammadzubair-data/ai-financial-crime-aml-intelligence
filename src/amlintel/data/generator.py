"""Synthetic AML transaction generator (AMLSim-style).

Why synthetic?  Real bank transactions cannot be published, and public AML
benchmarks (IBM AMLSim / "IBM Transactions for Anti Money Laundering") have
licence and size constraints that make them awkward for a public demo.  This
generator follows the same design idea as AMLSim: simulate a background of
ordinary customer behaviour, then inject known money-laundering typologies so
that ground truth exists for evaluation.

Design goals
------------
* Realistic background: preferential-attachment counterparties, payroll,
  merchants, cash usage, cross-border wires.
* Benign look-alikes on purpose: merchants (natural fan-in), employers (natural
  fan-out), salary sweepers (fast pass-through), cash-heavy small businesses
  (occasional near-threshold deposits).  Without these, any rule would look
  perfect and the evaluation would be meaningless.
* Organised rings: laundering is injected through persistent account networks
  that reuse members across months, plus opportunistic one-off patterns.

Every injected transaction is labelled with its typology, instance and network
so the platform can be evaluated end to end.  Labels are NEVER used as model
features.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from amlintel.config import CASH_THRESHOLD, HIGH_RISK_JURISDICTIONS, HOME_COUNTRIES, Profile

TYPOLOGIES = (
    "fan_out", "fan_in", "cycle", "structuring", "scatter_gather", "pass_through",
)
CASH = "CASH"
HOUR = np.timedelta64(1, "h")


@dataclass
class _Ctx:
    rng: np.random.Generator
    start: np.datetime64
    end: np.datetime64
    accounts: pd.DataFrame
    country: dict
    bg_start: np.datetime64 = None   # start of look-back history (background only)


# ---------------------------------------------------------------------------
# Accounts
# ---------------------------------------------------------------------------
def generate_accounts(p: Profile, rng: np.random.Generator) -> pd.DataFrame:
    n = p.n_accounts
    start = np.datetime64(p.start_date)
    ids = np.array([f"ACC{i:06d}" for i in range(1, n + 1)])

    ctype = rng.choice(["individual", "business"], size=n, p=[0.84, 0.16])
    kind = np.where(
        ctype == "business",
        rng.choice(["merchant", "employer", "trading"], size=n, p=[0.35, 0.25, 0.40]),
        "retail",
    )
    countries = list(HOME_COUNTRIES)
    probs = np.array(list(HOME_COUNTRIES.values()))
    country = rng.choice(countries, size=n, p=probs / probs.sum())

    # Most accounts pre-date the window; ~12% open during it.
    offset_days = rng.integers(-2200, 600, size=n)
    opened = start + offset_days.astype("timedelta64[D]")

    hr = np.isin(country, HIGH_RISK_JURISDICTIONS)
    kyc = np.where(
        hr,
        rng.choice(["low", "medium", "high"], size=n, p=[0.30, 0.45, 0.25]),
        rng.choice(["low", "medium", "high"], size=n, p=[0.72, 0.22, 0.06]),
    )
    activity = rng.lognormal(0.0, 0.6, size=n) * np.where(ctype == "business", 3.0, 1.0)
    popularity = rng.lognormal(0.0, 1.0, size=n) * np.where(kind == "merchant", 30.0, 1.0)
    sweeper = (ctype == "individual") & (rng.random(n) < 0.05)
    # Spending power varies a lot between customers; affluent customers and larger
    # businesses routinely move five- and six-figure sums for legitimate reasons.
    wealth = np.clip(rng.lognormal(0.0, 1.0, size=n), 0.1, 60.0)

    return pd.DataFrame({
        "account_id": ids,
        "customer_type": ctype,
        "business_kind": kind,
        "country": country,
        "opened_on": pd.to_datetime(opened),
        "kyc_risk": kyc,
        "_activity": activity,
        "_popularity": popularity,
        "_sweeper": sweeper,
        "_wealth": wealth,
    })


# ---------------------------------------------------------------------------
# Background (legitimate) behaviour
# ---------------------------------------------------------------------------
def _rand_times(ctx: _Ctx, lo: np.ndarray, size: int) -> np.ndarray:
    """Uniform timestamps between per-row lower bound and window end."""
    lo = np.maximum(lo, ctx.bg_start)
    span = (ctx.end - lo).astype("timedelta64[s]").astype(np.int64)
    span = np.maximum(span, 1)
    offs = (ctx.rng.random(size) * span).astype(np.int64)
    return lo + offs.astype("timedelta64[s]")


def _background(p: Profile, ctx: _Ctx) -> pd.DataFrame:
    rng, acc = ctx.rng, ctx.accounts
    n = len(acc)
    opened = acc["opened_on"].to_numpy().astype("datetime64[s]")
    ctype = acc["customer_type"].to_numpy()
    kind = acc["business_kind"].to_numpy()
    activity = acc["_activity"].to_numpy()
    pop = acc["_popularity"].to_numpy()
    pop_p = pop / pop.sum()
    wealth = acc["_wealth"].to_numpy()

    active_months = np.clip(
        (ctx.end - np.maximum(opened, ctx.bg_start)).astype("timedelta64[D]").astype(int) / 30.4,
        0, p.months + p.history_months,
    )
    frames = []

    # 1) ordinary transfers / card payments --------------------------------
    counts = rng.poisson(p.base_monthly_txn_rate * activity * active_months)
    snd = np.repeat(np.arange(n), counts)
    m = len(snd)
    n_contacts = rng.integers(3, 13, size=n)
    contacts = rng.choice(n, size=(n, 12), p=pop_p)
    # Most counterparties are domestic: re-draw ~80% of contact slots in-country.
    home = acc["country"].to_numpy()
    for c in np.unique(home):
        idx = np.flatnonzero(home == c)
        cp = pop[idx] / pop[idx].sum()
        rows = np.flatnonzero(home == c)
        mask = rng.random((len(rows), 12)) < 0.8
        draws = rng.choice(idx, size=(len(rows), 12), p=cp)
        contacts[rows] = np.where(mask, draws, contacts[rows])
    use_contact = rng.random(m) < 0.75
    pick = (rng.random(m) * n_contacts[snd]).astype(int)
    rcv = np.where(use_contact, contacts[snd, pick], rng.choice(n, size=m, p=pop_p))

    same = rcv == snd
    rcv[same] = (rcv[same] + 1) % n
    rcv_open = opened[rcv]
    ts = _rand_times(ctx, np.maximum(opened[snd], rcv_open), m)
    is_biz = ctype[snd] == "business"
    amt = np.where(
        is_biz,
        rng.lognormal(np.log(1100), 1.1, m),
        rng.lognormal(np.log(60), 1.05, m),
    ) * wealth[snd]
    roundme = rng.random(m) < 0.10
    amt = np.where(roundme, np.maximum(50, np.round(amt / 50) * 50), amt)
    to_merchant = kind[rcv] == "merchant"
    xb = acc["country"].to_numpy()[snd] != acc["country"].to_numpy()[rcv]
    channel = np.where(to_merchant & (amt < 2500), "card",
                       np.where(xb & (amt >= 1000), "wire", "transfer"))
    frames.append(pd.DataFrame({"src": snd, "dst": rcv, "timestamp": ts,
                                "amount": amt, "channel": channel}))

    # 2) payroll: employers pay a fixed staff list every month --------------
    employers = np.flatnonzero(kind == "employer")
    individuals = np.flatnonzero(ctype == "individual")
    pay_rows = []
    month_starts = np.array(
        [np.datetime64(p.start_date, "M") + np.timedelta64(k - p.history_months, "M")
         for k in range(p.months + p.history_months)]
    ).astype("datetime64[s]")
    salary_of = {}
    for e in employers:
        staff = rng.choice(individuals, size=rng.integers(8, 45), replace=False)
        base = rng.lognormal(np.log(2600), 0.35, size=len(staff))
        for s, b in zip(staff, base):
            salary_of.setdefault(s, []).append((e, b))
        for ms in month_starts:
            pay_t = ms + np.timedelta64(int(rng.integers(24, 28)), "D") + np.timedelta64(9, "h")
            if pay_t < opened[e]:
                continue
            for s, b in zip(staff, base):
                if pay_t < opened[s] or pay_t >= ctx.end:
                    continue
                pay_rows.append((e, s, pay_t, b * rng.uniform(0.97, 1.03), "transfer"))

    # 3) salary sweepers: move most of pay out within hours (benign pass-through)
    sweepers = np.flatnonzero(acc["_sweeper"].to_numpy())
    sweep_rows = []
    for s in sweepers:
        if s not in salary_of:
            continue
        target = next((c for c in contacts[s] if c != s), (s + 1) % n)
        for (e, b) in salary_of[s]:
            for ms in month_starts:
                pay_t = ms + np.timedelta64(25, "D") + np.timedelta64(9, "h")
                if pay_t < opened[s] or pay_t >= ctx.end - np.timedelta64(2, "D"):
                    continue
                if rng.random() < 0.8:
                    sweep_rows.append((s, target, pay_t + np.timedelta64(int(rng.integers(2, 20)), "h"),
                                       b * rng.uniform(0.70, 0.95), "transfer"))
    extra = pay_rows + sweep_rows
    if extra:
        e = np.array(extra, dtype=object)
        frames.append(pd.DataFrame({"src": e[:, 0].astype(int), "dst": e[:, 1].astype(int),
                                    "timestamp": e[:, 2].astype("datetime64[s]"),
                                    "amount": e[:, 3].astype(float), "channel": e[:, 4]}))

    # 4) cash deposits & withdrawals ----------------------------------------
    dep_rate = np.where(kind == "trading", 1.6, np.where(kind == "merchant", 2.0, 0.15))
    dep_counts = rng.poisson(dep_rate * active_months * np.sqrt(activity))
    d_acc = np.repeat(np.arange(n), dep_counts)
    biz_d = ctype[d_acc] == "business"
    d_amt = np.where(biz_d, rng.lognormal(np.log(2800), 0.75, len(d_acc)),
                     rng.lognormal(np.log(250), 1.0, len(d_acc))) * np.sqrt(wealth[d_acc])
    d_amt = np.round(d_amt / 10) * 10
    frames.append(pd.DataFrame({"src": -1, "dst": d_acc,
                                "timestamp": _rand_times(ctx, opened[d_acc], len(d_acc)),
                                "amount": d_amt, "channel": "cash_deposit"}))
    w_counts = rng.poisson(0.35 * active_months * np.sqrt(activity))
    w_acc = np.repeat(np.arange(n), w_counts)
    w_amt = np.round(rng.lognormal(np.log(140), 0.8, len(w_acc)) / 10) * 10
    frames.append(pd.DataFrame({"src": w_acc, "dst": -1,
                                "timestamp": _rand_times(ctx, opened[w_acc], len(w_acc)),
                                "amount": w_amt, "channel": "cash_withdrawal"}))

    # 5) benign look-alikes that classic monitoring struggles with ------------
    look = []
    # intercompany treasury loops among related companies (legitimate cycles)
    biz = np.flatnonzero(np.isin(kind, ["trading", "employer"]))
    for _ in range(max(2, n // 240)):
        grp = rng.choice(biz, size=int(rng.integers(3, 5)), replace=False)
        g_open = opened[grp].max()
        for ms in month_starts:
            if ms < g_open or rng.random() > 0.35:
                continue
            t = ms + np.timedelta64(int(rng.integers(0, 20)), "D") + np.timedelta64(10, "h")
            amt = float(rng.integers(20, 200)) * 1000
            for i, a_ in enumerate(grp):
                b_ = grp[(i + 1) % len(grp)]
                look.append((a_, b_, t, amt, "transfer"))
                amt *= rng.uniform(0.995, 1.0)
                t = t + np.timedelta64(int(rng.integers(20, 120)), "h")
    # life events: house sale / loan drawdown received and paid on within a day
    life = rng.choice(np.flatnonzero(ctype == "individual"), size=max(5, n // 60), replace=False)
    trading = np.flatnonzero(kind == "trading")
    for a_ in life:
        t = _rand_times(ctx, np.array([opened[a_]]), 1)[0]
        v = float(rng.uniform(30_000, 250_000))
        look.append((rng.choice(trading), a_, t, v, "transfer"))
        for d_ in rng.choice(trading, size=int(rng.integers(1, 3)), replace=False):
            look.append((a_, d_, t + np.timedelta64(int(rng.integers(2, 22)), "h"),
                         v * rng.uniform(0.45, 0.5), "transfer"))
    if look:
        e = np.array(look, dtype=object)
        frames.append(pd.DataFrame({"src": e[:, 0].astype(int), "dst": e[:, 1].astype(int),
                                    "timestamp": e[:, 2].astype("datetime64[s]"),
                                    "amount": e[:, 3].astype(float), "channel": e[:, 4]}))

    bg = pd.concat(frames, ignore_index=True)
    bg["amount"] = bg["amount"].clip(1, 400_000).round(2)
    bg["is_laundering"] = 0
    bg["typology"] = "none"
    bg["instance_id"] = ""
    bg["network_id"] = ""
    return bg


# ---------------------------------------------------------------------------
# Laundering typologies
# ---------------------------------------------------------------------------
class _Injector:
    def __init__(self, ctx: _Ctx):
        self.ctx = ctx
        self.rows: list[tuple] = []
        self.k = 0

    def _new_instance(self) -> str:
        self.k += 1
        return f"TYP-{self.k:05d}"

    def _add(self, s, d, t, a, ch, typ, inst, net):
        if t >= self.ctx.end:
            return
        self.rows.append((s, d, t, round(float(a), 2), ch, 1, typ, inst, net))

    def _ch(self, s, d):
        cs = self.ctx.country.get(s, "")
        cd = self.ctx.country.get(d, "")
        return "wire" if cs != cd else "transfer"

    def _h(self, lo, hi):
        return np.timedelta64(int(self.ctx.rng.uniform(lo, hi) * 3600), "s")

    # each typology takes (t0, actors) ------------------------------------
    def fan_out(self, t0, pool, hubs, exits, net):
        r, inst = self.ctx.rng, self._new_instance()
        src = r.choice(hubs)
        k = min(len(pool), int(r.integers(6, 13)))
        for d in r.choice(pool, size=k, replace=False):
            if d == src:
                continue
            self._add(src, d, t0 + self._h(0, 60), r.uniform(1500, 9600), self._ch(src, d),
                      "fan_out", inst, net)

    def fan_in(self, t0, pool, hubs, exits, net):
        r, inst = self.ctx.rng, self._new_instance()
        dst = r.choice(exits)
        k = min(len(pool), int(r.integers(6, 13)))
        for s in r.choice(pool, size=k, replace=False):
            if s == dst:
                continue
            self._add(s, dst, t0 + self._h(0, 80), r.uniform(1500, 9600), self._ch(s, dst),
                      "fan_in", inst, net)

    def cycle(self, t0, pool, hubs, exits, net):
        r, inst = self.ctx.rng, self._new_instance()
        L = int(r.integers(3, 7))
        members = list(r.choice(pool, size=min(L, len(pool)), replace=False))
        amt, t = r.uniform(15_000, 120_000), t0
        for i, s in enumerate(members):
            d = members[(i + 1) % len(members)]
            self._add(s, d, t, amt, self._ch(s, d), "cycle", inst, net)
            amt *= r.uniform(0.96, 0.99)
            t = t + self._h(2, 48)

    def structuring(self, t0, pool, hubs, exits, net):
        r, inst = self.ctx.rng, self._new_instance()
        a = r.choice(pool)
        total, t = 0.0, t0
        for _ in range(int(r.integers(4, 10))):
            v = r.uniform(0.90, 0.995) * CASH_THRESHOLD
            v = round(v / 10) * 10
            self._add(CASH, a, t, v, "cash_deposit", "structuring", inst, net)
            total += v
            t = t + self._h(6, 40)
        d = r.choice(exits)
        if d != a:
            self._add(a, d, t + self._h(12, 72), total * r.uniform(0.90, 0.97),
                      self._ch(a, d), "structuring", inst, net)

    def scatter_gather(self, t0, pool, hubs, exits, net):
        r, inst = self.ctx.rng, self._new_instance()
        src, sink = r.choice(hubs), r.choice(exits)
        mids = [m for m in r.choice(pool, size=min(len(pool), int(r.integers(3, 7))), replace=False)
                if m not in (src, sink)]
        if not mids:
            return
        total = r.uniform(40_000, 200_000)
        shares = r.dirichlet(np.ones(len(mids))) * total
        for m, v in zip(mids, shares):
            t1 = t0 + self._h(0, 24)
            self._add(src, m, t1, v, self._ch(src, m), "scatter_gather", inst, net)
            self._add(m, sink, t1 + self._h(6, 60), v * r.uniform(0.95, 0.99),
                      self._ch(m, sink), "scatter_gather", inst, net)

    def pass_through(self, t0, pool, hubs, exits, net):
        r, inst = self.ctx.rng, self._new_instance()
        a = r.choice(pool)
        t = t0
        for _ in range(int(r.integers(2, 5))):
            s = r.choice(hubs)
            if s == a:
                continue
            v = r.uniform(20_000, 150_000)
            self._add(s, a, t, v, self._ch(s, a), "pass_through", inst, net)
            outs = r.choice(exits, size=int(r.integers(1, 4)))
            split = r.dirichlet(np.ones(len(outs))) * v * r.uniform(0.90, 0.99)
            for d, sv in zip(outs, split):
                if d != a:
                    self._add(a, d, t + self._h(2, 20), sv, self._ch(a, d), "pass_through", inst, net)
            t = t + self._h(48, 200)


def _inject(p: Profile, ctx: _Ctx) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng, acc = ctx.rng, ctx.accounts
    ids = acc["account_id"].to_numpy()
    opened = acc["opened_on"].to_numpy().astype("datetime64[s]")
    ctype = acc["customer_type"].to_numpy()
    kind = acc["business_kind"].to_numpy()
    country = acc["country"].to_numpy()

    # Rings favour newer retail accounts (classic money-mule profile), but not exclusively.
    young = (opened > ctx.start - np.timedelta64(540, "D")) & (opened < ctx.start + np.timedelta64(120, "D"))
    w = np.where(young, 4.0, 1.0) * np.where(ctype == "individual", 1.5, 1.0)
    eligible = opened < ctx.start + np.timedelta64(120, "D")
    w = np.where(eligible, w, 0.0)
    taken = np.zeros(len(acc), dtype=bool)
    inj = _Injector(ctx)
    typ_names = list(TYPOLOGIES)
    ring_rows = []

    def pick(k, weights, extra_mask=None):
        ww = np.where(taken, 0.0, weights)
        if extra_mask is not None:
            ww = np.where(extra_mask, ww, 0.0)
        if ww.sum() == 0:
            return np.array([], dtype=int)
        k = min(k, int((ww > 0).sum()))
        out = rng.choice(len(acc), size=k, replace=False, p=ww / ww.sum())
        taken[out] = True
        return out

    for ring in range(p.n_rings):
        net = f"NET-{ring + 1:03d}"
        members = pick(int(rng.integers(*p.ring_size)), w)
        hubs = pick(int(rng.integers(1, 3)), np.where(eligible, 1.0, 0.0),
                    extra_mask=(kind == "trading") | (ctype == "individual"))
        exit_mask = np.isin(country, HIGH_RISK_JURISDICTIONS + ("AE", "SG"))
        exits = pick(int(rng.integers(1, 3)), np.where(eligible, 1.0, 0.0), extra_mask=exit_mask)
        if len(exits) == 0:
            exits = pick(1, np.where(eligible, 1.0, 0.0))
        all_m = np.concatenate([members, hubs, exits])
        latest_open = opened[all_m].max()
        lo = max(latest_open, ctx.start)
        room = max(int(((ctx.end - np.timedelta64(200, "D")) - lo).astype("timedelta64[D]").astype(int)), 1)
        r_start = lo + np.timedelta64(int(rng.integers(0, room)), "D")
        dur = np.timedelta64(int(rng.integers(180, 540)), "D")
        r_end = min(r_start + dur, ctx.end - np.timedelta64(10, "D"))
        if r_end <= r_start:
            continue
        # each ring has a preferred playbook
        pref = rng.dirichlet(np.ones(len(typ_names)) * 0.8)
        n_inst = int(rng.integers(*p.instances_per_ring))
        span = (r_end - r_start).astype("timedelta64[s]").astype(np.int64)
        for _ in range(n_inst):
            t0 = r_start + np.timedelta64(int(rng.random() * span), "s")
            typ = rng.choice(typ_names, p=pref)
            getattr(inj, typ)(t0, ids[members], ids[hubs], ids[exits], net)
        for role, arr in (("member", members), ("hub", hubs), ("exit", exits)):
            for a in arr:
                ring_rows.append((ids[a], net, role))

    # Lone, opportunistic patterns among otherwise ordinary customers.
    for _ in range(p.lone_instances):
        actors = pick(int(rng.integers(5, 10)), np.where(eligible, 1.0, 0.0))
        if len(actors) < 5:
            break
        latest_open = opened[actors].max()
        lo = max(latest_open, ctx.start)
        span = (ctx.end - np.timedelta64(10, "D") - lo).astype("timedelta64[s]").astype(np.int64)
        if span <= 0:
            continue
        t0 = lo + np.timedelta64(int(rng.random() * span), "s")
        typ = rng.choice(typ_names)
        a = ids[actors]
        getattr(inj, typ)(t0, a[2:], a[:1], a[1:2], "LONE")
        for x in a:
            ring_rows.append((x, "LONE", "opportunistic"))

    cols = ["src", "dst", "timestamp", "amount", "channel", "is_laundering",
            "typology", "instance_id", "network_id"]
    df = pd.DataFrame(inj.rows, columns=cols)
    roles = pd.DataFrame(ring_rows, columns=["account_id", "network_id", "network_role"])
    return df, roles


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def generate(p: Profile) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(accounts, transactions)`` DataFrames for a profile.

    Transactions include ``history_months`` of look-back history before
    ``start_date``.  History gives detectors context (e.g. which relationships are
    routine) and is excluded from features, models and evaluation.
    """
    rng = np.random.default_rng(p.seed)
    accounts = generate_accounts(p, rng)
    start = np.datetime64(p.start_date).astype("datetime64[s]")
    end = (np.datetime64(p.start_date, "M") + np.timedelta64(p.months, "M")).astype("datetime64[s]")
    bg_start = (np.datetime64(p.start_date, "M") - np.timedelta64(p.history_months, "M")).astype("datetime64[s]")
    ctx = _Ctx(rng, start, end, accounts, dict(zip(accounts.account_id, accounts.country)), bg_start)

    bg = _background(p, ctx)
    ids = accounts["account_id"].to_numpy()
    id_or_cash = np.append(ids, CASH)          # index -1 -> CASH
    bg["from_account"] = id_or_cash[bg["src"].to_numpy()]
    bg["to_account"] = id_or_cash[bg["dst"].to_numpy()]
    bg = bg.drop(columns=["src", "dst"])

    launder, roles = _inject(p, ctx)
    launder = launder.rename(columns={"src": "from_account", "dst": "to_account"})

    tx = pd.concat([bg, launder], ignore_index=True)
    tx["timestamp"] = pd.to_datetime(tx["timestamp"]).astype("datetime64[us]")
    tx = tx[(tx["timestamp"] >= pd.Timestamp(bg_start)) & (tx["timestamp"] < pd.Timestamp(end))]
    tx = tx[tx["from_account"] != tx["to_account"]]          # defensive: no self-transfers
    tx = tx.sort_values("timestamp", kind="stable").reset_index(drop=True)

    cmap = dict(zip(accounts.account_id, accounts.country))
    tx["from_country"] = tx["from_account"].map(cmap)
    tx["to_country"] = tx["to_account"].map(cmap)
    tx.loc[tx.from_account == CASH, "from_country"] = tx["to_country"]
    tx.loc[tx.to_account == CASH, "to_country"] = tx["from_country"]
    tx["is_cross_border"] = (tx["from_country"] != tx["to_country"]).astype(int)
    tx.insert(0, "txn_id", [f"TXN{i:09d}" for i in range(1, len(tx) + 1)])
    tx = tx[["txn_id", "timestamp", "from_account", "to_account", "amount", "channel",
             "from_country", "to_country", "is_cross_border",
             "is_laundering", "typology", "instance_id", "network_id"]]

    accounts = accounts.drop(columns=[c for c in accounts.columns if c.startswith("_")])
    accounts = accounts.merge(roles.drop_duplicates("account_id"), on="account_id", how="left")
    accounts["network_id"] = accounts["network_id"].fillna("")
    accounts["network_role"] = accounts["network_role"].fillna("")
    return accounts, tx
