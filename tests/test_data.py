import numpy as np
import pandas as pd

from amlintel.config import get_profile
from amlintel.data.generator import TYPOLOGIES, generate
from amlintel.data.pipeline import attach_labels, build_account_month_features
from amlintel.detection.models import FEATURES


def test_generator_is_deterministic():
    p = get_profile("test")
    a1, t1 = generate(p)
    a2, t2 = generate(p)
    pd.testing.assert_frame_equal(a1, a2)
    pd.testing.assert_frame_equal(t1, t2)


def test_transactions_are_well_formed(small_data):
    p, accounts, tx = small_data
    assert tx.txn_id.is_unique
    assert (tx.amount > 0).all()
    assert (tx.from_account != tx.to_account).all()
    assert tx.timestamp.is_monotonic_increasing
    known = set(accounts.account_id) | {"CASH"}
    assert set(tx.from_account) <= known and set(tx.to_account) <= known
    # cash only appears on cash channels
    assert set(tx.loc[tx.from_account == "CASH", "channel"]) == {"cash_deposit"}
    assert set(tx.loc[tx.to_account == "CASH", "channel"]) == {"cash_withdrawal"}


def test_every_typology_is_injected_inside_the_window(small_data):
    p, _, tx = small_data
    lau = tx[tx.is_laundering == 1]
    assert set(lau.typology) == set(TYPOLOGIES)
    assert lau.timestamp.min() >= pd.Timestamp(p.start_date)      # never in look-back history
    assert (tx.loc[tx.is_laundering == 0, "typology"] == "none").all()


def test_history_months_precede_window(small_data):
    p, _, tx = small_data
    assert tx.month_idx.min() == -p.history_months
    assert tx.month_idx.max() == p.months - 1


def test_features_one_row_per_account_month_and_no_leakage(small_data):
    p, accounts, tx = small_data
    f = build_account_month_features(tx, accounts, p.start_date)
    assert not f.duplicated(["account_id", "month"]).any()
    assert not any(c.startswith("label") or c in {"is_laundering", "typology", "network_id"} for c in f.columns)
    assert not any(c.startswith("label") for c in FEATURES)
    assert f.pass_through_ratio.between(0, 1).all()
    assert (f.in_out_balance.between(0, 1)).all()


def test_features_match_hand_computation(small_data):
    p, accounts, tx = small_data
    f = build_account_month_features(tx, accounts, p.start_date).set_index(["account_id", "month"])
    busy = tx[tx.from_account != "CASH"].groupby(["from_account", "month"]).size().idxmax()
    acct, month = busy
    t = tx[tx.month == month]
    out = t[t.from_account == acct]
    inn = t[t.to_account == acct]
    row = f.loc[(acct, month)]
    assert row.out_cnt == len(out)
    assert np.isclose(row.out_amt, out.amount.sum(), atol=0.01)
    assert row.in_cnt == len(inn)
    assert row.uniq_out_cp == out.loc[out.to_account != "CASH", "to_account"].nunique()
    near = inn[(inn.channel == "cash_deposit") & (inn.amount >= 9000) & (inn.amount < 10000)]
    assert row.near_thr_cnt == len(near)


def test_labels_mark_every_laundering_participant(small_data):
    p, accounts, tx = small_data
    win = tx[tx.month_idx >= 0]
    f = attach_labels(build_account_month_features(win, accounts, p.start_date), win)
    lau = win[win.is_laundering == 1]
    expected = set(zip(lau.from_account, lau.month)) | set(zip(lau.to_account, lau.month))
    expected = {e for e in expected if e[0] != "CASH"}
    got = set(zip(f.loc[f.label == 1, "account_id"], f.loc[f.label == 1, "month"]))
    assert got == expected


def test_pass_through_matches_brute_force(small_data):
    p, accounts, tx = small_data
    f = build_account_month_features(tx, accounts, p.start_date).set_index(["account_id", "month"])
    cand = f[f.pass_through_amt > 0].sort_values("pass_through_amt", ascending=False).head(5)
    for acct, month in cand.index:
        t = tx[(tx.from_account == acct) | (tx.to_account == acct)]
        ins = t[t.to_account == acct]
        outs = t[(t.from_account == acct) & (t.month == month)]
        total = 0.0
        for o in outs.itertuples():
            recent = ins[(ins.timestamp > o.timestamp - pd.Timedelta(hours=24)) & (ins.timestamp <= o.timestamp)]
            total += min(o.amount, recent.amount.sum())
        assert np.isclose(f.loc[(acct, month), "pass_through_amt"], round(total, 2), atol=0.02)
