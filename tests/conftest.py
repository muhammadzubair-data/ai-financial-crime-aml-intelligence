import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amlintel.config import ModelConfig, get_profile  # noqa: E402

TEST_CFG = ModelConfig(train_months=6, valid_months=2, capacity_per_month=10)


@pytest.fixture(scope="session")
def built(tmp_path_factory):
    """Run the full pipeline once on the small 'test' profile."""
    from amlintel.build import build
    out = tmp_path_factory.mktemp("build")
    manifest = build(get_profile("test"), cfg=TEST_CFG, out_dir=out)
    return out, manifest


@pytest.fixture(scope="session")
def small_data():
    from amlintel.data.generator import generate
    from amlintel.data.pipeline import add_month
    p = get_profile("test")
    accounts, tx = generate(p)
    return p, accounts, add_month(tx, p.start_date)


def make_tx(rows):
    """Tiny transaction frames for detector unit tests.

    rows: (from, to, 'YYYY-MM-DD HH:MM', amount, channel)
    """
    df = pd.DataFrame(rows, columns=["from_account", "to_account", "timestamp", "amount", "channel"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["txn_id"] = [f"T{i:04d}" for i in range(len(df))]
    df["month"] = df.timestamp.dt.strftime("%Y-%m")
    df["is_cross_border"] = 0
    df["from_country"] = "GB"
    df["to_country"] = "GB"
    return df
