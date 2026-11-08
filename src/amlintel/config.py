"""Central configuration for the AML intelligence platform.

Two data profiles are provided:

* ``demo``  - compact, reproducible dataset that ships with the repo and powers the
              public Streamlit deployment (fits comfortably in Streamlit Cloud memory).
* ``full``  - production-scale planning profile (100k accounts, multi-million
              transactions). Run it locally with ``python scripts/build_demo.py --profile full``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# Cash-reporting threshold used by the structuring logic. Synthetic, currency-neutral.
CASH_THRESHOLD = 10_000.0

# Placeholder codes for high-risk jurisdictions. Deliberately generic: the platform
# reads the real list from policy config in production; we do not label real countries.
HIGH_RISK_JURISDICTIONS = ("HR1", "HR2", "HR3")

HOME_COUNTRIES = {
    "GB": 0.34, "DE": 0.12, "AE": 0.10, "SG": 0.08, "US": 0.10,
    "PK": 0.06, "NG": 0.05, "FR": 0.06, "NL": 0.05,
    "HR1": 0.015, "HR2": 0.01, "HR3": 0.005,
}


@dataclass(frozen=True)
class Profile:
    name: str
    n_accounts: int
    months: int = 24
    history_months: int = 2         # look-back context before the analysis window
    start_date: str = "2024-11-01"
    n_rings: int = 16               # organised laundering networks
    ring_size: tuple[int, int] = (10, 24)
    instances_per_ring: tuple[int, int] = (10, 18)
    lone_instances: int = 30        # opportunistic one-off patterns outside rings
    base_monthly_txn_rate: float = 1.2
    seed: int = 42

    @property
    def out_dir(self) -> Path:
        return DATA_DIR / self.name


PROFILES = {
    "demo": Profile(name="demo", n_accounts=6_000),
    "test": Profile(name="test", n_accounts=900, months=12, n_rings=4,
                    instances_per_ring=(5, 8), lone_instances=8, seed=7),
    "full": Profile(name="full", n_accounts=100_000, n_rings=180,
                    instances_per_ring=(8, 16), lone_instances=600,
                    base_monthly_txn_rate=0.30),
}


@dataclass
class ModelConfig:
    # Month indexes (0-based) for the time-based split.
    train_months: int = 14          # months 0..13 used for fitting
    valid_months: int = 2           # months 14..15 for early stopping / calibration
    # Remaining months form the out-of-time test holdout.
    capacity_per_month: int = 50    # investigations the team can open per month (demo scale)
    # Share of truly suspicious account-months that carry a confirmed label for
    # training.  Real AML labels come from past investigations and SARs, so many
    # laundering cases are never labelled.  Evaluation always uses full truth.
    label_coverage: float = 0.7
    neg_sample_rate: float = 1.0    # share of training negatives used to fit LightGBM (1.0 = all)
    iforest_fit_rows: int = 200_000 # Isolation Forest fits on at most this many training rows
    blend_min_gain: float = 0.01   # validation precision gain a blend needs over the pure model
    random_state: int = 42
    lgbm_params: dict = field(default_factory=lambda: {
        "objective": "binary",
        "learning_rate": 0.04,
        "num_leaves": 31,
        "min_child_samples": 40,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 2.0,
        "verbose": -1,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 2,
    })
    hybrid_weights: dict = field(default_factory=lambda: {
        "lgbm": 0.70, "network": 0.15, "rules": 0.10, "anomaly": 0.05,
    })

    def to_dict(self) -> dict:
        return asdict(self)


def get_profile(name: str) -> Profile:
    if name not in PROFILES:
        raise KeyError(f"Unknown profile '{name}'. Choose from {sorted(PROFILES)}")
    return PROFILES[name]
