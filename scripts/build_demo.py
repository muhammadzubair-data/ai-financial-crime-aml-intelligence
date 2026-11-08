"""Rebuild the data, models and case database.

    python scripts/build_demo.py              # compact demo (ships with the repo)
    python scripts/build_demo.py --profile full   # 100k accounts, multi-million transactions
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amlintel.build import main  # noqa: E402

if __name__ == "__main__":
    main()
