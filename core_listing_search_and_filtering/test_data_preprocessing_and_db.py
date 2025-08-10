"""
Smoke tests for the data preprocessing and DB creation pipeline.

Run:
  python core_listing_search_and_filtering/test_data_preprocessing_and_db.py \
         --csv data/manchester_properties_for_sale.csv \
         --out-dir data/clean
"""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import sqlite3
import pandas as pd

# Import that works whether run as module (-m) or as a path script
try:
    from core_listing_search_and_filtering.data_preprocessing_and_db_creation import (
        run_pipeline,
    )
except ModuleNotFoundError:
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parents[1]))
    from core_listing_search_and_filtering.data_preprocessing_and_db_creation import (  # type: ignore
        run_pipeline,
    )


def run_tests(csv_path: Path, out_dir: Path) -> None:
    _, parquet_out, csv_out, sqlite_out = run_pipeline(csv_path, out_dir)

    # 1) Files exist and non-empty
    assert parquet_out.exists() and parquet_out.stat().st_size > 0
    assert csv_out.exists() and csv_out.stat().st_size > 0
    assert sqlite_out.exists() and sqlite_out.stat().st_size > 0

    # 2) Basic columns present
    df = pd.read_parquet(parquet_out)
    expected_cols = {
        "price_num",
        "bedrooms_int",
        "transactionType_norm",
        "propertySubType_norm",
        "hasVirtualTour_bool",
        "hasFloorplan_bool",
        "postcode_outward",
    }
    missing = expected_cols - set(df.columns)
    assert not missing, f"Missing cleaned cols: {missing}"

    # 3) Simple quality checks
    if "price_num" in df.columns:
        assert df["price_num"].dropna().ge(0).all()
    if "bedrooms_int" in df.columns:
        assert df["bedrooms_int"].dropna().ge(0).all()
    if "transactionType_norm" in df.columns:
        assert df["transactionType_norm"].isin(["buy", "rent"]).all()

    # 4) SQLite portal-style queries
    conn = sqlite3.connect(sqlite_out)
    cur = conn.cursor()

    def count(sql: str) -> int:
        return cur.execute(sql).fetchone()[0]

    # Should not crash and should return integers
    count("SELECT COUNT(1) FROM listings;")

    print("All tests passed. Clean data and SQLite are ready.")


def main():
    parser = argparse.ArgumentParser(
        description="Test data preprocessing and DB creation"
    )
    parser.add_argument(
        "--csv", type=Path, default=Path("data/manchester_properties_for_sale.csv")
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/clean"))
    args = parser.parse_args()
    run_tests(args.csv, args.out_dir)


if __name__ == "__main__":
    main()
