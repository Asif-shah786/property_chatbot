"""
Data loading, preprocessing, and SQLite DB creation for Manchester listings.

Usage (CLI):
  python "Core Listing Search & Filtering/data  preprocessing  and db creation.py" \
         --csv data/manchester_properties_for_sale.csv \
         --out-dir data/clean

This script:
  1) Loads a raw CSV
  2) Cleans/normalizes key fields and derives helpful columns
  3) Saves cleaned data to Parquet and CSV
  4) Builds a SQLite DB with useful single and composite indexes

It is importable and runnable.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# -----------------------------
# Normalization dictionaries
# -----------------------------
TXN_NORMALIZE = {
    "buy": "buy",
    "sale": "buy",
    "for sale": "buy",
    "sell": "buy",
    "rent": "rent",
    "to let": "rent",
    "let": "rent",
    "rental": "rent",
}

SUBTYPE_NORMALIZE = {
    "flat": "flat",
    "apartment": "flat",
    "maisonette": "flat",
    "semi": "semi-detached",
    "semi-detached": "semi-detached",
    "terraced": "terraced",
    "terrace": "terraced",
    "detached": "detached",
    "link-detached": "detached",
    "end terrace": "terraced",
    "mid terrace": "terraced",
    "bungalow": "bungalow",
    "studio": "studio",
    "house": "house",
    "cottage": "house",
    "mews": "house",
}

BOOL_COLS = [
    "premiumListing",
    "featuredProperty",
    "commercial",
    "development",
    "residential",
    "students",
    "auction",
    "feesApply",
    "onlineViewingsAvailable",
    "isRecent",
    "hasBrandPlus",
]

NUM_COLS = [
    "price",
    "displayPrice",
    "bedrooms",
    "latitude",
    "longitude",
    "numberOfImages",
    "numberOfFloorplans",
    "numberOfVirtualTours",
]

DATE_COLS = ["firstVisibleDate", "listingUpdateDate"]

POSTCODE_RE = re.compile(r"\b([A-Z]{1,2}\d{1,2}[A-Z]?)\s*\d[A-Z]{2}\b", re.I)


# -----------------------------
# Helpers
# -----------------------------
def to_bool(s):
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float)):
        return bool(int(s))
    s = str(s).strip().lower()
    if s in {"true", "t", "yes", "y", "1"}:
        return True
    if s in {"false", "f", "no", "n", "0"}:
        return False
    return np.nan


def norm_str(s):
    return np.nan if pd.isna(s) else re.sub(r"\s+", " ", str(s)).strip()


def norm_txn(s):
    s = norm_str(s)
    if s is np.nan:
        return np.nan
    key = str(s).lower()
    for k, v in TXN_NORMALIZE.items():
        if k in key:
            return v
    return key


def norm_subtype(s):
    s = norm_str(s)
    if s is np.nan:
        return np.nan
    key = str(s).lower()
    if key in SUBTYPE_NORMALIZE:
        return SUBTYPE_NORMALIZE[key]
    for k, v in SUBTYPE_NORMALIZE.items():
        if k in key:
            return v
    return key


def extract_postcode_outward(addr):
    if pd.isna(addr):
        return np.nan
    m = POSTCODE_RE.search(str(addr).upper())
    if not m:
        return np.nan
    return m.group(1)


def parse_date(s):
    return pd.to_datetime(s, errors="coerce", dayfirst=True)


# -----------------------------
# Core pipeline
# -----------------------------
def run_pipeline(raw_csv: Path, out_dir: Path) -> tuple[pd.DataFrame, Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_out = out_dir / "manchester_properties_clean.parquet"
    csv_out = out_dir / "manchester_properties_clean.csv"
    sqlite_out = out_dir / "manchester_properties.sqlite"

    df = pd.read_csv(raw_csv, low_memory=False)
    print(f"Raw shape: {df.shape}")

    # Identify available typed columns
    num_found = [c for c in NUM_COLS if c in df.columns]
    bool_found = [c for c in BOOL_COLS if c in df.columns]
    date_found = [c for c in DATE_COLS if c in df.columns]
    print("Numeric columns to coerce:", num_found)
    print("Boolean columns to coerce:", bool_found)
    print("Date columns to parse:", date_found)

    clean = df.copy()

    # Trim strings
    for c in clean.select_dtypes(include=["object"]).columns:
        clean[c] = clean[c].map(norm_str)

    # Coerce numerics
    for c in num_found:
        clean[c + "_num"] = pd.to_numeric(clean[c], errors="coerce")

    # Coerce booleans
    for c in bool_found:
        clean[c + "_bool"] = clean[c].map(to_bool)

    # Dates + derived
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    if "firstVisibleDate" in date_found:
        clean["firstVisibleDate_dt"] = parse_date(clean["firstVisibleDate"])
        fv = clean["firstVisibleDate_dt"].dt.tz_localize(None)
        today_naive = pd.Timestamp(today.date())
        clean["firstVisibleDate_iso"] = fv.dt.strftime("%Y-%m-%d")
        clean["days_on_portal"] = fv.apply(
            lambda d: (today_naive - d.normalize()).days if pd.notna(d) else np.nan
        )

    if "listingUpdateDate" in date_found:
        clean["listingUpdateDate_dt"] = parse_date(clean["listingUpdateDate"])
        clean["listingUpdateDate_iso"] = clean["listingUpdateDate_dt"].dt.strftime(
            "%Y-%m-%d"
        )

    # Normalize categoricals
    if "transactionType" in clean.columns:
        clean["transactionType_norm"] = clean["transactionType"].map(norm_txn)

    if "propertySubType" in clean.columns:
        clean["propertySubType_norm"] = clean["propertySubType"].map(norm_subtype)

    if "propertyTypeFullDescription" in clean.columns:
        clean["propertyTypeFullDescription_norm"] = clean[
            "propertyTypeFullDescription"
        ].str.lower()

    if "branchLocation" in clean.columns:
        clean["branchLocation_norm"] = clean["branchLocation"].str.title()

    # Amenities / bands
    if "numberOfVirtualTours_num" in clean.columns:
        clean["hasVirtualTour_bool"] = clean["numberOfVirtualTours_num"].fillna(0) > 0
    if "numberOfFloorplans_num" in clean.columns:
        clean["hasFloorplan_bool"] = clean["numberOfFloorplans_num"].fillna(0) > 0
    if "numberOfImages_num" in clean.columns:
        clean["hasImages_bool"] = clean["numberOfImages_num"].fillna(0) > 0
    if "auction_bool" not in clean.columns and "auction" in clean.columns:
        clean["auction_bool"] = clean["auction"].map(to_bool)

    if "price_num" in clean.columns:
        bins = [-1, 50000, 75000, 100000, 150000, 200000, 300000, 500000, 1e9]
        labels = [
            "<=50k",
            "50-75k",
            "75-100k",
            "100-150k",
            "150-200k",
            "200-300k",
            "300-500k",
            "500k+",
        ]
        clean["price_band"] = pd.cut(clean["price_num"], bins=bins, labels=labels)

    if "displayAddress" in clean.columns:
        clean["postcode_outward"] = clean["displayAddress"].map(
            extract_postcode_outward
        )

    if "bedrooms_num" in clean.columns:
        clean["bedrooms_int"] = (
            clean["bedrooms_num"].fillna(-1).astype(int).replace({-1: np.nan})
        )

    if "transactionType_norm" in clean.columns:
        mask_missing = clean["transactionType_norm"].isna()
        clean.loc[mask_missing, "transactionType_norm"] = "buy"

    # Dedup
    if "id" in clean.columns:
        dedup_cols = ["id"]
    else:
        dedup_cols = [
            c
            for c in ["displayAddress", "price_num", "bedrooms_int"]
            if c in clean.columns
        ]
    before = len(clean)
    clean = (
        clean.drop_duplicates(subset=dedup_cols)
        if dedup_cols
        else clean.drop_duplicates()
    )
    after = len(clean)
    print(f"Deduplicated: {before} -> {after}")

    # Save cleaned
    clean.to_parquet(parquet_out, index=False)
    clean.to_csv(csv_out, index=False)
    print("Saved:", parquet_out, "and", csv_out)

    # SQLite
    with sqlite3.connect(sqlite_out) as conn:
        clean.to_sql("listings", conn, if_exists="replace", index=False)
        cur = conn.cursor()
        idx_cols = [
            "transactionType_norm",
            "price_num",
            "bedrooms_int",
            "auction_bool",
            "students_bool",
            "hasVirtualTour_bool",
            "hasFloorplan_bool",
            "branchLocation_norm",
            "postcode_outward",
        ]
        for col in idx_cols:
            if col in clean.columns:
                cur.execute(
                    f'CREATE INDEX IF NOT EXISTS idx_{col} ON listings("{col}")'
                )

        if {"transactionType_norm", "price_num", "bedrooms_int"} <= set(clean.columns):
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_txn_price_beds "
                "ON listings(transactionType_norm, price_num, bedrooms_int)"
            )
        conn.commit()
    print("SQLite ready:", sqlite_out)

    return clean, parquet_out, csv_out, sqlite_out


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess property listings and build SQLite DB"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/manchester_properties_for_sale.csv"),
        help="Path to the raw CSV input",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/clean"),
        help="Directory to write cleaned data and SQLite",
    )
    args = parser.parse_args()

    run_pipeline(args.csv, args.out_dir)


if __name__ == "__main__":
    main()
