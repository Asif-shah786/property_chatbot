"""
Group 1 Hybrid Search (Structured filters + semantic rerank)

CLI usage:
  python -m core_listing_search_and_filtering.group1_hybrid_search \
      --query "2-bed flats in M3 under 150000 to buy" \
      --k 5

Or import and call:
  from core_listing_search_and_filtering.group1_hybrid_search import search_listings
  df = search_listings("auction properties with floorplan", k=10)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Chroma
import chromadb
from chromadb.config import Settings as ChromaSettings

# Embeddings (LangChain Azure)
try:
    from langchain_openai import AzureOpenAIEmbeddings
except Exception:  # fallback if user has community alias
    from langchain_community.embeddings import AzureOpenAIEmbeddings  # type: ignore

# Optional TF-IDF keyword support
try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

    _HAS_SKLEARN = True
except Exception:
    TfidfVectorizer = None  # type: ignore
    cosine_similarity = None  # type: ignore
    _HAS_SKLEARN = False


# ---------------- CONFIG ----------------
PARQUET_OUT = Path("data/clean/manchester_properties_clean.parquet")
SQLITE_OUT = Path("data/clean/manchester_properties.sqlite")

CHROMA_DIR = Path("data/vector_stores/manchester_properties")
CHROMA_COLLECTION = "gm_listings"

TEXT_FIELDS = [
    "heading",
    "summary",
    "displayAddress",
    "propertyTypeFullDescription",
    "formattedBranchName",
]

# Azure env (fallback to project's .env naming where possible)
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("azureKey") or ""
AZURE_OPENAI_ENDPOINT = (
    os.getenv("AZURE_OPENAI_ENDPOINT")
    or "https://asha-me2poe2i-eastus2.cognitiveservices.azure.com/"
)
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION") or "2024-02-01"
AZURE_EMBEDDING_DEPLOYMENT = (
    os.getenv("AZURE_EMBEDDING_DEPLOYMENT") or "text-embedding-3-small"
)

# City centre centroid (St Peter’s Square approx)
CITY_CENTRE_LAT, CITY_CENTRE_LON = 53.4780, -2.2445
DEFAULT_RADIUS_M = 2000


# ---------------- Utilities ----------------
def _ensure_text_blob(df: pd.DataFrame) -> pd.DataFrame:
    def mk_text_blob(row: pd.Series) -> str | None:
        parts: List[str] = []
        for c in TEXT_FIELDS:
            if c in row and pd.notna(row[c]):
                parts.append(str(row[c]))
        return " | ".join(parts) if parts else None

    if "text_blob" not in df.columns:
        df = df.copy()
        # Use list comprehension to avoid pandas typing complaints
        df["text_blob"] = [mk_text_blob(r) for _, r in df.iterrows()]
    return df


def _build_embeddings_client() -> AzureOpenAIEmbeddings:
    if not AZURE_OPENAI_API_KEY:
        raise RuntimeError(
            "Azure OpenAI API key not set (env AZURE_OPENAI_API_KEY or azureKey)"
        )
    return AzureOpenAIEmbeddings(
        azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        model=AZURE_EMBEDDING_DEPLOYMENT,
    )


# ---------------- Chroma setup ----------------
def _open_collection():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR), settings=ChromaSettings(anonymized_telemetry=False)
    )
    try:
        col = client.get_collection(CHROMA_COLLECTION)
    except Exception:
        col = client.create_collection(
            CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
        )
    return col


def upsert_missing_embeddings(df: pd.DataFrame, batch_size: int = 512) -> None:
    col = _open_collection()
    emb = _build_embeddings_client()
    have = col.count()
    existing: set[str] = set()
    if have:
        try:
            existing = set(col.get(ids=None, include=[])["ids"])  # type: ignore[assignment]
        except Exception:
            existing = set()

    todo = df[~df["id"].astype(str).isin(existing)]
    todo = todo[pd.notna(todo["text_blob"]) & (todo["text_blob"].str.len() > 0)]
    print(f"Embeddings: have={have}  missing={len(todo)}")

    for i in range(0, len(todo), batch_size):
        chunk = todo.iloc[i : i + batch_size]
        ids: List[str] = chunk["id"].astype(str).tolist()
        docs: List[str] = chunk["text_blob"].astype(str).tolist()

        metas: List[Dict[str, Any]] = []
        for _, r in chunk.iterrows():
            price_val = r.get("price_num")
            beds_val = r.get("bedrooms_int")
            lat_val = r.get("latitude_num")
            lon_val = r.get("longitude_num")

            metas.append(
                {
                    "id": str(r.get("id")),
                    "price_num": (
                        float(price_val)
                        if price_val is not None and pd.notna(price_val)
                        else 0.0
                    ),
                    "bedrooms_int": (
                        int(beds_val)
                        if beds_val is not None and pd.notna(beds_val)
                        else 0
                    ),
                    "transactionType_norm": (
                        str(r.get("transactionType_norm"))
                        if r.get("transactionType_norm") is not None
                        else ""
                    ),
                    "propertySubType_norm": (
                        str(r.get("propertySubType_norm"))
                        if r.get("propertySubType_norm") is not None
                        else ""
                    ),
                    "latitude_num": (
                        float(lat_val)
                        if lat_val is not None and pd.notna(lat_val)
                        else 0.0
                    ),
                    "longitude_num": (
                        float(lon_val)
                        if lon_val is not None and pd.notna(lon_val)
                        else 0.0
                    ),
                    "postcode_outward": (
                        str(r.get("postcode_outward"))
                        if r.get("postcode_outward") is not None
                        else ""
                    ),
                    "hasVirtualTour_bool": (
                        bool(r.get("hasVirtualTour_bool"))
                        if r.get("hasVirtualTour_bool") in [True, False]
                        else False
                    ),
                    "hasFloorplan_bool": (
                        bool(r.get("hasFloorplan_bool"))
                        if r.get("hasFloorplan_bool") in [True, False]
                        else False
                    ),
                    "auction_bool": (
                        bool(r.get("auction_bool"))
                        if r.get("auction_bool") in [True, False]
                        else False
                    ),
                }
            )
        vecs: List[List[float]] = emb.embed_documents(docs)
        col.add(ids=ids, documents=docs, embeddings=vecs, metadatas=metas)  # type: ignore[arg-type]
    print("Embedding sync complete.")


# ---------------- Parsing ----------------
AREA_TOKEN = re.compile(r"\b(m\d{1,2}|m\d{1,2}[a-z]|bl\d{1,2}|wa\d{1,2})\b", re.I)


def parse_price(q: str) -> Tuple[Any, Any]:
    q = q.lower()
    m_under = re.search(r"(?:under|below|<=?)\s*£?\s*([\d,]+)k?", q)
    m_between = re.search(
        r"(?:between)\s*£?\s*([\d,]+)\s*(?:and|-|to)\s*£?\s*([\d,]+)", q
    )
    m_over = re.search(r"(?:over|>=?)\s*£?\s*([\d,]+)k?", q)

    def num(x: str | None) -> float | None:
        if x is None:
            return None
        x2 = x.replace(",", "")
        return float(x2)

    if m_between:
        return num(m_between.group(1)), num(m_between.group(2))
    if m_under:
        return None, num(m_under.group(1))
    if m_over:
        return num(m_over.group(1)), None
    return None, None


def parse_beds(q: str) -> Tuple[Any, Any]:
    ql = q.lower()
    m_exact = re.search(r"(\d+)\s*bed(room)?", ql)
    m_range = re.search(r"(\d+)\s*-\s*(\d+)\s*beds?", ql)
    if m_range:
        return int(m_range.group(1)), int(m_range.group(2))
    if m_exact:
        b = int(m_exact.group(1))
        return b, b
    if "studio" in ql:
        return 0, 0
    return None, None


def parse_txn(q: str) -> str | None:
    ql = q.lower()
    if any(k in ql for k in ["rent", "to let", "let", "rental"]):
        return "rent"
    if any(k in ql for k in ["buy", "sale", "for sale", "purchase"]):
        return "buy"
    return None


def parse_flags(q: str) -> Dict[str, Any]:
    ql = q.lower()
    return {
        "auction_bool": True if "auction" in ql else None,
        "hasVirtualTour_bool": (
            True if ("virtual tour" in ql or "video tour" in ql) else None
        ),
        "hasFloorplan_bool": (
            True if ("floorplan" in ql or "floor plan" in ql) else None
        ),
        "student_like": True if "student" in ql else None,
        "quiet_like": (
            True if any(k in ql for k in ["quiet", "peaceful", "residential"]) else None
        ),
        "near_city": (
            True if re.search(r"\b(city\s*centre|city\s*center)\b", ql) else None
        ),
    }


def parse_areas(q: str) -> List[str]:
    pcs = AREA_TOKEN.findall(q)
    return list({pc.upper() for pc in pcs})


def parse_query(user_q: str) -> Dict[str, Any]:
    pmin, pmax = parse_price(user_q)
    bmin, bmax = parse_beds(user_q)
    txn = parse_txn(user_q)
    flags = parse_flags(user_q)
    areas = parse_areas(user_q)

    residue = user_q
    for pat in [
        r"(under|below|over|between)\s*£?\s*[\d,]+(\s*(and|-|to)\s*£?\s*[\d,]+)?k?",
        r"\d+\s*beds?",
        r"\d+\s*bed(room)?",
        r"studio",
        r"for sale|buy|to let|rent|rental|let",
        r"auction|virtual tour|video tour|floor ?plan",
        r"city\s*centre|city\s*center",
        r"\b(m\d{1,2}[a-z]?|bl\d{1,2}|wa\d{1,2})\b",
    ]:
        residue = re.sub(pat, " ", residue, flags=re.I)
    residue = re.sub(r"\s+", " ", residue).strip()

    return {
        "price_min": pmin,
        "price_max": pmax,
        "beds_min": bmin,
        "beds_max": bmax,
        "transactionType_norm": txn,
        "areas_outward": areas,
        **flags,
        "semantic_text": residue if residue else None,
    }


# ---------------- SQL + radius filter ----------------
def sql_candidates(q: Dict[str, Any], limit: int = 5000) -> pd.DataFrame:
    conn = sqlite3.connect(SQLITE_OUT)
    wh: List[str] = []
    params: List[Any] = []

    if q.get("transactionType_norm"):
        wh.append("transactionType_norm = ?")
        params.append(q["transactionType_norm"])

    if q.get("price_min") is not None:
        wh.append("price_num >= ?")
        params.append(float(q["price_min"]))
    if q.get("price_max") is not None:
        wh.append("price_num <= ?")
        params.append(float(q["price_max"]))

    if q.get("beds_min") is not None:
        wh.append("bedrooms_int >= ?")
        params.append(int(q["beds_min"]))
    if q.get("beds_max") is not None:
        wh.append("bedrooms_int <= ?")
        params.append(int(q["beds_max"]))

    if q.get("auction_bool") is True:
        wh.append("auction_bool = 1")
    if q.get("hasVirtualTour_bool") is True:
        wh.append("hasVirtualTour_bool = 1")
    if q.get("hasFloorplan_bool") is True:
        wh.append("hasFloorplan_bool = 1")

    if q.get("areas_outward"):
        marks = ",".join(["?"] * len(q["areas_outward"]))
        wh.append(f"postcode_outward IN ({marks})")
        params.extend(q["areas_outward"])

    where_sql = ("WHERE " + " AND ".join(wh)) if wh else ""
    sql = f"""
        SELECT id, displayAddress, price_num, bedrooms_int, propertySubType_norm,
               transactionType_norm, latitude_num, longitude_num, text_blob
        FROM listings
        {where_sql}
        LIMIT {int(limit)}
    """
    df = pd.read_sql(sql, conn, params=params)
    conn.close()
    return _ensure_text_blob(df)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


def apply_radius_city_centre(
    df_in: pd.DataFrame, enabled: bool, radius_m: int = DEFAULT_RADIUS_M
) -> pd.DataFrame:
    if not enabled:
        return df_in
    if "latitude_num" not in df_in.columns or "longitude_num" not in df_in.columns:
        return df_in.iloc[0:0]
    mask: List[bool] = []
    for lat, lon in zip(df_in["latitude_num"], df_in["longitude_num"]):
        if pd.isna(lat) or pd.isna(lon):
            mask.append(False)
        else:
            mask.append(
                haversine_m(CITY_CENTRE_LAT, CITY_CENTRE_LON, float(lat), float(lon))
                <= radius_m
            )
    return df_in.loc[mask]


# ---------------- Reranking ----------------
def _cosine(a: List[float] | np.ndarray, b: List[float] | np.ndarray) -> float:
    a2 = np.asarray(a)
    b2 = np.asarray(b)
    denom = float(np.linalg.norm(a2) * np.linalg.norm(b2))
    return float(np.dot(a2, b2) / denom) if denom else 0.0


def vector_rerank(
    cands: pd.DataFrame, semantic_text: str | None, alpha: float = 0.8
) -> pd.DataFrame:
    if cands.empty:
        return cands

    # Keyword score via TF-IDF (optional)
    kw_score = np.zeros(len(cands))
    if (
        _HAS_SKLEARN
        and "text_blob" in cands.columns
        and TfidfVectorizer is not None
        and cosine_similarity is not None
    ):
        try:
            tfidf = TfidfVectorizer(stop_words="english")
            X = tfidf.fit_transform(cands["text_blob"].fillna(""))
            if semantic_text and semantic_text.strip():
                qv = tfidf.transform([semantic_text])
                kw = cosine_similarity(qv, X).ravel()
                kw_score = kw
        except Exception:
            pass

    emb_score = np.zeros(len(cands))
    if semantic_text and semantic_text.strip():
        emb = _build_embeddings_client()
        q_emb = emb.embed_query(semantic_text)
        docs = cands["text_blob"].fillna("").tolist()
        d_embs = emb.embed_documents(docs)
        emb_score = np.array([_cosine(q_emb, d) for d in d_embs])

    score = (
        kw_score
        if not (semantic_text and semantic_text.strip())
        else alpha * emb_score + (1 - alpha) * kw_score
    )
    out = cands.copy()
    out["score"] = score
    out = out.sort_values("score", ascending=False)
    return out


# ---------------- End-to-end search ----------------
def search_listings(user_query: str, k: int = 10) -> pd.DataFrame:
    if not PARQUET_OUT.exists() or not SQLITE_OUT.exists():
        raise FileNotFoundError(
            "Cleaned data not found. Run the preprocessing step first."
        )

    # Load cleaned (ensure text blob)
    df = pd.read_parquet(PARQUET_OUT)
    df = _ensure_text_blob(df)

    # Sync embeddings (idempotent)
    upsert_missing_embeddings(df)

    q = parse_query(user_query)
    cands = sql_candidates(q, limit=4000)
    cands = apply_radius_city_centre(cands, enabled=bool(q.get("near_city")))
    if cands.empty:
        return cands

    ranked = vector_rerank(cands, semantic_text=q.get("semantic_text"))
    cols = [
        "id",
        "displayAddress",
        "price_num",
        "bedrooms_int",
        "propertySubType_norm",
        "transactionType_norm",
        "score",
    ]
    cols = [c for c in cols if c in ranked.columns]
    return ranked[cols].head(k)


def main():
    parser = argparse.ArgumentParser(
        description="Group 1 hybrid search over cleaned Manchester listings"
    )
    parser.add_argument(
        "--query", type=str, default="2-bed flats in M3 under 150000 to buy"
    )
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    res = search_listings(args.query, k=args.k)
    if res.empty:
        print("No results.")
    else:
        print(res.to_string(index=False))


if __name__ == "__main__":
    main()
