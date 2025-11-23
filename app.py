import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import feedparser

# Third-party: you need to install these:
# pip install sentence-transformers faiss-cpu
import faiss
from sentence_transformers import SentenceTransformer

# ---------------------- CONFIG & PATHS ---------------------- #

BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

FAISS_INDEX_PATH = DATA_DIR / "faiss_index.bin"
PROCESSED_DOCS_PATH = DATA_DIR / "processed_docs.parquet"
RAW_JSON_PATH = RAW_DIR / "rss_docs.json"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Example RSS feeds – tweak as you like
RSS_FEEDS = {
    "FT Markets": "https://www.ft.com/markets/rss",
    "Reuters Business": "https://feeds.reuters.com/reuters/businessNews",
    "Yahoo Finance (AAPL, MSFT, GOOGL)": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL,MSFT,GOOGL&region=US&lang=en-US",
}

# ---------------------- STREAMLIT PAGE CONFIG ---------------------- #

st.set_page_config(
    page_title="Finance Semantic Tracker",
    page_icon="💹",
    layout="wide",
)


# ---------------------- EMBEDDING MODEL (CACHED) ---------------------- #

@st.cache_resource
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# ---------------------- DATA MODELS ---------------------- #

@dataclass
class SearchResult:
    score: float
    title: str
    text_snippet: str
    source: str
    ticker: Optional[str]
    published_at: Optional[str]
    url: Optional[str]


# ---------------------- INGESTION: RSS FETCH & SAVE ---------------------- #

def parse_rss_feed(url: str, source_name: str) -> List[Dict[str, Any]]:
    feed = feedparser.parse(url)
    docs: List[Dict[str, Any]] = []

    for entry in feed.entries:
        published_iso = None
        if getattr(entry, "published_parsed", None):
            try:
                published_dt = datetime(*entry.published_parsed[:6])
                published_iso = published_dt.isoformat()
            except Exception:
                published_iso = None

        title = entry.get("title", "") or ""
        summary = entry.get("summary", "") or ""
        text = f"{title}\n\n{summary}".strip()

        docs.append(
            {
                "title": title,
                "text": text,
                "source": source_name,
                "ticker": None,  # can be filled later with ticker detection
                "published_at": published_iso,
                "url": entry.get("link", None),
            }
        )
    return docs


def fetch_all_rss(feeds: Dict[str, str]) -> List[Dict[str, Any]]:
    all_docs: List[Dict[str, Any]] = []
    for name, url in feeds.items():
        docs = parse_rss_feed(url, name)
        all_docs.extend(docs)
    return all_docs


def save_raw_docs(docs: List[Dict[str, Any]]) -> None:
    RAW_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RAW_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)


def load_raw_docs() -> List[Dict[str, Any]]:
    if not RAW_JSON_PATH.exists():
        return []
    with open(RAW_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------- CLEANING & INDEX BUILDING ---------------------- #

def clean_and_prepare(docs: List[Dict[str, Any]]) -> pd.DataFrame:
    if not docs:
        return pd.DataFrame()

    df = pd.DataFrame(docs)

    # Ensure expected columns exist
    for col in ["title", "text", "source", "ticker", "published_at", "url"]:
        if col not in df.columns:
            df[col] = None

    df["text"] = df["text"].astype(str).str.replace("\n", " ").str.strip()
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.len() > 50]  # drop ultra-short entries
    df = df.reset_index(drop=True)
    return df


def build_index_from_raw() -> (bool, str):
    docs = load_raw_docs()
    if not docs:
        return False, "No raw docs found. Fetch RSS feeds first."

    df = clean_and_prepare(docs)
    if df.empty:
        return False, "No usable documents after cleaning."

    model = get_embedder()
    texts = df["text"].tolist()
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    # Create FAISS index (cosine similarity via inner product on normalized vectors)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    df.to_parquet(PROCESSED_DOCS_PATH, index=False)

    return True, f"Index built with {len(df)} documents."


# ---------------------- SEMANTIC SEARCH ENGINE ---------------------- #

class FinanceSemanticEngine:
    def __init__(self, index: faiss.Index, doc_df: pd.DataFrame):
        self.index = index
        self.doc_df = doc_df
        self.model = get_embedder()

    def search(
        self,
        query: str,
        k: int = 10,
        ticker_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        if not query.strip():
            return []

        q_emb = self.model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, idxs = self.index.search(q_emb, k)
        scores = scores[0]
        idxs = idxs[0]

        results: List[SearchResult] = []
        ticker_filter_norm = ticker_filter.strip().upper() if ticker_filter else None
        source_filter_norm = source_filter.strip().lower() if source_filter else None

        for score, idx in zip(scores, idxs):
            if idx == -1:
                continue

            row = self.doc_df.iloc[int(idx)]

            # Apply optional filters
            row_ticker = (row.get("ticker") or "").strip().upper()
            row_source = (row.get("source") or "").strip().lower()

            if ticker_filter_norm and row_ticker != ticker_filter_norm:
                continue
            if source_filter_norm and source_filter_norm not in row_source:
                continue

            text = str(row.get("text", ""))
            snippet = text[:400] + ("..." if len(text) > 400 else "")

            results.append(
                SearchResult(
                    score=float(score),
                    title=str(row.get("title") or "Untitled"),
                    text_snippet=snippet,
                    source=str(row.get("source") or "unknown"),
                    ticker=row.get("ticker"),
                    published_at=row.get("published_at"),
                    url=row.get("url"),
                )
            )

        return results


@st.cache_resource
def load_engine() -> Optional[FinanceSemanticEngine]:
    if not (FAISS_INDEX_PATH.exists() and PROCESSED_DOCS_PATH.exists()):
        return None

    index = faiss.read_index(str(FAISS_INDEX_PATH))
    doc_df = pd.read_parquet(PROCESSED_DOCS_PATH)
    return FinanceSemanticEngine(index=index, doc_df=doc_df)


# ---------------------- UI: SIDEBAR (ADMIN & WATCHLIST) ---------------------- #

st.sidebar.title("⚙️ Admin & Watchlists")

st.sidebar.subheader("Data & Index")

if st.sidebar.button("1️⃣ Fetch latest RSS & save"):
    with st.spinner("Fetching RSS feeds..."):
        docs = fetch_all_rss(RSS_FEEDS)
        save_raw_docs(docs)
    st.sidebar.success(f"Fetched and saved {len(docs)} raw RSS items.")

if st.sidebar.button("2️⃣ Build / Rebuild semantic index"):
    with st.spinner("Building FAISS index... (first time can be slow)"):
        ok, msg = build_index_from_raw()
    if ok:
        # Clear cached engine so it reloads with the new index
        try:
            load_engine.clear()
        except Exception:
            pass
        st.sidebar.success(msg)
    else:
        st.sidebar.error(msg)

st.sidebar.markdown("---")
st.sidebar.subheader("Saved queries (watchlist)")

default_watchlist = {
    "AI chips & regulation": "AI chip export restrictions and impact on Nvidia margins",
    "Indian fintech in UK": "Indian fintech companies expanding to the UK and regulatory challenges",
    "US rates & banks": "How rising US interest rates are affecting regional bank liquidity and risk",
}

watch_choice = st.sidebar.selectbox(
    "Pick a saved query (optional)",
    options=[""] + list(default_watchlist.keys()),
)

# ---------------------- UI: MAIN SEARCH AREA ---------------------- #

st.title("💹 Finance Semantic Tracker")
st.write(
    "This app fetches finance news via RSS, builds a semantic vector index, "
    "and lets you **search by meaning**, not just keywords."
)

# Inputs
if "current_query" not in st.session_state:
    st.session_state["current_query"] = (
        default_watchlist[watch_choice] if watch_choice else ""
    )

if watch_choice:
    st.session_state["current_query"] = default_watchlist[watch_choice]

query = st.text_input(
    "Ask a question about markets, companies, or themes:",
    value=st.session_state.get("current_query", ""),
    placeholder="Example: What are the main risks mentioned recently about big tech valuations?",
)

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    k = st.slider("Number of results", min_value=3, max_value=30, value=10, step=1)
with col2:
    ticker_filter = st.text_input("Filter by ticker (optional)", value="")
with col3:
    source_filter = st.text_input("Filter by source keyword (optional)", value="")

st.markdown("---")

engine = load_engine()

if engine is None:
    st.warning(
        "No index found yet. Use the **sidebar**:\n\n"
        "1. Click **'Fetch latest RSS & save'**\n"
        "2. Then click **'Build / Rebuild semantic index'**\n\n"
        "After that, rerun your search."
    )
else:
    if st.button("Search 🔍"):
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("Running semantic search..."):
                results = engine.search(
                    query=query,
                    k=k,
                    ticker_filter=ticker_filter or None,
                    source_filter=source_filter or None,
                )

            st.subheader(f"Results for: “{query}”")

            if not results:
                st.info(
                    "No results found. Try a broader query or remove some filters."
                )
            else:
                for r in results:
                    with st.container():
                        st.markdown(f"### {r.title}")

                        meta_parts = []
                        if r.ticker:
                            meta_parts.append(f"**Ticker:** {r.ticker}")
                        if r.source:
                            meta_parts.append(f"**Source:** {r.source}")
                        if r.published_at:
                            try:
                                dt = datetime.fromisoformat(r.published_at)
                                meta_parts.append(f"**Date:** {dt.date().isoformat()}")
                            except Exception:
                                meta_parts.append(f"**Date:** {r.published_at}")

                        if meta_parts:
                            st.markdown(" | ".join(meta_parts))

                        st.markdown(
                            f"<small>Relevance score: {r.score:.3f}</small>",
                            unsafe_allow_html=True,
                        )

                        st.write(r.text_snippet)

                        if r.url:
                            st.markdown(f"[Open source link]({r.url})")

                        st.markdown("---")
