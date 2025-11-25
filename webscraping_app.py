import os
import re
import json
import time
import math
import hashlib
import urllib.parse
import urllib.robotparser
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup

# Optional (better text extraction)
try:
    import trafilatura
    HAS_TRAFILATURA = True
except Exception:
    HAS_TRAFILATURA = False

# Embeddings
from sentence_transformers import SentenceTransformer


# ---------------------------
# Helpers
# ---------------------------

DEFAULT_UA = "SemanticSearchBot/1.0 (educational; contact: you@example.com)"
SESSION = requests.Session()

def stable_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()[:16]

def normalize_ws(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_html_response(resp: requests.Response) -> bool:
    ctype = resp.headers.get("Content-Type", "")
    return "text/html" in ctype or ctype.startswith("text/")

def strip_nav_boilerplate(text: str) -> str:
    # light cleanup (keep simple & safe)
    text = re.sub(r"\b(Cookie settings|Accept cookies|Reject all|Privacy policy)\b", " ", text, flags=re.I)
    return normalize_ws(text)

def same_domain(a: str, b: str) -> bool:
    try:
        ha = urllib.parse.urlparse(a).netloc.lower()
        hb = urllib.parse.urlparse(b).netloc.lower()
        return ha == hb and ha != ""
    except Exception:
        return False

def canonicalize_url(url: str) -> str:
    """Remove fragments, normalize scheme, keep query (some sites encode content in query)."""
    u = urllib.parse.urlparse(url.strip())
    scheme = u.scheme or "https"
    netloc = u.netloc
    path = u.path or "/"
    return urllib.parse.urlunparse((scheme, netloc, path, "", u.query, ""))

def extract_links(base_url: str, html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for a in soup.select("a[href]"):
        href = a.get("href", "").strip()
        if not href:
            continue
        if href.startswith("mailto:") or href.startswith("javascript:") or href.startswith("tel:"):
            continue
        abs_url = urllib.parse.urljoin(base_url, href)
        abs_url = canonicalize_url(abs_url)
        out.append(abs_url)
    # de-dupe while preserving order
    seen = set()
    deduped = []
    for u in out:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped

def html_to_text(url: str, html: str) -> str:
    if HAS_TRAFILATURA:
        extracted = trafilatura.extract(html, url=url, include_comments=False, include_tables=True)
        if extracted and extracted.strip():
            return strip_nav_boilerplate(extracted)

    # Fallback with BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return strip_nav_boilerplate(text)

def get_title(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    t = soup.title.string if soup.title and soup.title.string else ""
    return normalize_ws(t)


def chunk_text(text: str, chunk_words: int = 220, overlap_words: int = 50) -> List[str]:
    """Chunk by words with overlap; good enough for web content."""
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(1, chunk_words - overlap_words)
    for start in range(0, len(words), step):
        end = min(len(words), start + chunk_words)
        chunk = " ".join(words[start:end]).strip()
        if len(chunk) >= 80:  # avoid tiny chunks
            chunks.append(chunk)
        if end >= len(words):
            break
    return chunks


def cos_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarities between vectors in a (n,d) and a single vector b (d,) or (1,d)."""
    if b.ndim == 1:
        b = b[None, :]
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T  # (n,1)


# ---------------------------
# Robots + fetch
# ---------------------------

@st.cache_data(show_spinner=False)
def get_robots_parser(root_url: str, user_agent: str) -> urllib.robotparser.RobotFileParser:
    parsed = urllib.parse.urlparse(root_url)
    robots_url = urllib.parse.urlunparse((parsed.scheme, parsed.netloc, "/robots.txt", "", "", ""))
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
    except Exception:
        # If robots fails, be conservative: allow (many sites block robots fetching)
        pass
    return rp

def allowed_by_robots(url: str, user_agent: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    root = urllib.parse.urlunparse((parsed.scheme, parsed.netloc, "/", "", "", ""))
    rp = get_robots_parser(root, user_agent)
    try:
        return rp.can_fetch(user_agent, url)
    except Exception:
        return True

def fetch_url(url: str, timeout: int, user_agent: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """Returns (html, final_url, status_code)"""
    headers = {"User-Agent": user_agent}
    try:
        resp = SESSION.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        status = resp.status_code
        final_url = canonicalize_url(resp.url)
        if status >= 400:
            return None, final_url, status
        if not is_html_response(resp):
            return None, final_url, status
        return resp.text, final_url, status
    except Exception:
        return None, None, None


# ---------------------------
# Data model
# ---------------------------

@dataclass
class Record:
    doc_id: str
    url: str
    title: str
    chunk_id: int
    text: str

@dataclass
class IndexBundle:
    model_name: str
    records: List[Record]
    embeddings: np.ndarray  # (n,d)


def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 32) -> np.ndarray:
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return vecs.astype(np.float32)


def search(bundle: IndexBundle, model: SentenceTransformer, query: str, top_k: int = 8) -> List[Tuple[float, Record]]:
    qv = embed_texts(model, [query], batch_size=1)[0]
    sims = cos_sim_matrix(bundle.embeddings, qv).reshape(-1)  # (n,)
    if len(sims) == 0:
        return []
    idx = np.argpartition(-sims, min(top_k, len(sims)-1))[:top_k]
    idx = idx[np.argsort(-sims[idx])]
    return [(float(sims[i]), bundle.records[i]) for i in idx]


# ---------------------------
# Crawler
# ---------------------------

def crawl_and_build_index(
    seed_urls: List[str],
    max_pages: int,
    max_depth: int,
    same_domain_only: bool,
    timeout: int,
    per_request_delay: float,
    concurrency: int,
    user_agent: str,
    chunk_words: int,
    overlap_words: int,
    model_name: str,
) -> Tuple[IndexBundle, Dict]:
    model = SentenceTransformer(model_name)

    seed_urls = [canonicalize_url(u) for u in seed_urls if u.strip()]
    seeds_set = set(seed_urls)

    allowed_domains: Set[str] = set()
    for u in seed_urls:
        p = urllib.parse.urlparse(u)
        if p.netloc:
            allowed_domains.add(p.netloc.lower())

    queue: List[Tuple[str, int]] = [(u, 0) for u in seed_urls]
    seen: Set[str] = set()
    pages_fetched = 0

    # Storage
    records: List[Record] = []
    docs_meta: Dict[str, Dict] = {}

    def eligible(url: str) -> bool:
        if url in seen:
            return False
        pu = urllib.parse.urlparse(url)
        if pu.scheme not in ("http", "https"):
            return False
        if not pu.netloc:
            return False
        if same_domain_only and pu.netloc.lower() not in allowed_domains:
            return False
        if not allowed_by_robots(url, user_agent):
            return False
        return True

    # BFS by depth; we’ll process in small batches for concurrency
    while queue and pages_fetched < max_pages:
        batch = []
        # gather up to concurrency eligible URLs, preserving depth order
        while queue and len(batch) < concurrency and pages_fetched + len(batch) < max_pages:
            url, depth = queue.pop(0)
            url = canonicalize_url(url)
            if depth > max_depth:
                continue
            if not eligible(url):
                continue
            seen.add(url)
            batch.append((url, depth))

        if not batch:
            continue

        # fetch concurrently
        futures = []
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            for (url, depth) in batch:
                futures.append(ex.submit(fetch_url, url, timeout, user_agent))

            for fut, (url, depth) in zip(futures, batch):
                html, final_url, status = fut.result()
                # politeness delay between completed requests (simple)
                if per_request_delay > 0:
                    time.sleep(per_request_delay)

                if not html or not final_url:
                    continue

                pages_fetched += 1

                title = get_title(html)
                text = html_to_text(final_url, html)
                text = normalize_ws(text)

                if len(text) < 300:
                    continue

                doc_id = stable_hash(final_url)

                docs_meta[doc_id] = {
                    "url": final_url,
                    "title": title,
                    "chars": len(text),
                    "depth": depth,
                    "status": status,
                }

                chunks = chunk_text(text, chunk_words=chunk_words, overlap_words=overlap_words)
                for i, ch in enumerate(chunks):
                    records.append(
                        Record(
                            doc_id=doc_id,
                            url=final_url,
                            title=title or final_url,
                            chunk_id=i,
                            text=ch,
                        )
                    )

                # enqueue new links if we can go deeper
                if depth < max_depth:
                    links = extract_links(final_url, html)
                    for link in links:
                        if link not in seen:
                            if same_domain_only:
                                if urllib.parse.urlparse(link).netloc.lower() in allowed_domains:
                                    queue.append((link, depth + 1))
                            else:
                                queue.append((link, depth + 1))

    # Build embeddings
    texts = [r.text for r in records]
    if not texts:
        bundle = IndexBundle(model_name=model_name, records=[], embeddings=np.zeros((0, 384), dtype=np.float32))
        return bundle, {"pages_fetched": pages_fetched, "chunks": 0, "docs": 0}

    embs = embed_texts(model, texts, batch_size=32)
    bundle = IndexBundle(model_name=model_name, records=records, embeddings=embs)

    stats = {
        "pages_fetched": pages_fetched,
        "docs": len(docs_meta),
        "chunks": len(records),
        "embedding_dim": int(embs.shape[1]),
    }
    return bundle, stats


def save_bundle(path: str, bundle: IndexBundle) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model_name": bundle.model_name,
        "records": [asdict(r) for r in bundle.records],
        "embeddings": bundle.embeddings.tolist(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

def load_bundle(path: str) -> IndexBundle:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    records = [Record(**r) for r in payload["records"]]
    embs = np.array(payload["embeddings"], dtype=np.float32)
    return IndexBundle(model_name=payload["model_name"], records=records, embeddings=embs)


def highlight_snippet(text: str, query: str, max_len: int = 300) -> str:
    q_words = [w for w in re.findall(r"[A-Za-z0-9_]+", query) if len(w) >= 3]
    if not q_words:
        return text[:max_len] + ("…" if len(text) > max_len else "")
    # simple highlight around first occurrence
    lower = text.lower()
    positions = [(lower.find(w.lower()), w) for w in q_words if lower.find(w.lower()) != -1]
    if not positions:
        return text[:max_len] + ("…" if len(text) > max_len else "")
    pos, w = sorted(positions, key=lambda x: x[0])[0]
    start = max(0, pos - max_len // 2)
    end = min(len(text), start + max_len)
    snippet = text[start:end]
    snippet = re.sub(r"\s+", " ", snippet).strip()
    return ("…" if start > 0 else "") + snippet + ("…" if end < len(text) else "")


# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="Web-Scraped Semantic Search", layout="wide")

st.title("🔎 Web-Scraped Semantic Search")
st.caption("Scrape pages → chunk → embed → semantic search (local embeddings). Please respect each site’s Terms and robots.txt.")

with st.sidebar:
    st.header("Crawl & Index")

    seed_text = st.text_area(
        "Seed URLs (one per line)",
        value="https://example.com/\n",
        height=120,
    )

    model_name = st.selectbox(
        "Embedding model",
        options=[
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
        ],
        index=0,
        help="MiniLM is fast; mpnet is stronger but heavier.",
    )

    max_pages = st.slider("Max pages", 5, 500, 60, step=5)
    max_depth = st.slider("Max depth", 0, 6, 2)
    same_domain_only = st.checkbox("Stay on same domain(s) as seeds", value=True)

    concurrency = st.slider("Concurrency", 1, 12, 6)
    timeout = st.slider("Request timeout (sec)", 4, 30, 12)
    delay = st.slider("Delay between requests (sec)", 0.0, 2.0, 0.25, step=0.05)

    st.subheader("Chunking")
    chunk_words = st.slider("Chunk size (words)", 120, 420, 220, step=20)
    overlap_words = st.slider("Overlap (words)", 0, 120, 50, step=10)

    user_agent = st.text_input("User-Agent", value=DEFAULT_UA)

    st.divider()
    index_path = st.text_input("Index file path", value="web_index.json")
    colA, colB = st.columns(2)
    build_clicked = colA.button("🧱 Scrape & Build Index", use_container_width=True)
    load_clicked = colB.button("📂 Load Index", use_container_width=True)

# Session state
if "bundle" not in st.session_state:
    st.session_state.bundle = None

if load_clicked:
    try:
        st.session_state.bundle = load_bundle(index_path)
        st.success(f"Loaded index from {index_path} ({len(st.session_state.bundle.records)} chunks).")
    except Exception as e:
        st.error(f"Failed to load index: {e}")

if build_clicked:
    seeds = [line.strip() for line in seed_text.splitlines() if line.strip()]
    if not seeds:
        st.error("Please provide at least one seed URL.")
    else:
        with st.spinner("Scraping and building embeddings…"):
            bundle, stats = crawl_and_build_index(
                seed_urls=seeds,
                max_pages=max_pages,
                max_depth=max_depth,
                same_domain_only=same_domain_only,
                timeout=int(timeout),
                per_request_delay=float(delay),
                concurrency=int(concurrency),
                user_agent=user_agent.strip() or DEFAULT_UA,
                chunk_words=int(chunk_words),
                overlap_words=int(overlap_words),
                model_name=model_name,
            )
            st.session_state.bundle = bundle

        if len(bundle.records) == 0:
            st.warning("Built an empty index (no extractable text). Try increasing max_pages/depth or different URLs.")
        else:
            try:
                save_bundle(index_path, bundle)
                st.success(f"Index built ✅ Pages: {stats['pages_fetched']}, Chunks: {stats['chunks']}. Saved to {index_path}")
            except Exception as e:
                st.warning(f"Built index, but failed to save: {e}")

bundle: Optional[IndexBundle] = st.session_state.bundle

st.divider()

left, right = st.columns([1.2, 1])

with left:
    st.header("Search")
    query = st.text_input("Ask a question / type a query", placeholder="e.g., pricing, requirements, architecture, refund policy…")
    top_k = st.slider("Top K results", 3, 20, 8)
    do_search = st.button("🔍 Search", use_container_width=False, disabled=(bundle is None or len(bundle.records) == 0))

    if do_search and bundle and query.strip():
        model = SentenceTransformer(bundle.model_name)
        results = search(bundle, model, query.strip(), top_k=int(top_k))

        if not results:
            st.info("No results.")
        else:
            st.subheader("Results")
            for score, rec in results:
                with st.container(border=True):
                    st.markdown(f"**{rec.title}**  \n{rec.url}")
                    st.caption(f"Similarity: {score:.3f} • Chunk #{rec.chunk_id} • Doc {rec.doc_id}")
                    st.write(highlight_snippet(rec.text, query, max_len=360))

with right:
    st.header("Index status")
    if not bundle:
        st.info("No index loaded yet. Build or load an index from the sidebar.")
    else:
        st.metric("Chunks", len(bundle.records))
        dim = int(bundle.embeddings.shape[1]) if bundle.embeddings.size else 0
        st.metric("Embedding dim", dim)
        st.caption(f"Embedding model: {bundle.model_name}")

        if len(bundle.records) > 0:
            # show a few sample sources
            urls = list(dict.fromkeys([r.url for r in bundle.records]))[:8]
            st.subheader("Sample sources")
            for u in urls:
                st.write("•", u)
