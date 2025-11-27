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
from datetime import datetime
from collections import Counter

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
# Enhanced Helpers
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
    # Enhanced cleanup patterns
    patterns = [
        r"\b(Cookie settings|Accept cookies|Reject all|Privacy policy|Terms of service)\b",
        r"\b(Skip to content|Jump to navigation|Back to top)\b",
        r"\b(Subscribe to newsletter|Sign up for updates)\b",
    ]
    for pattern in patterns:
        text = re.sub(pattern, " ", text, flags=re.I)
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

def should_skip_url(url: str, exclude_patterns: List[str]) -> bool:
    """Check if URL matches any exclusion patterns."""
    url_lower = url.lower()
    for pattern in exclude_patterns:
        if pattern.lower() in url_lower:
            return True
    # Common patterns to skip
    skip_extensions = ['.pdf', '.jpg', '.png', '.gif', '.zip', '.exe', '.mp4', '.mp3']
    return any(url_lower.endswith(ext) for ext in skip_extensions)

def extract_links(base_url: str, html: str, exclude_patterns: List[str] = None) -> List[str]:
    exclude_patterns = exclude_patterns or []
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
        
        if not should_skip_url(abs_url, exclude_patterns):
            out.append(abs_url)
    
    # de-dupe while preserving order
    seen = set()
    deduped = []
    for u in out:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped

def extract_metadata(html: str) -> Dict[str, str]:
    """Extract meta description and other metadata."""
    soup = BeautifulSoup(html, "html.parser")
    meta = {}
    
    # Description
    desc_tag = soup.find("meta", attrs={"name": "description"})
    if desc_tag:
        meta["description"] = desc_tag.get("content", "")
    
    # Keywords
    kw_tag = soup.find("meta", attrs={"name": "keywords"})
    if kw_tag:
        meta["keywords"] = kw_tag.get("content", "")
    
    # Author
    author_tag = soup.find("meta", attrs={"name": "author"})
    if author_tag:
        meta["author"] = author_tag.get("content", "")
    
    return meta

def html_to_text(url: str, html: str) -> str:
    if HAS_TRAFILATURA:
        extracted = trafilatura.extract(html, url=url, include_comments=False, include_tables=True)
        if extracted and extracted.strip():
            return strip_nav_boilerplate(extracted)

    # Fallback with BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header"]):
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

@st.cache_data(show_spinner=False, ttl=3600)
def get_robots_parser(root_url: str, user_agent: str) -> urllib.robotparser.RobotFileParser:
    parsed = urllib.parse.urlparse(root_url)
    robots_url = urllib.parse.urlunparse((parsed.scheme, parsed.netloc, "/robots.txt", "", "", ""))
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
    except Exception:
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

def fetch_url(url: str, timeout: int, user_agent: str) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[str]]:
    """Returns (html, final_url, status_code, error_msg)"""
    headers = {"User-Agent": user_agent}
    try:
        resp = SESSION.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        status = resp.status_code
        final_url = canonicalize_url(resp.url)
        if status >= 400:
            return None, final_url, status, f"HTTP {status}"
        if not is_html_response(resp):
            return None, final_url, status, "Non-HTML content"
        return resp.text, final_url, status, None
    except requests.Timeout:
        return None, None, None, "Timeout"
    except requests.RequestException as e:
        return None, None, None, f"Request error: {str(e)[:50]}"
    except Exception as e:
        return None, None, None, f"Error: {str(e)[:50]}"


# ---------------------------
# Enhanced Data model
# ---------------------------

@dataclass
class Record:
    doc_id: str
    url: str
    title: str
    chunk_id: int
    text: str
    metadata: Dict[str, str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class IndexBundle:
    model_name: str
    records: List[Record]
    embeddings: np.ndarray  # (n,d)
    metadata: Dict = None  # overall index metadata
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {
                "created_at": datetime.now().isoformat(),
                "version": "2.0"
            }


@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str):
    """Cache the embedding model to avoid reloading."""
    return SentenceTransformer(model_name)


def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 32) -> np.ndarray:
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return vecs.astype(np.float32)


def search(bundle: IndexBundle, model: SentenceTransformer, query: str, top_k: int = 8, 
           url_filter: str = None, min_score: float = 0.0) -> List[Tuple[float, Record]]:
    """Enhanced search with filtering options."""
    qv = embed_texts(model, [query], batch_size=1)[0]
    sims = cos_sim_matrix(bundle.embeddings, qv).reshape(-1)  # (n,)
    
    if len(sims) == 0:
        return []
    
    # Apply filters
    filtered_indices = []
    for i, (sim, rec) in enumerate(zip(sims, bundle.records)):
        if sim < min_score:
            continue
        if url_filter and url_filter.lower() not in rec.url.lower():
            continue
        filtered_indices.append(i)
    
    if not filtered_indices:
        return []
    
    filtered_sims = sims[filtered_indices]
    top_k_local = min(top_k, len(filtered_indices))
    idx = np.argpartition(-filtered_sims, top_k_local-1)[:top_k_local]
    idx = idx[np.argsort(-filtered_sims[idx])]
    
    return [(float(filtered_sims[i]), bundle.records[filtered_indices[i]]) for i in idx]


# ---------------------------
# Enhanced Crawler
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
    exclude_patterns: List[str] = None,
    min_text_length: int = 300,
    progress_callback=None,
) -> Tuple[IndexBundle, Dict]:
    """Enhanced crawler with progress tracking and better error handling."""
    
    model = load_embedding_model(model_name)
    exclude_patterns = exclude_patterns or []

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
    pages_failed = 0
    error_log = []

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
        if should_skip_url(url, exclude_patterns):
            return False
        if not allowed_by_robots(url, user_agent):
            return False
        return True

    # BFS by depth
    while queue and pages_fetched < max_pages:
        batch = []
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
                html, final_url, status, error = fut.result()
                
                # Update progress
                if progress_callback:
                    progress_callback(pages_fetched, max_pages, url)
                
                # politeness delay
                if per_request_delay > 0:
                    time.sleep(per_request_delay)

                if not html or not final_url:
                    pages_failed += 1
                    if error and len(error_log) < 50:  # Keep log manageable
                        error_log.append({"url": url, "error": error})
                    continue

                pages_fetched += 1

                title = get_title(html)
                text = html_to_text(final_url, html)
                text = normalize_ws(text)
                metadata = extract_metadata(html)

                if len(text) < min_text_length:
                    continue

                doc_id = stable_hash(final_url)

                docs_meta[doc_id] = {
                    "url": final_url,
                    "title": title,
                    "chars": len(text),
                    "depth": depth,
                    "status": status,
                    "metadata": metadata,
                    "fetched_at": datetime.now().isoformat(),
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
                            metadata=metadata,
                        )
                    )

                # enqueue new links if we can go deeper
                if depth < max_depth:
                    links = extract_links(final_url, html, exclude_patterns)
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
        bundle = IndexBundle(
            model_name=model_name, 
            records=[], 
            embeddings=np.zeros((0, 384), dtype=np.float32),
            metadata={"created_at": datetime.now().isoformat(), "version": "2.0"}
        )
        stats = {
            "pages_fetched": pages_fetched,
            "pages_failed": pages_failed,
            "docs": 0,
            "chunks": 0,
            "errors": error_log
        }
        return bundle, stats

    embs = embed_texts(model, texts, batch_size=32)
    bundle = IndexBundle(
        model_name=model_name, 
        records=records, 
        embeddings=embs,
        metadata={
            "created_at": datetime.now().isoformat(),
            "version": "2.0",
            "seed_urls": seed_urls,
            "docs_metadata": docs_meta,
        }
    )

    stats = {
        "pages_fetched": pages_fetched,
        "pages_failed": pages_failed,
        "docs": len(docs_meta),
        "chunks": len(records),
        "embedding_dim": int(embs.shape[1]),
        "errors": error_log,
        "unique_domains": len(set(urllib.parse.urlparse(r.url).netloc for r in records)),
    }
    return bundle, stats


def save_bundle(path: str, bundle: IndexBundle) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model_name": bundle.model_name,
        "records": [asdict(r) for r in bundle.records],
        "embeddings": bundle.embeddings.tolist(),
        "metadata": bundle.metadata,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

def load_bundle(path: str) -> IndexBundle:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    records = [Record(**r) for r in payload["records"]]
    embs = np.array(payload["embeddings"], dtype=np.float32)
    metadata = payload.get("metadata", {})
    return IndexBundle(model_name=payload["model_name"], records=records, embeddings=embs, metadata=metadata)


def export_results_to_markdown(results: List[Tuple[float, Record]], query: str) -> str:
    """Export search results to markdown format."""
    md = f"# Search Results for: {query}\n\n"
    md += f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
    md += "---\n\n"
    
    for i, (score, rec) in enumerate(results, 1):
        md += f"## Result {i} (Score: {score:.3f})\n\n"
        md += f"**Title:** {rec.title}\n\n"
        md += f"**URL:** {rec.url}\n\n"
        md += f"**Chunk:** {rec.chunk_id}\n\n"
        md += f"**Content:**\n\n{rec.text}\n\n"
        md += "---\n\n"
    
    return md


def highlight_snippet(text: str, query: str, max_len: int = 300) -> str:
    q_words = [w for w in re.findall(r"[A-Za-z0-9_]+", query) if len(w) >= 3]
    if not q_words:
        return text[:max_len] + ("…" if len(text) > max_len else "")
    
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
# Enhanced Streamlit UI
# ---------------------------

st.set_page_config(page_title="Advanced Web Semantic Search", layout="wide", page_icon="🔎")

# Custom CSS
st.markdown("""
<style>
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    .result-card {
        border-left: 3px solid #1f77b4;
        padding-left: 1rem;
        margin: 1rem 0;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔎 Advanced Web Semantic Search Engine")
st.caption("Scrape websites, extract content, create embeddings, and perform semantic search locally. Respect robots.txt and site terms.")

with st.sidebar:
    st.header("⚙️ Configuration")
    
    with st.expander("🌐 Crawl Settings", expanded=True):
        seed_text = st.text_area(
            "Seed URLs (one per line)",
            value="https://example.com/\n",
            height=120,
            help="Starting URLs for the crawler"
        )
        
        exclude_text = st.text_area(
            "Exclude URL patterns (optional, one per line)",
            value="",
            height=80,
            help="URLs containing these patterns will be skipped (e.g., /login, /admin)"
        )

        model_name = st.selectbox(
            "Embedding model",
            options=[
                "all-MiniLM-L6-v2",
                "all-mpnet-base-v2",
                "paraphrase-MiniLM-L6-v2",
            ],
            index=0,
            help="MiniLM is fast (384d); mpnet is more accurate but slower (768d)",
        )

        max_pages = st.slider("Max pages to crawl", 5, 1000, 60, step=5)
        max_depth = st.slider("Max crawl depth", 0, 8, 2, help="0 = seed URLs only")
        same_domain_only = st.checkbox("Stay on same domain(s)", value=True)
        min_text_length = st.slider("Min text length (chars)", 100, 1000, 300, step=50)

    with st.expander("🔧 Advanced Settings"):
        concurrency = st.slider("Concurrent requests", 1, 20, 6)
        timeout = st.slider("Request timeout (sec)", 4, 60, 12)
        delay = st.slider("Delay between requests (sec)", 0.0, 3.0, 0.25, step=0.05)
        
        st.subheader("Chunking")
        chunk_words = st.slider("Chunk size (words)", 120, 500, 220, step=20)
        overlap_words = st.slider("Overlap (words)", 0, 150, 50, step=10)
        
        user_agent = st.text_input("User-Agent", value=DEFAULT_UA)

    st.divider()
    
    st.subheader("💾 Index Management")
    index_path = st.text_input("Index file path", value="web_index.json")
    
    col1, col2 = st.columns(2)
    build_clicked = col1.button("🧱 Build Index", use_container_width=True, type="primary")
    load_clicked = col2.button("📂 Load Index", use_container_width=True)
    
    if st.session_state.get("bundle"):
        if st.button("💾 Save Current Index", use_container_width=True):
            try:
                save_bundle(index_path, st.session_state.bundle)
                st.success(f"Saved to {index_path}")
            except Exception as e:
                st.error(f"Save failed: {e}")

# Session state
if "bundle" not in st.session_state:
    st.session_state.bundle = None
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# Load index
if load_clicked:
    try:
        with st.spinner("Loading index..."):
            st.session_state.bundle = load_bundle(index_path)
        st.success(f"✅ Loaded {len(st.session_state.bundle.records)} chunks from {index_path}")
    except Exception as e:
        st.error(f"❌ Load failed: {e}")

# Build index
if build_clicked:
    seeds = [line.strip() for line in seed_text.splitlines() if line.strip()]
    exclude_patterns = [line.strip() for line in exclude_text.splitlines() if line.strip()]
    
    if not seeds:
        st.error("⚠️ Please provide at least one seed URL.")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(current, total, url):
            progress = min(current / total, 1.0) if total > 0 else 0
            progress_bar.progress(progress)
            status_text.text(f"Crawling... {current}/{total} pages | Current: {url[:60]}...")
        
        try:
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
                exclude_patterns=exclude_patterns,
                min_text_length=min_text_length,
                progress_callback=update_progress,
            )
            st.session_state.bundle = bundle
            
            progress_bar.empty()
            status_text.empty()
            
            if len(bundle.records) == 0:
                st.warning("⚠️ Built an empty index. Try adjusting settings or different URLs.")
            else:
                try:
                    save_bundle(index_path, bundle)
                    st.success(f"""
                    ✅ **Index built successfully!**
                    - Pages fetched: {stats['pages_fetched']}
                    - Pages failed: {stats['pages_failed']}
                    - Documents: {stats['docs']}
                    - Chunks: {stats['chunks']}
                    - Unique domains: {stats['unique_domains']}
                    - Saved to: {index_path}
                    """)
                    
                    if stats.get('errors'):
                        with st.expander(f"⚠️ View {len(stats['errors'])} errors"):
                            for err in stats['errors'][:20]:
                                st.text(f"• {err['url']}: {err['error']}")
                except Exception as e:
                    st.warning(f"Index built but save failed: {e}")
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Crawl failed: {e}")

bundle: Optional[IndexBundle] = st.session_state.bundle

st.divider()

# Main layout
left_col, right_col = st.columns([1.4, 1])

with left_col:
    st.header("🔍 Search")
    
    query = st.text_input(
        "Enter your search query",
        value=st.session_state.last_query,
        placeholder="e.g., pricing information, technical requirements, refund policy...",
        help="Ask questions or search for specific topics"
    )
    
    search_col1, search_col2, search_col3 = st.columns([2, 1, 1])
    
    with search_col1:
        top_k = st.slider("Results to show", 3, 30, 8)
    with search_col2:
        min_score = st.slider("Min score", 0.0, 0.9, 0.0, step=0.05)
    with search_col3:
        url_filter = st.text_input("Filter by URL", placeholder="optional")
    
    do_search = st.button(
        "🔍 Search", 
        use_container_width=False, 
        disabled=(bundle is None or len(bundle.records) == 0),
        type="primary"
    )

    if do_search and bundle and query.strip():
        st.session_state.last_query = query
        with st.spinner("Searching..."):
            model = load_embedding_model(bundle.model_name)
            results = search(
                bundle, 
                model, 
                query.strip(), 
                top_k=int(top_k),
                url_filter=url_filter if url_filter.strip() else None,
                min_score=float(min_score)
            )
            st.session_state.search_results = results

    # Display results
    if st.session_state.search_results is not None:
        results = st.session_state.search_results
        
        if not results:
            st.info("📭 No results found. Try adjusting your query or filters.")
        else:
            st.subheader(f"📊 Found {len(results)} results")
            
            # Export option
            if st.button("📥 Export Results to Markdown"):
                md_content = export_results_to_markdown(results, st.session_state.last_query)
                st.download_button(
                    label="Download Markdown",
                    data=md_content,
                    file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
            
            # Results display
            for i, (score, rec) in enumerate(results, 1):
                with st.container():
                    st.markdown(f"### {i}. {rec.title}")
                    
                    col_a, col_b, col_c = st.columns([2, 1, 1])
                    with col_a:
                        st.markdown(f"[🔗 {rec.url}]({rec.url})")
                    with col_b:
                        st.metric("Similarity", f"{score:.3f}")
                    with col_c:
                        st.caption(f"Chunk {rec.chunk_id}")
                    
                    snippet = highlight_snippet(rec.text, st.session_state.last_query, max_len=400)
                    st.write(snippet)
                    
                    # Metadata
                    if rec.metadata:
                        with st.expander("📄 Metadata"):
                            for k, v in rec.metadata.items():
                                if v:
                                    st.text(f"{k}: {v}")
                    
                    st.divider()

with right_col:
    st.header("📊 Index Status")
    
    if not bundle:
        st.info("💡 No index loaded. Build or load an index to begin searching.")
        
        st.markdown("""
        ### Quick Start:
        1. Enter seed URLs in the sidebar
        2. Adjust crawl settings as needed
        3. Click **Build Index**
        4. Start searching!
        """)
    else:
        # Statistics
        st.metric("Total Chunks", f"{len(bundle.records):,}")
        
        col_a, col_b = st.columns(2)
        with col_a:
            dim = int(bundle.embeddings.shape[1]) if bundle.embeddings.size else 0
            st.metric("Embedding Dim", dim)
        with col_b:
            unique_urls = len(set(r.url for r in bundle.records))
            st.metric("Unique Pages", unique_urls)
        
        st.caption(f"Model: `{bundle.model_name}`")
        
        # Index metadata
        if bundle.metadata:
            with st.expander("ℹ️ Index Information"):
                created = bundle.metadata.get("created_at", "Unknown")
                st.text(f"Created: {created}")
                
                if "seed_urls" in bundle.metadata:
                    st.text("Seeds:")
                    for url in bundle.metadata["seed_urls"][:5]:
                        st.text(f"  • {url}")
        
        # Sample sources
        if len(bundle.records) > 0:
            st.subheader("📑 Indexed Sources")
            urls = list(dict.fromkeys([r.url for r in bundle.records]))[:10]
            
            # Domain distribution
            domains = [urllib.parse.urlparse(r.url).netloc for r in bundle.records]
            domain_counts = Counter(domains).most_common(5)
            
            with st.expander(f"🌐 Top Domains ({len(set(domains))} total)"):
                for domain, count in domain_counts:
                    st.text(f"{domain}: {count} chunks")
            
            with st.expander(f"📄 Sample URLs ({len(urls)} shown)"):
                for u in urls:
                    st.markdown(f"• [{u}]({u})")

st.divider()
st.caption("Built with Streamlit • Respect robots.txt and site terms • Educational use only")