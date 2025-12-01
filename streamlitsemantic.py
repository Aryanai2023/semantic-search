from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Optional, Dict, Any

import streamlit as st
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class Defaults:
    persist_dir: str = "chroma_db"
    collection: str = "documents"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_chars: int = 1400
    chunk_overlap: int = 250
    batch_size: int = 128


# -----------------------------
# Helpers
# -----------------------------
def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

def chunk_text(text: str, max_chars: int, overlap: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    # normalize whitespace lightly
    text = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    n = len(text)

    chunks: List[str] = []
    start = 0
    while start < n:
        end = min(start + max_chars, n)
        window = text[start:end]

        # try to cut on a nicer boundary
        cut = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(". "))
        if cut > 200:  # avoid tiny early cuts
            end = start + cut + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break

        start = max(0, end - overlap)

    return chunks

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> List[Tuple[int, str]]:
    """Return list of (page_number_1_indexed, page_text)."""
    reader = PdfReader(st.runtime.uploaded_file_manager.BytesIO(pdf_bytes))
    pages: List[Tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        pages.append((i + 1, (page.extract_text() or "").strip()))
    return pages

def extract_text_from_upload(file) -> List[Tuple[Optional[int], str]]:
    """
    Returns list of (page, text). page=None for non-PDF.
    """
    suffix = Path(file.name).suffix.lower()
    raw = file.getvalue()

    if suffix == ".pdf":
        pages = extract_text_from_pdf_bytes(raw)
        return [(p, t) for p, t in pages if t]

    # txt/md
    try:
        text = raw.decode("utf-8", errors="ignore").strip()
    except Exception:
        text = ""
    return [(None, text)] if text else []

@st.cache_resource(show_spinner=False)
def load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)

@st.cache_resource(show_spinner=False)
def get_chroma_client(persist_dir: str):
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)

def get_collection(client, name: str):
    # Ensure cosine distance (works well with SBERT embeddings).
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )

def encode_texts(model: SentenceTransformer, texts: List[str], batch_size: int) -> List[List[float]]:
    # normalize_embeddings=True gives consistent cosine similarity behavior
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return emb.tolist()

def upsert_chunks(
    col,
    model: SentenceTransformer,
    source_name: str,
    page: Optional[int],
    chunks: List[str],
    batch_size: int,
):
    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    # deterministic IDs so re-indexing doesn't duplicate
    for idx, ch in enumerate(chunks):
        base = f"{source_name}|page={page}|chunk={idx}|{sha1(ch)}"
        ids.append(sha1(base))
        docs.append(ch)
        metas.append({
            "source": source_name,
            "page": page,
            "chunk": idx,
            "chars": len(ch),
        })

    # embeddings in batches
    B = batch_size
    for i in range(0, len(docs), B):
        batch_docs = docs[i:i+B]
        batch_ids = ids[i:i+B]
        batch_metas = metas[i:i+B]
        batch_emb = encode_texts(model, batch_docs, batch_size=min(B, 64))
        col.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_metas, embeddings=batch_emb)

def clear_collection(col):
    # delete all items
    col.delete(where={})


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Semantic Search (Streamlit)", layout="wide")

st.title("🔎 Semantic Search Engine (Streamlit-only)")
st.caption("Upload docs → build vector index → semantic search. Uses ChromaDB (persistent) + Sentence-Transformers.")

defaults = Defaults()

with st.sidebar:
    st.header("Settings")

    persist_dir = st.text_input("Persistent DB folder", defaults.persist_dir)
    collection_name = st.text_input("Collection name", defaults.collection)
    model_name = st.text_input("Embedding model", defaults.embedding_model)

    st.divider()
    chunk_chars = st.slider("Chunk size (chars)", 400, 4000, defaults.chunk_chars, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0, 800, defaults.chunk_overlap, step=50)
    batch_size = st.slider("Embed batch size", 16, 256, defaults.batch_size, step=16)

    st.divider()
    top_k = st.slider("Top-K results", 1, 25, 8)

    st.divider()
    st.subheader("Index tools")
    danger = st.checkbox("I understand clearing deletes indexed chunks", value=False)
    clear_btn = st.button("🧹 Clear index", type="secondary", use_container_width=True, disabled=not danger)

# init db
try:
    client = get_chroma_client(persist_dir)
    col = get_collection(client, collection_name)
except Exception as e:
    st.error(f"Failed to initialize ChromaDB: {e}")
    st.stop()

if clear_btn:
    try:
        clear_collection(col)
        st.success("Index cleared.")
        st.rerun()
    except Exception as e:
        st.error(f"Failed to clear index: {e}")

# load model
with st.spinner("Loading embedding model..."):
    try:
        model = load_model(model_name)
    except Exception as e:
        st.error(f"Failed to load embedding model '{model_name}': {e}")
        st.stop()

# Status row
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Indexed chunks", col.count())
with c2:
    st.write(f"**DB:** `{persist_dir}`")
with c3:
    st.write(f"**Model:** `{model_name}`")

st.divider()

# Ingestion
st.subheader("1) Upload & Index Documents")
uploads = st.file_uploader(
    "Upload PDFs / TXT / MD (multiple allowed)",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
)

index_btn = st.button("📥 Index uploaded files", type="primary", use_container_width=True, disabled=not uploads)

if index_btn and uploads:
    progress = st.progress(0)
    status = st.empty()

    total_files = len(uploads)
    done_files = 0
    total_chunks = 0

    for file in uploads:
        done_files += 1
        status.info(f"Reading: {file.name}")

        try:
            docs = extract_text_from_upload(file)
        except Exception as e:
            st.warning(f"Skipped {file.name}: {e}")
            continue

        file_chunks = 0
        for page, text in docs:
            chunks = chunk_text(text, chunk_chars, chunk_overlap)
            if not chunks:
                continue
            upsert_chunks(
                col=col,
                model=model,
                source_name=file.name,
                page=page,
                chunks=chunks,
                batch_size=batch_size,
            )
            file_chunks += len(chunks)

        total_chunks += file_chunks
        progress.progress(min(done_files / total_files, 1.0))

    status.success(f"Indexed ✅ {total_chunks} chunks from {total_files} file(s).")
    st.balloons()
    st.rerun()

st.divider()

# Search
st.subheader("2) Search")
q = st.text_input("Search query", placeholder="e.g., 'how does TCP congestion control work?'")
source_filter = st.text_input("Optional filter: source contains", placeholder="e.g., 'OperatingSystems.pdf'")

search_btn = st.button("🔍 Search", use_container_width=True, disabled=not q.strip())

if search_btn:
    if col.count() == 0:
        st.warning("Index is empty. Upload & index some documents first.")
    else:
        with st.spinner("Searching..."):
            q_emb = encode_texts(model, [q], batch_size=1)[0]
            res = col.query(
                query_embeddings=[q_emb],
                n_results=min(top_k * 5, 50),  # overfetch for filtering
                include=["documents", "metadatas", "distances", "ids"],
            )

        ids = res["ids"][0] if res.get("ids") else []
        docs = res["documents"][0] if res.get("documents") else []
        metas = res["metadatas"][0] if res.get("metadatas") else []
        dists = res["distances"][0] if res.get("distances") else []

        hits = []
        for _id, doc, meta, dist in zip(ids, docs, metas, dists):
            src = (meta or {}).get("source", "")
            if source_filter.strip() and source_filter.lower() not in str(src).lower():
                continue
            # For cosine space in Chroma, distance is "smaller is better".
            score = 1.0 - float(dist)
            hits.append((_id, doc, meta or {}, score))
            if len(hits) >= top_k:
                break

        if not hits:
            st.info("No results matched (try removing the source filter or rephrasing query).")
        else:
            st.write(f"Found **{len(hits)}** results:")
            for i, (_id, doc, meta, score) in enumerate(hits, 1):
                src = meta.get("source", "unknown")
                page = meta.get("page", None)
                chunk_i = meta.get("chunk", None)

                title_bits = [f"{i}. score={score:.3f}", src]
                if page is not None:
                    title_bits.append(f"p.{page}")
                if chunk_i is not None:
                    title_bits.append(f"chunk {chunk_i}")

                with st.expander(" | ".join(title_bits), expanded=(i == 1)):
                    st.code(doc[:4000] + ("…" if len(doc) > 4000 else ""), language="markdown")
                    st.caption(f"Chunk ID: {_id}")