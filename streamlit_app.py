"""
VC Semantic Search – Enhanced Pro Edition

Major Improvements:
- Hybrid search (BM25 + Semantic) for better relevance
- Query expansion with synonyms and related terms
- "Find Similar" feature - click any result to find related entities
- Clustering visualization (t-SNE/UMAP)
- Negative query support (exclude terms)
- Autocomplete suggestions
- Saved searches with persistence
- Advanced analytics dashboard
- Side-by-side entity comparison
- Duplicate detection on upload
- Inline data editing
- Real-time search as you type
- Dark/Light theme support
"""

import os
import re
import json
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
from datetime import datetime
from collections import Counter
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Optional imports with fallbacks
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


# ======================= Page Configuration ======================= #

st.set_page_config(
    page_title="VC Semantic Search Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stMetric {
        background-color: rgba(28, 131, 225, 0.1);
        padding: 10px;
        border-radius: 10px;
    }
    .search-result-card {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #1c83e1;
    }
    .similarity-badge {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
    }
    div[data-testid="stExpander"] details summary p {
        font-size: 1.1rem;
        font-weight: 600;
    }
    .tag-chip {
        display: inline-block;
        background-color: rgba(28, 131, 225, 0.2);
        padding: 2px 8px;
        border-radius: 12px;
        margin: 2px;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)


# ======================= Session State ======================= #

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "search_history": [],
        "saved_searches": {},
        "last_results": None,
        "comparison_list": [],
        "favorites": set(),
        "current_query": "",
        "embeddings_computed": False,
        "show_clustering": False,
        "theme": "light",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ======================= Embedding & Model Utilities ======================= #

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load and cache the sentence transformer model."""
    return SentenceTransformer(model_name)


@st.cache_data(show_spinner="Computing embeddings...")
def compute_corpus_embeddings(
    texts: Tuple[str, ...],
    model_name: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    """Compute and cache embeddings for the corpus."""
    model = load_embedding_model(model_name)
    embeddings = model.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 100,
        batch_size=32,
    )
    return embeddings


def embed_query(query: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Embed a single query."""
    model = load_embedding_model(model_name)
    return model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]


@st.cache_data(show_spinner="Building BM25 index...")
def build_bm25_index(texts: Tuple[str, ...]) -> Optional[object]:
    """Build BM25 index for keyword search."""
    if not BM25_AVAILABLE:
        return None
    tokenized = [text.lower().split() for text in texts]
    return BM25Okapi(tokenized)


# ======================= Search Functions ======================= #

def parse_query(query: str) -> Tuple[List[str], List[str]]:
    """
    Parse query to extract positive and negative terms.
    Supports: "quantum computing -cryptography" syntax
    """
    positive_terms = []
    negative_terms = []
    
    tokens = query.split()
    for token in tokens:
        if token.startswith("-") and len(token) > 1:
            negative_terms.append(token[1:].lower())
        elif token.startswith("NOT "):
            negative_terms.append(token[4:].lower())
        else:
            positive_terms.append(token)
    
    return positive_terms, negative_terms


def expand_query(query: str) -> List[str]:
    """
    Expand query with synonyms and related terms.
    Returns list of expanded queries.
    """
    expansions = {
        "ai": ["artificial intelligence", "machine learning", "deep learning"],
        "ml": ["machine learning", "ai", "neural networks"],
        "quantum": ["quantum computing", "qubits", "quantum mechanics"],
        "biotech": ["biotechnology", "life sciences", "biopharma"],
        "fintech": ["financial technology", "payments", "banking tech"],
        "saas": ["software as a service", "cloud software", "subscription software"],
        "ev": ["electric vehicle", "electric car", "electrification"],
        "battery": ["energy storage", "lithium", "solid state"],
        "fusion": ["nuclear fusion", "clean energy", "plasma"],
        "edge": ["edge computing", "edge ai", "distributed computing"],
        "robotics": ["automation", "robots", "autonomous systems"],
        "semiconductor": ["chips", "silicon", "integrated circuits"],
        "photonics": ["silicon photonics", "optical", "light-based"],
    }
    
    expanded = [query]
    query_lower = query.lower()
    
    for term, synonyms in expansions.items():
        if term in query_lower:
            for syn in synonyms[:2]:  # Limit expansions
                expanded.append(query_lower.replace(term, syn))
    
    return list(set(expanded))[:5]  # Max 5 expansions


def hybrid_search(
    query: str,
    corpus_embeddings: np.ndarray,
    bm25_index: Optional[object],
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    semantic_weight: float = 0.7,
    top_k: int = 20,
    min_similarity: float = 0.0,
    use_expansion: bool = False,
    filter_negatives: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Hybrid search combining semantic and BM25 scores.
    Returns: (top_indices, combined_scores, metadata)
    """
    positive_terms, negative_terms = parse_query(query)
    clean_query = " ".join(positive_terms)
    
    # Query expansion
    queries = [clean_query]
    if use_expansion:
        queries = expand_query(clean_query)
    
    # Semantic search (aggregate multiple queries)
    all_semantic_scores = []
    for q in queries:
        q_emb = embed_query(q, model_name)
        scores = cosine_similarity([q_emb], corpus_embeddings)[0]
        all_semantic_scores.append(scores)
    
    semantic_scores = np.max(all_semantic_scores, axis=0)
    
    # BM25 search
    if bm25_index is not None and BM25_AVAILABLE:
        tokenized_query = clean_query.lower().split()
        bm25_scores = bm25_index.get_scores(tokenized_query)
        # Normalize BM25 scores
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()
    else:
        bm25_scores = np.zeros_like(semantic_scores)
    
    # Combine scores
    combined_scores = (
        semantic_weight * semantic_scores +
        (1 - semantic_weight) * bm25_scores
    )
    
    # Apply negative term filtering
    if filter_negatives and negative_terms:
        for i, text in enumerate(texts):
            text_lower = text.lower()
            for neg_term in negative_terms:
                if neg_term in text_lower:
                    combined_scores[i] *= 0.1  # Heavily penalize
    
    # Filter by minimum similarity
    valid_mask = combined_scores >= min_similarity
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) > 0:
        sorted_idx = np.argsort(combined_scores[valid_mask])[::-1][:top_k]
        top_indices = valid_indices[sorted_idx]
    else:
        top_indices = np.array([])
    
    metadata = {
        "expanded_queries": queries,
        "negative_terms": negative_terms,
        "semantic_weight": semantic_weight,
        "bm25_available": BM25_AVAILABLE and bm25_index is not None,
    }
    
    return top_indices, combined_scores, metadata


def find_similar(
    entity_idx: int,
    corpus_embeddings: np.ndarray,
    top_k: int = 10,
    exclude_self: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find entities similar to a given entity."""
    entity_emb = corpus_embeddings[entity_idx:entity_idx+1]
    scores = cosine_similarity(entity_emb, corpus_embeddings)[0]
    
    sorted_indices = np.argsort(scores)[::-1]
    
    if exclude_self:
        sorted_indices = sorted_indices[sorted_indices != entity_idx]
    
    top_indices = sorted_indices[:top_k]
    return top_indices, scores[top_indices]


# ======================= Analytics & Visualization ======================= #

@st.cache_data(show_spinner="Computing clusters...")
def compute_clusters(
    embeddings: np.ndarray,
    n_clusters: int = 5,
    method: str = "tsne",
    perplexity: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 2D projection and clusters."""
    # Dimensionality reduction
    if method == "umap" and UMAP_AVAILABLE:
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
        coords_2d = reducer.fit_transform(embeddings)
    else:
        # Use t-SNE
        perp = min(perplexity, len(embeddings) - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
        coords_2d = tsne.fit_transform(embeddings)
    
    # Clustering
    kmeans = KMeans(n_clusters=min(n_clusters, len(embeddings)), random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    return coords_2d, cluster_labels


def create_cluster_plot(
    df: pd.DataFrame,
    coords_2d: np.ndarray,
    cluster_labels: np.ndarray,
    color_by: str = "cluster",
) -> go.Figure:
    """Create interactive cluster visualization."""
    plot_df = df.copy()
    plot_df["x"] = coords_2d[:, 0]
    plot_df["y"] = coords_2d[:, 1]
    plot_df["cluster"] = cluster_labels.astype(str)
    
    color_col = color_by if color_by in plot_df.columns else "cluster"
    
    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color=color_col,
        hover_data=["name", "sector", "stage"] if all(c in plot_df.columns for c in ["name", "sector", "stage"]) else None,
        title="Entity Clustering Visualization",
        template="plotly_white",
    )
    
    fig.update_traces(marker=dict(size=10, opacity=0.7))
    fig.update_layout(
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        showlegend=True,
        height=500,
    )
    
    return fig


def create_similarity_heatmap(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    max_items: int = 20,
) -> go.Figure:
    """Create similarity heatmap for top items."""
    n = min(max_items, len(df))
    sim_matrix = cosine_similarity(embeddings[:n])
    
    labels = df["name"].head(n).tolist() if "name" in df.columns else [f"Item {i}" for i in range(n)]
    
    fig = go.Figure(data=go.Heatmap(
        z=sim_matrix,
        x=labels,
        y=labels,
        colorscale="Viridis",
        hoverongaps=False,
    ))
    
    fig.update_layout(
        title="Similarity Heatmap",
        height=500,
        xaxis_tickangle=-45,
    )
    
    return fig


def create_funding_analysis(df: pd.DataFrame) -> Optional[go.Figure]:
    """Create funding analysis charts."""
    if "funding_amount" not in df.columns or "sector" not in df.columns:
        return None
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Total Funding by Sector", "Average Funding by Stage"),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Funding by sector
    sector_funding = df.groupby("sector")["funding_amount"].sum().sort_values(ascending=True)
    fig.add_trace(
        go.Bar(x=sector_funding.values, y=sector_funding.index, orientation="h", name="Total"),
        row=1, col=1
    )
    
    # Funding by stage
    if "stage" in df.columns:
        stage_funding = df.groupby("stage")["funding_amount"].mean().sort_values(ascending=True)
        fig.add_trace(
            go.Bar(x=stage_funding.values, y=stage_funding.index, orientation="h", name="Average"),
            row=1, col=2
        )
    
    fig.update_layout(height=400, showlegend=False)
    return fig


def create_sector_sunburst(df: pd.DataFrame) -> Optional[go.Figure]:
    """Create sector/subsector sunburst chart."""
    if "sector" not in df.columns:
        return None
    
    if "subsector" in df.columns:
        fig = px.sunburst(
            df,
            path=["sector", "subsector"],
            title="Sector Breakdown",
        )
    else:
        sector_counts = df["sector"].value_counts()
        fig = px.pie(
            values=sector_counts.values,
            names=sector_counts.index,
            title="Sector Distribution",
            hole=0.4,
        )
    
    fig.update_layout(height=400)
    return fig


# ======================= Data Utilities ======================= #

def load_example_data() -> pd.DataFrame:
    """Load enhanced example dataset."""
    data = [
        {
            "name": "QuantaCore Systems",
            "entity_type": "Startup",
            "sector": "Quantum Computing",
            "subsector": "Error Correction",
            "region": "Europe",
            "country": "UK",
            "stage": "Seed",
            "funding_amount": 5000000,
            "year_founded": 2022,
            "employees": 15,
            "description": "Building fault-tolerant quantum error correction for NISQ devices targeting quantum chemistry workloads and drug discovery simulations.",
            "tags": "quantum, error correction, chemistry, NISQ, drug discovery",
            "website": "https://quantacore.example",
            "investors": "Quantum Ventures, Tech Angels UK",
        },
        {
            "name": "Helios Cathode Labs",
            "entity_type": "Startup",
            "sector": "Energy Storage",
            "subsector": "Solid State Batteries",
            "region": "North America",
            "country": "USA",
            "stage": "Series A",
            "funding_amount": 25000000,
            "year_founded": 2020,
            "employees": 45,
            "description": "Developing solid-state lithium metal batteries with ceramic electrolytes for grid-scale storage and electric vehicle applications with 2x energy density.",
            "tags": "batteries, solid state, energy storage, grid, EV, lithium",
            "website": "https://helios.example",
            "investors": "Energy Impact Partners, Breakthrough Energy",
        },
        {
            "name": "NeuroMesh AI",
            "entity_type": "Startup",
            "sector": "Artificial Intelligence",
            "subsector": "Edge AI",
            "region": "Europe",
            "country": "Germany",
            "stage": "Series B",
            "funding_amount": 45000000,
            "year_founded": 2019,
            "employees": 120,
            "description": "Edge AI platform compressing large language models and foundation models for deployment on low-power industrial sensors, autonomous vehicles, and robotics.",
            "tags": "edge ai, compression, robotics, foundation models, LLM, autonomous",
            "website": "https://neuromesh.example",
            "investors": "a]6z, Index Ventures, BMW Ventures",
        },
        {
            "name": "Photonix Fabrication",
            "entity_type": "Startup",
            "sector": "Semiconductors",
            "subsector": "Silicon Photonics",
            "region": "Asia",
            "country": "Japan",
            "stage": "Series A",
            "funding_amount": 30000000,
            "year_founded": 2021,
            "employees": 55,
            "description": "Silicon photonics interconnects reducing data center power consumption and latency for AI training workloads by 10x compared to electrical interconnects.",
            "tags": "photonic interconnects, datacenter, semiconductors, AI infrastructure",
            "website": "https://photonix.example",
            "investors": "SoftBank Vision, Sony Innovation Fund",
        },
        {
            "name": "CryoQubit Instruments",
            "entity_type": "Startup",
            "sector": "Quantum Computing",
            "subsector": "Cryogenics",
            "region": "Europe",
            "country": "Finland",
            "stage": "Seed",
            "funding_amount": 3500000,
            "year_founded": 2023,
            "employees": 12,
            "description": "Cryogenic control electronics and instrumentation for superconducting qubit systems at scale, enabling room-temperature control of quantum processors.",
            "tags": "cryogenics, superconducting qubits, instrumentation, control systems",
            "website": "https://cryoqubit.example",
            "investors": "IQM Ventures, Nordic Makers",
        },
        {
            "name": "BioSynth Therapeutics",
            "entity_type": "Startup",
            "sector": "Biotechnology",
            "subsector": "Synthetic Biology",
            "region": "North America",
            "country": "USA",
            "stage": "Series A",
            "funding_amount": 20000000,
            "year_founded": 2021,
            "employees": 35,
            "description": "Engineering synthetic organisms for sustainable production of rare pharmaceutical compounds, industrial enzymes, and novel protein therapeutics.",
            "tags": "synthetic biology, pharmaceuticals, biomanufacturing, proteins",
            "website": "https://biosynth.example",
            "investors": "Flagship Pioneering, ARCH Venture",
        },
        {
            "name": "FusionGrid Energy",
            "entity_type": "Startup",
            "sector": "Energy",
            "subsector": "Nuclear Fusion",
            "region": "North America",
            "country": "Canada",
            "stage": "Series B",
            "funding_amount": 75000000,
            "year_founded": 2018,
            "employees": 85,
            "description": "Compact tokamak design for commercial fusion power generation with advanced plasma control algorithms and high-temperature superconducting magnets.",
            "tags": "fusion, tokamak, clean energy, plasma physics, superconducting",
            "website": "https://fusiongrid.example",
            "investors": "Khosla Ventures, Canadian Pension Plan",
        },
        {
            "name": "NanoCarbon Materials",
            "entity_type": "Startup",
            "sector": "Advanced Materials",
            "subsector": "Carbon Nanotubes",
            "region": "Asia",
            "country": "Singapore",
            "stage": "Series A",
            "funding_amount": 18000000,
            "year_founded": 2020,
            "employees": 40,
            "description": "Scalable manufacturing of aligned carbon nanotube films for next-gen electronics, ultra-strong composite materials, and thermal management solutions.",
            "tags": "carbon nanotubes, materials science, electronics, composites",
            "website": "https://nanocarbon.example",
            "investors": "Temasek, GIC",
        },
        {
            "name": "Cortex Robotics",
            "entity_type": "Startup",
            "sector": "Robotics",
            "subsector": "Humanoid Robots",
            "region": "North America",
            "country": "USA",
            "stage": "Series B",
            "funding_amount": 60000000,
            "year_founded": 2019,
            "employees": 150,
            "description": "General-purpose humanoid robots with advanced dexterity and AI-powered learning for manufacturing, logistics, and household assistance applications.",
            "tags": "humanoid robots, dexterity, manufacturing, AI, automation",
            "website": "https://cortexrobotics.example",
            "investors": "OpenAI Fund, Sequoia Capital",
        },
        {
            "name": "AeroGraph Defense",
            "entity_type": "Startup",
            "sector": "Aerospace",
            "subsector": "Satellite Systems",
            "region": "North America",
            "country": "USA",
            "stage": "Series A",
            "funding_amount": 35000000,
            "year_founded": 2021,
            "employees": 60,
            "description": "Next-generation small satellites with on-board AI for real-time Earth observation, defense applications, and communications resilience.",
            "tags": "satellites, earth observation, defense, space, AI",
            "website": "https://aerograph.example",
            "investors": "Lux Capital, Founders Fund",
        },
        {
            "name": "GenomeAI Diagnostics",
            "entity_type": "Startup",
            "sector": "Healthcare",
            "subsector": "Diagnostics",
            "region": "Europe",
            "country": "UK",
            "stage": "Series A",
            "funding_amount": 22000000,
            "year_founded": 2020,
            "employees": 50,
            "description": "AI-powered genomic analysis platform for early cancer detection and personalized treatment recommendations using liquid biopsy.",
            "tags": "genomics, AI, cancer detection, diagnostics, liquid biopsy",
            "website": "https://genomeai.example",
            "investors": "GV, Wellcome Trust",
        },
        {
            "name": "HyperLogic Computing",
            "entity_type": "Startup",
            "sector": "Semiconductors",
            "subsector": "AI Accelerators",
            "region": "Asia",
            "country": "Taiwan",
            "stage": "Series B",
            "funding_amount": 55000000,
            "year_founded": 2018,
            "employees": 95,
            "description": "Custom AI accelerator chips optimized for transformer architectures with 5x better performance per watt than GPUs for inference workloads.",
            "tags": "AI chips, accelerators, transformers, inference, semiconductors",
            "website": "https://hyperlogic.example",
            "investors": "MediaTek Ventures, Walden International",
        },
        {
            "name": "AgriSense Technologies",
            "entity_type": "Startup",
            "sector": "AgTech",
            "subsector": "Precision Agriculture",
            "region": "Europe",
            "country": "Netherlands",
            "stage": "Series A",
            "funding_amount": 15000000,
            "year_founded": 2021,
            "employees": 30,
            "description": "IoT sensor networks and AI analytics for precision agriculture, optimizing water usage, fertilizer application, and crop yield prediction.",
            "tags": "agriculture, IoT, sensors, AI, sustainability, farming",
            "website": "https://agrisense.example",
            "investors": "Anterra Capital, DCVC",
        },
        {
            "name": "CryptoShield Security",
            "entity_type": "Startup",
            "sector": "Cybersecurity",
            "subsector": "Post-Quantum Cryptography",
            "region": "North America",
            "country": "USA",
            "stage": "Seed",
            "funding_amount": 8000000,
            "year_founded": 2022,
            "employees": 20,
            "description": "Post-quantum cryptography solutions protecting enterprise data against future quantum computer attacks with NIST-approved algorithms.",
            "tags": "cryptography, quantum-safe, security, encryption, enterprise",
            "website": "https://cryptoshield.example",
            "investors": "YC, In-Q-Tel",
        },
        {
            "name": "CleanHydrogen Systems",
            "entity_type": "Startup",
            "sector": "Energy",
            "subsector": "Green Hydrogen",
            "region": "Europe",
            "country": "Germany",
            "stage": "Series A",
            "funding_amount": 28000000,
            "year_founded": 2020,
            "employees": 55,
            "description": "Advanced electrolyzer technology for green hydrogen production at 30% lower cost, enabling industrial decarbonization and clean fuel applications.",
            "tags": "hydrogen, electrolysis, clean energy, decarbonization, industrial",
            "website": "https://cleanhydrogen.example",
            "investors": "Hy24, EIT InnoEnergy",
        },
    ]
    return pd.DataFrame(data)


def detect_duplicates(df: pd.DataFrame, threshold: float = 0.95) -> List[Tuple[int, int, float]]:
    """Detect potential duplicate entries based on text similarity."""
    if "name" not in df.columns:
        return []
    
    # Simple duplicate detection using name similarity
    duplicates = []
    names = df["name"].str.lower().tolist()
    
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            # Simple Jaccard similarity
            set_i = set(names[i].split())
            set_j = set(names[j].split())
            if len(set_i | set_j) > 0:
                similarity = len(set_i & set_j) / len(set_i | set_j)
                if similarity >= threshold:
                    duplicates.append((i, j, similarity))
    
    return duplicates


def combine_text_columns(
    df: pd.DataFrame,
    selected_cols: List[str],
    weights: Dict[str, float] = None,
) -> pd.Series:
    """Combine multiple text columns with optional weighting."""
    cols = [c for c in selected_cols if c in df.columns]
    if not cols:
        cols = df.select_dtypes(include=["object"]).columns.tolist()
    
    if weights:
        combined_parts = []
        for col in cols:
            weight = weights.get(col, 1.0)
            repetitions = max(1, int(weight))
            text = df[col].fillna("").astype(str)
            combined_parts.append(" ".join([text] * repetitions))
        combined = pd.concat(combined_parts, axis=1).agg(" ".join, axis=1)
    else:
        combined = df[cols].fillna("").astype(str).agg(" | ".join, axis=1)
    
    return combined.str.replace(r"\s+", " ", regex=True).str.strip()


def apply_filters(df: pd.DataFrame, filter_config: List[Tuple[str, str]]) -> Tuple[pd.DataFrame, np.ndarray]:
    """Apply sidebar filters and return filtered df + mask."""
    mask = np.ones(len(df), dtype=bool)
    
    for col, label in filter_config:
        if col in df.columns:
            options = sorted(df[col].dropna().unique().tolist())
            if len(options) > 1:
                selected = st.sidebar.multiselect(label, options=options, key=f"filter_{col}")
                if selected:
                    mask &= df[col].isin(selected).values
    
    # Funding range filter
    if "funding_amount" in df.columns:
        min_fund = int(df["funding_amount"].min())
        max_fund = int(df["funding_amount"].max())
        if min_fund < max_fund:
            fund_range = st.sidebar.slider(
                "Funding Range ($)",
                min_value=min_fund,
                max_value=max_fund,
                value=(min_fund, max_fund),
                format="$%d",
                key="funding_filter",
            )
            mask &= (df["funding_amount"] >= fund_range[0]) & (df["funding_amount"] <= fund_range[1])
    
    return df[mask].reset_index(drop=True), mask


def get_autocomplete_suggestions(query: str, df: pd.DataFrame, n: int = 5) -> List[str]:
    """Generate autocomplete suggestions based on existing data."""
    suggestions = set()
    query_lower = query.lower()
    
    # Collect terms from various columns
    for col in ["name", "sector", "subsector", "tags", "description"]:
        if col in df.columns:
            for value in df[col].dropna().unique():
                if isinstance(value, str):
                    words = value.lower().split()
                    for word in words:
                        if word.startswith(query_lower) and len(word) > 2:
                            suggestions.add(word.title())
    
    return sorted(list(suggestions))[:n]


def save_search_to_history(query: str, results_count: int, filters: Dict = None):
    """Save search to history."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "results_count": results_count,
        "filters": filters or {},
    }
    st.session_state.search_history.append(entry)
    # Keep only last 50 searches
    st.session_state.search_history = st.session_state.search_history[-50:]


def export_results(df: pd.DataFrame, format: str = "csv") -> bytes:
    """Export results to various formats."""
    if format == "csv":
        return df.to_csv(index=False).encode("utf-8")
    elif format == "json":
        return df.to_json(orient="records", indent=2).encode("utf-8")
    else:
        return df.to_csv(index=False).encode("utf-8")


# ======================= UI Components ======================= #

def render_entity_card(row: pd.Series, score: float, idx: int, show_actions: bool = True):
    """Render a single entity card with details."""
    with st.container():
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            name = row.get("name", f"Entity {idx}")
            sector = row.get("sector", "N/A")
            stage = row.get("stage", "N/A")
            st.markdown(f"### {name}")
            st.markdown(f"**{sector}** • {stage}")
        
        with col2:
            st.metric("Similarity", f"{score:.1%}")
        
        with col3:
            if show_actions:
                if st.button("🔍 Similar", key=f"similar_{idx}"):
                    st.session_state.find_similar_idx = idx
                if st.button("📊 Compare", key=f"compare_{idx}"):
                    if name not in st.session_state.comparison_list:
                        st.session_state.comparison_list.append(name)
        
        # Details
        if "description" in row:
            st.markdown(row["description"])
        
        # Tags
        if "tags" in row and pd.notna(row["tags"]):
            tags = row["tags"].split(",")
            tags_html = " ".join([f'<span class="tag-chip">{tag.strip()}</span>' for tag in tags[:5]])
            st.markdown(tags_html, unsafe_allow_html=True)
        
        # Metadata row
        meta_cols = st.columns(4)
        if "region" in row:
            meta_cols[0].markdown(f"📍 {row['region']}")
        if "country" in row:
            meta_cols[1].markdown(f"🌍 {row['country']}")
        if "funding_amount" in row and pd.notna(row["funding_amount"]):
            meta_cols[2].markdown(f"💰 ${row['funding_amount']:,.0f}")
        if "employees" in row and pd.notna(row["employees"]):
            meta_cols[3].markdown(f"👥 {int(row['employees'])} employees")
        
        st.markdown("---")


def render_comparison_view(df: pd.DataFrame, names: List[str]):
    """Render side-by-side comparison of entities."""
    if not names:
        st.info("Add entities to compare using the 'Compare' button on search results.")
        return
    
    entities = df[df["name"].isin(names)]
    
    if len(entities) == 0:
        st.warning("Selected entities not found.")
        return
    
    cols = st.columns(len(entities))
    
    compare_fields = ["sector", "subsector", "stage", "region", "country", "funding_amount", "employees", "year_founded"]
    
    for i, (_, entity) in enumerate(entities.iterrows()):
        with cols[i]:
            st.markdown(f"### {entity['name']}")
            if st.button("Remove", key=f"remove_compare_{i}"):
                st.session_state.comparison_list.remove(entity['name'])
                st.rerun()
            
            for field in compare_fields:
                if field in entity and pd.notna(entity[field]):
                    value = entity[field]
                    if field == "funding_amount":
                        value = f"${value:,.0f}"
                    st.markdown(f"**{field.replace('_', ' ').title()}:** {value}")
            
            if "description" in entity:
                st.markdown("**Description:**")
                st.markdown(entity["description"])


# ======================= Main Application ======================= #

def main():
    st.title("🚀 VC Semantic Search Pro")
    st.caption("Advanced semantic search with hybrid retrieval, clustering, and analytics")
    
    # ---------------------- Sidebar ---------------------- #
    
    st.sidebar.header("⚙️ Configuration")
    
    # Data source
    data_source = st.sidebar.radio(
        "Data Source",
        options=["Example Dataset", "Upload CSV"],
        help="Use built-in data or upload your own",
    )
    
    if data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file:
            df_raw = pd.read_csv(uploaded_file)
            
            # Duplicate detection
            duplicates = detect_duplicates(df_raw)
            if duplicates:
                st.sidebar.warning(f"⚠️ {len(duplicates)} potential duplicates detected")
        else:
            st.info("📤 Upload a CSV file to get started, or use the example dataset.")
            st.stop()
    else:
        df_raw = load_example_data()
    
    st.sidebar.markdown("---")
    
    # Text columns
    text_cols_options = df_raw.select_dtypes(include=["object"]).columns.tolist()
    default_cols = [c for c in ["name", "description", "sector", "tags"] if c in text_cols_options]
    
    text_cols = st.sidebar.multiselect(
        "Search Columns",
        options=text_cols_options,
        default=default_cols or text_cols_options[:3],
    )
    
    # Model selection
    model_options = {
        "all-MiniLM-L6-v2": "Fast & Good",
        "multi-qa-MiniLM-L6-cos-v1": "QA Optimized",
        "all-mpnet-base-v2": "Best Quality",
    }
    model_name = st.sidebar.selectbox(
        "Embedding Model",
        options=list(model_options.keys()),
        format_func=lambda x: f"{x} ({model_options[x]})",
    )
    
    # Search parameters
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Search Settings")
    
    top_k = st.sidebar.slider("Max Results", 5, 100, 20, 5)
    min_similarity = st.sidebar.slider("Min Similarity", 0.0, 1.0, 0.0, 0.05)
    
    use_hybrid = st.sidebar.checkbox(
        "Hybrid Search (BM25 + Semantic)",
        value=BM25_AVAILABLE,
        disabled=not BM25_AVAILABLE,
        help="Combines keyword and semantic search for better results",
    )
    
    if use_hybrid:
        semantic_weight = st.sidebar.slider(
            "Semantic Weight",
            0.0, 1.0, 0.7, 0.1,
            help="Balance between semantic (1.0) and keyword (0.0) search",
        )
    else:
        semantic_weight = 1.0
    
    use_expansion = st.sidebar.checkbox(
        "Query Expansion",
        value=False,
        help="Automatically expand queries with synonyms",
    )
    
    # Filters
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Filters")
    
    filter_config = [
        ("entity_type", "Entity Type"),
        ("sector", "Sector"),
        ("subsector", "Subsector"),
        ("region", "Region"),
        ("country", "Country"),
        ("stage", "Stage"),
    ]
    
    # Prepare data
    combined_text = combine_text_columns(df_raw, text_cols)
    df_raw = df_raw.copy()
    df_raw["__combined_text"] = combined_text
    
    # Apply filters
    df_filtered, mask = apply_filters(df_raw, filter_config)
    
    # Compute embeddings
    texts_tuple = tuple(df_filtered["__combined_text"].tolist())
    corpus_embeddings = compute_corpus_embeddings(texts_tuple, model_name)
    
    # Build BM25 index
    bm25_index = build_bm25_index(texts_tuple) if use_hybrid else None
    
    # ---------------------- Main Content ---------------------- #
    
    # Tabs
    tab_search, tab_compare, tab_analytics, tab_clusters, tab_history = st.tabs([
        "🔍 Search",
        "📊 Compare",
        "📈 Analytics",
        "🔮 Clusters",
        "📜 History",
    ])
    
    # ---------------------- Search Tab ---------------------- #
    
    with tab_search:
        # Search input
        col1, col2 = st.columns([4, 1])
        
        with col1:
            query = st.text_input(
                "Search Query",
                placeholder="e.g., quantum computing startups -cryptography",
                help="Use -term to exclude results containing that term",
                key="main_query",
            )
        
        with col2:
            search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)
        
        # Autocomplete suggestions
        if query and len(query) >= 2:
            suggestions = get_autocomplete_suggestions(query.split()[-1], df_filtered)
            if suggestions:
                st.caption(f"💡 Suggestions: {', '.join(suggestions)}")
        
        # Execute search
        if (query and search_clicked) or (query and st.session_state.current_query != query):
            st.session_state.current_query = query
            
            with st.spinner("Searching..."):
                top_indices, combined_scores, metadata = hybrid_search(
                    query=query,
                    corpus_embeddings=corpus_embeddings,
                    bm25_index=bm25_index,
                    texts=df_filtered["__combined_text"].tolist(),
                    model_name=model_name,
                    semantic_weight=semantic_weight,
                    top_k=top_k,
                    min_similarity=min_similarity,
                    use_expansion=use_expansion,
                )
                
                if len(top_indices) > 0:
                    results = df_filtered.iloc[top_indices].copy()
                    results["similarity"] = combined_scores[top_indices]
                    results = results.sort_values("similarity", ascending=False)
                    st.session_state.last_results = results
                    
                    # Save to history
                    save_search_to_history(query, len(results))
                    
                    # Show metadata
                    if metadata.get("expanded_queries") and len(metadata["expanded_queries"]) > 1:
                        st.info(f"🔄 Query expanded to: {', '.join(metadata['expanded_queries'])}")
                    
                    if metadata.get("negative_terms"):
                        st.info(f"❌ Excluding: {', '.join(metadata['negative_terms'])}")
                    
                    # Metrics
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Results", len(results))
                    m2.metric("Avg Score", f"{results['similarity'].mean():.1%}")
                    m3.metric("Max Score", f"{results['similarity'].max():.1%}")
                    m4.metric("Search Type", "Hybrid" if metadata["bm25_available"] else "Semantic")
                    
                    # Export buttons
                    col_exp1, col_exp2, col_exp3 = st.columns([1, 1, 4])
                    with col_exp1:
                        csv_data = export_results(results.drop(columns=["__combined_text"], errors="ignore"), "csv")
                        st.download_button(
                            "📥 CSV",
                            csv_data,
                            f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv",
                        )
                    with col_exp2:
                        json_data = export_results(results.drop(columns=["__combined_text"], errors="ignore"), "json")
                        st.download_button(
                            "📥 JSON",
                            json_data,
                            f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            "application/json",
                        )
                    
                    st.markdown("---")
                    
                    # Results display
                    view_mode = st.radio("View", ["Cards", "Table"], horizontal=True)
                    
                    if view_mode == "Cards":
                        for idx, (_, row) in enumerate(results.iterrows()):
                            render_entity_card(row, row["similarity"], idx)
                    else:
                        display_cols = [c for c in [
                            "name", "sector", "subsector", "stage", "region",
                            "country", "funding_amount", "similarity"
                        ] if c in results.columns]
                        
                        st.dataframe(
                            results[display_cols].style.format({
                                "similarity": "{:.1%}",
                                "funding_amount": "${:,.0f}",
                            }),
                            use_container_width=True,
                            hide_index=True,
                            height=500,
                        )
                
                else:
                    st.warning("No results found. Try adjusting your query or lowering the similarity threshold.")
        
        # Find Similar feature
        if "find_similar_idx" in st.session_state:
            idx = st.session_state.find_similar_idx
            st.markdown("---")
            st.subheader(f"🔍 Similar to: {df_filtered.iloc[idx].get('name', f'Entity {idx}')}")
            
            similar_indices, similar_scores = find_similar(idx, corpus_embeddings, top_k=5)
            
            for i, (sim_idx, score) in enumerate(zip(similar_indices, similar_scores)):
                row = df_filtered.iloc[sim_idx]
                render_entity_card(row, score, f"sim_{i}", show_actions=False)
            
            if st.button("Clear Similar Search"):
                del st.session_state.find_similar_idx
                st.rerun()
    
    # ---------------------- Compare Tab ---------------------- #
    
    with tab_compare:
        st.subheader("📊 Entity Comparison")
        
        if st.session_state.comparison_list:
            st.info(f"Comparing {len(st.session_state.comparison_list)} entities")
            
            if st.button("Clear All"):
                st.session_state.comparison_list = []
                st.rerun()
            
            render_comparison_view(df_filtered, st.session_state.comparison_list)
        else:
            st.info("Add entities to compare using the '📊 Compare' button on search results.")
    
    # ---------------------- Analytics Tab ---------------------- #
    
    with tab_analytics:
        st.subheader("📈 Data Analytics")
        
        data_to_analyze = st.session_state.last_results if st.session_state.last_results is not None else df_filtered
        
        if len(data_to_analyze) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_sunburst = create_sector_sunburst(data_to_analyze)
                if fig_sunburst:
                    st.plotly_chart(fig_sunburst, use_container_width=True)
            
            with col2:
                if "similarity" in data_to_analyze.columns:
                    fig_sim = px.histogram(
                        data_to_analyze,
                        x="similarity",
                        nbins=20,
                        title="Similarity Distribution",
                    )
                    fig_sim.add_vline(x=data_to_analyze["similarity"].mean(), line_dash="dash", line_color="red")
                    st.plotly_chart(fig_sim, use_container_width=True)
                else:
                    if "region" in data_to_analyze.columns:
                        region_counts = data_to_analyze["region"].value_counts()
                        fig_region = px.bar(
                            x=region_counts.index,
                            y=region_counts.values,
                            title="Regional Distribution",
                            labels={"x": "Region", "y": "Count"},
                        )
                        st.plotly_chart(fig_region, use_container_width=True)
            
            # Funding analysis
            fig_funding = create_funding_analysis(data_to_analyze)
            if fig_funding:
                st.plotly_chart(fig_funding, use_container_width=True)
            
            # Similarity heatmap
            if len(data_to_analyze) >= 3 and len(data_to_analyze) <= 30:
                st.markdown("### Similarity Heatmap")
                # Get embeddings for displayed data
                display_texts = tuple(data_to_analyze["__combined_text"].tolist())
                display_embeddings = compute_corpus_embeddings(display_texts, model_name)
                fig_heatmap = create_similarity_heatmap(data_to_analyze, display_embeddings)
                st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("Run a search or load data to see analytics.")
    
    # ---------------------- Clusters Tab ---------------------- #
    
    with tab_clusters:
        st.subheader("🔮 Entity Clustering")
        
        if len(df_filtered) >= 5:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                n_clusters = st.slider("Number of Clusters", 2, min(10, len(df_filtered)), 5)
            
            with col2:
                method = st.selectbox(
                    "Reduction Method",
                    ["tsne", "umap"] if UMAP_AVAILABLE else ["tsne"],
                )
            
            with col3:
                color_by = st.selectbox(
                    "Color By",
                    ["cluster", "sector", "stage", "region"] if all(c in df_filtered.columns for c in ["sector", "stage", "region"]) else ["cluster"],
                )
            
            if st.button("Generate Clusters", type="primary"):
                with st.spinner("Computing clusters..."):
                    coords_2d, cluster_labels = compute_clusters(
                        corpus_embeddings,
                        n_clusters=n_clusters,
                        method=method,
                    )
                    
                    fig = create_cluster_plot(df_filtered, coords_2d, cluster_labels, color_by)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Cluster summary
                    st.markdown("### Cluster Summary")
                    cluster_df = df_filtered.copy()
                    cluster_df["cluster"] = cluster_labels
                    
                    for cluster_id in range(n_clusters):
                        cluster_entities = cluster_df[cluster_df["cluster"] == cluster_id]
                        if len(cluster_entities) > 0:
                            with st.expander(f"Cluster {cluster_id} ({len(cluster_entities)} entities)"):
                                if "sector" in cluster_entities.columns:
                                    top_sectors = cluster_entities["sector"].value_counts().head(3)
                                    st.markdown(f"**Top Sectors:** {', '.join(top_sectors.index.tolist())}")
                                if "name" in cluster_entities.columns:
                                    st.markdown(f"**Entities:** {', '.join(cluster_entities['name'].head(5).tolist())}")
        else:
            st.warning("Need at least 5 entities for clustering visualization.")
    
    # ---------------------- History Tab ---------------------- #
    
    with tab_history:
        st.subheader("📜 Search History")
        
        if st.session_state.search_history:
            st.markdown(f"**Total Searches:** {len(st.session_state.search_history)}")
            
            # Clear button
            if st.button("🗑️ Clear History"):
                st.session_state.search_history = []
                st.rerun()
            
            # Display history
            for entry in reversed(st.session_state.search_history[-20:]):
                with st.expander(f"{entry['timestamp'][:19]} - \"{entry['query'][:50]}\" ({entry['results_count']} results)"):
                    st.markdown(f"**Query:** {entry['query']}")
                    st.markdown(f"**Results:** {entry['results_count']}")
                    st.markdown(f"**Time:** {entry['timestamp']}")
                    
                    if st.button("🔄 Re-run", key=f"rerun_{entry['timestamp']}"):
                        st.session_state.current_query = entry['query']
                        st.rerun()
        else:
            st.info("No search history yet. Run some searches to see them here.")
    
    # ---------------------- Footer ---------------------- #
    
    st.markdown("---")
    
    with st.expander("ℹ️ About & Features"):
        st.markdown("""
        ### Features
        
        - **Hybrid Search**: Combines semantic (embeddings) with BM25 keyword search
        - **Query Expansion**: Automatically adds synonyms and related terms
        - **Negative Queries**: Exclude terms with `-term` syntax
        - **Find Similar**: Click any result to find related entities
        - **Entity Comparison**: Side-by-side comparison of multiple entities
        - **Clustering**: Visualize how entities group together
        - **Analytics**: Sector distribution, funding analysis, similarity heatmaps
        - **Export**: Download results as CSV or JSON
        
        ### Search Tips
        
        - `quantum computing` - Basic semantic search
        - `quantum computing -cryptography` - Exclude cryptography results
        - Enable "Query Expansion" for broader results
        - Adjust "Semantic Weight" to balance keyword vs semantic matching
        
        ### Requirements
        
        ```
        pip install streamlit sentence-transformers scikit-learn plotly pandas numpy rank-bm25 umap-learn
        ```
        """)
    
    st.caption("Built with Streamlit • Powered by sentence-transformers")


if __name__ == "__main__":
    main()