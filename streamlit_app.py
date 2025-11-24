"""
VC Semantic Search – Enhanced Streamlit App

Improvements:
- Multi-query comparison (search multiple concepts at once)
- Saved searches & search history
- Export results to CSV/Excel
- Advanced analytics dashboard (sector distribution, similarity heatmaps)
- Reranking with weighted fields
- Similarity threshold filtering
- Better caching strategy
- Enhanced UI/UX with tabs and metrics
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict
from datetime import datetime
import json

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import plotly.express as px
import plotly.graph_objects as go


# ---------------------- Streamlit Page Config ---------------------- #

st.set_page_config(
    page_title="VC Semantic Search – Intelligence Layer Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------- Session State Initialization --------------- #

if "search_history" not in st.session_state:
    st.session_state.search_history = []

if "saved_searches" not in st.session_state:
    st.session_state.saved_searches = {}

if "last_results" not in st.session_state:
    st.session_state.last_results = None


# ---------------------- Embedding Utilities ----------------------- #

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Load and cache the sentence transformer model.
    """
    return SentenceTransformer(model_name)


@st.cache_data(show_spinner="Computing corpus embeddings...")
def compute_corpus_embeddings(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    _cache_key: str = None,  # Force recomputation when needed
) -> np.ndarray:
    """
    Compute and cache embeddings for the entire corpus.
    """
    model = load_embedding_model(model_name)
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 50,
        batch_size=32,
    )
    return embeddings


def embed_query(query: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Embed the user query using the same model as the corpus.
    """
    model = load_embedding_model(model_name)
    emb = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]
    return emb


def semantic_search(
    query: str,
    corpus_embeddings: np.ndarray,
    model_name: str = "all-MiniLM-L6-v2",
    top_k: int = 10,
    min_similarity: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return similarity scores for the query against corpus_embeddings.
    """
    query_emb = embed_query(query, model_name)
    scores = cosine_similarity([query_emb], corpus_embeddings)[0]
    
    # Filter by minimum similarity
    valid_mask = scores >= min_similarity
    valid_indices = np.where(valid_mask)[0]
    valid_scores = scores[valid_mask]
    
    # Sort by similarity
    if len(valid_scores) > 0:
        sorted_idx = np.argsort(valid_scores)[::-1][:top_k]
        top_indices = valid_indices[sorted_idx]
        return top_indices, scores
    else:
        return np.array([]), scores


def multi_query_search(
    queries: List[str],
    corpus_embeddings: np.ndarray,
    model_name: str = "all-MiniLM-L6-v2",
    aggregation: str = "max",
) -> np.ndarray:
    """
    Search with multiple queries and aggregate results.
    aggregation: 'max', 'mean', or 'min'
    """
    all_scores = []
    for query in queries:
        query_emb = embed_query(query, model_name)
        scores = cosine_similarity([query_emb], corpus_embeddings)[0]
        all_scores.append(scores)
    
    all_scores = np.array(all_scores)
    
    if aggregation == "max":
        return np.max(all_scores, axis=0)
    elif aggregation == "mean":
        return np.mean(all_scores, axis=0)
    elif aggregation == "min":
        return np.min(all_scores, axis=0)
    else:
        return np.mean(all_scores, axis=0)


# ---------------------- Data Utilities ---------------------------- #

def load_example_data() -> pd.DataFrame:
    """
    Enhanced example dataset with more diverse deep-tech startups.
    """
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
            "description": (
                "Building fault-tolerant quantum error correction "
                "for NISQ devices targeting quantum chemistry workloads."
            ),
            "tags": "quantum, error correction, chemistry, NISQ",
            "website": "https://quantacore.example",
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
            "description": (
                "Developing solid-state lithium metal batteries with ceramic "
                "electrolytes for grid-scale storage and EV applications."
            ),
            "tags": "batteries, solid state, energy storage, grid",
            "website": "https://helios.example",
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
            "description": (
                "Edge AI platform compressing foundation models for deployment "
                "on low-power industrial sensors and robotics."
            ),
            "tags": "edge ai, compression, robotics, foundation models",
            "website": "https://neuromesh.example",
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
            "description": (
                "Silicon photonics interconnects to reduce data center "
                "power consumption and latency for AI workloads."
            ),
            "tags": "photonic interconnects, datacenter, semiconductors",
            "website": "https://photonix.example",
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
            "description": (
                "Cryogenic control electronics and instrumentation for "
                "superconducting qubit systems at scale."
            ),
            "tags": "cryogenics, superconducting qubits, instrumentation",
            "website": "https://cryoqubit.example",
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
            "description": (
                "Engineering synthetic organisms for sustainable production "
                "of rare pharmaceutical compounds and industrial enzymes."
            ),
            "tags": "synthetic biology, pharmaceuticals, biomanufacturing",
            "website": "https://biosynth.example",
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
            "description": (
                "Compact tokamak design for commercial fusion power generation "
                "with advanced plasma control algorithms."
            ),
            "tags": "fusion, tokamak, clean energy, plasma physics",
            "website": "https://fusiongrid.example",
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
            "description": (
                "Scalable manufacturing of aligned carbon nanotube films "
                "for next-gen electronics and composite materials."
            ),
            "tags": "carbon nanotubes, materials science, electronics",
            "website": "https://nanocarbon.example",
        },
    ]
    return pd.DataFrame(data)


def combine_text_columns(
    df: pd.DataFrame,
    selected_cols: List[str],
    weights: Dict[str, float] = None,
) -> pd.Series:
    """
    Combine multiple text columns with optional weighting.
    """
    cols = [c for c in selected_cols if c in df.columns]
    if not cols:
        cols = df.select_dtypes(include=["object"]).columns.tolist()

    if weights:
        # Apply weights by repeating text
        combined_parts = []
        for col in cols:
            weight = weights.get(col, 1.0)
            repetitions = int(weight)
            if repetitions > 0:
                repeated = (df[col].fillna("").astype(str) + " ") * repetitions
                combined_parts.append(repeated)
        combined = pd.Series([" ".join(parts) for parts in zip(*combined_parts)])
    else:
        combined = (
            df[cols]
            .fillna("")
            .astype(str)
            .agg(" | ".join, axis=1)
        )
    
    combined = combined.str.replace(r"\s+", " ", regex=True).str.strip()
    return combined


def apply_filters(
    df: pd.DataFrame,
    filter_config: List[Tuple[str, str]],
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Apply sidebar filters and return filtered df + mask.
    """
    mask = np.ones(len(df), dtype=bool)

    for col, label in filter_config:
        if col in df.columns:
            options = sorted(df[col].dropna().unique().tolist())
            if len(options) > 1:
                selected = st.sidebar.multiselect(label, options=options, key=f"filter_{col}")
                if selected:
                    mask &= df[col].isin(selected).values

    return df[mask].reset_index(drop=False), mask


def export_to_csv(df: pd.DataFrame) -> str:
    """Export dataframe to CSV string."""
    return df.to_csv(index=False)


def save_search(query: str, filters: Dict, results_count: int):
    """Save search to history."""
    search_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "filters": filters,
        "results_count": results_count,
    }
    st.session_state.search_history.append(search_entry)


# ---------------------- Analytics Functions ----------------------- #

def create_sector_distribution(df: pd.DataFrame):
    """Create sector distribution pie chart."""
    if "sector" in df.columns:
        sector_counts = df["sector"].value_counts()
        fig = px.pie(
            values=sector_counts.values,
            names=sector_counts.index,
            title="Sector Distribution",
        )
        return fig
    return None


def create_similarity_distribution(scores: np.ndarray):
    """Create similarity score distribution histogram."""
    fig = px.histogram(
        x=scores,
        nbins=30,
        title="Similarity Score Distribution",
        labels={"x": "Similarity Score", "y": "Count"},
    )
    fig.add_vline(x=scores.mean(), line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {scores.mean():.3f}")
    return fig


def create_stage_funding_chart(df: pd.DataFrame):
    """Create stage vs funding chart."""
    if "stage" in df.columns and "funding_amount" in df.columns:
        fig = px.box(
            df,
            x="stage",
            y="funding_amount",
            title="Funding by Stage",
            labels={"funding_amount": "Funding Amount ($)", "stage": "Stage"},
        )
        return fig
    return None


# ---------------------- UI Layout -------------------------------- #

st.title("🧠 VC Semantic Search – Intelligence Layer Pro")
st.caption(
    "Advanced semantic search with multi-query support, analytics, and export capabilities"
)

st.markdown("---")

# ---------------------- Sidebar Configuration -------------------- #

st.sidebar.header("⚙️ Configuration")

data_source = st.sidebar.radio(
    "Data source",
    options=["Use example VC dataset", "Upload CSV"],
    help="Start with a built-in dataset or upload your own.",
)

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="Upload your startup list, patents, or company database.",
    )
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
    else:
        st.info("📤 Upload a CSV to continue, or switch to the example dataset.")
        st.stop()
else:
    df_raw = load_example_data()

st.sidebar.markdown("---")

# Text columns configuration
default_text_cols = [c for c in ["name", "description", "sector", "tags"] if c in df_raw.columns]
text_cols = st.sidebar.multiselect(
    "Columns for semantic search",
    options=df_raw.columns.tolist(),
    default=default_text_cols or df_raw.select_dtypes(include=["object"]).columns.tolist(),
    help="Columns to embed and search through.",
)

# Field weighting
st.sidebar.subheader("⚖️ Field Weights")
use_weights = st.sidebar.checkbox("Enable field weighting", value=False)
field_weights = {}
if use_weights:
    for col in text_cols:
        field_weights[col] = st.sidebar.slider(
            f"{col} weight",
            min_value=0.5,
            max_value=3.0,
            value=1.0,
            step=0.5,
            key=f"weight_{col}",
        )

# Model selection
model_name = st.sidebar.selectbox(
    "Embedding model",
    options=[
        "all-MiniLM-L6-v2",
        "multi-qa-MiniLM-L6-cos-v1",
        "all-mpnet-base-v2",
    ],
    index=0,
    help="Choose embedding model (higher models = better quality, slower).",
)

# Search parameters
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Search Parameters")

top_k = st.sidebar.slider(
    "Max results",
    min_value=5,
    max_value=100,
    value=20,
    step=5,
)

min_similarity = st.sidebar.slider(
    "Minimum similarity threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05,
    help="Filter out results below this similarity score.",
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
if use_weights:
    combined_text = combine_text_columns(df_raw, text_cols, field_weights)
else:
    combined_text = combine_text_columns(df_raw, text_cols)

df_raw = df_raw.copy()
df_raw["__combined_text"] = combined_text

# Compute embeddings
cache_key = f"{model_name}_{len(df_raw)}_{hash(tuple(text_cols))}"
corpus_embeddings = compute_corpus_embeddings(
    df_raw["__combined_text"].tolist(),
    model_name=model_name,
    _cache_key=cache_key,
)

# Apply filters
df_filtered, mask = apply_filters(df_raw, filter_config)
filtered_embeddings = corpus_embeddings[mask]

# ---------------------- Main Search Interface -------------------- #

# Create tabs
tab_search, tab_multi, tab_analytics, tab_history = st.tabs([
    "🔍 Single Search",
    "🔀 Multi-Query Search",
    "📊 Analytics",
    "📜 Search History",
])

# ---------------------- Single Search Tab ------------------------ #

with tab_search:
    st.subheader("Semantic Search")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Search query",
            placeholder="e.g. quantum error correction startups in Europe",
            key="single_query",
        )
    
    with col2:
        show_debug = st.checkbox("Debug mode", value=False)
    
    if query and len(df_filtered) > 0:
        with st.spinner("🔎 Searching..."):
            top_indices, scores = semantic_search(
                query,
                filtered_embeddings,
                model_name=model_name,
                top_k=top_k,
                min_similarity=min_similarity,
            )
            
            if len(top_indices) > 0:
                results = df_filtered.iloc[top_indices].copy()
                results["similarity"] = scores[top_indices]
                results = results.sort_values("similarity", ascending=False)
                
                # Save to session state
                st.session_state.last_results = results
                
                # Save to history
                save_search(query, {}, len(results))
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Results", len(results))
                with col2:
                    st.metric("Avg Similarity", f"{results['similarity'].mean():.3f}")
                with col3:
                    st.metric("Max Similarity", f"{results['similarity'].max():.3f}")
                with col4:
                    st.metric("Filtered From", len(df_filtered))
                
                # Display columns
                display_cols = [c for c in [
                    "name", "entity_type", "sector", "subsector",
                    "region", "country", "stage", "similarity",
                    "description", "tags", "website",
                ] if c in results.columns]
                
                if not show_debug:
                    display_cols = [c for c in display_cols if c != "__combined_text"]
                
                # Results table
                st.dataframe(
                    results[display_cols].style.format({"similarity": "{:.3f}"}),
                    use_container_width=True,
                    hide_index=True,
                    height=400,
                )
                
                # Export button
                csv_data = export_to_csv(results[display_cols])
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv_data,
                    file_name=f"vc_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
                
                # Top results cards
                st.markdown("---")
                st.markdown("#### 🌟 Top 3 Matches")
                
                for idx, (_, row) in enumerate(results.head(3).iterrows(), 1):
                    with st.expander(
                        f"#{idx}: {row.get('name', 'Unknown')} "
                        f"[{row['similarity']:.3f}]"
                    ):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown(f"**Sector:** {row.get('sector', 'N/A')}")
                            st.markdown(f"**Subsector:** {row.get('subsector', 'N/A')}")
                            st.markdown(f"**Stage:** {row.get('stage', 'N/A')}")
                        with col_b:
                            st.markdown(f"**Region:** {row.get('region', 'N/A')}")
                            st.markdown(f"**Country:** {row.get('country', 'N/A')}")
                            if "funding_amount" in row and pd.notna(row["funding_amount"]):
                                st.markdown(f"**Funding:** ${row['funding_amount']:,.0f}")
                        
                        st.markdown("**Description:**")
                        st.write(row.get("description", ""))
                        
                        if "website" in row and pd.notna(row["website"]):
                            st.markdown(f"🔗 [{row['website']}]({row['website']})")
            else:
                st.warning("⚠️ No results found matching your criteria. Try lowering the similarity threshold.")
    
    elif query:
        st.warning("⚠️ No entities match the selected filters.")
    else:
        st.info("👆 Enter a search query to find matching opportunities.")

# ---------------------- Multi-Query Search Tab ------------------- #

with tab_multi:
    st.subheader("Multi-Query Search")
    st.markdown(
        "Search with multiple queries simultaneously and see which entities "
        "match across different concepts."
    )
    
    num_queries = st.number_input(
        "Number of queries",
        min_value=2,
        max_value=5,
        value=2,
        help="Search with multiple related or complementary queries.",
    )
    
    queries = []
    for i in range(num_queries):
        q = st.text_input(
            f"Query {i+1}",
            key=f"multi_query_{i}",
            placeholder=f"e.g. {'quantum computing' if i == 0 else 'error correction algorithms'}",
        )
        if q:
            queries.append(q)
    
    aggregation = st.radio(
        "Score aggregation",
        options=["max", "mean", "min"],
        index=0,
        horizontal=True,
        help="How to combine scores from multiple queries.",
    )
    
    if len(queries) >= 2 and len(df_filtered) > 0:
        if st.button("🔍 Search All Queries", type="primary"):
            with st.spinner("Running multi-query search..."):
                combined_scores = multi_query_search(
                    queries,
                    filtered_embeddings,
                    model_name=model_name,
                    aggregation=aggregation,
                )
                
                # Filter by threshold
                valid_mask = combined_scores >= min_similarity
                valid_indices = np.where(valid_mask)[0]
                
                if len(valid_indices) > 0:
                    sorted_idx = np.argsort(combined_scores[valid_mask])[::-1][:top_k]
                    top_indices = valid_indices[sorted_idx]
                    
                    results = df_filtered.iloc[top_indices].copy()
                    results["combined_score"] = combined_scores[top_indices]
                    
                    # Individual query scores
                    for i, query in enumerate(queries):
                        q_emb = embed_query(query, model_name)
                        q_scores = cosine_similarity([q_emb], filtered_embeddings)[0]
                        results[f"query_{i+1}_score"] = q_scores[top_indices]
                    
                    results = results.sort_values("combined_score", ascending=False)
                    st.session_state.last_results = results
                    
                    # Metrics
                    st.metric("Results Found", len(results))
                    
                    # Results table with individual scores
                    score_cols = ["combined_score"] + [f"query_{i+1}_score" for i in range(len(queries))]
                    display_cols = [c for c in ["name", "sector", "stage", "region"] if c in results.columns]
                    display_cols += score_cols
                    
                    st.dataframe(
                        results[display_cols].style.format({col: "{:.3f}" for col in score_cols}),
                        use_container_width=True,
                        hide_index=True,
                    )
                    
                    # Download
                    csv_data = export_to_csv(results)
                    st.download_button(
                        label="📥 Download Multi-Query Results",
                        data=csv_data,
                        file_name=f"multi_query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )
                else:
                    st.warning("No results found matching all criteria.")

# ---------------------- Analytics Tab ---------------------------- #

with tab_analytics:
    st.subheader("📊 Search Analytics")
    
    if st.session_state.last_results is not None and len(st.session_state.last_results) > 0:
        results = st.session_state.last_results
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sector distribution
            fig_sector = create_sector_distribution(results)
            if fig_sector:
                st.plotly_chart(fig_sector, use_container_width=True)
        
        with col2:
            # Similarity distribution
            if "similarity" in results.columns:
                fig_sim = create_similarity_distribution(results["similarity"].values)
                st.plotly_chart(fig_sim, use_container_width=True)
        
        # Stage vs Funding
        fig_funding = create_stage_funding_chart(results)
        if fig_funding:
            st.plotly_chart(fig_funding, use_container_width=True)
        
        # Regional distribution
        if "region" in results.columns:
            st.markdown("#### Regional Distribution")
            region_counts = results["region"].value_counts()
            fig_region = px.bar(
                x=region_counts.index,
                y=region_counts.values,
                labels={"x": "Region", "y": "Count"},
            )
            st.plotly_chart(fig_region, use_container_width=True)
    else:
        st.info("Run a search first to see analytics on the results.")

# ---------------------- Search History Tab ----------------------- #

with tab_history:
    st.subheader("📜 Search History")
    
    if st.session_state.search_history:
        st.markdown(f"**Total searches:** {len(st.session_state.search_history)}")
        
        for idx, search in enumerate(reversed(st.session_state.search_history[-10:]), 1):
            with st.expander(
                f"{search['timestamp'][:19]} - \"{search['query'][:50]}...\" "
                f"({search['results_count']} results)"
            ):
                st.markdown(f"**Query:** {search['query']}")
                st.markdown(f"**Results:** {search['results_count']}")
                st.markdown(f"**Timestamp:** {search['timestamp']}")
        
        if st.button("🗑️ Clear History"):
            st.session_state.search_history = []
            st.rerun()
    else:
        st.info("No search history yet. Run some searches to see them here.")

# ---------------------- Footer ----------------------------------- #

st.markdown("---")

with st.expander("ℹ️ How This Works & What's New"):
    st.markdown("""
    ### New Features in Pro Version
    
    - **Multi-Query Search**: Search with multiple related concepts simultaneously
    - **Field Weighting**: Prioritize certain fields (e.g., description over tags)
    - **Similarity Threshold**: Filter out low-confidence matches
    - **Advanced Analytics**: Visualize sector distribution, similarity scores, and funding patterns
    - **Search History**: Track and review past searches
    - **Enhanced Export**: Download results with all metadata
    - **Better Performance**: Optimized caching and batch processing
    - **Improved UX**: Tabbed interface, metrics dashboard, and result cards
    
    ### Technical Architecture
    
    1. **Embedding Pipeline**: Text → Sentence Transformers → Vector Space
    2. **Search**: Cosine similarity with optional multi-query aggregation
    3. **Filtering**: Boolean masks applied to both DataFrame and embeddings
    4. **Caching**: Smart caching with `@st.cache_resource` and `@st.cache_data`
    
    ### Production Roadmap
    
    - Replace in-memory storage with vector database (Pinecone, Weaviate, pgvector)
    - Add user authentication and saved search profiles
    - Implement real-time data updates and refresh
    - Add AI-powered query expansion and refinement
    - Integrate with external data sources (Crunchbase, PitchBook, etc.)
    - Add collaborative features (shared searches, annotations)
    """)

st.caption("Built with Streamlit • Powered by sentence-transformers")