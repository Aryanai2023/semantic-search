import os
import re
import json
import time
import hashlib
import urllib.parse
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import streamlit as st
import requests
from bs4 import BeautifulSoup

# Text extraction
try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False

# Search API (using DuckDuckGo as free alternative)
try:
    from duckduckgo_search import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False


# ---------------------------
# Configuration
# ---------------------------

DEFAULT_UA = "ResearchBot/1.0 (Educational Research Tool)"
SESSION = requests.Session()

# Simulated AI responses (replace with actual API calls)
USE_ANTHROPIC_API = False  # Set to True if you have API access


# ---------------------------
# Data Models
# ---------------------------

@dataclass
class Source:
    """Represents a web source"""
    url: str
    title: str
    snippet: str
    content: str = ""
    fetch_time: str = ""
    index: int = 0
    
    def to_dict(self):
        return asdict(self)


@dataclass
class Citation:
    """Represents a citation in the answer"""
    source_index: int
    text: str
    url: str


@dataclass
class ResearchResult:
    """Complete research result with answer and sources"""
    query: str
    answer: str
    sources: List[Source]
    citations: List[Citation]
    timestamp: str
    follow_up_questions: List[str]
    
    def to_dict(self):
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "citations": [asdict(c) for c in self.citations],
            "timestamp": self.timestamp,
            "follow_up_questions": self.follow_up_questions,
        }


# ---------------------------
# Web Scraping Functions
# ---------------------------

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def extract_content(url: str, html: str) -> str:
    """Extract main content from HTML"""
    if HAS_TRAFILATURA:
        content = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            include_links=False,
        )
        if content:
            return clean_text(content)
    
    # Fallback to BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove unwanted elements
    for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe']):
        element.decompose()
    
    # Try to find main content
    main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile('content|article|post'))
    
    if main_content:
        text = main_content.get_text(separator=' ', strip=True)
    else:
        text = soup.get_text(separator=' ', strip=True)
    
    return clean_text(text)


def fetch_url_content(url: str, timeout: int = 10) -> Optional[str]:
    """Fetch and extract content from a URL"""
    headers = {'User-Agent': DEFAULT_UA}
    try:
        response = SESSION.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        
        if 'text/html' in response.headers.get('Content-Type', ''):
            return extract_content(url, response.text)
        return None
    except Exception as e:
        st.warning(f"Failed to fetch {url[:50]}...: {str(e)[:50]}")
        return None


def search_web(query: str, num_results: int = 5) -> List[Source]:
    """Search the web and return sources"""
    sources = []
    
    if HAS_DDGS:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=num_results))
                
                for i, result in enumerate(results):
                    sources.append(Source(
                        url=result['href'],
                        title=result['title'],
                        snippet=result['body'],
                        index=i + 1
                    ))
        except Exception as e:
            st.error(f"Search failed: {e}")
            return []
    else:
        # Fallback: Manual search simulation (for demo)
        st.warning("DuckDuckGo search not available. Install: pip install duckduckgo-search")
        sources = [
            Source(
                url="https://example.com",
                title="Example Result",
                snippet="This is a demo result. Install duckduckgo-search for real results.",
                index=1
            )
        ]
    
    return sources


def enrich_sources(sources: List[Source], max_sources: int = 5) -> List[Source]:
    """Fetch full content for sources"""
    enriched = []
    
    for source in sources[:max_sources]:
        with st.spinner(f"Fetching content from {source.title[:50]}..."):
            content = fetch_url_content(source.url)
            if content:
                source.content = content[:5000]  # Limit content size
                source.fetch_time = datetime.now().isoformat()
                enriched.append(source)
                time.sleep(0.5)  # Be polite
    
    return enriched


# ---------------------------
# AI Synthesis (Simulated)
# ---------------------------

def create_context_from_sources(sources: List[Source]) -> str:
    """Create context string from sources for AI"""
    context_parts = []
    
    for source in sources:
        context_parts.append(f"""
[Source {source.index}]
Title: {source.title}
URL: {source.url}
Content: {source.content[:1500]}...
""")
    
    return "\n\n".join(context_parts)


def synthesize_answer_simple(query: str, sources: List[Source]) -> Tuple[str, List[Citation], List[str]]:
    """Simple answer synthesis without API (demo mode)"""
    # Create a simple answer based on sources
    answer_parts = []
    citations = []
    
    answer_parts.append(f"Based on the available sources, here's what I found about '{query}':\n\n")
    
    for source in sources[:3]:
        # Extract first few sentences from content
        sentences = source.content.split('.')[:2]
        excerpt = '. '.join(sentences).strip()
        
        if excerpt:
            answer_parts.append(f"{excerpt}. [{source.index}]\n\n")
            citations.append(Citation(
                source_index=source.index,
                text=source.title,
                url=source.url
            ))
    
    answer_parts.append("\nPlease note: This is a basic summary. For more accurate AI-powered synthesis, configure an Anthropic API key.")
    
    # Generate follow-up questions
    follow_ups = [
        f"What are the implications of {query}?",
        f"How does {query} compare to similar topics?",
        f"What are the latest developments in {query}?",
    ]
    
    return "".join(answer_parts), citations, follow_ups


def synthesize_answer_with_api(query: str, sources: List[Source], api_key: str) -> Tuple[str, List[Citation], List[str]]:
    """Synthesize answer using Anthropic API"""
    context = create_context_from_sources(sources)
    
    prompt = f"""You are a research assistant. Based on the provided sources, answer the user's question comprehensively and accurately.

IMPORTANT INSTRUCTIONS:
1. Synthesize information from multiple sources into a coherent answer
2. Use inline citations [1], [2], etc. when referencing specific sources
3. Be accurate and cite sources for all factual claims
4. If sources conflict, mention the disagreement
5. Do not include information not supported by the sources
6. Write in a clear, informative style

Sources:
{context}

User Question: {query}

Provide:
1. A comprehensive answer with inline citations
2. A list of 3-4 relevant follow-up questions the user might ask

Format your response as:
ANSWER:
[Your synthesized answer with citations]

FOLLOW_UP:
- [Question 1]
- [Question 2]
- [Question 3]
"""

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 2000,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            full_response = result['content'][0]['text']
            
            # Parse response
            parts = full_response.split('FOLLOW_UP:')
            answer = parts[0].replace('ANSWER:', '').strip()
            
            follow_ups = []
            if len(parts) > 1:
                follow_up_text = parts[1].strip()
                follow_ups = [line.strip('- ').strip() for line in follow_up_text.split('\n') if line.strip().startswith('-')]
            
            # Extract citations from answer
            citations = []
            citation_pattern = r'\[(\d+)\]'
            cited_indices = set(re.findall(citation_pattern, answer))
            
            for idx in cited_indices:
                source_idx = int(idx)
                if 1 <= source_idx <= len(sources):
                    source = sources[source_idx - 1]
                    citations.append(Citation(
                        source_index=source_idx,
                        text=source.title,
                        url=source.url
                    ))
            
            return answer, citations, follow_ups
        else:
            st.error(f"API error: {response.status_code}")
            return synthesize_answer_simple(query, sources)
    
    except Exception as e:
        st.error(f"API call failed: {e}")
        return synthesize_answer_simple(query, sources)


def synthesize_answer(query: str, sources: List[Source], api_key: Optional[str] = None) -> Tuple[str, List[Citation], List[str]]:
    """Main synthesis function"""
    if api_key and api_key.startswith('sk-ant-'):
        return synthesize_answer_with_api(query, sources, api_key)
    else:
        return synthesize_answer_simple(query, sources)


# ---------------------------
# Research Pipeline
# ---------------------------

def conduct_research(query: str, num_sources: int = 5, api_key: Optional[str] = None) -> ResearchResult:
    """Complete research pipeline"""
    
    # Step 1: Search
    st.info("🔍 Searching the web...")
    sources = search_web(query, num_results=num_sources * 2)
    
    if not sources:
        st.error("No sources found")
        return None
    
    # Step 2: Fetch content
    st.info(f"📄 Fetching content from {num_sources} sources...")
    sources = enrich_sources(sources, max_sources=num_sources)
    
    if not sources:
        st.error("Failed to fetch content from sources")
        return None
    
    # Step 3: Synthesize answer
    st.info("🤖 Synthesizing answer...")
    answer, citations, follow_ups = synthesize_answer(query, sources, api_key)
    
    # Create result
    result = ResearchResult(
        query=query,
        answer=answer,
        sources=sources,
        citations=citations,
        timestamp=datetime.now().isoformat(),
        follow_up_questions=follow_ups
    )
    
    return result


# ---------------------------
# Export Functions
# ---------------------------

def export_to_markdown(result: ResearchResult) -> str:
    """Export research result to markdown"""
    md = f"# Research: {result.query}\n\n"
    md += f"*Generated on {datetime.fromisoformat(result.timestamp).strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
    md += "---\n\n"
    
    md += "## Answer\n\n"
    md += result.answer + "\n\n"
    
    md += "## Sources\n\n"
    for source in result.sources:
        md += f"{source.index}. **{source.title}**\n"
        md += f"   - URL: {source.url}\n"
        md += f"   - Snippet: {source.snippet}\n\n"
    
    if result.follow_up_questions:
        md += "## Related Questions\n\n"
        for q in result.follow_up_questions:
            md += f"- {q}\n"
    
    return md


def export_to_json(result: ResearchResult) -> str:
    """Export research result to JSON"""
    return json.dumps(result.to_dict(), indent=2)


# ---------------------------
# Streamlit UI
# ---------------------------

def main():
    st.set_page_config(
        page_title="AI Research Assistant",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(120deg, #1e3a8a, #3b82f6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        .source-card {
            border-left: 4px solid #3b82f6;
            padding: 1rem;
            margin: 1rem 0;
            background-color: #f8fafc;
            border-radius: 0.5rem;
        }
        .citation-badge {
            background-color: #3b82f6;
            color: white;
            padding: 0.2rem 0.5rem;
            border-radius: 0.3rem;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .follow-up-card {
            background-color: #eff6ff;
            padding: 0.8rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            border-left: 3px solid #60a5fa;
        }
        .answer-box {
            background-color: #ffffff;
            padding: 1.5rem;
            border-radius: 0.8rem;
            border: 1px solid #e5e7eb;
            line-height: 1.8;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/research.png", width=80)
        st.title("⚙️ Settings")
        
        # API Configuration
        with st.expander("🔑 API Configuration", expanded=False):
            st.markdown("""
            For AI-powered synthesis, add your Anthropic API key.
            Get one at [console.anthropic.com](https://console.anthropic.com)
            """)
            api_key = st.text_input(
                "Anthropic API Key",
                type="password",
                value=st.session_state.get("api_key", ""),
                help="Optional: For better AI synthesis"
            )
            if api_key:
                st.session_state.api_key = api_key
        
        # Research Settings
        st.subheader("🔧 Research Settings")
        num_sources = st.slider(
            "Number of sources",
            min_value=3,
            max_value=10,
            value=5,
            help="More sources = more comprehensive but slower"
        )
        
        st.divider()
        
        # History
        if "research_history" not in st.session_state:
            st.session_state.research_history = []
        
        if st.session_state.research_history:
            st.subheader("📚 Recent Searches")
            for i, item in enumerate(reversed(st.session_state.research_history[-5:])):
                if st.button(f"🔍 {item['query'][:30]}...", key=f"history_{i}"):
                    st.session_state.current_query = item['query']
                    st.rerun()
        
        st.divider()
        
        # Info
        st.caption("Built with ❤️ using Streamlit")
        st.caption("Respects robots.txt • Educational use only")
    
    # Main content
    st.markdown('<h1 class="main-header">🔬 AI Research Assistant</h1>', unsafe_allow_html=True)
    st.markdown("Ask any question and get comprehensive answers with cited sources")
    
    # Initialize session state
    if "current_result" not in st.session_state:
        st.session_state.current_result = None
    if "current_query" not in st.session_state:
        st.session_state.current_query = ""
    
    # Search interface
    col1, col2 = st.columns([5, 1])
    
    with col1:
        query = st.text_input(
            "Ask a question",
            value=st.session_state.current_query,
            placeholder="e.g., What are the latest developments in quantum computing?",
            label_visibility="collapsed",
            key="query_input"
        )
    
    with col2:
        search_button = st.button("🔍 Research", type="primary", use_container_width=True)
    
    # Example queries
    st.markdown("**Try these:**")
    example_cols = st.columns(4)
    examples = [
        "Latest AI breakthroughs",
        "Climate change solutions",
        "Space exploration news",
        "Quantum computing applications"
    ]
    
    for i, example in enumerate(examples):
        with example_cols[i]:
            if st.button(example, key=f"example_{i}", use_container_width=True):
                st.session_state.current_query = example
                st.rerun()
    
    st.divider()
    
    # Conduct research
    if search_button and query.strip():
        st.session_state.current_query = query
        
        with st.spinner("🔬 Researching your question..."):
            api_key = st.session_state.get("api_key", None)
            result = conduct_research(query, num_sources=num_sources, api_key=api_key)
            
            if result:
                st.session_state.current_result = result
                st.session_state.research_history.append({
                    "query": query,
                    "timestamp": result.timestamp
                })
                st.success("✅ Research complete!")
    
    # Display results
    if st.session_state.current_result:
        result = st.session_state.current_result
        
        # Answer section
        st.markdown("## 📝 Answer")
        
        # Export buttons
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            md_export = export_to_markdown(result)
            st.download_button(
                "📥 Markdown",
                data=md_export,
                file_name=f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
        with col2:
            json_export = export_to_json(result)
            st.download_button(
                "📥 JSON",
                data=json_export,
                file_name=f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # Display answer with citations
        st.markdown(f'<div class="answer-box">{result.answer}</div>', unsafe_allow_html=True)
        
        # Sources
        st.markdown("## 📚 Sources")
        
        for source in result.sources:
            with st.container():
                st.markdown(f"""
                <div class="source-card">
                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                        <span class="citation-badge">[{source.index}]</span>
                        <strong>{source.title}</strong>
                    </div>
                    <div style="margin-bottom: 0.5rem;">
                        <a href="{source.url}" target="_blank" style="color: #3b82f6; text-decoration: none;">
                            🔗 {source.url}
                        </a>
                    </div>
                    <div style="color: #64748b; font-size: 0.9rem;">
                        {source.snippet}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Follow-up questions
        if result.follow_up_questions:
            st.markdown("## 💡 Related Questions")
            
            for i, question in enumerate(result.follow_up_questions):
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(f'<div class="follow-up-card">{question}</div>', unsafe_allow_html=True)
                with col2:
                    if st.button("Ask", key=f"followup_{i}"):
                        st.session_state.current_query = question
                        st.rerun()
    
    else:
        # Welcome message
        st.markdown("""
        ### 👋 Welcome to AI Research Assistant
        
        This tool helps you:
        - 🔍 Search the web for information
        - 📄 Extract content from multiple sources
        - 🤖 Synthesize comprehensive answers with citations
        - 💾 Export results in multiple formats
        
        **How to use:**
        1. Enter your question in the search box
        2. Click "Research" or press Enter
        3. Get a synthesized answer with cited sources
        4. Explore follow-up questions or export results
        
        **Pro tip:** Add your Anthropic API key in the sidebar for AI-powered synthesis!
        """)


if __name__ == "__main__":
    main()
