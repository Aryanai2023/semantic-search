"""
Flask-based AI Research Assistant (Perplexity-Style)
Alternative implementation using Flask instead of Streamlit
"""

import os
import re
import json
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from flask import Flask, render_template, request, jsonify, send_file
import requests
from bs4 import BeautifulSoup

# Text extraction
try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False

# Search API
try:
    from duckduckgo_search import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

DEFAULT_UA = "ResearchBot/1.0 (Educational Research Tool)"
SESSION = requests.Session()


# ---------------------------
# Helper Functions
# ---------------------------

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_content(url: str, html: str) -> str:
    """Extract main content from HTML"""
    if HAS_TRAFILATURA:
        content = trafilatura.extract(
            html, url=url, include_comments=False, 
            include_tables=True, include_links=False
        )
        if content:
            return clean_text(content)
    
    soup = BeautifulSoup(html, 'html.parser')
    for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
        element.decompose()
    
    main = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile('content'))
    text = main.get_text(separator=' ', strip=True) if main else soup.get_text(separator=' ', strip=True)
    return clean_text(text)


def fetch_url_content(url: str, timeout: int = 10) -> Optional[str]:
    """Fetch and extract content from URL"""
    headers = {'User-Agent': DEFAULT_UA}
    try:
        response = SESSION.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        
        if 'text/html' in response.headers.get('Content-Type', ''):
            return extract_content(url, response.text)
        return None
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return None


def search_web(query: str, num_results: int = 5) -> List[Dict]:
    """Search the web and return sources"""
    sources = []
    
    if HAS_DDGS:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=num_results))
                
                for i, result in enumerate(results):
                    sources.append({
                        'index': i + 1,
                        'url': result['href'],
                        'title': result['title'],
                        'snippet': result['body'],
                        'content': '',
                    })
        except Exception as e:
            print(f"Search failed: {e}")
    
    return sources


def enrich_sources(sources: List[Dict], max_sources: int = 5) -> List[Dict]:
    """Fetch full content for sources"""
    enriched = []
    
    for source in sources[:max_sources]:
        content = fetch_url_content(source['url'])
        if content:
            source['content'] = content[:5000]
            source['fetch_time'] = datetime.now().isoformat()
            enriched.append(source)
            time.sleep(0.5)
    
    return enriched


def synthesize_answer(query: str, sources: List[Dict], api_key: Optional[str] = None) -> Tuple[str, List[Dict], List[str]]:
    """Synthesize answer from sources"""
    
    if api_key and api_key.startswith('sk-ant-'):
        return synthesize_with_api(query, sources, api_key)
    
    # Simple synthesis
    answer_parts = [f"Based on the available sources for '{query}':\n\n"]
    citations = []
    
    for source in sources[:3]:
        sentences = source['content'].split('.')[:2]
        excerpt = '. '.join(sentences).strip()
        
        if excerpt:
            answer_parts.append(f"{excerpt}. [{source['index']}]\n\n")
            citations.append({
                'index': source['index'],
                'title': source['title'],
                'url': source['url']
            })
    
    follow_ups = [
        f"What are the implications of {query}?",
        f"How has {query} evolved over time?",
        f"What are the latest developments in {query}?",
    ]
    
    return "".join(answer_parts), citations, follow_ups


def synthesize_with_api(query: str, sources: List[Dict], api_key: str) -> Tuple[str, List[Dict], List[str]]:
    """Synthesize using Anthropic API"""
    context_parts = []
    for source in sources:
        context_parts.append(f"""
[Source {source['index']}]
Title: {source['title']}
URL: {source['url']}
Content: {source['content'][:1500]}...
""")
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""You are a research assistant. Answer the user's question using the provided sources.

INSTRUCTIONS:
1. Synthesize information from sources into a coherent answer
2. Use inline citations [1], [2], etc.
3. Be accurate and cite all factual claims
4. If sources conflict, mention it
5. Only use information from the sources

Sources:
{context}

Question: {query}

Provide:
ANSWER:
[Your answer with citations]

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
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            full_response = result['content'][0]['text']
            
            parts = full_response.split('FOLLOW_UP:')
            answer = parts[0].replace('ANSWER:', '').strip()
            
            follow_ups = []
            if len(parts) > 1:
                follow_up_text = parts[1].strip()
                follow_ups = [line.strip('- ').strip() 
                             for line in follow_up_text.split('\n') 
                             if line.strip().startswith('-')]
            
            citations = []
            cited_indices = set(re.findall(r'\[(\d+)\]', answer))
            
            for idx in cited_indices:
                source_idx = int(idx)
                if 1 <= source_idx <= len(sources):
                    source = sources[source_idx - 1]
                    citations.append({
                        'index': source_idx,
                        'title': source['title'],
                        'url': source['url']
                    })
            
            return answer, citations, follow_ups
    
    except Exception as e:
        print(f"API call failed: {e}")
    
    return synthesize_answer(query, sources, None)


# ---------------------------
# Flask Routes
# ---------------------------

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/api/research', methods=['POST'])
def research():
    """Research endpoint"""
    data = request.json
    query = data.get('query', '').strip()
    num_sources = int(data.get('num_sources', 5))
    api_key = data.get('api_key', None)
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    try:
        # Search
        sources = search_web(query, num_results=num_sources * 2)
        
        if not sources:
            return jsonify({'error': 'No sources found'}), 404
        
        # Fetch content
        sources = enrich_sources(sources, max_sources=num_sources)
        
        if not sources:
            return jsonify({'error': 'Failed to fetch content'}), 500
        
        # Synthesize
        answer, citations, follow_ups = synthesize_answer(query, sources, api_key)
        
        result = {
            'query': query,
            'answer': answer,
            'sources': sources,
            'citations': citations,
            'follow_up_questions': follow_ups,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/markdown', methods=['POST'])
def export_markdown():
    """Export to markdown"""
    data = request.json
    
    md = f"# Research: {data['query']}\n\n"
    md += f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
    md += "---\n\n"
    md += "## Answer\n\n"
    md += data['answer'] + "\n\n"
    md += "## Sources\n\n"
    
    for source in data['sources']:
        md += f"{source['index']}. **{source['title']}**\n"
        md += f"   - URL: {source['url']}\n"
        md += f"   - Snippet: {source['snippet']}\n\n"
    
    if data.get('follow_up_questions'):
        md += "## Related Questions\n\n"
        for q in data['follow_up_questions']:
            md += f"- {q}\n"
    
    return jsonify({'markdown': md})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
