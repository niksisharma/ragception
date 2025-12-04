"""Graph Module - Knowledge Graph Generation for Research Papers"""

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import List, Dict, Optional
import logging
import tempfile

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

try:
    from graphiti_core import Graphiti
    from graphiti_core.nodes import EpisodeType
    GRAPHITI_AVAILABLE = True
except ImportError:
    GRAPHITI_AVAILABLE = False

try:
    from neo4j import AsyncGraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False

logger = logging.getLogger(__name__)

def get_config():
    """Get configuration from environment or defaults"""
    return {
        "NEO4J_URI": os.environ.get("NEO4J_URI", ""),
        "NEO4J_USER": os.environ.get("NEO4J_USER", "neo4j"),
        "NEO4J_PASSWORD": os.environ.get("NEO4J_PASSWORD", ""),
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "")
    }

async def extract_concepts_llm(text: str, client: AsyncOpenAI) -> List[str]:
    """Extract key technical concepts using LLM"""
    prompt = f"""Extract the key technical concepts, methods, and technologies from this research abstract.
Return ONLY a JSON array of concept strings. Focus on:
- Technical terms and methodologies
- Model names and frameworks
- Key technologies mentioned
- Research contributions

Abstract: {text}

Return format: ["concept1", "concept2", ...]
"""
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a technical concept extractor. Return only valid JSON arrays."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        concepts = json.loads(content)
        return sorted(concepts)
    except Exception as e:
        logger.error(f"Error extracting concepts: {e}")
        return []

def create_simple_graph_html(papers: List[Dict], show_shared: bool = True) -> str:
    """
    Create a simple knowledge graph visualization without Neo4j
    Uses only Pyvis for visualization based on paper concepts
    """
    if not PYVIS_AVAILABLE:
        return "<p>Pyvis not installed. Run: pip install pyvis</p>"
    
    if not papers:
        return "<p>No papers to visualize</p>"
    
    # Create network
    net = Network(
        height="600px", 
        width="100%", 
        bgcolor="#0f172a", 
        font_color="white",
        directed=False
    )
    
    net.barnes_hut(
        gravity=-3000,
        central_gravity=0.3,
        spring_length=200,
        spring_strength=0.01,
        damping=0.09
    )
    
    # Track concepts across papers
    concept_to_papers = {}
    paper_concepts = {}
    
    # Add paper nodes and collect concepts
    for i, paper in enumerate(papers):
        paper_id = f"paper_{i}"
        title = paper.get('title', f'Paper {i+1}')[:50]
        abstract = paper.get('abstract', '')
        
        # Extract simple concepts from title and abstract
        concepts = extract_simple_concepts(title + " " + abstract)
        paper_concepts[paper_id] = concepts
        
        # Track which papers mention each concept
        for concept in concepts:
            if concept not in concept_to_papers:
                concept_to_papers[concept] = []
            concept_to_papers[concept].append(paper_id)
        
        # Add paper node
        net.add_node(
            paper_id,
            label=title,
            title=f"<b>{title}</b><br><br>{abstract[:200]}...",
            color="#3b82f6",
            size=35,
            shape="dot"
        )
    
    # Add concept nodes and edges
    added_concepts = set()
    
    for concept, paper_ids in concept_to_papers.items():
        # Determine if shared or unique
        is_shared = len(paper_ids) > 1
        
        if show_shared and is_shared:
            color = "#f59e0b"  # Orange for shared
            size = 25
        elif not show_shared and not is_shared:
            color = "#10b981"  # Green for unique
            size = 20
        elif show_shared and not is_shared:
            color = "#6b7280"  # Gray for unique when showing shared
            size = 15
        else:
            continue
        
        concept_id = f"concept_{concept.replace(' ', '_')}"
        
        if concept_id not in added_concepts:
            net.add_node(
                concept_id,
                label=concept[:25],
                title=f"<b>Concept:</b> {concept}<br><b>Mentioned in:</b> {len(paper_ids)} papers",
                color=color,
                size=size,
                shape="dot"
            )
            added_concepts.add(concept_id)
        
        # Add edges from papers to concepts
        for paper_id in paper_ids:
            net.add_edge(
                paper_id,
                concept_id,
                color="#475569",
                width=2 if is_shared else 1
            )
    
    # Generate HTML
    html = net.generate_html()
    
    # Fix for Streamlit iframe display
    html = html.replace(
        '<head>',
        '<head><style>body { margin: 0; }</style>'
    )
    
    return html


def extract_simple_concepts(text: str) -> List[str]:
    """Extract simple concepts using keyword matching"""
    concept_keywords = [
        "retrieval augmented generation", "RAG", "large language model", "LLM",
        "transformer", "attention", "BERT", "GPT", "embedding", "vector",
        "neural network", "deep learning", "machine learning", "NLP",
        "fine-tuning", "pre-training", "encoder", "decoder", "tokenization",
        "semantic search", "knowledge graph", "entity extraction", "QA",
        "question answering", "summarization", "classification", "clustering",
        "reinforcement learning", "RL", "reward", "agent", "policy",
        "benchmark", "evaluation", "dataset", "corpus", "annotation",
        "hallucination", "factual", "grounding", "retrieval", "reranking",
        "context window", "prompt", "few-shot", "zero-shot", "in-context",
        "chain of thought", "reasoning", "inference", "generation",
        "cross-encoder", "bi-encoder", "dense retrieval", "sparse retrieval",
        "BM25", "TF-IDF", "cosine similarity", "semantic similarity"
    ]
    
    text_lower = text.lower()
    found_concepts = []
    
    for concept in concept_keywords:
        if concept.lower() in text_lower:
            found_concepts.append(concept)
    
    # Also extract capitalized terms (likely proper nouns/methods)
    words = text.split()
    for word in words:
        if len(word) > 3 and word[0].isupper() and word not in found_concepts:
            clean_word = word.strip('.,;:()[]')
            if clean_word.isalpha() and len(clean_word) > 3:
                found_concepts.append(clean_word)
    
    return list(set(found_concepts))[:20]

class KnowledgeGraphBuilder:
    """Builds knowledge graphs from research papers using Neo4j and Graphiti"""
    
    def __init__(self):
        self.config = get_config()
        self.client = None
        self.graphiti = None
        self.driver = None
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialize connections"""
        if not all([GRAPHITI_AVAILABLE, NEO4J_AVAILABLE, self.config["NEO4J_URI"]]):
            logger.warning("Neo4j/Graphiti not available or not configured")
            return False
        
        try:
            self.client = AsyncOpenAI(api_key=self.config["OPENAI_API_KEY"])
            self.graphiti = Graphiti(
                self.config["NEO4J_URI"],
                self.config["NEO4J_USER"],
                self.config["NEO4J_PASSWORD"]
            )
            self.driver = AsyncGraphDatabase.driver(
                self.config["NEO4J_URI"],
                auth=(self.config["NEO4J_USER"], self.config["NEO4J_PASSWORD"])
            )
            self.initialized = True
            logger.info("Knowledge graph connections initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize graph connections: {e}")
            return False
    
    async def close(self):
        """Close connections"""
        if self.graphiti:
            await self.graphiti.close()
        if self.driver:
            await self.driver.close()
    
    async def build_graph_from_papers(self, papers: List[Dict]) -> Dict:
        """Build knowledge graph from search result papers"""
        if not self.initialized:
            if not await self.initialize():
                return {"error": "Could not initialize graph connections"}
        
        results = {
            "papers_processed": 0,
            "concepts_extracted": 0,
            "common_concepts": [],
            "unique_concepts": {}
        }
        
        try:
            # Clear existing data
            async with self.driver.session() as session:
                await session.run("MATCH (n) DETACH DELETE n")
            
            await self.graphiti.build_indices_and_constraints()
            
            # Process each paper
            papers_with_concepts = []
            
            for paper in papers:
                title = paper.get('title', 'Untitled')
                abstract = paper.get('abstract', '')
                arxiv_id = paper.get('arxiv_id', '')
                
                # Extract concepts
                concepts = await extract_concepts_llm(abstract, self.client)
                
                papers_with_concepts.append({
                    "title": title,
                    "abstract": abstract,
                    "arxiv_id": arxiv_id,
                    "concepts": concepts
                })
                
                # Add to graph
                await self.graphiti.add_episode(
                    name=title,
                    episode_body=abstract,
                    source=EpisodeType.text,
                    source_description=f"Research Paper: {arxiv_id}",
                    reference_time=datetime.now(timezone.utc),
                )
                
                results["papers_processed"] += 1
                results["concepts_extracted"] += len(concepts)
            
            # Calculate common and unique concepts
            if len(papers_with_concepts) >= 2:
                all_concept_sets = [set(p['concepts']) for p in papers_with_concepts]
                common = set.intersection(*all_concept_sets)
                results["common_concepts"] = sorted(common)
                
                for p in papers_with_concepts:
                    other_concepts = set.union(*[
                        all_concept_sets[j] 
                        for j in range(len(all_concept_sets)) 
                        if papers_with_concepts[j]['title'] != p['title']
                    ])
                    unique = set(p['concepts']) - other_concepts
                    results["unique_concepts"][p['title']] = sorted(unique)
            
            return results
            
        except Exception as e:
            logger.error(f"Error building graph: {e}")
            return {"error": str(e)}
    
    async def create_visualization(self, output_file: str = None) -> str:
        """Create interactive visualization"""
        if not self.initialized or not PYVIS_AVAILABLE:
            return ""
        
        if output_file is None:
            output_file = os.path.join(tempfile.gettempdir(), "kg_viz.html")
        
        try:
            query = """
            MATCH (n)-[r]->(m) 
            WHERE NOT n.name CONTAINS 'Comparison' AND NOT m.name CONTAINS 'Comparison'
            RETURN n, r, m LIMIT 100
            """
            
            async with self.driver.session() as session:
                result = await session.run(query)
                records = await result.data()
            
            if not records:
                return ""
            
            net = Network(
                height="600px",
                width="100%",
                bgcolor="#0f172a",
                font_color="white",
                directed=True
            )
            
            added_nodes = set()
            
            for record in records:
                source, target, rel = record['n'], record['m'], record['r']
                
                source_id = str(source.element_id) if hasattr(source, 'element_id') else str(hash(str(source)))
                target_id = str(target.element_id) if hasattr(target, 'element_id') else str(hash(str(target)))
                
                source_name = source.get('name', 'Unknown') if hasattr(source, 'get') else 'Unknown'
                target_name = target.get('name', 'Unknown') if hasattr(target, 'get') else 'Unknown'
                
                source_labels = list(source.labels) if hasattr(source, 'labels') else []
                target_labels = list(target.labels) if hasattr(target, 'labels') else []
                
                source_type = source_labels[0] if source_labels else 'Unknown'
                target_type = target_labels[0] if target_labels else 'Unknown'
                
                if source_id not in added_nodes:
                    color = "#3b82f6" if source_type == "Episodic" else "#ef4444"
                    size = 30 if source_type == "Episodic" else 20
                    net.add_node(source_id, label=str(source_name)[:40], color=color, size=size)
                    added_nodes.add(source_id)
                
                if target_id not in added_nodes:
                    color = "#3b82f6" if target_type == "Episodic" else "#ef4444"
                    size = 30 if target_type == "Episodic" else 20
                    net.add_node(target_id, label=str(target_name)[:40], color=color, size=size)
                    added_nodes.add(target_id)
                
                edge_type = rel.type if hasattr(rel, 'type') else 'RELATED'
                net.add_edge(source_id, target_id, title=edge_type, color="#64748b")
            
            net.save_graph(output_file)
            
            with open(output_file, 'r') as f:
                return f.read()
                
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            return ""

def create_graph_for_streamlit(papers: List[Dict], graph_type: str = "simple") -> str:
    """
    Create graph visualization HTML for Streamlit display
    
    Args:
        papers: List of paper dicts with 'title' and 'abstract'
        graph_type: "simple" (no Neo4j) or "full" (with Neo4j)
    
    Returns:
        HTML string for the graph
    """
    if graph_type == "simple" or not GRAPHITI_AVAILABLE:
        return create_simple_graph_html(papers, show_shared=True)
    else:
        # For full graph, would need async handling
        return create_simple_graph_html(papers, show_shared=True)


def get_graph_stats(papers: List[Dict]) -> Dict:
    """Get statistics about the graph that would be created"""
    concept_to_papers = {}
    
    for i, paper in enumerate(papers):
        title = paper.get('title', f'Paper {i+1}')
        abstract = paper.get('abstract', '')
        concepts = extract_simple_concepts(title + " " + abstract)
        
        for concept in concepts:
            if concept not in concept_to_papers:
                concept_to_papers[concept] = []
            concept_to_papers[concept].append(title)
    
    shared = [c for c, p in concept_to_papers.items() if len(p) > 1]
    unique = [c for c, p in concept_to_papers.items() if len(p) == 1]
    
    return {
        "total_papers": len(papers),
        "total_concepts": len(concept_to_papers),
        "shared_concepts": len(shared),
        "unique_concepts": len(unique),
        "shared_list": shared[:10],
        "concept_details": concept_to_papers
    }
