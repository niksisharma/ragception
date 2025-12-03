"""
Vector Store - Embeddings management with OpenAI and SQLite backend
Author: Amaan
"""

import numpy as np
import os
import json
from typing import List, Dict, Optional
import logging
from database_manager import DatabaseManager
from pdf_parser import PDFParser

# OpenAI import
try:
    from openai import OpenAI
except ImportError:
    print("OpenAI library not installed. Please run: pip install openai")
    raise

logger = logging.getLogger(__name__)

class VectorStore:
    """Manages embeddings using OpenAI API and SQLite for storage"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.parser = PDFParser()
        
        # Load configuration
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        
        # Get API key - try multiple sources
        api_key = self._get_api_key()
        
        if not api_key:
            raise ValueError(
                "OpenAI API key not found! Please either:\n"
                "1. Create .streamlit/secrets.toml with your key\n"
                "2. Set environment variable: export OPENAI_API_KEY='your-key'\n"
                "3. Pass it directly when initializing VectorStore"
            )
        
        self.client = OpenAI(api_key=api_key)
        self.embedding_model = self.config.get('embedding_model', 'text-embedding-3-small')
        
        # Embedding dimensions for different models
        self.embedding_dimensions = {
            'text-embedding-3-small': 1536,
            'text-embedding-3-large': 3072,
            'text-embedding-ada-002': 1536
        }
        
        logger.info(f"Using OpenAI {self.embedding_model} for embeddings")
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from multiple sources"""
        # Try Streamlit secrets first
        try:
            import streamlit as st
            if 'OPENAI_API_KEY' in st.secrets:
                return st.secrets['OPENAI_API_KEY']
        except (ImportError, FileNotFoundError):
            pass
        
        # Try loading from secrets.toml directly (for non-Streamlit usage)
        try:
            import toml
            secrets_path = '.streamlit/secrets.toml'
            if os.path.exists(secrets_path):
                secrets = toml.load(secrets_path)
                if 'OPENAI_API_KEY' in secrets:
                    return secrets['OPENAI_API_KEY']
        except (ImportError, FileNotFoundError):
            pass
        
        # Fall back to environment variable
        return os.environ.get('OPENAI_API_KEY')
    
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for text using OpenAI API"""
        try:
            # Truncate text if too long (OpenAI has token limits)
            # text-embedding-3-small supports up to 8191 tokens
            # Roughly 1 token = 4 characters, so limit to ~32000 chars to be safe
            if len(text) > 32000:
                text = text[:32000]
                logger.warning("Text truncated to fit token limit")
            
            # Create embedding using OpenAI API
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text,
                encoding_format="float"
            )
            
            # Extract embedding vector
            embedding = response.data[0].embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            # Return a zero vector as fallback
            dim = self.embedding_dimensions.get(self.embedding_model, 1536)
            return [0.0] * dim
    
    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for multiple texts efficiently (OpenAI supports batch)"""
        try:
            # Truncate texts if needed
            texts = [text[:32000] if len(text) > 32000 else text for text in texts]
            
            # OpenAI can handle multiple texts in one API call (more efficient)
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=texts,
                encoding_format="float"
            )
            
            # Extract all embeddings
            embeddings = [data.embedding for data in response.data]
            
            logger.info(f"Created {len(embeddings)} embeddings in batch")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error creating batch embeddings: {e}")
            # Return zero vectors as fallback
            dim = self.embedding_dimensions.get(self.embedding_model, 1536)
            return [[0.0] * dim for _ in texts]
    
    def process_paper(self, arxiv_id: str) -> bool:
        """Create and store embeddings for a paper"""
        try:
            # Get chunks from parser
            chunks = self.parser.prepare_chunks_for_embedding(arxiv_id)
            
            if not chunks:
                logger.warning(f"No chunks for {arxiv_id}")
                return False
            
            # Extract texts for batch processing
            chunk_texts = [chunk['text'] for chunk in chunks]
            
            # Create embeddings in batch (more efficient for OpenAI API)
            if len(chunk_texts) > 1:
                embeddings = self.create_embeddings_batch(chunk_texts)
            else:
                embeddings = [self.create_embedding(chunk_texts[0])]
            
            # Store each embedding
            for chunk, embedding in zip(chunks, embeddings):
                self.db.store_embedding(
                    paper_id=arxiv_id,
                    chunk_index=chunk['index'],
                    chunk_text=chunk['text'],
                    embedding=embedding,
                    chunk_type=chunk['type']
                )
            
            logger.info(f"Created {len(chunks)} embeddings for {arxiv_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {arxiv_id}: {e}")
            return False
    
    def process_all_papers(self, limit=50) -> Dict:
        """Process all papers that need embeddings"""
        # Get papers that have been parsed but not embedded
        self.db.cursor.execute("""
        SELECT arxiv_id FROM papers 
        WHERE processed = 1 AND embedding_created = 0
        LIMIT ?
        """, (limit,))
        
        papers = [row[0] for row in self.db.cursor.fetchall()]
        
        results = {
            'total': len(papers),
            'success': 0,
            'failed': [],
            'total_api_calls': 0,
            'estimated_cost': 0
        }
        
        for arxiv_id in papers:
            if self.process_paper(arxiv_id):
                results['success'] += 1
                # Rough estimate: ~5 chunks per paper
                results['total_api_calls'] += 5
            else:
                results['failed'].append(arxiv_id)
        
        # Calculate estimated cost (text-embedding-3-small: $0.02 per 1M tokens)
        # Rough estimate: 500 tokens per chunk, 5 chunks per paper
        total_tokens = results['success'] * 5 * 500
        results['estimated_cost'] = (total_tokens / 1_000_000) * 0.02
        
        logger.info(f"Processed {results['success']}/{results['total']} papers")
        logger.info(f"Estimated API cost: ${results['estimated_cost']:.4f}")
        
        return results
    
    def semantic_search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for similar papers using embeddings"""
        # Create query embedding using OpenAI
        query_embedding = np.array(self.create_embedding(query))
        
        # Get all embeddings from database
        self.db.cursor.execute("""
        SELECT paper_id, chunk_index, chunk_text, embedding
        FROM embeddings
        """)
        
        results = []
        for row in self.db.cursor.fetchall():
            if row[3]:  # If embedding exists
                import pickle
                chunk_embedding = np.array(pickle.loads(row[3]))
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, chunk_embedding)
                
                results.append({
                    'paper_id': row[0],
                    'chunk_index': row[1],
                    'chunk_text': row[2],
                    'similarity': similarity
                })
        
        # Sort by similarity and get top results
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Group by paper and get unique papers
        seen_papers = set()
        unique_results = []
        
        for result in results:
            if result['paper_id'] not in seen_papers:
                # Get paper details
                paper = self.db.get_paper(result['paper_id'])
                if paper:
                    unique_results.append({
                        'arxiv_id': paper['arxiv_id'],
                        'title': paper['title'],
                        'abstract': paper['abstract'][:200] + '...',
                        'similarity': result['similarity'],
                        'relevant_chunk': result['chunk_text'][:200] + '...'
                    })
                    seen_papers.add(result['paper_id'])
                    
                    if len(unique_results) >= n_results:
                        break
        
        return unique_results
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def get_embedding_stats(self) -> Dict:
        """Get statistics about embeddings and API usage"""
        stats = self.db.get_stats()
        
        # Add OpenAI-specific stats
        stats['embedding_model'] = self.embedding_model
        stats['embedding_dimensions'] = self.embedding_dimensions.get(self.embedding_model, 'unknown')
        
        # Estimate costs
        if stats['total_chunks'] > 0:
            # Rough estimate: 500 tokens per chunk
            estimated_tokens = stats['total_chunks'] * 500
            stats['estimated_tokens_used'] = estimated_tokens
            stats['estimated_cost_usd'] = (estimated_tokens / 1_000_000) * 0.02
        
        return stats