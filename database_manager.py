"""
Database Manager - Core SQLite implementation
Author: Amaan
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Core database manager for all RAG bot data"""
    
    def __init__(self, db_path="./data/ragbot.db"):
        self.db_path = db_path
        
        # Create the directory if it doesn't exist
        db_dir = os.path.dirname(db_path)
        if db_dir:  # Only create if there's actually a directory path
            os.makedirs(db_dir, exist_ok=True)
            if not os.path.exists(db_path):
                logger.info(f"Created database directory: {db_dir}")
        
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        self._create_tables()
        logger.info(f"Database initialized at {db_path}")
    
    def _create_tables(self):
        """Create all necessary tables"""
        
        # Main papers table
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            arxiv_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            abstract TEXT,
            authors TEXT,  -- JSON array stored as text
            published_date DATETIME,
            categories TEXT,  -- JSON array
            pdf_url TEXT,
            pdf_downloaded BOOLEAN DEFAULT 0,
            full_text TEXT,
            sections TEXT,  -- JSON object
            processed BOOLEAN DEFAULT 0,
            embedding_created BOOLEAN DEFAULT 0,
            summary_generated BOOLEAN DEFAULT 0,
            fetched_date DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Embeddings table
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id TEXT,
            chunk_index INTEGER,
            chunk_text TEXT,
            chunk_type TEXT,
            embedding BLOB,
            created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (paper_id) REFERENCES papers(arxiv_id)
        )
        """)
        
        # Pipeline runs tracking
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time DATETIME,
            end_time DATETIME,
            papers_fetched INTEGER,
            papers_processed INTEGER,
            status TEXT,
            error_message TEXT
        )
        """)
        
        # Create indices for performance
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_papers_processed ON papers(processed)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_paper ON embeddings(paper_id)")
        
        self.conn.commit()
    
    def insert_paper(self, paper_data: Dict) -> bool:
        """Insert or update a paper"""
        try:
            # Convert lists to JSON strings
            authors = json.dumps(paper_data.get('authors', []))
            categories = json.dumps(paper_data.get('categories', []))
            
            self.cursor.execute("""
            INSERT OR REPLACE INTO papers (
                arxiv_id, title, abstract, authors, published_date,
                categories, pdf_url
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                paper_data['arxiv_id'],
                paper_data['title'],
                paper_data.get('abstract'),
                authors,
                paper_data.get('published_date'),
                categories,
                paper_data.get('pdf_url')
            ))
            
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error inserting paper: {e}")
            self.conn.rollback()
            return False
    
    def update_paper_content(self, arxiv_id: str, full_text: str, sections: Dict) -> bool:
        """Update paper with parsed content"""
        try:
            sections_json = json.dumps(sections)
            
            self.cursor.execute("""
            UPDATE papers 
            SET full_text = ?, sections = ?, processed = 1
            WHERE arxiv_id = ?
            """, (full_text, sections_json, arxiv_id))
            
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error updating paper: {e}")
            return False
    
    def get_paper(self, arxiv_id: str) -> Optional[Dict]:
        """Get complete paper data"""
        self.cursor.execute("SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,))
        row = self.cursor.fetchone()
        
        if not row:
            return None
        
        paper = dict(row)
        # Parse JSON fields
        paper['authors'] = json.loads(paper['authors']) if paper['authors'] else []
        paper['categories'] = json.loads(paper['categories']) if paper['categories'] else []
        paper['sections'] = json.loads(paper['sections']) if paper['sections'] else {}
        
        return paper
    
    def get_unprocessed_papers(self, limit: int = 50) -> List[Dict]:
        """Get papers that need processing"""
        self.cursor.execute("""
        SELECT * FROM papers
        WHERE processed = 0 AND pdf_downloaded = 1
        ORDER BY fetched_date DESC
        LIMIT ?
        """, (limit,))
        
        papers = []
        for row in self.cursor.fetchall():
            paper = dict(row)
            paper['authors'] = json.loads(paper['authors']) if paper['authors'] else []
            paper['categories'] = json.loads(paper['categories']) if paper['categories'] else []
            papers.append(paper)
        
        return papers
    
    def get_papers_for_summarization(self, limit: int = 20) -> List[Dict]:
        """Get papers ready for Nikita's summarization"""
        self.cursor.execute("""
        SELECT * FROM papers
        WHERE processed = 1 AND summary_generated = 0 AND full_text IS NOT NULL
        LIMIT ?
        """, (limit,))
        
        papers = []
        for row in self.cursor.fetchall():
            paper = dict(row)
            paper['authors'] = json.loads(paper['authors']) if paper['authors'] else []
            paper['categories'] = json.loads(paper['categories']) if paper['categories'] else []
            paper['sections'] = json.loads(paper['sections']) if paper['sections'] else {}
            papers.append(paper)
        
        return papers
    
    def store_embedding(self, paper_id: str, chunk_index: int, 
                       chunk_text: str, embedding: List[float], chunk_type: str) -> bool:
        """Store embedding for a chunk"""
        try:
            import pickle
            embedding_blob = pickle.dumps(embedding)
            
            self.cursor.execute("""
            INSERT INTO embeddings (paper_id, chunk_index, chunk_text, embedding, chunk_type)
            VALUES (?, ?, ?, ?, ?)
            """, (paper_id, chunk_index, chunk_text, embedding_blob, chunk_type))
            
            # Mark paper as having embeddings
            self.cursor.execute("""
            UPDATE papers SET embedding_created = 1 WHERE arxiv_id = ?
            """, (paper_id,))
            
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error storing embedding: {e}")
            return False
    
    def search_papers(self, query: str, limit: int = 10) -> List[Dict]:
        """Simple text search in papers"""
        self.cursor.execute("""
        SELECT arxiv_id, title, abstract, published_date
        FROM papers
        WHERE (title LIKE ? OR abstract LIKE ?) AND processed = 1
        ORDER BY published_date DESC
        LIMIT ?
        """, (f"%{query}%", f"%{query}%", limit))
        
        return [dict(row) for row in self.cursor.fetchall()]
    
    def get_papers_for_vector_search(self, limit: int = 100) -> List[Dict]:
        """Get papers with embeddings for vector search"""
        self.cursor.execute("""
        SELECT DISTINCT p.arxiv_id, p.title, p.abstract, p.authors
        FROM papers p
        JOIN embeddings e ON p.arxiv_id = e.paper_id
        WHERE p.embedding_created = 1
        LIMIT ?
        """, (limit,))
        
        papers = []
        for row in self.cursor.fetchall():
            paper = dict(row)
            paper['authors'] = json.loads(paper['authors']) if paper['authors'] else []
            papers.append(paper)
        
        return papers
    
    def get_embeddings_for_paper(self, paper_id: str) -> List[Dict]:
        """Get all embeddings for a paper"""
        self.cursor.execute("""
        SELECT chunk_index, chunk_text, embedding, chunk_type
        FROM embeddings
        WHERE paper_id = ?
        ORDER BY chunk_index
        """, (paper_id,))
        
        import pickle
        embeddings = []
        for row in self.cursor.fetchall():
            embeddings.append({
                'chunk_index': row[0],
                'chunk_text': row[1],
                'embedding': pickle.loads(row[2]) if row[2] else None,
                'chunk_type': row[3]
            })
        
        return embeddings
    
    def log_pipeline_run(self, start_time, end_time, papers_fetched, papers_processed, status, error=None):
        """Log pipeline execution"""
        self.cursor.execute("""
        INSERT INTO pipeline_runs (start_time, end_time, papers_fetched, papers_processed, status, error_message)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (start_time, end_time, papers_fetched, papers_processed, status, error))
        self.conn.commit()
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        stats = {}
        
        self.cursor.execute("SELECT COUNT(*) FROM papers")
        stats['total_papers'] = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT COUNT(*) FROM papers WHERE processed = 1")
        stats['processed_papers'] = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT COUNT(*) FROM papers WHERE embedding_created = 1")
        stats['papers_with_embeddings'] = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT COUNT(*) FROM embeddings")
        stats['total_chunks'] = self.cursor.fetchone()[0]
        
        return stats
    
    def close(self):
        """Close database connection"""
        self.conn.close()