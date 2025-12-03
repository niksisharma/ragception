"""
ArXiv Bot - Fetches papers and stores in SQLite
Author: Amaan
"""

import arxiv
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArxivBot:
    """Fetches LLM/RAG papers from arXiv and stores in SQLite"""
    
    def __init__(self, config_path="config.json"):
        # Load configuration from JSON
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            pipeline_cfg = self.config.get("pipeline", {})
            self.default_days_back = pipeline_cfg.get("days_back", 7)
            self.default_max_results = pipeline_cfg.get("max_papers_per_run", 100)
        else:
            # Default config if file doesn't exist
            self.config = {
                "arxiv_categories": ["cs.CL", "cs.LG", "cs.AI"],
                "keywords": ["retrieval augmented generation", "RAG", "large language model", "LLM"],
                "pdf_directory": "./data/pdfs",
                "database_path": "./data/ragbot.db"
            }
        
        # Import here to avoid circular imports
        from database_manager import DatabaseManager
        
        # Initialize database
        self.db = DatabaseManager()
        
        # Create PDF directory
        self.pdf_dir = self.config.get('pdf_directory', './data/pdfs')
        os.makedirs(self.pdf_dir, exist_ok=True)
        
        logger.info("ArxivBot initialized with SQLite backend")
    
    def fetch_recent_papers(self, days_back=None, max_results=None) -> Dict:
        """Main method to fetch recent papers"""
        if days_back is None:
            days_back = self.config_days_back
        if max_results is None:
            max_results = self.config_max_results
        logger.info(f"Fetching papers from last {days_back} days...")
        
        start_time = datetime.now()
        results = {
            'papers_found': 0,
            'papers_stored': 0,
            'pdfs_downloaded': 0,
            'errors': []
        }
        
        try:
            # Build search query
            categories = self.config.get('arxiv_categories', ['cs.CL'])
            category_query = " OR ".join([f"cat:{cat}" for cat in categories])
            
            # Search arXiv
            search = arxiv.Search(
                query=category_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            for paper in search.results():
                # Check date
                if paper.published.replace(tzinfo=None) < cutoff_date:
                    break
                
                # Check if relevant
                if self._is_relevant(paper):
                    results['papers_found'] += 1
                    
                    # Extract paper data
                    paper_data = self._extract_paper_data(paper)
                    
                    # Store in database
                    if self.db.insert_paper(paper_data):
                        results['papers_stored'] += 1
                        
                        # Download PDF
                        if self._download_pdf(paper):
                            results['pdfs_downloaded'] += 1
                            
                            # Mark PDF as downloaded
                            self.db.cursor.execute(
                                "UPDATE papers SET pdf_downloaded = 1 WHERE arxiv_id = ?",
                                (paper_data['arxiv_id'],)
                            )
                            self.db.conn.commit()
                    
                    # Be polite to arXiv
                    time.sleep(0.5)
            
            # Log successful run
            self.db.log_pipeline_run(
                start_time, datetime.now(),
                results['papers_found'],
                results['papers_stored'],
                'SUCCESS'
            )
            
        except Exception as e:
            logger.error(f"Error fetching papers: {e}")
            results['errors'].append(str(e))
            
            # Log failed run
            self.db.log_pipeline_run(
                start_time, datetime.now(),
                results['papers_found'],
                results['papers_stored'],
                'FAILED', str(e)
            )
        
        logger.info(f"Fetch complete: {results['papers_stored']} papers stored")
        return results
    
    def _is_relevant(self, paper) -> bool:
        """Check if paper matches our keywords"""
        text = (paper.title + " " + paper.summary).lower()
        
        keywords = self.config.get('keywords', ['rag', 'llm'])
        for keyword in keywords:
            if keyword.lower() in text:
                return True
        return False
    
    def _extract_paper_data(self, paper) -> Dict:
        """Extract paper metadata"""
        return {
            'arxiv_id': paper.entry_id.split('/')[-1],
            'title': paper.title,
            'abstract': paper.summary,
            'authors': [author.name for author in paper.authors],
            'published_date': paper.published.isoformat(),
            'categories': paper.categories,
            'pdf_url': paper.pdf_url
        }
    
    def _download_pdf(self, paper) -> bool:
        """Download PDF file"""
        try:
            arxiv_id = paper.entry_id.split('/')[-1]
            pdf_path = os.path.join(self.pdf_dir, f"{arxiv_id}.pdf")
            
            # Skip if exists
            if os.path.exists(pdf_path):
                logger.info(f"PDF already exists: {arxiv_id}")
                return True
            
            logger.info(f"Downloading PDF: {arxiv_id}")
            paper.download_pdf(dirpath=self.pdf_dir, filename=f"{arxiv_id}.pdf")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading PDF: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """Get fetching statistics"""
        stats = self.db.get_stats()
        
        # Add PDF count
        if os.path.exists(self.pdf_dir):
            pdf_files = len([f for f in os.listdir(self.pdf_dir) if f.endswith('.pdf')])
            stats['pdf_files'] = pdf_files
        else:
            stats['pdf_files'] = 0
        
        return stats

# Test the import when running directly
if __name__ == "__main__":
    print("Testing ArxivBot...")
    bot = ArxivBot()
    print("ArxivBot created successfully!")
    stats = bot.get_statistics()
    print(f"Current stats: {stats}")