"""
Pipeline Orchestrator - Coordinates all components
Author: Amaan
"""

import json
import os
from datetime import datetime
import logging
import schedule
import time
from typing import Dict

# Import components with error handling
try:
    from database_manager import DatabaseManager
    from arxiv_bot import ArxivBot
    from pdf_parser import PDFParser
    from vector_store import VectorStore
except ImportError as e:
    print(f"Error importing components: {e}")
    print("Make sure all component files are in the same directory.")
    import sys
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    """Coordinates the entire data pipeline"""
    
    def __init__(self):
        logger.info("Initializing Pipeline Orchestrator...")
        
        # Load configuration
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        
        # Initialize components
        self.db = DatabaseManager()
        self.arxiv_bot = ArxivBot()
        self.pdf_parser = PDFParser()
        self.vector_store = VectorStore()
        
        logger.info("Orchestrator ready!")
    
    def run_complete_pipeline(self) -> Dict:
        """Run the entire pipeline end-to-end"""
        logger.info("="*60)
        logger.info("STARTING COMPLETE PIPELINE")
        logger.info("="*60)
        
        start_time = datetime.now()
        results = {
            'start_time': start_time.isoformat(),
            'steps': {}
        }
        
        try:
            # Step 1: Fetch new papers
            logger.info("Step 1: Fetching papers from arXiv...")
            fetch_results = self.arxiv_bot.fetch_recent_papers(
                days_back=self.config.get('days_back', 7),
                max_results=self.config.get('max_papers_per_run', 50)
            )
            results['steps']['fetch'] = fetch_results
            logger.info(f"✓ Fetched {fetch_results['papers_stored']} papers")
            
            # Step 2: Parse PDFs
            logger.info("Step 2: Parsing PDF documents...")
            parse_results = self.pdf_parser.parse_all_unprocessed()
            results['steps']['parse'] = parse_results
            logger.info(f"✓ Parsed {parse_results['success']} papers")
            
            # Step 3: Create embeddings
            logger.info("Step 3: Creating embeddings...")
            embedding_results = self.vector_store.process_all_papers()
            results['steps']['embeddings'] = embedding_results
            logger.info(f"✓ Created embeddings for {embedding_results['success']} papers")
            logger.info(f"  Estimated OpenAI API cost: ${embedding_results['estimated_cost']:.4f}")
            
            results['status'] = 'SUCCESS'
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            results['status'] = 'FAILED'
            results['error'] = str(e)
        
        results['end_time'] = datetime.now().isoformat()
        
        # Save pipeline results
        self._save_results(results)
        
        # Log to database
        end_time = datetime.now()
        papers_fetched = results['steps'].get('fetch', {}).get('papers_found', 0)
        papers_processed = results['steps'].get('embeddings', {}).get('success', 0)
        self.db.log_pipeline_run(
            start_time, end_time, papers_fetched, papers_processed,
            results['status'], results.get('error')
        )
        
        logger.info("="*60)
        logger.info(f"PIPELINE COMPLETE - Status: {results['status']}")
        logger.info("="*60)
        
        return results
    
    def search_papers(self, query: str, n_results: int = 5) -> Dict:
        """Search papers using vector similarity"""
        results = self.vector_store.semantic_search(query, n_results)
        
        return {
            'query': query,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
    
    def schedule_weekly_run(self):
        """Schedule pipeline to run weekly"""
        # Schedule for every Sunday at 2 AM
        schedule.every().sunday.at("02:00").do(self.run_complete_pipeline)
        
        logger.info("Pipeline scheduled for weekly runs (Sundays at 2 AM)")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def get_status(self) -> Dict:
        """Get current pipeline status"""
        stats = self.db.get_stats()
        
        # Get embedding stats
        embedding_stats = self.vector_store.get_embedding_stats()
        stats.update(embedding_stats)
        
        # Get last run info
        self.db.cursor.execute("""
        SELECT start_time, end_time, status, papers_fetched, papers_processed
        FROM pipeline_runs
        ORDER BY id DESC
        LIMIT 1
        """)
        
        last_run = self.db.cursor.fetchone()
        
        if last_run:
            stats['last_run'] = {
                'start_time': last_run[0],
                'end_time': last_run[1],
                'status': last_run[2],
                'papers_fetched': last_run[3],
                'papers_processed': last_run[4]
            }
        else:
            stats['last_run'] = None
        
        return stats
    
    def get_recent_papers(self, limit: int = 20) -> list:
        """Get recently added papers"""
        self.db.cursor.execute("""
        SELECT arxiv_id, title, abstract, published_date, pdf_downloaded, processed, embedding_created
        FROM papers
        ORDER BY fetched_date DESC
        LIMIT ?
        """, (limit,))
        
        papers = []
        for row in self.db.cursor.fetchall():
            papers.append({
                'arxiv_id': row[0],
                'title': row[1],
                'abstract': row[2][:300] + '...' if row[2] and len(row[2]) > 300 else row[2],
                'published_date': row[3],
                'pdf_downloaded': bool(row[4]),
                'processed': bool(row[5]),
                'has_embeddings': bool(row[6])
            })
        
        return papers
    
    def _save_results(self, results: Dict):
        """Save pipeline results to file"""
        os.makedirs("./data/pipeline_runs", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"./data/pipeline_runs/run_{timestamp}.json"
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)


def main():
    """Main entry point"""
    print("""
    ╔═══════════════════════════════════════╗
    ║   RAG Research Bot Pipeline Manager   ║
    ╚═══════════════════════════════════════╝
    """)
    
    orchestrator = PipelineOrchestrator()
    
    while True:
        print("\n" + "="*40)
        print("What would you like to do?")
        print("1. Run complete pipeline")
        print("2. Check pipeline status")
        print("3. Search papers")
        print("4. View recent papers")
        print("5. Schedule weekly runs")
        print("6. Exit")
        print("="*40)
        
        try:
            choice = input("\nEnter choice (1-6): ")
            
            if choice == "1":
                print("\n🚀 Running complete pipeline...")
                results = orchestrator.run_complete_pipeline()
                print(f"\n✅ Pipeline completed with status: {results['status']}")
                
                if results['status'] == 'SUCCESS':
                    fetch = results['steps'].get('fetch', {})
                    parse = results['steps'].get('parse', {})
                    embed = results['steps'].get('embeddings', {})
                    
                    print(f"\nSummary:")
                    print(f"  📥 Papers fetched: {fetch.get('papers_stored', 0)}")
                    print(f"  📄 Papers parsed: {parse.get('success', 0)}")
                    print(f"  🔮 Embeddings created: {embed.get('success', 0)}")
                    print(f"  💰 API cost: ${embed.get('estimated_cost', 0):.4f}")
                
            elif choice == "2":
                status = orchestrator.get_status()
                print("\n📊 Pipeline Status:")
                print(f"  Total papers: {status['total_papers']}")
                print(f"  Processed papers: {status['processed_papers']}")
                print(f"  Papers with embeddings: {status['papers_with_embeddings']}")
                print(f"  Total embedding chunks: {status['total_chunks']}")
                print(f"  Embedding model: {status.get('embedding_model', 'N/A')}")
                print(f"  Total API cost estimate: ${status.get('estimated_cost_usd', 0):.4f}")
                
                if status['last_run']:
                    print(f"\n  Last run: {status['last_run']['start_time']}")
                    print(f"  Status: {status['last_run']['status']}")
                    print(f"  Papers processed: {status['last_run']['papers_processed']}")
                
            elif choice == "3":
                query = input("\n🔍 Enter search query: ")
                results = orchestrator.search_papers(query)
                
                if results['results']:
                    print(f"\nFound {len(results['results'])} relevant papers:")
                    for i, paper in enumerate(results['results'], 1):
                        print(f"\n{i}. {paper['title']}")
                        print(f"   ArXiv ID: {paper['arxiv_id']}")
                        print(f"   Similarity: {paper['similarity']:.3f}")
                        print(f"   Abstract: {paper['abstract']}")
                else:
                    print("\nNo relevant papers found.")
            
            elif choice == "4":
                papers = orchestrator.get_recent_papers(10)
                print(f"\n📚 Recent Papers ({len(papers)} shown):")
                for i, paper in enumerate(papers, 1):
                    status_icons = []
                    if paper['pdf_downloaded']: status_icons.append("📥")
                    if paper['processed']: status_icons.append("📄")
                    if paper['has_embeddings']: status_icons.append("🔮")
                    
                    print(f"\n{i}. {paper['title'][:80]}...")
                    print(f"   ArXiv: {paper['arxiv_id']} | Status: {' '.join(status_icons)}")
                    print(f"   Published: {paper['published_date'][:10] if paper['published_date'] else 'N/A'}")
            
            elif choice == "5":
                print("\n⏰ Starting scheduler... (Press Ctrl+C to stop)")
                orchestrator.schedule_weekly_run()
            
            elif choice == "6":
                print("\n👋 Goodbye!")
                break
            
            else:
                print("\n❌ Invalid choice. Please try again.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or check the logs for details.")

if __name__ == "__main__":
    main()