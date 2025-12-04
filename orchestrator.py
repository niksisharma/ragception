# """Pipeline Orchestrator - Coordinates all components"""

# import json
# import os
# from datetime import datetime
# import logging
# import schedule
# import time
# from typing import Dict

# try:
#     from database_manager import DatabaseManager
#     from arxiv_bot import ArxivBot
#     from pdf_parser import PDFParser
#     from vector_store import VectorStore
#     from paper_summarizer import PaperSummarizer
# except ImportError as e:
#     print(f"Error importing components: {e}")
#     print("Make sure all component files are in the same directory.")
#     import sys
#     sys.exit(1)

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# class PipelineOrchestrator:
#     """Coordinates the entire data pipeline"""
    
#     def __init__(self):
#         logger.info("Initializing Pipeline Orchestrator...")

#         with open('config.json', 'r') as f:
#             self.config = json.load(f)

#         self.db = DatabaseManager()
#         self.arxiv_bot = ArxivBot()
#         self.pdf_parser = PDFParser()
#         self.vector_store = VectorStore()

#         try:
#             self.summarizer = PaperSummarizer()
#             self.summarizer_enabled = True
#             logger.info("Paper Summarizer initialized successfully")
#         except Exception as e:
#             logger.warning(f"Paper Summarizer not available: {e}")
#             self.summarizer = None
#             self.summarizer_enabled = False

#         logger.info("Orchestrator ready!")

#     def run_complete_pipeline(self) -> Dict:
#         """Run the entire pipeline end-to-end"""
#         logger.info("="*60)
#         logger.info("STARTING COMPLETE PIPELINE")
#         logger.info("="*60)

#         start_time = datetime.now()
#         results = {
#             'start_time': start_time.isoformat(),
#             'steps': {}
#         }

#         try:
#             logger.info("Step 1: Fetching papers from arXiv...")
#             fetch_results = self.arxiv_bot.fetch_recent_papers(
#                 days_back=self.config.get('days_back', 7),
#                 max_results=self.config.get('max_papers_per_run', 50)
#             )
#             results['steps']['fetch'] = fetch_results
#             logger.info(f"✓ Fetched {fetch_results['papers_stored']} papers")

#             logger.info("Step 2: Parsing PDF documents...")
#             parse_results = self.pdf_parser.parse_all_unprocessed()
#             results['steps']['parse'] = parse_results
#             logger.info(f"✓ Parsed {parse_results['success']} papers")

#             logger.info("Step 3: Creating embeddings...")
#             embedding_results = self.vector_store.process_all_papers()
#             results['steps']['embeddings'] = embedding_results
#             logger.info(f"✓ Created embeddings for {embedding_results['success']} papers")
#             logger.info(f"  Estimated OpenAI API cost: ${embedding_results['estimated_cost']:.4f}")

#             if self.summarizer_enabled and self.summarizer:
#                 logger.info("Step 4: Generating structured summaries...")
#                 summary_results = self.summarizer.generate_summaries_batch(
#                     limit=self.config.get('max_papers_per_run', 50)
#                 )
#                 results['steps']['summaries'] = summary_results
#                 logger.info(f"✓ Generated summaries for {summary_results['success']} papers")
#                 logger.info(f"  Estimated OpenAI API cost: ${summary_results['estimated_cost']:.4f}")
#             else:
#                 logger.info("Step 4: Skipping summary generation (disabled or unavailable)")
#                 results['steps']['summaries'] = {'skipped': True}

#             results['status'] = 'SUCCESS'

#         except Exception as e:
#             logger.error(f"Pipeline failed: {e}")
#             results['status'] = 'FAILED'
#             results['error'] = str(e)

#         results['end_time'] = datetime.now().isoformat()
#         self._save_results(results)

#         end_time = datetime.now()
#         papers_fetched = results['steps'].get('fetch', {}).get('papers_found', 0)
#         papers_processed = results['steps'].get('embeddings', {}).get('success', 0)
#         self.db.log_pipeline_run(
#             start_time, end_time, papers_fetched, papers_processed,
#             results['status'], results.get('error')
#         )

#         logger.info("="*60)
#         logger.info(f"PIPELINE COMPLETE - Status: {results['status']}")
#         logger.info("="*60)

#         return results
    
#     def search_papers(self, query: str, n_results: int = 5) -> Dict:
#         """Search papers using vector similarity"""
#         results = self.vector_store.semantic_search(query, n_results)
        
#         return {
#             'query': query,
#             'results': results,
#             'timestamp': datetime.now().isoformat()
#         }
    
#     def schedule_weekly_run(self):
#         """Schedule pipeline to run weekly"""
#         # Schedule for every Sunday at 2 AM
#         schedule.every().sunday.at("02:00").do(self.run_complete_pipeline)
        
#         logger.info("Pipeline scheduled for weekly runs (Sundays at 2 AM)")
        
#         while True:
#             schedule.run_pending()
#             time.sleep(60)  # Check every minute
    
#     def get_status(self) -> Dict:
#         """Get current pipeline status"""
#         stats = self.db.get_stats()

#         # Get embedding stats
#         embedding_stats = self.vector_store.get_embedding_stats()
#         stats.update(embedding_stats)

#         # Get summary stats (if summarizer available)
#         if self.summarizer_enabled and self.summarizer:
#             summary_stats = self.summarizer.get_summary_stats()
#             stats.update(summary_stats)

#         # Get last run info
#         stats['last_run'] = self.db.get_last_pipeline_run()
        
#         return stats
    
#     def get_recent_papers(self, limit: int = 20) -> list:
#         """Get recently added papers"""
#         self.db.cursor.execute("""
#         SELECT arxiv_id, title, abstract, published_date, pdf_downloaded, processed, embedding_created, summary_generated
#         FROM papers
#         ORDER BY fetched_date DESC
#         LIMIT ?
#         """, (limit,))

#         papers = []
#         for row in self.db.cursor.fetchall():
#             papers.append({
#                 'arxiv_id': row[0],
#                 'title': row[1],
#                 'abstract': row[2][:300] + '...' if row[2] and len(row[2]) > 300 else row[2],
#                 'published_date': row[3],
#                 'pdf_downloaded': bool(row[4]),
#                 'processed': bool(row[5]),
#                 'has_embeddings': bool(row[6]),
#                 'has_summary': bool(row[7])
#             })
        
#         return papers
    
#     def _save_results(self, results: Dict):
#         """Save pipeline results to file"""
#         os.makedirs("./data/pipeline_runs", exist_ok=True)
        
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filepath = f"./data/pipeline_runs/run_{timestamp}.json"
        
#         with open(filepath, 'w') as f:
#             json.dump(results, f, indent=2)


# def main():
#     """Main entry point"""
#     print("""
#     ╔═══════════════════════════════════════╗
#     ║   RAG Research Bot Pipeline Manager   ║
#     ╚═══════════════════════════════════════╝
#     """)
    
#     orchestrator = PipelineOrchestrator()
    
#     while True:
#         print("\n" + "="*40)
#         print("What would you like to do?")
#         print("1. Run complete pipeline")
#         print("2. Check pipeline status")
#         print("3. Search papers")
#         print("4. View recent papers")
#         print("5. Schedule weekly runs")
#         print("6. Exit")
#         print("="*40)
        
#         try:
#             choice = input("\nEnter choice (1-6): ")
            
#             if choice == "1":
#                 print("\n🚀 Running complete pipeline...")
#                 results = orchestrator.run_complete_pipeline()
#                 print(f"\n✅ Pipeline completed with status: {results['status']}")
                
#                 if results['status'] == 'SUCCESS':
#                     fetch = results['steps'].get('fetch', {})
#                     parse = results['steps'].get('parse', {})
#                     embed = results['steps'].get('embeddings', {})
                    
#                     print(f"\nSummary:")
#                     print(f"  📥 Papers fetched: {fetch.get('papers_stored', 0)}")
#                     print(f"  📄 Papers parsed: {parse.get('success', 0)}")
#                     print(f"  🔮 Embeddings created: {embed.get('success', 0)}")
#                     print(f"  💰 API cost: ${embed.get('estimated_cost', 0):.4f}")
                
#             elif choice == "2":
#                 status = orchestrator.get_status()
#                 print("\n📊 Pipeline Status:")
#                 print(f"  Total papers: {status['total_papers']}")
#                 print(f"  Processed papers: {status['processed_papers']}")
#                 print(f"  Papers with embeddings: {status['papers_with_embeddings']}")
#                 print(f"  Total embedding chunks: {status['total_chunks']}")
#                 print(f"  Embedding model: {status.get('embedding_model', 'N/A')}")
#                 print(f"  Total API cost estimate: ${status.get('estimated_cost_usd', 0):.4f}")
                
#                 if status['last_run']:
#                     print(f"\n  Last run: {status['last_run']['start_time']}")
#                     print(f"  Status: {status['last_run']['status']}")
#                     print(f"  Papers processed: {status['last_run']['papers_processed']}")
                
#             elif choice == "3":
#                 query = input("\n🔍 Enter search query: ")
#                 results = orchestrator.search_papers(query)
                
#                 if results['results']:
#                     print(f"\nFound {len(results['results'])} relevant papers:")
#                     for i, paper in enumerate(results['results'], 1):
#                         print(f"\n{i}. {paper['title']}")
#                         print(f"   ArXiv ID: {paper['arxiv_id']}")
#                         print(f"   Similarity: {paper['similarity']:.3f}")
#                         print(f"   Abstract: {paper['abstract']}")
#                 else:
#                     print("\nNo relevant papers found.")
            
#             elif choice == "4":
#                 papers = orchestrator.get_recent_papers(10)
#                 print(f"\n📚 Recent Papers ({len(papers)} shown):")
#                 for i, paper in enumerate(papers, 1):
#                     status_icons = []
#                     if paper['pdf_downloaded']: status_icons.append("📥")
#                     if paper['processed']: status_icons.append("📄")
#                     if paper['has_embeddings']: status_icons.append("🔮")
                    
#                     print(f"\n{i}. {paper['title'][:80]}...")
#                     print(f"   ArXiv: {paper['arxiv_id']} | Status: {' '.join(status_icons)}")
#                     print(f"   Published: {paper['published_date'][:10] if paper['published_date'] else 'N/A'}")
            
#             elif choice == "5":
#                 print("\n⏰ Starting scheduler... (Press Ctrl+C to stop)")
#                 orchestrator.schedule_weekly_run()
            
#             elif choice == "6":
#                 print("\n👋 Goodbye!")
#                 break
            
#             else:
#                 print("\n❌ Invalid choice. Please try again.")
                
#         except KeyboardInterrupt:
#             print("\n\n👋 Interrupted by user. Goodbye!")
#             break
#         except Exception as e:
#             print(f"\n❌ Error: {e}")
#             print("Please try again or check the logs for details.")

# if __name__ == "__main__":
#     main()

"""Pipeline Orchestrator - Coordinates all components"""

import json
import os
from datetime import datetime
import logging
import schedule
import time
from typing import Dict

try:
    from database_manager import DatabaseManager
    from arxiv_bot import ArxivBot
    from pdf_parser import PDFParser
    from vector_store import VectorStore
    from paper_summarizer import PaperSummarizer
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

        with open('config.json', 'r') as f:
            self.config = json.load(f)

        self.db = DatabaseManager()
        self.arxiv_bot = ArxivBot()
        self.pdf_parser = PDFParser()
        self.vector_store = VectorStore()

        try:
            self.summarizer = PaperSummarizer()
            self.summarizer_enabled = True
            logger.info("Paper Summarizer initialized successfully")
        except Exception as e:
            logger.warning(f"Paper Summarizer not available: {e}")
            self.summarizer = None
            self.summarizer_enabled = False

        logger.info("Orchestrator ready!")

    def run_complete_pipeline(self, max_papers: int = None) -> Dict:
        """Run the entire pipeline end-to-end
        
        Args:
            max_papers: Override for max papers to process (uses config if None)
        """
        logger.info("="*60)
        logger.info("STARTING COMPLETE PIPELINE")
        logger.info("="*60)

        start_time = datetime.now()
        results = {
            'start_time': start_time.isoformat(),
            'steps': {}
        }

        # Get limits from config or override
        days_back = self.config.get('days_back', 7)
        paper_limit = max_papers or self.config.get('max_papers_per_run', 50)
        
        logger.info(f"Configuration: days_back={days_back}, max_papers={paper_limit}")

        try:
            # Step 1: Fetch papers
            logger.info("Step 1: Fetching papers from arXiv...")
            fetch_results = self.arxiv_bot.fetch_recent_papers(
                days_back=days_back,
                max_results=paper_limit
            )
            results['steps']['fetch'] = fetch_results
            logger.info(f"✓ Fetched {fetch_results['papers_stored']} papers")

            # Step 2: Parse PDFs - NOW USES CONFIG LIMIT
            logger.info("Step 2: Parsing PDF documents...")
            parse_results = self.pdf_parser.parse_all_unprocessed(limit=paper_limit)
            results['steps']['parse'] = parse_results
            logger.info(f"✓ Parsed {parse_results['success']} papers")

            # Step 3: Create embeddings - NOW USES CONFIG LIMIT
            logger.info("Step 3: Creating embeddings...")
            embedding_results = self.vector_store.process_all_papers(limit=paper_limit)
            results['steps']['embeddings'] = embedding_results
            logger.info(f"✓ Created embeddings for {embedding_results['success']} papers")
            logger.info(f"  Estimated OpenAI API cost: ${embedding_results['estimated_cost']:.4f}")

            # Step 4: Generate summaries - NOW USES CONFIG LIMIT
            if self.summarizer_enabled and self.summarizer:
                logger.info("Step 4: Generating structured summaries...")
                summary_results = self.summarizer.generate_summaries_batch(limit=paper_limit)
                results['steps']['summaries'] = summary_results
                logger.info(f"✓ Generated summaries for {summary_results['success']} papers")
                logger.info(f"  Estimated OpenAI API cost: ${summary_results['estimated_cost']:.4f}")
            else:
                logger.info("Step 4: Skipping summary generation (disabled or unavailable)")
                results['steps']['summaries'] = {'skipped': True}

            results['status'] = 'SUCCESS'
            
            # Log remaining backlog
            self._log_backlog_status()

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            results['status'] = 'FAILED'
            results['error'] = str(e)

        results['end_time'] = datetime.now().isoformat()
        self._save_results(results)

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
    
    def _log_backlog_status(self):
        """Log the current backlog of unprocessed papers"""
        # Count papers needing parsing
        self.db.cursor.execute("""
            SELECT COUNT(*) FROM papers 
            WHERE processed = 0 AND pdf_downloaded = 1
        """)
        unparsed = self.db.cursor.fetchone()[0]
        
        # Count papers needing embeddings
        self.db.cursor.execute("""
            SELECT COUNT(*) FROM papers 
            WHERE processed = 1 AND embedding_created = 0
        """)
        unembedded = self.db.cursor.fetchone()[0]
        
        # Count papers needing summaries
        self.db.cursor.execute("""
            SELECT COUNT(*) FROM papers 
            WHERE processed = 1 AND summary_generated = 0
        """)
        unsummarized = self.db.cursor.fetchone()[0]
        
        if unparsed > 0 or unembedded > 0 or unsummarized > 0:
            logger.info("="*40)
            logger.info("BACKLOG STATUS:")
            logger.info(f"  Papers awaiting parsing: {unparsed}")
            logger.info(f"  Papers awaiting embeddings: {unembedded}")
            logger.info(f"  Papers awaiting summaries: {unsummarized}")
            logger.info("  Run pipeline again to process more.")
            logger.info("="*40)

    def process_backlog(self, batch_size: int = 100) -> Dict:
        """Process any backlog of papers that need parsing/embedding/summarization
        
        This is useful for catching up after fetching many papers.
        """
        logger.info("="*60)
        logger.info("PROCESSING BACKLOG")
        logger.info("="*60)
        
        results = {
            'parse': None,
            'embeddings': None,
            'summaries': None
        }
        
        # Parse backlog
        logger.info(f"Parsing up to {batch_size} papers...")
        results['parse'] = self.pdf_parser.parse_all_unprocessed(limit=batch_size)
        logger.info(f"✓ Parsed {results['parse']['success']} papers")
        
        # Embed backlog
        logger.info(f"Creating embeddings for up to {batch_size} papers...")
        results['embeddings'] = self.vector_store.process_all_papers(limit=batch_size)
        logger.info(f"✓ Embedded {results['embeddings']['success']} papers")
        
        # Summarize backlog
        if self.summarizer_enabled and self.summarizer:
            logger.info(f"Generating summaries for up to {batch_size} papers...")
            results['summaries'] = self.summarizer.generate_summaries_batch(limit=batch_size)
            logger.info(f"✓ Summarized {results['summaries']['success']} papers")
        
        self._log_backlog_status()
        
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
        schedule.every().sunday.at("02:00").do(self.run_complete_pipeline)
        
        logger.info("Pipeline scheduled for weekly runs (Sundays at 2 AM)")
        
        while True:
            schedule.run_pending()
            time.sleep(60)
    
    def get_status(self) -> Dict:
        """Get current pipeline status"""
        stats = self.db.get_stats()

        # Get embedding stats
        embedding_stats = self.vector_store.get_embedding_stats()
        stats.update(embedding_stats)

        # Get summary stats
        if self.summarizer_enabled and self.summarizer:
            summary_stats = self.summarizer.get_summary_stats()
            stats.update(summary_stats)

        # Get backlog counts
        self.db.cursor.execute("SELECT COUNT(*) FROM papers WHERE processed = 0 AND pdf_downloaded = 1")
        stats['backlog_unparsed'] = self.db.cursor.fetchone()[0]
        
        self.db.cursor.execute("SELECT COUNT(*) FROM papers WHERE processed = 1 AND embedding_created = 0")
        stats['backlog_unembedded'] = self.db.cursor.fetchone()[0]
        
        self.db.cursor.execute("SELECT COUNT(*) FROM papers WHERE processed = 1 AND summary_generated = 0")
        stats['backlog_unsummarized'] = self.db.cursor.fetchone()[0]

        # Get last run info
        stats['last_run'] = self.db.get_last_pipeline_run()
        
        return stats
    
    def get_recent_papers(self, limit: int = 20) -> list:
        """Get recently added papers"""
        self.db.cursor.execute("""
        SELECT arxiv_id, title, abstract, published_date, pdf_downloaded, processed, embedding_created, summary_generated
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
                'has_embeddings': bool(row[6]),
                'has_summary': bool(row[7])
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
        print("5. Process backlog (catch up)")
        print("6. Schedule weekly runs")
        print("7. Exit")
        print("="*40)
        
        try:
            choice = input("\nEnter choice (1-7): ")
            
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
                print(f"\n  📋 Backlog:")
                print(f"    Awaiting parsing: {status.get('backlog_unparsed', 0)}")
                print(f"    Awaiting embeddings: {status.get('backlog_unembedded', 0)}")
                print(f"    Awaiting summaries: {status.get('backlog_unsummarized', 0)}")
                
                if status['last_run']:
                    print(f"\n  Last run: {status['last_run']['start_time']}")
                    print(f"  Status: {status['last_run']['status']}")
                
            elif choice == "3":
                query = input("\n🔍 Enter search query: ")
                results = orchestrator.search_papers(query)
                
                if results['results']:
                    print(f"\nFound {len(results['results'])} relevant papers:")
                    for i, paper in enumerate(results['results'], 1):
                        print(f"\n{i}. {paper['title']}")
                        print(f"   ArXiv ID: {paper['arxiv_id']}")
                        print(f"   Similarity: {paper['similarity']:.3f}")
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
                    if paper['has_summary']: status_icons.append("📝")
                    
                    print(f"\n{i}. {paper['title'][:80]}...")
                    print(f"   ArXiv: {paper['arxiv_id']} | Status: {' '.join(status_icons)}")
            
            elif choice == "5":
                batch = input("\nBatch size (default 100): ").strip()
                batch_size = int(batch) if batch else 100
                print(f"\n⚡ Processing backlog (batch size: {batch_size})...")
                results = orchestrator.process_backlog(batch_size=batch_size)
                print("\n✅ Backlog processing complete!")
                print(f"  Parsed: {results['parse']['success']}")
                print(f"  Embedded: {results['embeddings']['success']}")
                if results['summaries']:
                    print(f"  Summarized: {results['summaries']['success']}")
            
            elif choice == "6":
                print("\n⏰ Starting scheduler... (Press Ctrl+C to stop)")
                orchestrator.schedule_weekly_run()
            
            elif choice == "7":
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