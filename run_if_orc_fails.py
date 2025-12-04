from orchestrator import PipelineOrchestrator

orchestrator = PipelineOrchestrator()

print("🔄 Running complete processing pipeline...")
print("="*50)

print("\n📄 Step 1: Parsing PDFs...")
parse_results = orchestrator.pdf_parser.parse_all_unprocessed(limit=64)
print(f"✅ Parsed {parse_results['success']} papers")

print("\n🔮 Step 2: Creating OpenAI embeddings...")
embedding_results = orchestrator.vector_store.process_all_papers(limit=64)
print(f"✅ Created embeddings for {embedding_results['success']} papers")
print(f"💰 Estimated cost: ${embedding_results['estimated_cost']:.4f}")

print("\n✨ Pipeline complete! You can now search.")