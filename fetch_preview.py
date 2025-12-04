"""Preview Fetch - See what papers would be fetched without downloading or using API tokens"""

import arxiv
import json
from datetime import datetime, timedelta
from typing import Dict, List
import time

def preview_fetch(show_titles=True, test_days=None, test_max_results=None):
    """Preview what papers would be fetched based on config"""
    print("🔍 FETCH PREVIEW - No downloads, No API costs!")
    print("="*60)

    with open('config.json', 'r') as f:
        config = json.load(f)

    days_back = test_days if test_days else config.get('days_back', 7)
    max_results = test_max_results if test_max_results else config.get('max_papers_per_run', 50)
    
    print(f"📋 Configuration:")
    print(f"  Days back: {days_back}")
    print(f"  Max results: {max_results}")
    print(f"  Categories: {', '.join(config.get('arxiv_categories', []))}")
    print(f"  Keywords: {', '.join(config.get('keywords', [])[:5])}...")
    print("="*60)
    
    # Build search query (same as arxiv_bot)
    categories = config.get('arxiv_categories', ['cs.CL'])
    category_query = " OR ".join([f"cat:{cat}" for cat in categories])
    
    # Search arXiv
    search = arxiv.Search(
        query=category_query,
        max_results=max_results * 2,  # Get extra to account for filtering
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    cutoff_date = datetime.now() - timedelta(days=days_back)
    keywords = [kw.lower() for kw in config.get('keywords', ['rag', 'llm'])]
    
    relevant_papers = []
    all_papers_count = 0
    
    print("⏳ Searching arXiv (this takes a moment)...")
    
    for paper in search.results():
        all_papers_count += 1
        
        # Check date
        if paper.published.replace(tzinfo=None) < cutoff_date:
            break
        
        # Check relevance (same logic as arxiv_bot)
        text = (paper.title + " " + paper.summary).lower()
        is_relevant = any(keyword in text for keyword in keywords)
        
        if is_relevant:
            relevant_papers.append({
                'arxiv_id': paper.entry_id.split('/')[-1],
                'title': paper.title,
                'published': paper.published.strftime('%Y-%m-%d'),
                'categories': paper.categories,
                'matching_keywords': [kw for kw in keywords if kw in text][:3]
            })
            
            if len(relevant_papers) >= max_results:
                break
    
    print(f"\n📊 RESULTS PREVIEW:")
    print(f"  Total papers scanned: {all_papers_count}")
    print(f"  Papers within date range: {all_papers_count}")
    print(f"  Papers matching keywords: {len(relevant_papers)}")
    print(f"  Would be fetched: {min(len(relevant_papers), max_results)}")
    
    if relevant_papers and show_titles:
        print(f"\n📚 PAPERS THAT WOULD BE FETCHED (showing first 10):")
        print("-"*60)
        
        for i, paper in enumerate(relevant_papers[:10], 1):
            print(f"\n{i}. {paper['title'][:80]}...")
            print(f"   ID: {paper['arxiv_id']} | Published: {paper['published']}")
            print(f"   Matched: {', '.join(paper['matching_keywords'])}")
    
    # Cost estimation
    estimated_papers = min(len(relevant_papers), max_results)
    estimated_chunks = estimated_papers * 5  # ~5 chunks per paper
    estimated_tokens = estimated_chunks * 500  # ~500 tokens per chunk
    estimated_cost = (estimated_tokens / 1_000_000) * 0.02
    
    print(f"\n💰 COST ESTIMATION (if you run full pipeline):")
    print(f"  Papers to process: {estimated_papers}")
    print(f"  Estimated chunks: {estimated_chunks}")
    print(f"  Estimated tokens: {estimated_tokens:,}")
    print(f"  Estimated OpenAI cost: ${estimated_cost:.4f}")
    
    print(f"\n✅ Preview complete! No PDFs downloaded, no API tokens used.")
    
    return {
        'total_scanned': all_papers_count,
        'relevant_found': len(relevant_papers),
        'would_fetch': min(len(relevant_papers), max_results),
        'estimated_cost': estimated_cost,
        'papers': relevant_papers[:max_results]
    }

def test_different_configs():
    """Test different configurations to find optimal settings"""
    
    print("\n🧪 TESTING DIFFERENT CONFIGURATIONS")
    print("="*60)
    
    test_configs = [
        {'days': 7, 'max': 20},
        {'days': 14, 'max': 50},
        {'days': 30, 'max': 100},
    ]
    
    results = []
    for config in test_configs:
        print(f"\nTesting: {config['days']} days, max {config['max']} papers")
        print("-"*40)
        result = preview_fetch(
            show_titles=False,
            test_days=config['days'],
            test_max_results=config['max']
        )
        results.append({
            'config': config,
            'found': result['relevant_found'],
            'fetched': result['would_fetch'],
            'cost': result['estimated_cost']
        })
        time.sleep(1)  # Be nice to arXiv
    
    print("\n📊 COMPARISON SUMMARY:")
    print("-"*60)
    print(f"{'Days':<6} {'Max':<6} {'Found':<8} {'Fetched':<8} {'Est. Cost':<10}")
    print("-"*60)
    for r in results:
        print(f"{r['config']['days']:<6} {r['config']['max']:<6} "
              f"{r['found']:<8} {r['fetched']:<8} ${r['cost']:<10.4f}")

def check_specific_terms():
    """Check how many papers match specific search terms"""
    
    print("\n🔎 CHECKING SPECIFIC TERMS")
    print("="*60)
    
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    terms_to_check = [
        'hallucination',
        'RAG',
        'retrieval augmented',
        'context window',
        'fine-tuning',
        'prompt engineering'
    ]
    
    days_back = config.get('days_back', 7)
    categories = config.get('arxiv_categories', ['cs.CL'])
    category_query = " OR ".join([f"cat:{cat}" for cat in categories])
    
    # Get recent papers
    search = arxiv.Search(
        query=category_query,
        max_results=200,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    cutoff_date = datetime.now() - timedelta(days=days_back)
    papers_text = []
    
    print(f"Fetching last {days_back} days of papers for analysis...")
    
    for paper in search.results():
        if paper.published.replace(tzinfo=None) < cutoff_date:
            break
        papers_text.append((paper.title + " " + paper.summary).lower())
    
    print(f"Analyzing {len(papers_text)} papers...")
    print("\nTerm frequency in recent papers:")
    print("-"*40)
    
    for term in terms_to_check:
        count = sum(1 for text in papers_text if term.lower() in text)
        percentage = (count / len(papers_text)) * 100 if papers_text else 0
        print(f"{term:<20} : {count:3} papers ({percentage:5.1f}%)")

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════╗
    ║     FETCH PREVIEW - No Cost Testing       ║
    ╚═══════════════════════════════════════════╝
    """)
    
    print("\nWhat would you like to do?")
    print("1. Preview fetch with current config")
    print("2. Test different configurations")
    print("3. Check specific term frequencies")
    print("4. Quick preview (no titles)")
    
    choice = input("\nEnter choice (1-4): ")
    
    if choice == "1":
        preview_fetch(show_titles=True)
    elif choice == "2":
        test_different_configs()
    elif choice == "3":
        check_specific_terms()
    elif choice == "4":
        preview_fetch(show_titles=False)
    else:
        # Default to quick preview
        preview_fetch(show_titles=False)