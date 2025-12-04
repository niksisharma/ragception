"""Reparse existing summaries with the improved parser"""

from database_manager import DatabaseManager
from paper_summarizer import PaperSummarizer
import re

db = DatabaseManager()
summarizer = PaperSummarizer()

print("Fetching summaries with score < 50...")

# Get all summaries with low scores
db.cursor.execute("""
    SELECT paper_id, raw_summary
    FROM paper_summaries
    WHERE structure_score < 50
""")

summaries_to_reparse = db.cursor.fetchall()
print(f"Found {len(summaries_to_reparse)} summaries to reparse\n")

success_count = 0
failed_count = 0

for paper_id, raw_summary in summaries_to_reparse:
    if not raw_summary:
        print(f"Skipping {paper_id}: no raw summary")
        failed_count += 1
        continue

    # Re-parse the summary
    sections = summarizer._parse_summary_sections(raw_summary)
    structure_score = summarizer._validate_summary(sections)

    # Update in database
    try:
        db.store_paper_summary(
            paper_id=paper_id,
            title=sections.get('title', ''),
            authors=sections.get('authors', ''),
            date=sections.get('date', ''),
            abstract_summary=sections.get('abstract', ''),
            methodology=sections.get('methodology', ''),
            results=sections.get('results', ''),
            related_work=sections.get('related_work', ''),
            raw_summary=raw_summary,
            structure_score=structure_score
        )
        success_count += 1
        print(f"[OK] {paper_id}: score {structure_score:.0f}%")
    except Exception as e:
        print(f"[FAIL] {paper_id}: {e}")
        failed_count += 1

print(f"\nReparsing complete!")
print(f"  Success: {success_count}")
print(f"  Failed: {failed_count}")

# Show new stats
stats = db.get_summary_stats()
print(f"\nNew average structure score: {stats['avg_structure_score']:.1f}%")
