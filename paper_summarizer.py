"""Paper Summarizer - Generates structured summaries using fine-tuned model"""

import json
import os
import re
from typing import Dict, List, Optional
import logging
from openai import OpenAI
from database_manager import DatabaseManager

logger = logging.getLogger(__name__)

class PaperSummarizer:
    """Generates structured summaries for research papers using fine-tuned model"""

    def __init__(self):
        self.db = DatabaseManager()

        with open('config.json', 'r') as f:
            self.config = json.load(f)

        api_key = self._get_api_key()

        if not api_key:
            raise ValueError(
                "OpenAI API key not found! Please set OPENAI_API_KEY environment variable "
                "or add it to .streamlit/secrets.toml"
            )

        self.client = OpenAI(api_key=api_key)
        self.fine_tuned_model = self.config.get('fine_tuned_model')
        self.fallback_model = self.config.get('fallback_model', 'gpt-4o-mini')
        self.enabled = self.config.get('summarization_enabled', True)
        self.temperature = self.config.get('summary_temperature', 0.2)
        self.max_chars = self.config.get('summary_max_chars', 4000)

        if not self.fine_tuned_model:
            logger.warning("No fine-tuned model configured. Will use fallback model.")

        logger.info(f"Paper Summarizer initialized with model: {self.fine_tuned_model or self.fallback_model}")

    def _get_api_key(self) -> Optional[str]:
        """Get API key from multiple sources"""
        # Try Streamlit secrets first
        try:
            import streamlit as st
            if 'OPENAI_API_KEY' in st.secrets:
                return st.secrets['OPENAI_API_KEY']
        except (ImportError, FileNotFoundError):
            pass

        # Try loading from secrets.toml directly
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

    def generate_summary(self, paper_id: str) -> bool:
        """Generate structured summary for a single paper"""
        if not self.enabled:
            logger.info("Summarization is disabled in config")
            return False

        try:
            # Check if summary already exists
            existing = self.db.get_paper_summary(paper_id)
            if existing:
                logger.info(f"Summary already exists for {paper_id}")
                return True

            # Get paper data
            paper = self.db.get_paper(paper_id)
            if not paper:
                logger.error(f"Paper not found: {paper_id}")
                return False

            # Get full text (prefer full_text, fallback to abstract)
            text = paper.get('full_text') or paper.get('abstract')
            if not text:
                logger.error(f"No text available for {paper_id}")
                return False

            # Truncate text if needed
            if len(text) > self.max_chars:
                text = text[:self.max_chars]
                logger.warning(f"Text truncated to {self.max_chars} chars for {paper_id}")

            # Build prompt
            prompt = self._build_prompt(text)

            # Choose model
            model = self.fine_tuned_model if self.fine_tuned_model else self.fallback_model

            # Generate summary
            logger.info(f"Generating summary for {paper_id} using {model}")
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature
            )

            raw_summary = response.choices[0].message.content

            # Parse summary sections
            sections = self._parse_summary_sections(raw_summary)

            # Validate and calculate structure score
            structure_score = self._validate_summary(sections)

            # Store in database
            self.db.store_paper_summary(
                paper_id=paper_id,
                title=sections.get('title', paper.get('title', 'Unknown')),
                authors=sections.get('authors', ''),
                date=sections.get('date', ''),
                abstract_summary=sections.get('abstract', ''),
                methodology=sections.get('methodology', ''),
                results=sections.get('results', ''),
                related_work=sections.get('related_work', ''),
                raw_summary=raw_summary,
                structure_score=structure_score
            )

            # Update papers table
            self.db.mark_summary_generated(paper_id)

            logger.info(f"Summary generated for {paper_id} (score: {structure_score:.1f}%)")
            return True

        except Exception as e:
            logger.error(f"Error generating summary for {paper_id}: {e}")
            return False

    def generate_summaries_batch(self, limit: int = 50) -> Dict:
        """Generate summaries for multiple papers that don't have them yet"""
        # Get papers that need summaries
        papers = self.db.get_papers_without_summaries(limit)

        results = {
            'total': len(papers),
            'success': 0,
            'failed': [],
            'skipped': 0,
            'estimated_cost': 0.0
        }

        for paper in papers:
            paper_id = paper['arxiv_id']

            # Check if we should skip (no text available)
            if not paper.get('full_text') and not paper.get('abstract'):
                results['skipped'] += 1
                logger.warning(f"Skipping {paper_id}: no text available")
                continue

            if self.generate_summary(paper_id):
                results['success'] += 1
            else:
                results['failed'].append(paper_id)

        # Estimate cost (rough estimate based on tokens)
        # Fine-tuned models typically cost more than base models
        # Assuming ~1000 tokens per summary * $0.012 per 1K tokens for gpt-4o-mini fine-tuned
        avg_tokens_per_summary = 1000
        cost_per_1k_tokens = 0.012
        results['estimated_cost'] = (results['success'] * avg_tokens_per_summary / 1000) * cost_per_1k_tokens

        logger.info(f"Batch summary generation: {results['success']}/{results['total']} successful")
        logger.info(f"Estimated cost: ${results['estimated_cost']:.4f}")

        return results

    def _build_prompt(self, text: str) -> str:
        """Build the standardized prompt for summary generation"""
        prompt = f"""You are a research paper summarization expert.

Generate a structured summary with EXACTLY these sections in this order:

**Title:**
[paper title]

**Authors:**
[author names]

**Date:**
[publication date]

**Abstract:**
[2-3 sentence summary]

**Methodology:**
[2-3 bullet points on methods used]

**Results:**
[2-3 bullet points on key findings]

**Related Work:**
[2-3 bullet points on prior research]

### PAPER TEXT:
{text}

Remember: Use the EXACT section headers shown above."""

        return prompt

    def _parse_summary_sections(self, summary_text: str) -> Dict[str, str]:
        """Extract sections from the generated summary"""
        sections = {}

        # Define section patterns (case-insensitive, flexible formatting)
        # Try both **Section:** and Section: formats
        section_patterns = {
            'title': [
                r'\*\*Title:\*\*\s*(.*?)(?=\n\n|\*\*|\n[A-Z]|$)',
                r'Title:\s*(.*?)(?=\n\n|\n[A-Z]|$)'
            ],
            'authors': [
                r'\*\*Authors:\*\*\s*(.*?)(?=\n\n|\*\*|\n[A-Z]|$)',
                r'Authors:\s*(.*?)(?=\n\n|\n[A-Z]|$)'
            ],
            'date': [
                r'\*\*Date:\*\*\s*(.*?)(?=\n\n|\*\*|\n[A-Z]|$)',
                r'Date:\s*(.*?)(?=\n\n|\n[A-Z]|$)'
            ],
            'abstract': [
                r'\*\*Abstract:\*\*\s*(.*?)(?=\n\n|\*\*|\n[A-Z][a-z]+:|$)',
                r'Abstract:\s*(.*?)(?=\n\n|\n[A-Z][a-z]+:|$)'
            ],
            'methodology': [
                r'\*\*Methodology:\*\*\s*(.*?)(?=\n\n|\*\*|\n[A-Z][a-z]+:|$)',
                r'Methodology:\s*(.*?)(?=\n\n|\n[A-Z][a-z]+:|$)'
            ],
            'results': [
                r'\*\*Results:\*\*\s*(.*?)(?=\n\n|\*\*|\n[A-Z][a-z]+:|$)',
                r'Results:\s*(.*?)(?=\n\n|\n[A-Z][a-z]+:|$)'
            ],
            'related_work': [
                r'\*\*Related Work:\*\*\s*(.*?)(?=\n\n|\*\*|$)',
                r'Related Work:\s*(.*?)(?=\n\n|$)'
            ]
        }

        for section_name, patterns in section_patterns.items():
            content = ''
            for pattern in patterns:
                match = re.search(pattern, summary_text, re.DOTALL | re.IGNORECASE)
                if match:
                    content = match.group(1).strip()
                    break

            sections[section_name] = content
            if not content:
                logger.warning(f"Section '{section_name}' not found in summary")

        return sections

    def _validate_summary(self, sections: Dict[str, str]) -> float:
        """Validate summary structure and return quality score (0-100)"""
        required_sections = ['title', 'authors', 'date', 'abstract', 'methodology', 'results', 'related_work']

        sections_found = sum(1 for section in required_sections if sections.get(section))
        structure_score = (sections_found / len(required_sections)) * 100

        return structure_score

    def get_summary_stats(self) -> Dict:
        """Get statistics about generated summaries"""
        stats = self.db.get_summary_stats()

        # Add configuration info
        stats['fine_tuned_model'] = self.fine_tuned_model or 'Not configured'
        stats['fallback_model'] = self.fallback_model
        stats['enabled'] = self.enabled

        return stats

    def regenerate_summary(self, paper_id: str, force: bool = False) -> bool:
        """Regenerate summary for a paper (useful if model is updated)"""
        if force:
            # Delete existing summary
            self.db.delete_paper_summary(paper_id)

        return self.generate_summary(paper_id)
