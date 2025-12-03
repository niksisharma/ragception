"""
PDF Parser - Extracts text and sections from papers
Author: Amaan
"""

import PyPDF2
import re
import os
from typing import Dict, List
import logging
from database_manager import DatabaseManager

logger = logging.getLogger(__name__)

class PDFParser:
    """Extracts text from PDFs and stores in SQLite"""
    
    def __init__(self, pdf_dir="./data/pdfs"):
        self.pdf_dir = pdf_dir
        self.db = DatabaseManager()
    
    def parse_paper(self, arxiv_id: str) -> bool:
        """Parse a single paper and store in database"""
        pdf_path = os.path.join(self.pdf_dir, f"{arxiv_id}.pdf")
        
        if not os.path.exists(pdf_path):
            logger.error(f"PDF not found: {pdf_path}")
            return False
        
        try:
            # Extract text from PDF
            full_text = self._extract_text(pdf_path)
            
            # Extract sections
            sections = self._extract_sections(full_text)
            
            # Update database
            self.db.update_paper_content(arxiv_id, full_text, sections)
            
            logger.info(f"Parsed paper: {arxiv_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error parsing {arxiv_id}: {e}")
            return False
    
    def parse_all_unprocessed(self, limit=50) -> Dict:
        """Parse all unprocessed papers"""
        papers = self.db.get_unprocessed_papers(limit)
        
        results = {
            'total': len(papers),
            'success': 0,
            'failed': []
        }
        
        for paper in papers:
            if self.parse_paper(paper['arxiv_id']):
                results['success'] += 1
            else:
                results['failed'].append(paper['arxiv_id'])
        
        logger.info(f"Parsed {results['success']}/{results['total']} papers")
        return results
    
    def _extract_text(self, pdf_path: str) -> str:
        """Extract all text from PDF"""
        text_pages = []
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num in range(len(pdf_reader.pages)):
                try:
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    text_pages.append(text)
                except Exception as e:
                    logger.warning(f"Could not extract page {page_num}: {e}")
        
        return "\n".join(text_pages)
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract standard paper sections"""
        sections = {}
        
        # Common section patterns
        patterns = {
            'abstract': r'(?i)\babstract\b.*?(?=\n\s*\n|\b(?:introduction|1\.|keywords)\b)',
            'introduction': r'(?i)\b(?:1\.?\s*)?introduction\b.*?(?=\n\s*(?:2\.|related|background|method))',
            'methodology': r'(?i)\b(?:3\.?\s*)?(?:method|methodology|approach)\b.*?(?=\n\s*(?:4\.|experiment|evaluation|results))',
            'results': r'(?i)\b(?:4\.?\s*)?(?:results|experiments|evaluation)\b.*?(?=\n\s*(?:5\.|discussion|conclusion|related))',
            'conclusion': r'(?i)\b(?:5\.?\s*)?(?:conclusion|summary)\b.*?(?=\n\s*(?:references|acknowledgment|\Z))'
        }
        
        for section_name, pattern in patterns.items():
            match = re.search(pattern, text, re.DOTALL)
            if match:
                section_text = match.group()
                # Clean and truncate
                section_text = section_text.strip()[:5000]  # Limit section length
                sections[section_name] = section_text
        
        return sections
    
    def prepare_chunks_for_embedding(self, arxiv_id: str) -> List[Dict]:
        """Prepare text chunks for vector embedding"""
        paper = self.db.get_paper(arxiv_id)
        
        if not paper or not paper.get('full_text'):
            return []
        
        chunks = []
        
        # Chunk 1: Title + Abstract (most important)
        intro_chunk = f"Title: {paper['title']}\n"
        intro_chunk += f"Authors: {', '.join(paper['authors'][:3])}\n"
        intro_chunk += f"Abstract: {paper.get('abstract', '')}"
        
        chunks.append({
            'text': intro_chunk[:1000],
            'type': 'intro',
            'index': 0
        })
        
        # Chunk sections
        for section_name, section_text in paper.get('sections', {}).items():
            if section_text and len(section_text) > 100:
                chunks.append({
                    'text': f"Section: {section_name}\n{section_text[:1000]}",
                    'type': section_name,
                    'index': len(chunks)
                })
        
        # If no sections, chunk the full text
        if len(chunks) == 1 and paper.get('full_text'):
            text = paper['full_text']
            chunk_size = 1000
            overlap = 200
            
            for i in range(0, len(text), chunk_size - overlap):
                chunk_text = text[i:i + chunk_size]
                if len(chunk_text) > 100:
                    chunks.append({
                        'text': chunk_text,
                        'type': 'content',
                        'index': len(chunks)
                    })
        
        return chunks