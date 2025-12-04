"""Reranker Module - LLM-based reranking for improved search results"""

import json
import os
import logging
from typing import List, Dict, Optional

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("OpenAI library required. Run: pip install openai")

logger = logging.getLogger(__name__)


class LLMReranker:
    """Reranks search results using LLM for better relevance"""
    
    def __init__(self, api_key: str = None):
        # Get API key
        if not api_key:
            api_key = self._get_api_key()
        
        if not api_key:
            raise ValueError("OpenAI API key not found")
        
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-3.5-turbo"  # Cheaper model for reranking
        logger.info("LLM Reranker initialized")
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment or secrets file"""
        # Try environment variable first
        api_key = os.environ.get('OPENAI_API_KEY')
        if api_key:
            return api_key
        
        # Try secrets.toml
        try:
            import toml
            secrets_path = '.streamlit/secrets.toml'
            if os.path.exists(secrets_path):
                secrets = toml.load(secrets_path)
                return secrets.get('OPENAI_API_KEY')
        except Exception:
            pass
        
        return None
    
    def rerank(
        self, 
        query: str, 
        candidates: List[Dict], 
        top_k: int = 5,
        user_interests: List[str] = None
    ) -> List[Dict]:
        """
        Rerank candidates using LLM scoring
        
        Args:
            query: User's search query
            candidates: List of paper dicts with 'title', 'abstract', 'arxiv_id'
            top_k: Number of results to return
            user_interests: Optional list of user's research interests
            
        Returns:
            Reranked list of papers with added 'rerank_score' field
        """
        if not candidates:
            return []
        
        if len(candidates) <= top_k:
            # No need to rerank if we have fewer candidates than requested
            for paper in candidates:
                paper['rerank_score'] = paper.get('similarity', 0.5)
            return candidates
        
        try:
            # Build the prompt for reranking
            prompt = self._build_rerank_prompt(query, candidates, user_interests)
            
            # Call LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a research paper relevance evaluator. Score papers based on relevance to the query. Respond ONLY with valid JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent scoring
                max_tokens=1000
            )
            
            # Parse response
            response_text = response.choices[0].message.content.strip()
            scores = self._parse_scores(response_text, len(candidates))
            
            # Add scores to candidates
            for i, paper in enumerate(candidates):
                paper['rerank_score'] = scores.get(i, 0.5)
            
            # Sort by rerank score
            reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
            
            logger.info(f"Reranked {len(candidates)} papers, returning top {top_k}")
            return reranked[:top_k]
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}, falling back to original order")
            # Return original candidates if reranking fails
            return candidates[:top_k]
    
    def _build_rerank_prompt(
        self, 
        query: str, 
        candidates: List[Dict], 
        user_interests: List[str] = None
    ) -> str:
        """Build the prompt for LLM reranking"""
        
        prompt_parts = [
            f"Query: {query}",
            ""
        ]
        
        if user_interests:
            prompt_parts.append(f"User's research interests: {', '.join(user_interests)}")
            prompt_parts.append("")
        
        prompt_parts.append("Papers to evaluate:")
        prompt_parts.append("")
        
        for i, paper in enumerate(candidates):
            title = paper.get('title', 'Untitled')
            abstract = paper.get('abstract', '')[:300]  # Truncate for token limits
            prompt_parts.append(f"[{i}] Title: {title}")
            prompt_parts.append(f"    Abstract: {abstract}...")
            prompt_parts.append("")
        
        prompt_parts.extend([
            "Score each paper's relevance to the query from 0.0 to 1.0.",
            "Consider: topic match, methodology relevance, recency value.",
            "",
            "Respond with JSON only, format:",
            '{"scores": {"0": 0.9, "1": 0.7, "2": 0.5, ...}}'
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_scores(self, response_text: str, num_candidates: int) -> Dict[int, float]:
        """Parse LLM response to extract scores"""
        try:
            # Clean response (remove markdown code blocks if present)
            clean_text = response_text.strip()
            if clean_text.startswith("```"):
                clean_text = clean_text.split("```")[1]
                if clean_text.startswith("json"):
                    clean_text = clean_text[4:]
            
            # Parse JSON
            data = json.loads(clean_text)
            
            # Extract scores
            scores_dict = data.get("scores", data)
            
            # Convert to int keys and float values
            scores = {}
            for key, value in scores_dict.items():
                try:
                    idx = int(key)
                    score = float(value)
                    scores[idx] = min(max(score, 0.0), 1.0)  # Clamp to [0, 1]
                except (ValueError, TypeError):
                    continue
            
            return scores
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse reranker response: {e}")
            # Return default scores
            return {i: 0.5 for i in range(num_candidates)}
    
    def rerank_with_explanation(
        self, 
        query: str, 
        candidates: List[Dict], 
        top_k: int = 5
    ) -> tuple[List[Dict], str]:
        """
        Rerank and provide explanation for the ranking
        
        Returns:
            Tuple of (reranked papers, explanation string)
        """
        if not candidates:
            return [], "No papers to rank."
        
        try:
            prompt = f"""Query: {query}

Papers:
"""
            for i, paper in enumerate(candidates[:10]):  # Limit to 10 for explanation
                prompt += f"\n[{i}] {paper.get('title', 'Untitled')}"
            
            prompt += """

Rank these papers by relevance. Respond with:
1. JSON scores: {"scores": {"0": 0.9, ...}}
2. Brief explanation of top 3 picks

Format:
```json
{"scores": {...}}
```
Explanation: ..."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a research paper evaluator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=800
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Extract JSON and explanation
            if "```json" in response_text:
                json_part = response_text.split("```json")[1].split("```")[0]
                explanation_part = response_text.split("```")[-1].strip()
            else:
                json_part = response_text
                explanation_part = ""
            
            # Parse scores
            scores = self._parse_scores(json_part, len(candidates))
            
            # Add scores and sort
            for i, paper in enumerate(candidates):
                paper['rerank_score'] = scores.get(i, 0.5)
            
            reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
            
            # Clean explanation
            explanation = explanation_part.replace("Explanation:", "").strip()
            if not explanation:
                explanation = f"Top papers ranked by relevance to: {query}"
            
            return reranked[:top_k], explanation
            
        except Exception as e:
            logger.error(f"Reranking with explanation failed: {e}")
            return candidates[:top_k], "Ranked by embedding similarity."


class SimpleReranker:
    """Fallback reranker using keyword matching (no API calls)"""
    
    def rerank(
        self, 
        query: str, 
        candidates: List[Dict], 
        top_k: int = 5
    ) -> List[Dict]:
        """Rerank using simple keyword matching"""
        
        query_words = set(query.lower().split())
        
        for paper in candidates:
            title = paper.get('title', '').lower()
            abstract = paper.get('abstract', '').lower()
            
            # Count keyword matches
            text = title + " " + abstract
            matches = sum(1 for word in query_words if word in text)
            
            # Boost title matches
            title_matches = sum(1 for word in query_words if word in title)
            
            # Calculate simple score
            base_score = paper.get('similarity', 0.5)
            keyword_boost = (matches * 0.05) + (title_matches * 0.1)
            
            paper['rerank_score'] = min(base_score + keyword_boost, 1.0)
        
        # Sort by score
        reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
        
        return reranked[:top_k]
