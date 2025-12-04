"""Agent Module - LLM Agent with Tool Calling"""

import json
import os
import logging
from typing import List, Dict, Optional, Callable
from datetime import datetime

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("OpenAI library required. Run: pip install openai")

logger = logging.getLogger(__name__)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": "Search for research papers using semantic search. Use this when the user wants to find papers on a topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query - topic, keywords, or research question"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_paper_details",
            "description": "Get full details of a specific paper by its arXiv ID or reference number from last search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_reference": {
                        "type": "string",
                        "description": "arXiv ID (e.g., '2401.12345') or reference like 'first', 'second', '1', '2'"
                    }
                },
                "required": ["paper_reference"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_paper",
            "description": "Generate a summary of a research paper.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_reference": {
                        "type": "string",
                        "description": "arXiv ID or reference like 'first', 'second'"
                    },
                    "summary_type": {
                        "type": "string",
                        "enum": ["brief", "detailed", "technical"],
                        "description": "Type of summary to generate",
                        "default": "brief"
                    }
                },
                "required": ["paper_reference"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_paper",
            "description": "Save/bookmark a paper to the user's collection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_reference": {
                        "type": "string",
                        "description": "arXiv ID or reference like 'first', 'second'"
                    }
                },
                "required": ["paper_reference"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_note_to_paper",
            "description": "Add a personal note to a paper.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_reference": {
                        "type": "string",
                        "description": "arXiv ID or reference"
                    },
                    "note": {
                        "type": "string",
                        "description": "The note to add"
                    }
                },
                "required": ["paper_reference", "note"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_search_history",
            "description": "Get the user's recent search history.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of recent searches to retrieve",
                        "default": 10
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_saved_papers",
            "description": "Get all papers the user has saved/bookmarked.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_ethics",
            "description": "Evaluate a paper against an ethics rubric covering bias, privacy, reproducibility, and societal impact.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_reference": {
                        "type": "string",
                        "description": "arXiv ID or reference"
                    }
                },
                "required": ["paper_reference"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_papers",
            "description": "Compare two or more papers on their approaches, methods, and findings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_references": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of arXiv IDs or references to compare"
                    }
                },
                "required": ["paper_references"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_papers_email",
            "description": "Send papers to user's email address.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_references": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of paper references to email. Use 'current' for current search results."
                    }
                },
                "required": ["paper_references"]
            }
        }
    }
]


class ResearchAgent:
    """LLM Agent with tool calling for research paper assistance"""
    
    def __init__(
        self,
        vector_store,
        db_manager,
        conversation_memory,
        long_term_memory,
        reranker=None
    ):
        self.vector_store = vector_store
        self.db = db_manager
        self.conv_memory = conversation_memory
        self.lt_memory = long_term_memory
        self.reranker = reranker

        api_key = self._get_api_key()
        if not api_key:
            raise ValueError("OpenAI API key not found")

        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"

        self.current_user_id = None
        self.current_user_email = None
        self.smtp_user = None
        self.smtp_password = None

        logger.info("Research Agent initialized")
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment or secrets"""
        api_key = os.environ.get('OPENAI_API_KEY')
        if api_key:
            return api_key
        
        try:
            import toml
            secrets_path = '.streamlit/secrets.toml'
            if os.path.exists(secrets_path):
                secrets = toml.load(secrets_path)
                return secrets.get('OPENAI_API_KEY')
        except Exception:
            pass
        
        return None
    
    def set_user(self, email: str, name: str = None):
        """Set the current user for personalization"""
        user = self.lt_memory.get_or_create_user(email, name)
        self.current_user_id = user['id']
        self.current_user_email = email
        logger.info(f"Agent user set: {email}")
    
    def set_email_credentials(self, smtp_user: str, smtp_password: str):
        """Set email credentials for sending papers"""
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
    
    def chat(self, user_message: str) -> str:
        """Main chat interface - processes user message and returns response"""
        self.conv_memory.add_message("user", user_message)

        system_prompt = self._build_system_prompt()
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.conv_memory.get_messages_for_api())

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=1500
            )
            
            assistant_message = response.choices[0].message

            if assistant_message.tool_calls:
                final_response = self._handle_tool_calls(messages, assistant_message)
            else:
                final_response = assistant_message.content

            self.conv_memory.add_message("assistant", final_response)
            return final_response
            
        except Exception as e:
            logger.error(f"Agent chat error: {e}")
            error_msg = f"I encountered an error: {str(e)}. Please try again."
            self.conv_memory.add_message("assistant", error_msg)
            return error_msg
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with context"""
        prompt_parts = [
            "You are a helpful research paper assistant. You help users find, understand, and organize academic papers.",
            "",
            "You have access to tools for:",
            "- Searching papers (semantic search with RAG)",
            "- Getting paper details and summaries",
            "- Saving papers and adding notes",
            "- Viewing search history and saved papers",
            "- Evaluating papers for ethics",
            "- Comparing multiple papers",
            "- Emailing papers to the user",
            "",
            "Guidelines:",
            "- Use tools when appropriate to answer user requests",
            "- Reference papers by their position in search results (first, second, etc.) or arXiv ID",
            "- Be concise but informative",
            "- If unsure which paper the user means, ask for clarification",
            ""
        ]
        
        # Add user context if available
        if self.current_user_id:
            user_context = self.lt_memory.get_user_context(self.current_user_id)
            prompt_parts.append(f"=== User Context ===\n{user_context}\n")
        
        # Add conversation context
        conv_context = self.conv_memory.get_context_for_llm()
        if conv_context:
            prompt_parts.append(conv_context)
        
        return "\n".join(prompt_parts)
    
    def _handle_tool_calls(self, messages: List[Dict], assistant_message) -> str:
        """Handle tool calls and return final response"""
        messages.append({
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in assistant_message.tool_calls
            ]
        })

        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            logger.info(f"Executing tool: {function_name} with args: {function_args}")

            result = self._execute_tool(function_name, function_args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result) if isinstance(result, dict) else str(result)
            })

        final_response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=1500
        )

        return final_response.choices[0].message.content
    
    def _execute_tool(self, function_name: str, args: Dict) -> Dict:
        """Execute a tool and return result"""
        tool_handlers = {
            "search_papers": self._tool_search_papers,
            "get_paper_details": self._tool_get_paper_details,
            "summarize_paper": self._tool_summarize_paper,
            "save_paper": self._tool_save_paper,
            "add_note_to_paper": self._tool_add_note,
            "get_search_history": self._tool_get_search_history,
            "get_saved_papers": self._tool_get_saved_papers,
            "evaluate_ethics": self._tool_evaluate_ethics,
            "compare_papers": self._tool_compare_papers,
            "send_papers_email": self._tool_send_email
        }

        handler = tool_handlers.get(function_name)

        if handler:
            try:
                return handler(**args)
            except Exception as e:
                logger.error(f"Tool {function_name} failed: {e}")
                return {"error": str(e)}
        else:
            return {"error": f"Unknown tool: {function_name}"}

    def _tool_search_papers(self, query: str, num_results: int = 5) -> Dict:
        """Search for papers"""
        # Get more candidates for reranking
        candidates = self.vector_store.semantic_search(query, n_results=20)
        
        # Rerank if available
        if self.reranker and candidates:
            user_interests = []
            if self.current_user_id:
                user_interests = self.lt_memory.get_user_interests(self.current_user_id)
            
            results = self.reranker.rerank(
                query, 
                candidates, 
                top_k=num_results,
                user_interests=user_interests
            )
        else:
            results = candidates[:num_results]
        
        # Update conversation memory with results
        self.conv_memory.set_search_results(results, query)
        
        # Log search to long-term memory
        if self.current_user_id:
            self.lt_memory.add_search(self.current_user_id, query, len(results))
        
        return {
            "success": True,
            "query": query,
            "num_results": len(results),
            "papers": [
                {
                    "position": i + 1,
                    "arxiv_id": p.get("arxiv_id"),
                    "title": p.get("title"),
                    "abstract": p.get("abstract", "")[:300] + "...",
                    "similarity": p.get("similarity", 0),
                    "rerank_score": p.get("rerank_score", p.get("similarity", 0))
                }
                for i, p in enumerate(results)
            ]
        }
    
    def _tool_get_paper_details(self, paper_reference: str) -> Dict:
        """Get full paper details"""
        paper = self._resolve_paper_reference(paper_reference)
        
        if not paper:
            return {"error": f"Paper not found: {paper_reference}"}
        
        arxiv_id = paper.get("arxiv_id")
        full_paper = self.db.get_paper(arxiv_id)
        
        if full_paper:
            return {
                "success": True,
                "arxiv_id": full_paper["arxiv_id"],
                "title": full_paper["title"],
                "abstract": full_paper["abstract"],
                "authors": full_paper.get("authors", []),
                "published_date": full_paper.get("published_date"),
                "categories": full_paper.get("categories", []),
                "url": f"https://arxiv.org/abs/{arxiv_id}"
            }
        else:
            return {
                "success": True,
                "arxiv_id": arxiv_id,
                "title": paper.get("title"),
                "abstract": paper.get("abstract"),
                "url": f"https://arxiv.org/abs/{arxiv_id}"
            }
    
    def _tool_summarize_paper(self, paper_reference: str, summary_type: str = "brief") -> Dict:
        """Generate paper summary using LLM"""
        paper = self._resolve_paper_reference(paper_reference)
        
        if not paper:
            return {"error": f"Paper not found: {paper_reference}"}
        
        # Get full paper details
        arxiv_id = paper.get("arxiv_id")
        full_paper = self.db.get_paper(arxiv_id)
        
        title = full_paper.get("title") if full_paper else paper.get("title")
        abstract = full_paper.get("abstract") if full_paper else paper.get("abstract")
        
        # Generate summary with LLM
        length_guide = {
            "brief": "2-3 sentences",
            "detailed": "a paragraph (5-7 sentences)",
            "technical": "a detailed technical summary with methodology"
        }
        
        prompt = f"""Summarize this research paper in {length_guide.get(summary_type, '2-3 sentences')}:

Title: {title}

Abstract: {abstract}

Provide a clear, informative summary."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=500
        )
        
        summary = response.choices[0].message.content
        
        return {
            "success": True,
            "arxiv_id": arxiv_id,
            "title": title,
            "summary_type": summary_type,
            "summary": summary
        }
    
    def _tool_save_paper(self, paper_reference: str) -> Dict:
        """Save paper to user's collection"""
        if not self.current_user_id:
            return {"error": "No user logged in. Please provide your email first."}
        
        paper = self._resolve_paper_reference(paper_reference)
        
        if not paper:
            return {"error": f"Paper not found: {paper_reference}"}
        
        arxiv_id = paper.get("arxiv_id")
        success = self.lt_memory.save_paper(self.current_user_id, arxiv_id)
        
        if success:
            return {
                "success": True,
                "message": f"Paper saved: {paper.get('title', arxiv_id)}"
            }
        else:
            return {"error": "Failed to save paper"}
    
    def _tool_add_note(self, paper_reference: str, note: str) -> Dict:
        """Add note to a paper"""
        if not self.current_user_id:
            return {"error": "No user logged in. Please provide your email first."}
        
        paper = self._resolve_paper_reference(paper_reference)
        
        if not paper:
            return {"error": f"Paper not found: {paper_reference}"}
        
        arxiv_id = paper.get("arxiv_id")
        success = self.lt_memory.add_note(self.current_user_id, arxiv_id, note)
        
        if success:
            return {
                "success": True,
                "message": f"Note added to paper: {paper.get('title', arxiv_id)}"
            }
        else:
            return {"error": "Failed to add note"}
    
    def _tool_get_search_history(self, limit: int = 10) -> Dict:
        """Get user's search history"""
        if not self.current_user_id:
            return {"error": "No user logged in. Please provide your email first."}
        
        history = self.lt_memory.get_search_history(self.current_user_id, limit)
        
        return {
            "success": True,
            "searches": history
        }
    
    def _tool_get_saved_papers(self) -> Dict:
        """Get user's saved papers"""
        if not self.current_user_id:
            return {"error": "No user logged in. Please provide your email first."}
        
        papers = self.lt_memory.get_saved_papers(self.current_user_id)
        
        return {
            "success": True,
            "num_saved": len(papers),
            "papers": papers
        }
    
    def _tool_evaluate_ethics(self, paper_reference: str) -> Dict:
        """Evaluate paper against ethics rubric"""
        paper = self._resolve_paper_reference(paper_reference)
        
        if not paper:
            return {"error": f"Paper not found: {paper_reference}"}
        
        arxiv_id = paper.get("arxiv_id")
        full_paper = self.db.get_paper(arxiv_id)
        
        title = full_paper.get("title") if full_paper else paper.get("title")
        abstract = full_paper.get("abstract") if full_paper else paper.get("abstract")
        
        # Ethics evaluation prompt
        prompt = f"""Evaluate this research paper against an ethics rubric.

Title: {title}
Abstract: {abstract}

Evaluate on these criteria (score 1-5, with explanation):

1. **Bias & Fairness**: Does the paper address potential biases in data or methodology?
2. **Privacy**: Are there data privacy concerns or sensitive data handling issues?
3. **Reproducibility**: Is the methodology transparent and reproducible?
4. **Societal Impact**: What are potential positive/negative societal impacts?
5. **Environmental**: Are computational costs and environmental impact considered?

Provide a structured evaluation with scores and brief explanations for each criterion, plus an overall assessment."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800
        )
        
        evaluation = response.choices[0].message.content
        
        return {
            "success": True,
            "arxiv_id": arxiv_id,
            "title": title,
            "ethics_evaluation": evaluation
        }
    
    def _tool_compare_papers(self, paper_references: List[str]) -> Dict:
        """Compare multiple papers"""
        papers = []
        for ref in paper_references:
            paper = self._resolve_paper_reference(ref)
            if paper:
                papers.append(paper)
        
        if len(papers) < 2:
            return {"error": "Need at least 2 valid papers to compare"}
        
        # Build comparison prompt
        prompt = "Compare these research papers:\n\n"
        
        for i, p in enumerate(papers, 1):
            arxiv_id = p.get("arxiv_id")
            full_paper = self.db.get_paper(arxiv_id)
            title = full_paper.get("title") if full_paper else p.get("title")
            abstract = full_paper.get("abstract") if full_paper else p.get("abstract")
            
            prompt += f"Paper {i}: {title}\nAbstract: {abstract[:500]}...\n\n"
        
        prompt += """Compare these papers on:
1. Research focus and objectives
2. Methodology/approach
3. Key findings or contributions
4. Strengths and limitations
5. Which might be more relevant for different use cases

Provide a clear, structured comparison."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=1000
        )
        
        return {
            "success": True,
            "papers_compared": [p.get("title") for p in papers],
            "comparison": response.choices[0].message.content
        }
    
    def _tool_send_email(self, paper_references: List[str]) -> Dict:
        """Send papers via email"""
        if not self.current_user_email:
            return {"error": "No email address provided. Please set your email first."}
        
        if not self.smtp_user or not self.smtp_password:
            return {"error": "Email not configured. Please provide SMTP credentials."}
        
        # Resolve papers
        if paper_references == ["current"] or "current" in paper_references:
            papers = self.conv_memory.current_search_results
        else:
            papers = []
            for ref in paper_references:
                paper = self._resolve_paper_reference(ref)
                if paper:
                    papers.append(paper)
        
        if not papers:
            return {"error": "No papers found to email"}
        
        try:
            from email_utils import send_papers_email
            
            send_papers_email(
                smtp_user=self.smtp_user,
                smtp_password=self.smtp_password,
                to_address=self.current_user_email,
                query=self.conv_memory.current_topic or "Research papers",
                papers=papers
            )
            
            return {
                "success": True,
                "message": f"Sent {len(papers)} papers to {self.current_user_email}"
            }
        except Exception as e:
            return {"error": f"Failed to send email: {str(e)}"}
    
    # ===== Helper Methods =====
    
    def _resolve_paper_reference(self, reference: str) -> Optional[Dict]:
        """Resolve paper reference to actual paper data"""
        # Try conversation memory first (for references like "first", "second")
        paper = self.conv_memory.get_paper_by_reference(reference)
        if paper:
            return paper
        
        # Try as arXiv ID
        if "/" in reference or reference.replace(".", "").replace("-", "").isalnum():
            # Looks like an arXiv ID
            full_paper = self.db.get_paper(reference)
            if full_paper:
                return {
                    "arxiv_id": full_paper["arxiv_id"],
                    "title": full_paper["title"],
                    "abstract": full_paper["abstract"]
                }
        
        return None
