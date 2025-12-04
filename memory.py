"""Memory Module - Short-term (Conversation) and Long-term (Persistent) Memory"""

import json
from datetime import datetime
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class ConversationMemory:
    """Short-term memory for maintaining conversation context"""
    
    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self.messages: List[Dict] = []
        self.current_search_results: List[Dict] = []
        self.current_topic: Optional[str] = None
        self.session_start = datetime.now()
    
    def add_message(self, role: str, content: str):
        """Add a message to conversation history"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only recent messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def set_search_results(self, results: List[Dict], topic: str):
        """Store current search results for reference"""
        self.current_search_results = results
        self.current_topic = topic
    
    def get_paper_by_reference(self, reference: str) -> Optional[Dict]:
        """Get paper by number reference (e.g., 'first', 'second', '1', '2')"""
        if not self.current_search_results:
            return None
        
        # Map references to indices
        ref_map = {
            'first': 0, '1': 0, 'one': 0,
            'second': 1, '2': 1, 'two': 1,
            'third': 2, '3': 2, 'three': 2,
            'fourth': 3, '4': 3, 'four': 3,
            'fifth': 4, '5': 4, 'five': 4,
            'last': -1
        }
        
        ref_lower = reference.lower().strip()
        
        if ref_lower in ref_map:
            idx = ref_map[ref_lower]
            if idx < len(self.current_search_results):
                return self.current_search_results[idx]
        
        return None
    
    def get_context_for_llm(self) -> str:
        """Get formatted context for LLM prompt"""
        context_parts = []
        
        # Add conversation history
        if self.messages:
            context_parts.append("=== Recent Conversation ===")
            for msg in self.messages[-10:]:  # Last 10 messages
                role = "User" if msg["role"] == "user" else "Assistant"
                context_parts.append(f"{role}: {msg['content'][:500]}")
        
        # Add current search context
        if self.current_search_results:
            context_parts.append(f"\n=== Current Search Results (Topic: {self.current_topic}) ===")
            for i, paper in enumerate(self.current_search_results, 1):
                context_parts.append(f"{i}. {paper.get('title', 'Untitled')} (ID: {paper.get('arxiv_id', 'N/A')})")
        
        return "\n".join(context_parts)
    
    def get_messages_for_api(self) -> List[Dict]:
        """Get messages formatted for OpenAI API"""
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in self.messages[-10:]
        ]
    
    def clear(self):
        """Clear conversation memory"""
        self.messages = []
        self.current_search_results = []
        self.current_topic = None


class LongTermMemory:
    """Long-term persistent memory using SQLite"""
    
    def __init__(self, db_manager):
        self.db = db_manager
        self._ensure_tables()
    
    def _ensure_tables(self):
        """Create user-related tables if they don't exist"""
        self.db.cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            name TEXT,
            interests TEXT,
            expertise_level TEXT DEFAULT 'intermediate',
            preferred_style TEXT DEFAULT 'detailed',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_active DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        self.db.cursor.execute("""
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            query TEXT NOT NULL,
            results_count INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES user_profiles(id)
        )
        """)
        
        self.db.cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            arxiv_id TEXT NOT NULL,
            is_saved BOOLEAN DEFAULT 0,
            is_liked BOOLEAN DEFAULT 0,
            notes TEXT,
            tags TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES user_profiles(id),
            UNIQUE(user_id, arxiv_id)
        )
        """)
        
        self.db.conn.commit()
        logger.info("Long-term memory tables initialized")

    def get_or_create_user(self, email: str, name: str = None) -> Dict:
        """Get existing user or create new one"""
        self.db.cursor.execute(
            "SELECT * FROM user_profiles WHERE email = ?", (email,)
        )
        row = self.db.cursor.fetchone()
        
        if row:
            # Update last active
            self.db.cursor.execute(
                "UPDATE user_profiles SET last_active = ? WHERE email = ?",
                (datetime.now(), email)
            )
            self.db.conn.commit()
            return dict(row)

        self.db.cursor.execute(
            "INSERT INTO user_profiles (email, name) VALUES (?, ?)",
            (email, name)
        )
        self.db.conn.commit()
        return self.get_user_by_email(email)
    
    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """Get user by email"""
        self.db.cursor.execute(
            "SELECT * FROM user_profiles WHERE email = ?", (email,)
        )
        row = self.db.cursor.fetchone()
        return dict(row) if row else None
    
    def update_user_interests(self, user_id: int, interests: List[str]):
        """Update user's research interests"""
        interests_json = json.dumps(interests)
        self.db.cursor.execute(
            "UPDATE user_profiles SET interests = ? WHERE id = ?",
            (interests_json, user_id)
        )
        self.db.conn.commit()
    
    def get_user_interests(self, user_id: int) -> List[str]:
        """Get user's research interests"""
        self.db.cursor.execute(
            "SELECT interests FROM user_profiles WHERE id = ?", (user_id,)
        )
        row = self.db.cursor.fetchone()
        if row and row[0]:
            return json.loads(row[0])
        return []

    def add_search(self, user_id: int, query: str, results_count: int):
        """Record a search in history"""
        self.db.cursor.execute(
            "INSERT INTO search_history (user_id, query, results_count) VALUES (?, ?, ?)",
            (user_id, query, results_count)
        )
        self.db.conn.commit()
    
    def get_search_history(self, user_id: int, limit: int = 10) -> List[Dict]:
        """Get user's recent search history"""
        self.db.cursor.execute("""
            SELECT query, results_count, timestamp 
            FROM search_history 
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (user_id, limit))
        
        return [
            {"query": row[0], "results_count": row[1], "timestamp": row[2]}
            for row in self.db.cursor.fetchall()
        ]
    
    def get_frequent_topics(self, user_id: int, limit: int = 5) -> List[str]:
        """Get user's most frequently searched topics"""
        self.db.cursor.execute("""
            SELECT query, COUNT(*) as count
            FROM search_history
            WHERE user_id = ?
            GROUP BY query
            ORDER BY count DESC
            LIMIT ?
        """, (user_id, limit))
        
        return [row[0] for row in self.db.cursor.fetchall()]

    def save_paper(self, user_id: int, arxiv_id: str) -> bool:
        """Save/bookmark a paper for user"""
        try:
            self.db.cursor.execute("""
                INSERT INTO user_papers (user_id, arxiv_id, is_saved)
                VALUES (?, ?, 1)
                ON CONFLICT(user_id, arxiv_id) 
                DO UPDATE SET is_saved = 1, timestamp = CURRENT_TIMESTAMP
            """, (user_id, arxiv_id))
            self.db.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving paper: {e}")
            return False
    
    def unsave_paper(self, user_id: int, arxiv_id: str) -> bool:
        """Remove paper from saved"""
        try:
            self.db.cursor.execute("""
                UPDATE user_papers SET is_saved = 0 
                WHERE user_id = ? AND arxiv_id = ?
            """, (user_id, arxiv_id))
            self.db.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error unsaving paper: {e}")
            return False
    
    def like_paper(self, user_id: int, arxiv_id: str, liked: bool = True) -> bool:
        """Like or unlike a paper"""
        try:
            self.db.cursor.execute("""
                INSERT INTO user_papers (user_id, arxiv_id, is_liked)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id, arxiv_id) 
                DO UPDATE SET is_liked = ?, timestamp = CURRENT_TIMESTAMP
            """, (user_id, arxiv_id, liked, liked))
            self.db.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error liking paper: {e}")
            return False
    
    def add_note(self, user_id: int, arxiv_id: str, note: str) -> bool:
        """Add or update note on a paper"""
        try:
            self.db.cursor.execute("""
                INSERT INTO user_papers (user_id, arxiv_id, notes)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id, arxiv_id) 
                DO UPDATE SET notes = ?, timestamp = CURRENT_TIMESTAMP
            """, (user_id, arxiv_id, note, note))
            self.db.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error adding note: {e}")
            return False
    
    def get_saved_papers(self, user_id: int) -> List[Dict]:
        """Get all saved papers for user"""
        self.db.cursor.execute("""
            SELECT up.arxiv_id, up.notes, up.timestamp, p.title, p.abstract
            FROM user_papers up
            LEFT JOIN papers p ON up.arxiv_id = p.arxiv_id
            WHERE up.user_id = ? AND up.is_saved = 1
            ORDER BY up.timestamp DESC
        """, (user_id,))
        
        return [
            {
                "arxiv_id": row[0],
                "notes": row[1],
                "saved_at": row[2],
                "title": row[3] or "Unknown",
                "abstract": row[4][:200] + "..." if row[4] else "No abstract"
            }
            for row in self.db.cursor.fetchall()
        ]
    
    def get_liked_papers(self, user_id: int) -> List[Dict]:
        """Get all liked papers for user"""
        self.db.cursor.execute("""
            SELECT up.arxiv_id, p.title, p.abstract
            FROM user_papers up
            LEFT JOIN papers p ON up.arxiv_id = p.arxiv_id
            WHERE up.user_id = ? AND up.is_liked = 1
            ORDER BY up.timestamp DESC
        """, (user_id,))
        
        return [
            {
                "arxiv_id": row[0],
                "title": row[1] or "Unknown",
                "abstract": row[2][:200] + "..." if row[2] else "No abstract"
            }
            for row in self.db.cursor.fetchall()
        ]
    
    def get_paper_note(self, user_id: int, arxiv_id: str) -> Optional[str]:
        """Get note for a specific paper"""
        self.db.cursor.execute("""
            SELECT notes FROM user_papers
            WHERE user_id = ? AND arxiv_id = ?
        """, (user_id, arxiv_id))
        
        row = self.db.cursor.fetchone()
        return row[0] if row else None
    
    def is_paper_saved(self, user_id: int, arxiv_id: str) -> bool:
        """Check if paper is saved by user"""
        self.db.cursor.execute("""
            SELECT is_saved FROM user_papers
            WHERE user_id = ? AND arxiv_id = ?
        """, (user_id, arxiv_id))
        
        row = self.db.cursor.fetchone()
        return bool(row and row[0])

    def get_user_context(self, user_id: int) -> str:
        """Get formatted user context for LLM"""
        context_parts = []
        
        # User profile
        self.db.cursor.execute(
            "SELECT name, interests, expertise_level FROM user_profiles WHERE id = ?",
            (user_id,)
        )
        profile = self.db.cursor.fetchone()
        
        if profile:
            context_parts.append(f"User: {profile[0] or 'Unknown'}")
            if profile[1]:
                interests = json.loads(profile[1])
                context_parts.append(f"Interests: {', '.join(interests)}")
            context_parts.append(f"Expertise: {profile[2]}")
        
        # Recent searches
        recent_searches = self.get_search_history(user_id, limit=5)
        if recent_searches:
            context_parts.append(f"\nRecent searches: {', '.join([s['query'] for s in recent_searches])}")
        
        # Saved papers count
        saved = self.get_saved_papers(user_id)
        context_parts.append(f"Saved papers: {len(saved)}")
        
        return "\n".join(context_parts)
