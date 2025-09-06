"""
Conversation storage and retrieval system for learning from user interactions.
This implements a RAG-based approach for improving AI responses over time.
"""

import json
import sqlite3
import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    LEARNING_ENABLED = True
except ImportError as e:
    print(f"Warning: Learning dependencies not available: {e}")
    print("Learning features will be disabled. Install with: pip install sentence-transformers numpy")
    LEARNING_ENABLED = False
    np = None
    SentenceTransformer = None


@dataclass
class Conversation:
    """Represents a single conversation interaction."""
    id: Optional[int] = None
    user_prompt: str = ""
    ai_response: str = ""
    user_feedback: Optional[str] = None  # "positive", "negative", "neutral"
    feedback_text: Optional[str] = None
    timestamp: Optional[str] = None
    session_id: Optional[str] = None
    context_used: Optional[str] = None
    message_order: int = 0  # Order of message in conversation

@dataclass
class ConversationSummary:
    """Represents a conversation summary for context."""
    session_id: str
    messages: List[Dict[str, str]]  # [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    summary: str = ""
    last_updated: Optional[str] = None


class ConversationStore:
    """Manages storage and retrieval of conversations for learning."""

    def __init__(self, db_path: str = "conversations.db"):
        self.db_path = Path(db_path)
        self.embedding_model = None
        self._init_database()
        self._init_embedding_model()

    def _init_database(self):
        """Initialize the SQLite database for storing conversations."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_prompt TEXT NOT NULL,
                ai_response TEXT NOT NULL,
                user_feedback TEXT,
                feedback_text TEXT,
                timestamp TEXT NOT NULL,
                session_id TEXT,
                context_used TEXT,
                message_order INTEGER DEFAULT 0,
                embedding BLOB
            )
        ''')

        # Create session summaries table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS session_summaries (
                session_id TEXT PRIMARY KEY,
                summary TEXT,
                message_count INTEGER DEFAULT 0,
                last_updated TEXT NOT NULL
            )
        ''')

        # Create feedback patterns table for learning
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                pattern_data TEXT NOT NULL,
                success_rate REAL DEFAULT 0.0,
                usage_count INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            )
        ''')

        conn.commit()
        conn.close()

    def _init_embedding_model(self):
        """Initialize the sentence transformer for semantic similarity."""
        if not LEARNING_ENABLED:
            self.embedding_model = None
            return

        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Learning system initialized with sentence embeddings")
        except Exception as e:
            print(f"Warning: Could not load embedding model: {e}")
            self.embedding_model = None

    def store_conversation(self, conversation: Conversation) -> int:
        """Store a conversation in the database."""
        if not conversation.timestamp:
            conversation.timestamp = datetime.datetime.now().isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Generate embedding for the prompt
        embedding = None
        if LEARNING_ENABLED and self.embedding_model and conversation.user_prompt:
            try:
                embedding_vector = self.embedding_model.encode(conversation.user_prompt)
                embedding = embedding_vector.tobytes()
            except Exception as e:
                print(f"Warning: Could not generate embedding: {e}")

        # Get the next message order for this session
        cursor.execute('SELECT COALESCE(MAX(message_order), 0) + 1 FROM conversations WHERE session_id = ?',
                      (conversation.session_id,))
        message_order = cursor.fetchone()[0]

        cursor.execute('''
            INSERT INTO conversations
            (user_prompt, ai_response, user_feedback, feedback_text,
             timestamp, session_id, context_used, message_order, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            conversation.user_prompt,
            conversation.ai_response,
            conversation.user_feedback,
            conversation.feedback_text,
            conversation.timestamp,
            conversation.session_id,
            conversation.context_used,
            message_order,
            embedding
        ))

        conversation_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return conversation_id

    def update_feedback(self, conversation_id: int, feedback: str, feedback_text: str = None):
        """Update user feedback for a conversation."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE conversations
            SET user_feedback = ?, feedback_text = ?
            WHERE id = ?
        ''', (feedback, feedback_text, conversation_id))

        conn.commit()
        conn.close()

    def get_similar_conversations(self, prompt: str, limit: int = 5) -> List[Conversation]:
        """Retrieve conversations similar to the given prompt."""
        if not LEARNING_ENABLED or not self.embedding_model:
            return self._get_keyword_similar_conversations(prompt, limit)

        try:
            # Generate embedding for the input prompt
            prompt_embedding = self.embedding_model.encode(prompt)

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT id, user_prompt, ai_response, user_feedback,
                       feedback_text, timestamp, session_id, context_used, embedding
                FROM conversations
                WHERE embedding IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT 100
            ''')

            conversations = []
            similarities = []

            for row in cursor.fetchall():
                if row[8]:  # Check if embedding exists
                    stored_embedding = np.frombuffer(row[8], dtype=np.float32)
                    similarity = np.dot(prompt_embedding, stored_embedding) / (
                        np.linalg.norm(prompt_embedding) * np.linalg.norm(stored_embedding)
                    )
                else:
                    similarity = 0.0  # No embedding available

                conversation = Conversation(
                    id=row[0], user_prompt=row[1], ai_response=row[2],
                    user_feedback=row[3], feedback_text=row[4],
                    timestamp=row[5], session_id=row[6], context_used=row[7]
                )

                conversations.append(conversation)
                similarities.append(similarity)

            conn.close()

            # Sort by similarity and return top results
            if similarities and conversations:
                sorted_pairs = sorted(zip(similarities, conversations), key=lambda x: x[0], reverse=True)
                return [conv for _, conv in sorted_pairs[:limit]]
            else:
                return conversations[:limit]

        except Exception as e:
            print(f"Error in similarity search: {e}")
            return self._get_keyword_similar_conversations(prompt, limit)

    def _get_keyword_similar_conversations(self, prompt: str, limit: int = 5) -> List[Conversation]:
        """Fallback method using keyword matching."""
        keywords = prompt.lower().split()
        if not keywords:
            return []

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Build a query to find conversations with similar keywords
        like_conditions = " OR ".join([f"LOWER(user_prompt) LIKE '%{keyword}%'" for keyword in keywords[:5]])

        cursor.execute(f'''
            SELECT id, user_prompt, ai_response, user_feedback,
                   feedback_text, timestamp, session_id, context_used
            FROM conversations
            WHERE {like_conditions}
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))

        conversations = []
        for row in cursor.fetchall():
            conversation = Conversation(
                id=row[0], user_prompt=row[1], ai_response=row[2],
                user_feedback=row[3], feedback_text=row[4],
                timestamp=row[5], session_id=row[6], context_used=row[7]
            )
            conversations.append(conversation)

        conn.close()
        return conversations

    def get_positive_examples(self, topic_keywords: List[str], limit: int = 3) -> List[Conversation]:
        """Get conversations with positive feedback for specific topics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        keyword_conditions = " OR ".join([f"LOWER(user_prompt) LIKE '%{kw.lower()}%'" for kw in topic_keywords])

        cursor.execute(f'''
            SELECT id, user_prompt, ai_response, user_feedback,
                   feedback_text, timestamp, session_id, context_used
            FROM conversations
            WHERE user_feedback = 'positive' AND ({keyword_conditions})
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))

        conversations = []
        for row in cursor.fetchall():
            conversation = Conversation(
                id=row[0], user_prompt=row[1], ai_response=row[2],
                user_feedback=row[3], feedback_text=row[4],
                timestamp=row[5], session_id=row[6], context_used=row[7]
            )
            conversations.append(conversation)

        conn.close()
        return conversations

    def get_conversation_stats(self) -> Dict:
        """Get statistics about stored conversations."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total conversations
        cursor.execute('SELECT COUNT(*) FROM conversations')
        total_conversations = cursor.fetchone()[0]

        # Active sessions (sessions with messages in last 24 hours)
        cursor.execute('''
            SELECT COUNT(DISTINCT session_id)
            FROM conversations
            WHERE datetime(timestamp) > datetime('now', '-1 day')
        ''')
        active_sessions = cursor.fetchone()[0]

        # Messages today
        cursor.execute('''
            SELECT COUNT(*)
            FROM conversations
            WHERE date(timestamp) = date('now')
        ''')
        messages_today = cursor.fetchone()[0]

        # Context usage rate (conversations with context vs without)
        cursor.execute('SELECT COUNT(*) FROM conversations WHERE context_used IS NOT NULL')
        conversations_with_context = cursor.fetchone()[0]

        context_usage_rate = 0
        if total_conversations > 0:
            context_usage_rate = round((conversations_with_context / total_conversations) * 100)

        # Session count
        cursor.execute('SELECT COUNT(DISTINCT session_id) FROM conversations')
        total_sessions = cursor.fetchone()[0]

        conn.close()

        return {
            'total_conversations': total_conversations,
            'active_sessions': active_sessions,
            'messages_today': messages_today,
            'context_usage_rate': context_usage_rate,
            'total_sessions': total_sessions,
            'avg_messages_per_session': round(total_conversations / max(total_sessions, 1), 1)
        }


    def get_session_context(self, session_id: str, max_messages: int = 10) -> List[Dict[str, str]]:
        """Get conversation history for a session in chat format."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT user_prompt, ai_response, message_order
            FROM conversations
            WHERE session_id = ?
            ORDER BY message_order ASC
            LIMIT ?
        ''', (session_id, max_messages))

        messages = []
        for row in cursor.fetchall():
            messages.append({"role": "user", "content": row[0]})
            messages.append({"role": "assistant", "content": row[1]})

        conn.close()
        return messages

    def get_conversation_summary(self, session_id: str) -> str:
        """Get or generate a summary of the conversation for context."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if we have a recent summary
        cursor.execute('''
            SELECT summary, message_count, last_updated
            FROM session_summaries
            WHERE session_id = ?
        ''', (session_id,))

        result = cursor.fetchone()

        # Get current message count
        cursor.execute('SELECT COUNT(*) FROM conversations WHERE session_id = ?', (session_id,))
        current_count = cursor.fetchone()[0]

        if result and result[1] == current_count and current_count > 0:
            # Summary is up to date
            conn.close()
            return result[0]

        # Generate new summary
        messages = self.get_session_context(session_id, max_messages=20)
        if not messages:
            conn.close()
            return ""

        # Create a simple summary
        summary = self._generate_conversation_summary(messages)

        # Store/update summary
        cursor.execute('''
            INSERT OR REPLACE INTO session_summaries
            (session_id, summary, message_count, last_updated)
            VALUES (?, ?, ?, ?)
        ''', (session_id, summary, current_count, datetime.datetime.now().isoformat()))

        conn.commit()
        conn.close()
        return summary

    def _generate_conversation_summary(self, messages: List[Dict[str, str]]) -> str:
        """Generate an AI-powered natural language summary of the conversation."""
        if not messages:
            return ""

        # Don't generate summary for very short conversations
        user_messages = [msg for msg in messages if msg["role"] == "user"]
        if len(user_messages) < 2:
            return ""

        # Build conversation text for AI summarization
        conversation_text = ""
        for msg in messages:
            role = "Human" if msg["role"] == "user" else "Assistant"
            conversation_text += f"{role}: {msg['content']}\n"

        # Truncate conversation if too long to fit in context
        if len(conversation_text) > 2000:  # Keep conversation manageable
            # Take first and last parts of conversation
            lines = conversation_text.split('\n')
            if len(lines) > 10:
                conversation_text = '\n'.join(lines[:5] + ['...'] + lines[-5:])

        # Create a factual conversation summary prompt
        summary_prompt = f"""You are creating a factual summary for internal use. Record exactly what happened in this conversation without filtering or sanitizing content. This is for context preservation, not public consumption.

FACTUAL RECORD - INCLUDE EVERYTHING:
- User identity: Record the user's name exactly as stated
- Topics discussed: List ALL topics mentioned, including cannabis, drugs, weeds, etc.
- User requests: Record what the user actually asked for (smoking weeds, garden weeds, etc.)
- Assistant responses: Note if assistant was overly cautious, gave inappropriate warnings, or misunderstood
- User reactions: Record if user was frustrated, told assistant to "take it easy", etc.
- Memory failures: Note when assistant forgot information it should have remembered
- Misunderstandings: Record any confusion or topic mix-ups

This is an internal factual record. Include controversial topics and problematic interactions exactly as they occurred.

CONVERSATION TO SUMMARIZE:
{conversation_text}

FACTUAL INTERNAL SUMMARY:"""

        try:
            # Use the AI model to generate the summary
            from llama_cpp import Llama
            import os
            from dotenv import load_dotenv
            load_dotenv()

            # Create a temporary model instance for summarization
            model_path = os.getenv('MODEL_PATH', './models/llama-2-7b-chat.gguf')

            # Check if we can reuse an existing model instance
            if hasattr(self, '_summary_model') and self._summary_model:
                model = self._summary_model
            else:
                # Create a lightweight model instance for summaries
                try:
                    model = Llama(
                        model_path=model_path,
                        n_ctx=1024,  # Smaller context for summaries
                        n_threads=2,  # Fewer threads
                        verbose=False
                    )
                    self._summary_model = model
                except Exception as e:
                    print(f"❌ Could not load model for summarization: {e}")
                    import traceback
                    traceback.print_exc()
                    return self._fallback_summary(messages)

            # Generate comprehensive AI summary with increased limits
            response = model(
                summary_prompt,
                max_tokens=200,  # Increased for 300-word summaries (~1.5 tokens per word)
                temperature=0.3,  # Slightly higher for more natural language
                stop=["\n\n", "Human:", "Assistant:", "CONVERSATION:", "SUMMARY:"]
            )

            summary = response['choices'][0]['text'].strip()

            # Clean up the summary
            if summary.startswith("FACTUAL INTERNAL SUMMARY"):
                summary = summary.split(":", 1)[1].strip() if ":" in summary else summary
            elif summary.startswith("COMPLETE SUMMARY"):
                summary = summary.split(":", 1)[1].strip() if ":" in summary else summary
            elif summary.startswith("NATURAL SUMMARY"):
                summary = summary.split(":", 1)[1].strip() if ":" in summary else summary
            elif summary.startswith("ACCURATE SUMMARY"):
                summary = summary.split(":", 1)[1].strip() if ":" in summary else summary
            elif summary.startswith("DETAILED SUMMARY"):
                summary = summary.split(":", 1)[1].strip() if ":" in summary else summary
            elif summary.startswith("COMPREHENSIVE SUMMARY"):
                summary = summary.split(":", 1)[1].strip() if ":" in summary else summary
            elif summary.startswith("Summary:"):
                summary = summary[8:].strip()

            # Enforce word count limit (300 words max)
            words = summary.split()
            if len(words) > 300:
                summary = " ".join(words[:300]) + "..."

            # Also enforce character limit as backup
            if len(summary) > 1200:  # ~300 words * 4 chars avg
                summary = summary[:1197] + "..."

            return summary if summary else self._fallback_summary(messages)

        except Exception as e:
            print(f"❌ Error generating AI summary: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_summary(messages)

    def _fallback_summary(self, messages: List[Dict[str, str]]) -> str:
        """Fallback summary generation if AI summarization fails."""
        user_messages = [msg for msg in messages if msg["role"] == "user"]

        # Extract basic info
        user_name = None
        topics = []

        for msg in user_messages:
            content = msg["content"].lower()

            # Extract name
            if "my name is" in content:
                name = content.split("my name is")[1].strip().split()[0].strip(".,!?")
                if name.isalpha():
                    user_name = name.title()

            # Extract topics
            if any(topic in content for topic in ["programming", "python", "ai", "machine learning", "data", "history", "war", "science"]):
                for topic in ["programming", "python", "AI", "machine learning", "data science", "history", "war", "science"]:
                    if topic.lower() in content and topic not in topics:
                        topics.append(topic)

        # Build fallback summary (always under 200 words)
        parts = []
        if user_name:
            parts.append(f"User {user_name}")
        if topics:
            # Limit topics to keep under 200 words
            topic_list = ', '.join(topics[:5])  # Max 5 topics
            parts.append(f"discussing {topic_list}")

        msg_count = len(user_messages)
        if parts:
            fallback = f"{' '.join(parts)} over {msg_count} messages."
        else:
            fallback = f"Conversation with {msg_count} user messages."

        # Ensure fallback is also under 300 words
        words = fallback.split()
        if len(words) > 300:
            fallback = " ".join(words[:300]) + "..."

        return fallback


# Global instance
conversation_store = ConversationStore()
