from werkzeug.security import generate_password_hash, check_password_hash
import sqlalchemy.orm as so
import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship
from flask_login import UserMixin
from app import db, login
import os
import json
import requests
import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime
from collections import defaultdict
import fitz  # PyMuPDF engine
import uuid
from rank_bm25 import BM25Okapi


class User(UserMixin, db.Model):
    """Semantic normalization: Replaces the 'Student' model to accurately encapsulate all roles."""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True, nullable=False)
    email = db.Column(db.String(120), index=True, unique=True, nullable=False)
    student_id = db.Column(db.String(32), unique=True, nullable=True)
    role = db.Column(db.String(20), default='student', nullable=False)  # student, faculty, admin
    password_hash = db.Column(db.String(256), nullable=False)
    is_approved = db.Column(db.Boolean, default=False, nullable=False)

    # Relationships
    conversations = db.relationship('Conversation', backref='user', lazy='dynamic', cascade="all, delete-orphan")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Conversation(db.Model):
    """Encapsulates multi-turn conversational threads to isolate search history scopes."""
    __tablename__ = 'conversations'

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title = db.Column(db.String(100), nullable=False, default="New Conversation")
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    messages = db.relationship('ChatMessage', backref='conversation', lazy='dynamic', cascade="all, delete-orphan")


class Document(db.Model):
    """New Asset Inventory Table: Pairs SQLite directly with physical OS and ChromaDB files."""
    __tablename__ = 'documents'

    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), unique=True, nullable=False)
    clearance_level = db.Column(db.String(50), default='public', nullable=False)
    status = db.Column(db.String(50), default='Processing', nullable=False)  # Processing, Ready, Failed
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    citations = db.relationship('Citation', backref='document', lazy='dynamic', cascade="all, delete")


class ChatMessage(db.Model):
    """Maintains clean logs strictly limited to conversational texts."""
    __tablename__ = 'chat_messages'

    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.String(36), db.ForeignKey('conversations.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)  # Retained for legacy schema safety
    sender = db.Column(db.String(10), nullable=False)  # user, bot
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow, nullable=False)

    # Relationships
    citations = db.relationship('Citation', backref='message', lazy='dynamic', cascade="all, delete-orphan")


class Citation(db.Model):
    """Normalized Intersection Table: Resolves structural mapping between messages and documents."""
    __tablename__ = 'citations'

    id = db.Column(db.Integer, primary_key=True)
    message_id = db.Column(db.Integer, db.ForeignKey('chat_messages.id'), nullable=False)

    # ON DELETE SET NULL ensures old chat history text is preserved even if the source document is purged
    document_id = db.Column(db.Integer, db.ForeignKey('documents.id', ondelete='SET NULL'), nullable=True)
    page_number = db.Column(db.Integer, nullable=False)


# =========================================================================
# RETRIEVAL-AUGMENTED GENERATION (RAG) ENGINE - 3-TIER OPTIMIZED ARCHITECTURE
# =========================================================================
class RagEngine:
    def __init__(self):
        base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        self.chroma_path = os.path.join(base_dir, 'chroma_db')
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        self.default_ef = embedding_functions.DefaultEmbeddingFunction()

        # =====================================================================
        # LAYER 3: TWO-TIER VECTOR STORAGE COLLECTIONS
        # =====================================================================
        # Primary full-archive vector collection
        self.archive_collection = self.chroma_client.get_or_create_collection(
            "uni_regs_archive",
            embedding_function=self.default_ef
        )
        # Hot-tier collection containing only high-frequency document chunks
        self.hot_collection = self.chroma_client.get_or_create_collection(
            "uni_regs_hot",
            embedding_function=self.default_ef
        )

        # =====================================================================
        # LAYER 1: SEMANTIC QUERY CACHE COLLECTION
        # =====================================================================
        self._init_cache_collection()

        # =====================================================================
        # LAYER 2: IN-MEMORY BUFFER & FREQUENCY TRACKER STATE
        # =====================================================================
        self.doc_access_counts = defaultdict(int)  # Tracks access hits per filename
        self.HOT_PROMOTION_THRESHOLD = 5  # Accesses before promoting to Hot Collection
        self.hot_filenames = set()  # Active hot files

        # In-memory RAM state for BM25 to avoid Disk I/O
        self.in_memory_bm25 = None
        self.in_memory_corpus_docs = []
        self.in_memory_corpus_metas = []
        self.in_memory_corpus_ids = []

        # Build RAM BM25 buffer on startup
        self._refresh_ram_buffer()

    def _init_cache_collection(self):
        """Safely fetches or creates the active Layer 1 semantic query cache collection handle."""
        self.cache_collection = self.chroma_client.get_or_create_collection(
            "query_response_cache",
            embedding_function=self.default_ef
        )

    # -------------------------------------------------------------------------
    # LAYER 2 HELPERS: RAM BUFFERING & DYNAMIC HOT PROMOTION
    # -------------------------------------------------------------------------
    def _refresh_ram_buffer(self):
        """Loads all documents from Archive collection into RAM for sub-millisecond BM25 scoring."""
        try:
            payload = self.archive_collection.get()
            if payload and payload['documents']:
                self.in_memory_corpus_docs = payload['documents']
                self.in_memory_corpus_metas = payload['metadatas']
                self.in_memory_corpus_ids = payload['ids']

                STOP_WORDS = {
                    "how", "to", "use", "the", "a", "of", "and", "is", "for", "on", "in",
                    "at", "by", "with", "about", "an", "it", "this", "that", "your", "can", "you", "i",
                    "when", "where", "what", "who", "which", "why", "whose", "whom", "are", "was", "were",
                    "do", "does", "did", "could", "would", "should", "has", "have", "had", "been", "will", "shall"
                }

                tokenized_corpus = [
                    [w for w in doc.lower().split() if w not in STOP_WORDS]
                    for doc in self.in_memory_corpus_docs
                ]
                self.in_memory_bm25 = BM25Okapi(tokenized_corpus)
                print(f"[*] [Layer 2] Refreshed RAM BM25 Buffer: {len(self.in_memory_corpus_docs)} chunks indexed.")
            else:
                self.in_memory_bm25 = None
                self.in_memory_corpus_docs = []
                self.in_memory_corpus_metas = []
                self.in_memory_corpus_ids = []
        except Exception as e:
            print(f"[!] [Layer 2] RAM buffer build exception: {str(e)}")

    def _track_and_promote_hot_docs(self, cited_filenames):
        """Increments file hit counts and promotes high-frequency documents to Layer 3 Hot Collection."""
        for filename in cited_filenames:
            self.doc_access_counts[filename] += 1
            count = self.doc_access_counts[filename]

            if count >= self.HOT_PROMOTION_THRESHOLD and filename not in self.hot_filenames:
                print(f"[*] [Layer 2 & 3] PROMOTION TRIGGERED: '{filename}' reached {count} hits. Syncing to Hot Tier.")
                self.hot_filenames.add(filename)

                payload = self.archive_collection.get(where={"source": filename})
                if payload and payload['documents']:
                    self.hot_collection.add(
                        documents=payload['documents'],
                        metadatas=payload['metadatas'],
                        ids=payload['ids']
                    )
                    print(f"[*] [Layer 3] Added {len(payload['documents'])} chunks of '{filename}' into Hot Tier.")

    # -------------------------------------------------------------------------
    # LAYER 1 HELPERS: SEMANTIC QUERY CACHE
    # -------------------------------------------------------------------------
    def _check_semantic_cache(self, search_query, distance_threshold=0.15):
        """Layer 1: Checks cache vector store. Distance <= 0.15 means near-identical meaning."""
        try:
            results = self.cache_collection.query(query_texts=[search_query], n_results=1)
            if results and results['ids'] and results['ids'][0]:
                distance = results['distances'][0][0]
                if distance <= distance_threshold:
                    metadata = results['metadatas'][0][0]
                    # Retrieve the cached answer from metadata (or fallback to document for backward compatibility)
                    cached_answer = metadata.get('answer', results['documents'][0][0])
                    citations = json.loads(metadata.get('citations', '[]'))
                    print(
                        f"[*] [Layer 1] CACHE HIT! Query Distance ({distance:.4f}) <= Threshold ({distance_threshold}).")
                    return cached_answer, citations
        except Exception as e:
            if "does not exist" in str(e).lower():
                print("[!] Stale cache handle detected in lookup. Re-binding Layer 1 collection...")
                self._init_cache_collection()
            else:
                print(f"[!] [Layer 1] Cache lookup exception: {str(e)}")
        return None, None

    def _write_semantic_cache(self, search_query, answer, citations):
        """Layer 1: Saves query-response pair to vector cache with dynamic stale-handle recovery."""
        try:
            cache_id = f"cache_{uuid.uuid4()}"
            self.cache_collection.add(
                documents=[search_query],  # Vectorize search_query so vector search compares Query vs Query
                metadatas=[{
                    "original_query": search_query,
                    "answer": answer,        # Store generated answer string inside metadata
                    "citations": json.dumps(citations),
                    "cached_at": datetime.utcnow().isoformat()
                }],
                ids=[cache_id]
            )
            print(f"[*] [Layer 1] Saved query-response pair to cache ({cache_id}).")
        except Exception as e:
            if "does not exist" in str(e).lower():
                print("[!] Stale cache handle detected in write. Re-binding Layer 1 collection...")
                self._init_cache_collection()
                try:
                    cache_id = f"cache_{uuid.uuid4()}"
                    self.cache_collection.add(
                        documents=[search_query],  # Vectorize search_query
                        metadatas=[{
                            "original_query": search_query,
                            "answer": answer,       # Store generated answer string inside metadata
                            "citations": json.dumps(citations),
                            "cached_at": datetime.utcnow().isoformat()
                        }],
                        ids=[cache_id]
                    )
                    print(f"[*] [Layer 1] Saved query-response pair to cache after recovery ({cache_id}).")
                except Exception as retry_e:
                    print(f"[!] [Layer 1] Cache write retry exception: {str(retry_e)}")
            else:
                print(f"[!] [Layer 1] Cache write exception: {str(e)}")

    # -------------------------------------------------------------------------
    # DOCUMENT INGESTION & DELETION HANDLERS
    # -------------------------------------------------------------------------
    def process_and_index_pdf(self, pdf_path, clearance_level="public"):
        """Parses target PDF assets page by page using PyMuPDF and registers vectors in Archive Collection."""
        if not os.path.exists(pdf_path):
            print(f"Ingestion Alert: Specified target path does not exist: {pdf_path}")
            return

        filename = os.path.basename(pdf_path)
        doc = fitz.open(pdf_path)
        chunk_count = 0

        WINDOW_SIZE = 150  # ~200 tokens
        WINDOW_OVERLAP = 30

        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            if not text or not text.strip():
                continue
            words = text.split()

            i = 0
            while i < len(words):
                chunk_words = words[i: i + WINDOW_SIZE]
                clean_chunk = " ".join(chunk_words)

                if len(clean_chunk) > 30:
                    unique_chunk_id = f"{filename}_p{page_num + 1}_c{chunk_count}"

                    self.archive_collection.add(
                        documents=[clean_chunk],
                        metadatas=[{
                            "source": filename,
                            "page": page_num + 1,
                            "clearance": clearance_level
                        }],
                        ids=[unique_chunk_id]
                    )
                    chunk_count += 1

                i += (WINDOW_SIZE - WINDOW_OVERLAP)

        doc.close()
        print(f"Processed '{filename}' with a clearance '{clearance_level}': Vectorized {chunk_count} chunks into Archive Store.")

        # Flush outdated caches and rebuild RAM BM25 buffer
        self.clear_all_caches()

    def delete_source(self, filename):
        """Purges all vector chunks associated with a specific file from both vector tiers and flushes cache."""
        try:
            self.archive_collection.delete(where={"source": filename})
            try:
                self.hot_collection.delete(where={"source": filename})
            except Exception:
                pass

            if filename in self.hot_filenames:
                self.hot_filenames.remove(filename)

            self.clear_all_caches()
            return True
        except Exception as e:
            print(f"Error deleting from ChromaDB: {str(e)}")
            return False

    def get_all_sources(self):
        """Fetches all unique file sources and their respective clearance tiers from ChromaDB Archive."""
        try:
            results = self.archive_collection.get(include=['metadatas'])
            if not results or 'metadatas' not in results or not results['metadatas']:
                return []

            source_dict = {}
            for meta in results['metadatas']:
                if meta and 'source' in meta:
                    filename = meta['source']
                    clearance = meta.get('clearance', 'public')
                    source_dict[filename] = clearance

            unique_sources = [
                {"filename": fname, "clearance": tier}
                for fname, tier in sorted(source_dict.items())
            ]
            return unique_sources
        except Exception as e:
            print(f"Error fetching ChromaDB sources: {str(e)}")
            return []

    def condense_query(self, question, chat_history, api_key):
        """
        Query Rewriting Pipeline.
        Examines chat history and current question to generate a standalone query.
        """
        if not chat_history:
            return question

        history_str = ""
        for turn in chat_history[-3:]:
            history_str += f"User: {turn.get('user', '')}\nAssistant: {turn.get('bot', '')}\n"

        condense_prompt = (
            "Given the following conversation history and a follow-up question, "
            "rephrase the follow-up question to be a standalone search query. "
            "Do NOT answer the question. Only output the rephrased search query. "
            "If it is already standalone, output it exactly as-is.\n\n"
            f"Chat History:\n{history_str}\n"
            f"Follow-up Question: {question}\n"
            "Standalone Query:"
        )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": condense_prompt}],
            "temperature": 0.0
        }

        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=5
            )
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"[!] Error in query condensation: {str(e)}")

        return question

    # -------------------------------------------------------------------------
    # MAIN SEARCH & GENERATION ENGINE (INTEGRATED WITH 3-TIER ARCHITECTURE)
    # -------------------------------------------------------------------------
    def search_and_generate(self, question, allowed_clearances, chat_history=None):
        """Executes 3-Tier optimized retrieval, RRF with Vector Veto, and forwards to Groq API."""
        api_key = os.environ.get('GROQ_API_KEY')
        if not api_key:
            return "Configuration Error: GROQ_API_KEY missing from system .env file.", []

        # PRE-RETRIEVAL RUNTIME: Condense query using conversational context
        search_query = self.condense_query(question, chat_history, api_key)
        print(f"[*] Original Question: {question}")
        print(f"[*] Rewritten Standalone Query: {search_query}")

        # =====================================================================
        # TIER 1 CHECK: SEMANTIC CACHE LOOKUP
        # =====================================================================
        cached_answer, cached_citations = self._check_semantic_cache(search_query)
        if cached_answer is not None:
            return cached_answer, cached_citations

        # =====================================================================
        #  TIER 3 CHECK: TWO-TIERED VECTOR ROUTING
        # =====================================================================
        vector_results = None
        used_hot_tier = False

        if self.hot_collection.count() > 0:
            hot_results = self.hot_collection.query(
                query_texts=[search_query],
                n_results=10,
                where={"clearance": {"$in": allowed_clearances}}
            )
            # Evaluate whether Hot Tier yielded high-confidence match (Distance <= 1.10)
            if hot_results and hot_results['distances'] and hot_results['distances'][0]:
                best_hot_dist = hot_results['distances'][0][0]
                if best_hot_dist <= 1.10:
                    vector_results = hot_results
                    used_hot_tier = True
                    print(f"[*] [Layer 3] ROUTING HIT: Answered via Hot Collection (Distance: {best_hot_dist:.4f}).")

        # Fall back to Full Archive Collection if Hot Tier misses or is empty
        if not used_hot_tier:
            vector_results = self.archive_collection.query(
                query_texts=[search_query],
                n_results=20,
                where={"clearance": {"$in": allowed_clearances}}
            )
            print("[*] [Layer 3] ROUTING FALLBACK: Executed search across Full Archive Vector Store.")

        # =====================================================================
        #  TIER 2 CHECK: IN-MEMORY BM25 SCORING
        # =====================================================================
        if not self.in_memory_corpus_docs:
            return "I am sorry, I don't have relevant information", []

        # Filter RAM indices by user clearance tier
        valid_indices = [
            i for i, meta in enumerate(self.in_memory_corpus_metas)
            if meta.get('clearance') in allowed_clearances
        ]

        STOP_WORDS = {
            "how", "to", "use", "the", "a", "of", "and", "is", "for", "on", "in",
            "at", "by", "with", "about", "an", "it", "this", "that", "your", "can", "you", "i",
            "when", "where", "what", "who", "which", "why", "whose", "whom", "are", "was", "were",
            "do", "does", "did", "could", "would", "should", "has", "have", "had", "been", "will", "shall"
        }
        tokenized_query = [w for w in search_query.lower().split() if w not in STOP_WORDS]
        if not tokenized_query:
            tokenized_query = search_query.lower().split()

        raw_bm25_scores = self.in_memory_bm25.get_scores(tokenized_query)
        sparse_ranked_indices = [
            i for i in sorted(valid_indices, key=lambda i: raw_bm25_scores[i], reverse=True)
            if raw_bm25_scores[i] > 0
        ]

        # =====================================================================
        # RRF AGGREGATION & VECTOR VETO GUARDRAILS
        # =====================================================================
        rrf_scores = {}
        K_CONSTANT = 60
        DISTANCE_THRESHOLD = 1.25
        banned_global_indices = set()

        for rank, local_idx in enumerate(sparse_ranked_indices[:20]):
            rrf_scores[local_idx] = rrf_scores.get(local_idx, 0.0) + (1.0 / (K_CONSTANT + (rank + 1)))

        if vector_results and vector_results['ids'] and vector_results['ids'][0]:
            raw_ids = vector_results['ids'][0]
            raw_metas = vector_results['metadatas'][0]
            raw_distances = vector_results.get('distances', [[]])[0]

            for rank, chunk_id in enumerate(raw_ids):
                dist = raw_distances[rank] if rank < len(raw_distances) else 0.0
                meta = raw_metas[rank]

                print(
                    f"[*] Vector Match Check -> File: {meta.get('source')} | Page: {meta.get('page')} | Distance Score: {dist:.4f}")

                if chunk_id in self.in_memory_corpus_ids:
                    global_idx = self.in_memory_corpus_ids.index(chunk_id)
                    if dist <= DISTANCE_THRESHOLD:
                        rrf_scores[global_idx] = rrf_scores.get(global_idx, 0.0) + (1.0 / (K_CONSTANT + (rank + 1)))
                    else:
                        print(
                            f"[!] Noise Suppressed: File {meta.get('source')} (Score {dist:.4f} > {DISTANCE_THRESHOLD})")
                        banned_global_indices.add(global_idx)

        # Apply Vector Veto
        for banned_idx in banned_global_indices:
            if banned_idx in rrf_scores:
                del rrf_scores[banned_idx]

        fused_sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        top_fused_indices = fused_sorted_indices[:4]

        valid_documents = []
        citations_list = []
        cited_filenames = set()

        for idx in top_fused_indices:
            valid_documents.append(self.in_memory_corpus_docs[idx])
            meta = self.in_memory_corpus_metas[idx]
            filename = meta.get('source')
            page = meta.get('page', 1)

            if filename:
                cited_filenames.add(filename)
                source_tag = (filename, page)
                if source_tag not in [(c['filename'], c['page']) for c in citations_list]:
                    citations_list.append({"filename": filename, "page": page})

        if not valid_documents:
            return "I am sorry, I don't have relevant information.", []

        # Trigger dynamic promotion logic for hit documents
        self._track_and_promote_hot_docs(cited_filenames)

        # =====================================================================
        # GROQ LLM RESPONSE GENERATION
        # =====================================================================
        context_string = "\n\n".join(valid_documents)
        system_instruction = (
            "You are an academic regulations assistant. Answer the user's question relying strictly on the provided context. "
            "If the answer cannot be found within the provided context, or if the context is insufficient, you MUST reply with EXACTLY: "
            "\"I am sorry. I don't have relevant information\" and nothing else."
        )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        messages = [{"role": "system", "content": system_instruction}]

        if chat_history:
            for turn in chat_history[-5:]:
                messages.append({"role": "user", "content": turn.get("user", "")})
                messages.append({"role": "assistant", "content": turn.get("bot", "")})

        messages.append({
            "role": "user",
            "content": f"Context:\n{context_string}\n\nQuestion: {question}"
        })

        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": messages,
            "temperature": 0.0
        }

        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=30
            )

            if response.status_code == 200:
                answer_text = response.json()['choices'][0]['message']['content']

                #  NEGATIVE RESPONSE GUARD: Do not cache "information not found" fallbacks
                negative_indicators = [
                    "cannot locate",
                    "cannot find",
                    "not mentioned",
                    "provided context does not",
                    "no relevant documentation"
                ]

                is_negative_answer = any(phrase in answer_text.lower() for phrase in negative_indicators)

                if not is_negative_answer:
                    #  TIER 1 WRITE: Store only successful answers in cache
                    self._write_semantic_cache(search_query, answer_text, citations_list)
                else:
                    print("[*] [Layer 1] Skipped caching negative/fallback response.")

                return answer_text, citations_list
            else:
                return f"Cloud Ingestion Warning: Groq API returned status code {response.status_code}. Detail: {response.text}", []
        except Exception as e:
            return f"Cloud Connection Outage: Unable to connect to Groq endpoints. Detail: {str(e)}", []

    # -------------------------------------------------------------------------
    # FLUSH AND INVALIDATION HANDLER
    # -------------------------------------------------------------------------
    def clear_all_caches(self):
        """Flushes Layer 1 semantic query cache and rebuilds Layer 2 RAM BM25 index."""
        try:
            try:
                self.chroma_client.delete_collection("query_response_cache")
            except Exception:
                pass  # Safely ignore if collection didn't exist

            # Re-bind cache collection handle
            self._init_cache_collection()

            # Refresh RAM BM25 buffer
            self._refresh_ram_buffer()
            print("[*] All cache layers successfully invalidated and synchronized.")
            return True
        except Exception as e:
            print(f"[!] Cache flush exception: {str(e)}")
            return False