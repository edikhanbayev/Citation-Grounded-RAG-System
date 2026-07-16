from werkzeug.security import generate_password_hash, check_password_hash
import sqlalchemy.orm as so
import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship
from flask_login import UserMixin
from app import db, login
import os
import pypdf
import requests
import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime
import fitz  # PyMuPDF engine
import uuid


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


# ==========================================
# RETRIEVAL-AUGMENTED GENERATION (RAG) ENGINE
# ==========================================
class RagEngine:
    def __init__(self):
        base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        self.chroma_path = os.path.join(base_dir, 'chroma_db')
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        self.default_ef = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.chroma_client.get_or_create_collection(
            "uni_regs",
            embedding_function=self.default_ef
        )

    def process_and_index_pdf(self, pdf_path, clearance_level="public"):
        """Parses target PDF assets page by page using PyMuPDF and registers vectors in ChromaDB."""
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

                    self.collection.add(
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
        print(f"Processed '{filename}': Vectorized {chunk_count} chunks.")

    def condense_query(self, question, chat_history, api_key):
        """
        Query Rewriting Pipeline.
        Examines chat history and the current question to generate a standalone query.
        """
        if not chat_history:
            return question  # Fallback directly to the raw question if history is empty

        history_str = ""
        for turn in chat_history[-3:]:  # Limit to the last 3 exchanges to keep rewrite hyper-focused
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

        return question  # Graceful fallback

    def search_and_generate(self, question, allowed_clearances, chat_history=None):
        """Executes parallel Dense and Sparse lookups, fuses rankings, and forwards context to Groq."""
        from rank_bm25 import BM25Okapi

        api_key = os.environ.get('GROQ_API_KEY')
        if not api_key:
            return "Configuration Error: GROQ_API_KEY missing from system .env file.", []

        # PRE-RETRIEVAL RUNTIME: Condense current user query using isolated history context
        search_query = self.condense_query(question, chat_history, api_key)
        print(f"[*] Original Question: {question}")
        print(f"[*] Rewritten Standalone Query: {search_query}")

        # STEP 1: Fetch matching clearance documents
        corpus_payload = self.collection.get(
            where={"clearance": {"$in": allowed_clearances}}
        )

        if not corpus_payload or not corpus_payload['documents']:
            return "I am sorry, but I cannot locate relevant documentation parameters in my verified database context.", []

        corpus_docs = corpus_payload['documents']
        corpus_metas = corpus_payload['metadatas']
        corpus_ids = corpus_payload['ids']

        # STEP 2: Lexical BM25 Scoring with Noise/Stop-Word Filtering
        #  FIX: Expanded list to filter out common question interrogatives as well
        STOP_WORDS = {
            "how", "to", "use", "the", "a", "of", "and", "is", "for", "on", "in",
            "at", "by", "with", "about", "an", "it", "this", "that", "your", "can", "you", "i",
            "when", "where", "what", "who", "which", "why", "whose", "whom", "are", "was", "were",
            "do", "does", "did", "could", "would", "should", "has", "have", "had", "been", "will", "shall"
        }
        tokenized_query = [word for word in search_query.lower().split() if word not in STOP_WORDS]

        # Fall back to the original standalone query if stripping leaves it completely empty
        if not tokenized_query:
            tokenized_query = search_query.lower().split()

        tokenized_corpus = [doc.lower().split() for doc in corpus_docs]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(tokenized_query)

        # Only allow index candidates with a STRICTLY POSITIVE score (> 0)
        sparse_ranked_indices = [
            i for i in sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
            if bm25_scores[i] > 0
        ]

        # STEP 3: Dense Vector Lookup (Using rewritten query)
        db_results = self.collection.query(
            query_texts=[search_query],
            n_results=20,
            where={"clearance": {"$in": allowed_clearances}}
        )

        # STEP 4: Reciprocal Rank Fusion (RRF) Fusing with Vector Veto Guardrails
        rrf_scores = {}
        K_CONSTANT = 60
        DISTANCE_THRESHOLD = 1.13
        banned_global_indices = set()  # 🔥 Track verified semantic noise to filter out BM25 leaks

        # Map BM25 Ranks (Only score elements that actually matched keywords)
        for rank, local_idx in enumerate(sparse_ranked_indices[:20]):
            rrf_scores[local_idx] = rrf_scores.get(local_idx, 0.0) + (1.0 / (K_CONSTANT + (rank + 1)))

        # Map Vector Ranks
        if db_results and db_results['ids'] and db_results['ids'][0]:
            raw_ids = db_results['ids'][0]
            raw_docs = db_results['documents'][0]
            raw_metas = db_results['metadatas'][0]
            raw_distances = db_results.get('distances', [[]])[0]

            for rank, chunk_id in enumerate(raw_ids):
                dist = raw_distances[rank] if rank < len(raw_distances) else 0.0
                meta = raw_metas[rank]

                print(
                    f"[*] Vector Match Check -> File: {meta.get('source')} | Page: {meta.get('page')} | Distance Score: {dist:.4f}")

                if chunk_id in corpus_ids:
                    global_idx = corpus_ids.index(chunk_id)
                    if dist <= DISTANCE_THRESHOLD:
                        rrf_scores[global_idx] = rrf_scores.get(global_idx, 0.0) + (1.0 / (K_CONSTANT + (rank + 1)))
                    else:
                        print(
                            f"[!] Noise Detected: Suppressing chunk from {meta.get('source')} (Score {dist:.4f} exceeds {DISTANCE_THRESHOLD})")
                        banned_global_indices.add(global_idx)

        #  VECTOR VETO ACTIVATION: Purge any index from RRF scores that vector search flagged as noise
        for banned_idx in banned_global_indices:
            if banned_idx in rrf_scores:
                del rrf_scores[banned_idx]

        # STEP 5: Sort and slice Top 4 context blocks
        fused_sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        top_fused_indices = fused_sorted_indices[:4]

        valid_documents = []
        citations_list = []
        seen_sources = set()

        for idx in top_fused_indices:
            valid_documents.append(corpus_docs[idx])
            meta = corpus_metas[idx]

            filename = meta.get('source')
            page = meta.get('page', 1)

            if filename:
                source_tag = (filename, page)
                if source_tag not in seen_sources:
                    seen_sources.add(source_tag)
                    citations_list.append({
                        "filename": filename,
                        "page": page
                    })

        if not valid_documents:
            return "I am sorry, but I cannot locate relevant documentation parameters in my verified database context.", []

        context_string = "\n\n".join(valid_documents)

        # STEP 6: Execute generation with chat context memory injected
        system_instruction = (
            "You are an academic regulations assistant. Answer the user's question relying strictly on the provided context. "
            "If the answer cannot be found within the background context, reply that you cannot locate the information."
        )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Inject chat context memory directly into the final generation messages array
        messages = [{"role": "system", "content": system_instruction}]

        if chat_history:
            for turn in chat_history[-5:]:  # Pass last 5 exchanges to maintain memory boundary safety
                messages.append({"role": "user", "content": turn.get("user", "")})
                messages.append({"role": "assistant", "content": turn.get("bot", "")})

        # Final question containing context
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
                return response.json()['choices'][0]['message']['content'], citations_list
            else:
                return f"Cloud Ingestion Warning: Groq API returned status code {response.status_code}. Detail: {response.text}", []
        except Exception as e:
            return f"Cloud Connection Outage: Unable to connect to Groq endpoints. Detail: {str(e)}", []

    def get_all_sources(self):
        """Fetches all unique file sources and their respective clearance tiers from ChromaDB."""
        try:
            results = self.collection.get(include=['metadatas'])
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

    def delete_source(self, filename):
        """Purges all vector chunks associated with a specific file."""
        try:
            self.collection.delete(where={"source": filename})
            return True
        except Exception as e:
            print(f"Error deleting from ChromaDB: {str(e)}")
            return False