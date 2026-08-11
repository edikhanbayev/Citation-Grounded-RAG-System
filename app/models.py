import os, re, json, time, requests, chromadb, uuid
from datetime import datetime, timezone
from collections import defaultdict
from werkzeug.security import generate_password_hash, check_password_hash
import sqlalchemy.orm as so
import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship
from flask_login import UserMixin
from app import db, login
from chromadb.utils import embedding_functions
import fitz  # PyMuPDF engine
from rank_bm25 import BM25Okapi

STOP_WORDS = {
    "how", "to", "use", "the", "a", "of", "and", "is", "for", "on", "in",
    "at", "by", "with", "about", "an", "it", "this", "that", "your", "can", "you", "i",
    "when", "where", "what", "who", "which", "why", "whose", "whom", "are", "was", "were",
    "do", "does", "did", "could", "would", "should", "has", "have", "had", "been", "will", "shall"
}

class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True, nullable=False)
    email = db.Column(db.String(120), index=True, unique=True, nullable=False)
    student_id = db.Column(db.String(32), unique=True, nullable=True)
    role = db.Column(db.String(20), default='student', nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    is_approved = db.Column(db.Boolean, default=False, nullable=False)

    conversations = db.relationship('Conversation', backref='user', lazy='dynamic', cascade="all, delete-orphan")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Conversation(db.Model):
    __tablename__ = 'conversations'

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title = db.Column(db.String(100), nullable=False, default="New Conversation")
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    messages = db.relationship('ChatMessage', backref='conversation', lazy='dynamic', cascade="all, delete-orphan")


class Document(db.Model):
    __tablename__ = 'documents'

    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), unique=True, nullable=False)
    clearance_level = db.Column(db.String(50), default='public', nullable=False)
    status = db.Column(db.String(50), default='Processing', nullable=False)
    uploaded_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    citations = db.relationship('Citation', backref='document', lazy='dynamic', cascade="all, delete")


class ChatMessage(db.Model):
    __tablename__ = 'chat_messages'

    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.String(36), db.ForeignKey('conversations.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    sender = db.Column(db.String(10), nullable=False)
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, index=True, default=lambda: datetime.now(timezone.utc), nullable=False)

    citations = db.relationship('Citation', backref='message', lazy='dynamic', cascade="all, delete-orphan")


class Citation(db.Model):
    __tablename__ = 'citations'

    id = db.Column(db.Integer, primary_key=True)
    message_id = db.Column(db.Integer, db.ForeignKey('chat_messages.id'), nullable=False)
    document_id = db.Column(db.Integer, db.ForeignKey('documents.id', ondelete='SET NULL'), nullable=True)
    page_number = db.Column(db.Integer, nullable=False)


class RagEngine:
    def __init__(self):
        base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        self.chroma_path = os.path.join(base_dir, 'chroma_db')
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        self.default_ef = embedding_functions.DefaultEmbeddingFunction()

        self.archive_collection = self.chroma_client.get_or_create_collection(
            "uni_regs_archive",
            embedding_function=self.default_ef
        )
        self.hot_collection = self.chroma_client.get_or_create_collection(
            "uni_regs_hot",
            embedding_function=self.default_ef
        )

        self._init_cache_collection()

        self.doc_access_counts = defaultdict(int)
        self.HOT_PROMOTION_THRESHOLD = 5
        self.hot_filenames = set()

        # Hot tier: Load hot tier state from persistent disk storage on startup
        try:
            hot_payload = self.hot_collection.get(include=['metadatas'])
            if hot_payload and hot_payload.get('metadatas'):
                self.hot_filenames = {m['source'] for m in hot_payload['metadatas'] if m and 'source' in m}
                print(f"[*] [Hot Tier] Hydrated {len(self.hot_filenames)} active hot document source(s) from disk.")
        except Exception as e:
            print(f"[!] [Hot Tier] State hydration exception: {str(e)}")

        self.in_memory_bm25 = None
        self.in_memory_corpus_docs = []
        self.in_memory_corpus_metas = []
        self.in_memory_corpus_ids = []

        self._refresh_ram_buffer()

    def _init_cache_collection(self):
        self.cache_collection = self.chroma_client.get_or_create_collection(
            "query_response_cache",
            embedding_function=self.default_ef
        )

    @staticmethod
    def tokenize_text(doc):
        if not doc or not isinstance(doc, str):
            return []
        return re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}|[a-zA-Z0-9]+', doc.lower())

    def _refresh_ram_buffer(self):
        try:
            payload = self.archive_collection.get()
            docs = payload.get('documents', []) if payload else []
            metas = payload.get('metadatas', []) if payload else []
            ids = payload.get('ids', []) if payload else []

            if docs:
                self.in_memory_corpus_docs = docs
                self.in_memory_corpus_metas = metas
                self.in_memory_corpus_ids = ids

                tokenized_corpus = []
                for doc in self.in_memory_corpus_docs:
                    tokens = [w for w in self.tokenize_text(doc) if w not in STOP_WORDS]
                    if not tokens:
                        tokens = self.tokenize_text(doc)
                    tokenized_corpus.append(tokens)

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
        for filename in cited_filenames:
            self.doc_access_counts[filename] += 1
            count = self.doc_access_counts[filename]

            if count >= self.HOT_PROMOTION_THRESHOLD and filename not in self.hot_filenames:
                print(f"[*] PROMOTION TRIGGERED: '{filename}' reached {count} hits.")
                self.hot_filenames.add(filename)

                payload = self.archive_collection.get(where={"source": filename})
                if payload and payload['documents']:
                    # FIX 2 (HOT TIER): Use upsert to safely write without duplicate ID errors
                    self.hot_collection.upsert(
                        documents=payload['documents'],
                        metadatas=payload['metadatas'],
                        ids=payload['ids']
                    )

    def _check_semantic_cache(self, search_query, allowed_clearances, distance_threshold=0.18):
        # (CACHE RBAC): Validate user's allowed clearances against required clearance tier
        try:
            results = self.cache_collection.query(query_texts=[search_query], n_results=1)
            if results and results['ids'] and results['ids'][0]:
                distance = results['distances'][0][0]
                if distance <= distance_threshold:
                    metadata = results['metadatas'][0][0]

                    # RBAC Clearance Enforcement
                    required_clearance = metadata.get('required_clearance', 'restricted')
                    if required_clearance not in allowed_clearances:
                        print(f"[*] [Layer 1] CACHE HIT BLOCKED! Query requires '{required_clearance}' clearance, user has {allowed_clearances}.")
                        return None, None

                    cached_answer = metadata.get('answer', results['documents'][0][0])
                    citations = json.loads(metadata.get('citations', '[]'))
                    print(f"[*] [Layer 1] CACHE HIT! Query Distance ({distance:.4f}) <= Threshold ({distance_threshold}).")
                    return cached_answer, citations
        except Exception as e:
            if "does not exist" in str(e).lower():
                self._init_cache_collection()
            else:
                print(f"[!] [Layer 1] Cache lookup exception: {str(e)}")
        return None, None

    def _write_semantic_cache(self, search_query, answer, citations):
        # CACHE RBAC: Store the highest clearance level required by citations into cache metadata
        try:
            cache_id = f"cache_{uuid.uuid4()}"
            source_files = ",".join(sorted(list(set(c['filename'] for c in citations if isinstance(c, dict) and c.get('filename')))))

            cited_clearances = [c.get('clearance', 'public') for c in citations if isinstance(c, dict)]
            if 'restricted' in cited_clearances:
                required_clearance = 'restricted'
            elif 'internal' in cited_clearances:
                required_clearance = 'internal'
            else:
                required_clearance = 'public'

            self.cache_collection.add(
                documents=[search_query],
                metadatas=[{
                    "original_query": search_query,
                    "answer": answer,
                    "citations": json.dumps(citations),
                    "source_files": source_files,
                    "required_clearance": required_clearance,
                    "cached_at": datetime.now(timezone.utc).isoformat()
                }],
                ids=[cache_id]
            )
            print(f"[*] [Layer 1] Saved query-response pair to cache ({cache_id}) [Required Clearance: '{required_clearance}'].")
        except Exception as e:
            print(f"[!] [Layer 1] Cache write exception: {str(e)}")

    def invalidate_cache_for_source(self, filename):
        try:
            cached_items = self.cache_collection.get(include=['metadatas'])
            if cached_items and cached_items.get('metadatas'):
                ids_to_delete = []
                for cache_id, meta in zip(cached_items['ids'], cached_items['metadatas']):
                    if meta and 'source_files' in meta:
                        sources = [s.strip() for s in meta['source_files'].split(',') if s.strip()]
                        if filename in sources:
                            ids_to_delete.append(cache_id)

                if ids_to_delete:
                    self.cache_collection.delete(ids=ids_to_delete)
                    print(f"[*] [Layer 1] Purged {len(ids_to_delete)} cache entries for '{filename}'.")
        except Exception as e:
            print(f"[!] Selective cache invalidation exception: {str(e)}")

    def process_and_index_pdf(self, pdf_path, clearance_level="public"):
        if not os.path.exists(pdf_path):
            print(f"Ingestion Alert: Specified target path does not exist: {pdf_path}")
            return

        filename = os.path.basename(pdf_path)
        doc = fitz.open(pdf_path)
        chunk_count = 0

        WINDOW_SIZE = 150
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
        print(f"Processed '{filename}' with clearance '{clearance_level}': Vectorized {chunk_count} chunks.")

        self.invalidate_cache_for_source(filename)
        self._refresh_ram_buffer()

    def delete_source(self, filename):
        try:
            self.archive_collection.delete(where={"source": filename})
            try:
                self.hot_collection.delete(where={"source": filename})
            except Exception:
                pass

            if filename in self.hot_filenames:
                self.hot_filenames.remove(filename)

            # HOT TIER: Reset hit tracking counter on file deletion
            self.doc_access_counts.pop(filename, None)

            self.invalidate_cache_for_source(filename)
            self._refresh_ram_buffer()
            return True
        except Exception as e:
            print(f"Error deleting from ChromaDB: {str(e)}")
            return False

    def get_all_sources(self):
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

            return [{"filename": fname, "clearance": tier} for fname, tier in sorted(source_dict.items())]
        except Exception as e:
            print(f"Error fetching ChromaDB sources: {str(e)}")
            return []

    def condense_query(self, question, chat_history, api_key):
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

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
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

    def _format_chunk_payload(self, idx: int) -> dict:
        # CACHE RBAC: Include chunk clearance in formatted payload dictionary
        return {
            "filename": self.in_memory_corpus_metas[idx].get('source'),
            "page": self.in_memory_corpus_metas[idx].get('page', 1),
            "clearance": self.in_memory_corpus_metas[idx].get('clearance', 'public'),
            "text": self.in_memory_corpus_docs[idx]
        }

    def retrieve(self, query: str, allowed_clearances: list, mode: str = "hybrid", top_k: int = 4) -> list:
        if not self.in_memory_corpus_docs:
            return []

        # 1. DENSE RETRIEVAL (Query Hot Tier First, Fall Back to Primary Archive)
        dense_indices_with_ranks = []
        if mode in ("hybrid", "dense_only"):
            vector_results = None
            HOT_SIMILARITY_THRESHOLD = 0.85  # Strict distance cutoff for hot collection hits
            L2_CUTOFF_THRESHOLD = 1.40

            # HOT TIER: Query hot collection first if hot documents exist
            if self.hot_filenames:
                try:
                    hot_query_res = self.hot_collection.query(
                        query_texts=[query],
                        n_results=15,
                        where={"clearance": {"$in": allowed_clearances}}
                    )
                    if hot_query_res and hot_query_res['ids'] and hot_query_res['ids'][0]:
                        raw_distances = hot_query_res['distances'][0]
                        if raw_distances and min(raw_distances) <= HOT_SIMILARITY_THRESHOLD:
                            print(f"[*] [Hot Tier] HIT! Best similarity distance: {min(raw_distances):.4f} <= {HOT_SIMILARITY_THRESHOLD}")
                            vector_results = hot_query_res
                        else:
                            print(f"[*] [Hot Tier] LOW CONFIDENCE ({min(raw_distances):.4f} > {HOT_SIMILARITY_THRESHOLD}). Falling back to Archive.")
                except Exception as e:
                    print(f"[!] [Hot Tier] Query exception, falling back to Archive: {str(e)}")

            # Fall back to primary archive if Hot Tier produced no high-confidence hits
            if vector_results is None:
                vector_results = self.archive_collection.query(
                    query_texts=[query],
                    n_results=15,
                    where={"clearance": {"$in": allowed_clearances}}
                )

            id_to_idx_map = {cid: idx for idx, cid in enumerate(self.in_memory_corpus_ids)}

            if vector_results and vector_results['ids'] and vector_results['ids'][0]:
                raw_ids = vector_results['ids'][0]
                raw_distances = vector_results['distances'][0]

                valid_rank = 1
                for chunk_id, dist in zip(raw_ids, raw_distances):
                    if dist > L2_CUTOFF_THRESHOLD:
                        continue
                    if chunk_id in id_to_idx_map:
                        global_idx = id_to_idx_map[chunk_id]
                        dense_indices_with_ranks.append((global_idx, valid_rank))
                        valid_rank += 1

            if mode == "dense_only":
                top_dense_indices = [idx for idx, _ in dense_indices_with_ranks[:top_k]]
                return [self._format_chunk_payload(idx) for idx in top_dense_indices]

        # 2. SPARSE RETRIEVAL (BM25 Lexical Store)
        sparse_indices_with_ranks = []
        if mode in ("hybrid", "sparse_only"):
            valid_indices = [
                i for i, meta in enumerate(self.in_memory_corpus_metas)
                if meta.get('clearance') in allowed_clearances
            ]

            tokenized_query = [w for w in self.tokenize_text(query) if w not in STOP_WORDS]
            if not tokenized_query:
                tokenized_query = self.tokenize_text(query)

            raw_bm25_scores = self.in_memory_bm25.get_scores(tokenized_query)
            sorted_sparse_indices = [
                i for i in sorted(valid_indices, key=lambda i: raw_bm25_scores[i], reverse=True)
                if raw_bm25_scores[i] > 0
            ]

            sparse_indices_with_ranks = [(idx, rank + 1) for rank, idx in enumerate(sorted_sparse_indices[:15])]

            if mode == "sparse_only":
                top_sparse_indices = [idx for idx, _ in sparse_indices_with_ranks[:top_k]]
                return [self._format_chunk_payload(idx) for idx in top_sparse_indices]

        # 3. HYBRID RECIPROCAL RANK FUSION (RRF)
        rrf_scores = defaultdict(float)
        K_CONSTANT = 60

        for global_idx, rank in sparse_indices_with_ranks:
            rrf_scores[global_idx] += 1.0 / (K_CONSTANT + rank)

        for global_idx, rank in dense_indices_with_ranks:
            rrf_scores[global_idx] += 1.0 / (K_CONSTANT + rank)

        fused_sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        top_fused_indices = fused_sorted_indices[:top_k]

        return [self._format_chunk_payload(idx) for idx in top_fused_indices]

    def generate_response(self, question: str, retrieved_chunks: list, chat_history: list = None, api_key: str = None) -> tuple:
        if not retrieved_chunks:
            return "I am sorry, I don't have relevant information.", []

        valid_documents = []
        chunk_ref_map = {}

        # (CACHE RBAC): Pass chunk clearance into chunk_ref_map for cache writing
        for i, chunk in enumerate(retrieved_chunks, start=1):
            filename = chunk.get('filename')
            page = chunk.get('page', 1)
            clearance = chunk.get('clearance', 'public')
            doc_text = chunk.get('text', '')

            ref_tag = f"REF{i}"
            chunk_ref_map[ref_tag] = {"filename": filename, "page": page, "clearance": clearance}
            valid_documents.append(f"[{ref_tag}] Document: {filename} (Page {page})\n{doc_text}")

        context_string = "\n\n".join(valid_documents)
        system_instruction = (
            "You are an academic regulations assistant. Answer the user's question relying strictly on the provided context.\n"
            "CRITICAL INSTRUCTION: For any statement derived from the context, append its reference tag (e.g., [REF1], [REF2]) at the end of the sentence.\n"
            "If the answer cannot be found within the provided context, or if the context is insufficient, you MUST reply with EXACTLY:\n"
            "\"I am sorry, I don't have relevant information.\" and nothing else."
        )

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
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

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=30
                )

                if response.status_code == 200:
                    answer_text = response.json()['choices'][0]['message']['content']

                    negative_indicators = [
                        "don't have relevant information",
                        "do not have relevant information",
                        "cannot locate",
                        "cannot find",
                        "not mentioned",
                        "provided context does not",
                        "no relevant documentation"
                    ]

                    if any(phrase in answer_text.lower() for phrase in negative_indicators):
                        return "I am sorry, I don't have relevant information.", []

                    referenced_tags = set(re.findall(r'\[REF(\d+)\]', answer_text))

                    actual_citations = []
                    for tag_num in sorted(referenced_tags, key=int):
                        ref_key = f"REF{tag_num}"
                        if ref_key in chunk_ref_map:
                            cit = chunk_ref_map[ref_key]
                            if cit not in actual_citations:
                                actual_citations.append(cit)

                    clean_answer_text = re.sub(r'\s*\[REF\d+\]', '', answer_text)
                    return clean_answer_text, actual_citations

                elif response.status_code == 429:
                    time.sleep((attempt + 1) * 3)
                else:
                    return "I am sorry, I don't have relevant information.", []

            except Exception as e:
                if attempt == max_retries - 1:
                    return f"Cloud Connection Outage: {str(e)}", []
                time.sleep(2)

        return "I am sorry, I don't have relevant information.", []

    def search_and_generate(self, question, allowed_clearances, chat_history=None, mode="hybrid"):
        api_key = os.environ.get('GROQ_API_KEY')
        if not api_key:
            return "Configuration Error: GROQ_API_KEY missing from system .env file.", [], []

        search_query = self.condense_query(question, chat_history, api_key)
        print(f"[*] Original Question: {question}")
        print(f"[*] Rewritten Standalone Query: {search_query}")

        # CACHE RBAC: Pass allowed_clearances directly to semantic cache checker
        cached_answer, cached_citations = self._check_semantic_cache(search_query, allowed_clearances)
        if cached_answer is not None:
            return cached_answer, cached_citations, []

        retrieved_chunks = self.retrieve(search_query, allowed_clearances, mode=mode, top_k=4)
        if not retrieved_chunks:
            return "I am sorry, I don't have relevant information.", [], []

        cited_filenames = {chunk['filename'] for chunk in retrieved_chunks if chunk.get('filename')}
        self._track_and_promote_hot_docs(cited_filenames)

        answer_text, actual_citations = self.generate_response(question, retrieved_chunks, chat_history, api_key)

        negative_indicators = ["don't have relevant information", "cannot locate", "cannot find", "cloud connection outage"]
        is_negative = any(phrase in answer_text.lower() for phrase in negative_indicators)

        if not is_negative and actual_citations:
            self._write_semantic_cache(search_query, answer_text, actual_citations)

        return answer_text, actual_citations, retrieved_chunks

    def clear_all_caches(self):
        try:
            try:
                self.chroma_client.delete_collection("query_response_cache")
            except Exception:
                pass
            self._init_cache_collection()
            self._refresh_ram_buffer()
            return True
        except Exception as e:
            print(f"[!] Cache flush exception: {str(e)}")
            return False

    def clear_hot_documents(self) -> bool:
        """
        Completely purges all documents from the hot tier collection on disk
        and resets in-memory tracking state.
        """
        try:
            self.chroma_client.delete_collection("uni_regs_hot")
            self.hot_collection = self.chroma_client.get_or_create_collection(
                "uni_regs_hot",
                embedding_function=self.default_ef
            )

            self.hot_filenames.clear()
            self.doc_access_counts.clear()

            print("[*] [Hot Tier] Successfully cleared all hot documents and reset access counters.")
            return True

        except Exception as e:
            print(f"[!] Exception occurred while clearing hot documents: {str(e)}")
            return False