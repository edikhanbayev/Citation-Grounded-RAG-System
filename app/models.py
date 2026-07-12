from werkzeug.security import generate_password_hash, check_password_hash
import sqlalchemy.orm as so
import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship
from flask_login import UserMixin
from app import db, login
import os
import pypdf  # Actively utilized below by the ingestion engine to parse PDFs
import requests
import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime

class User(UserMixin, db.Model):
    """Semantic normalization: Replaces the 'Student' model to accurately encapsulate all roles."""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True, nullable=False)
    email = db.Column(db.String(120), index=True, unique=True, nullable=False)
    student_id = db.Column(db.String(32), unique=True, nullable=True)  # Nullable for non-student accounts
    role = db.Column(db.String(20), default='student', nullable=False)  # student, faculty, admin
    password_hash = db.Column(db.String(256), nullable=False)
    is_approved = db.Column(db.Boolean, default=False, nullable=False)

    # Relationships
    messages = db.relationship('ChatMessage', backref='author', lazy='dynamic', cascade="all, delete-orphan")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


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
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
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

@login.user_loader
def load_user(id):
    return db.session.get(User, int(id))


# ==========================================
# RETRIEVAL-AUGMENTED GENERATION (RAG) ENGINE
# ==========================================
class RagEngine:
    def __init__(self):
        # Establish vector engine storage tracks safely relative to current file paths
        base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        self.chroma_path = os.path.join(base_dir, 'chroma_db')
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        self.default_ef = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.chroma_client.get_or_create_collection(
            "uni_regs",
            embedding_function=self.default_ef
        )

    def process_and_index_pdf(self, pdf_path, clearance_level="public"):
        """
        Actively parses target PDF assets page by page using pypdf,
        extracts text contexts, and registers vectors in ChromaDB.
        """
        if not os.path.exists(pdf_path):
            print(f"Ingestion Alert: Specified target path does not exist: {pdf_path}")
            return

        filename = os.path.basename(pdf_path)
        reader = pypdf.PdfReader(pdf_path)
        chunk_count = 0

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text or not text.strip():
                continue

            # Chunking strategies: split page contents by paragraphs
            paragraphs = text.split('\n\n')
            for para in paragraphs:
                clean_para = para.strip()
                if len(clean_para) < 30:
                    continue  # Filter out empty lines, page numbers, or short headers

                # Generate a strict unique ID for each chunked slice
                unique_chunk_id = f"{filename}_p{page_num + 1}_c{chunk_count}"

                # Commit text embeddings and structural metadata directly to ChromaDB
                self.collection.add(
                    documents=[clean_para],
                    metadatas=[{
                        "source": filename,
                        "page": page_num + 1,
                        "clearance": clearance_level
                    }],
                    ids=[unique_chunk_id]
                )
                chunk_count += 1

        print(
            f"Successfully processed '{filename}': Vectorized {chunk_count} chunks under '{clearance_level}' clearance.")

    def search_and_generate(self, question, allowed_clearances):
        """
        Queries ChromaDB, applies a strict vector distance threshold cutoff
        to filter out irrelevant padding chunks, and forwards high-confidence
        contexts to the Groq Cloud API Gateway.
        """
        db_results = self.collection.query(
            query_texts=[question],
            n_results=4,
            where={"clearance": {"$in": allowed_clearances}}
        )

        # Guard clause: ensure valid baseline vectors exist in the return payload
        if not db_results or not db_results['documents'] or not db_results['documents'][0]:
            return "I am sorry, but I cannot locate relevant documentation parameters in my verified database context.", []

        # Extract raw matched parameters from the vector space return payload
        raw_docs = db_results['documents'][0]
        raw_metas = db_results['metadatas'][0]
        raw_distances = db_results.get('distances', [[]])[0]  # 🚀 Extract vector distance arrays

        valid_documents = []
        citations_list = []
        seen_sources = set()

        # 🎯 SEMANTIC GUARDRAIL THRESHOLD
        # For Chroma's default L2 space, 1.15 is the ideal dividing line
        # to separate high-confidence answers from legacy padding noise.
        # Chunks with distances greater than 1.15 are treated as irrelevant noise and dropped.
        DISTANCE_THRESHOLD = 1.15

        for i in range(len(raw_docs)):
            doc = raw_docs[i]
            meta = raw_metas[i]
            # Safeguard array bounds for distance outputs
            dist = raw_distances[i] if i < len(raw_distances) else 0.0

            # Terminal diagnostic printout so you can track exact scores in real time
            print(
                f"[*] Vector Match Check -> File: {meta.get('source')} | Page: {meta.get('page')} | Distance Score: {dist:.4f}")

            # 🚀 Dynamic Trimming: Only accept chunks that pass our strict distance threshold
            if dist <= DISTANCE_THRESHOLD:
                valid_documents.append(doc)

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
            else:
                print(
                    f"[!] Noise Detected: Suppressing filler chunk from {meta.get('source')} (Score {dist:.4f} exceeds threshold {DISTANCE_THRESHOLD})")

        # Secondary guard check: if no chunks were strong enough to pass the threshold cutoff
        if not valid_documents:
            return "I am sorry, but I cannot locate relevant documentation parameters in my verified database context.", []

        # Assemble the cleaner context string consisting ONLY of high-confidence results
        context_string = "\n\n".join(valid_documents)

        system_instruction = (
            "You are an academic regulations assistant. Answer the user's question relying strictly on the provided context. "
            "If the answer cannot be found within the background context, reply that you cannot locate the information."
        )

        # Fetch Groq cloud environment token credentials
        api_key = os.environ.get('GROQ_API_KEY')
        if not api_key:
            return "Configuration Error: GROQ_API_KEY missing from system .env file.", []

        # Set up Groq HTTP Headers & OpenAI compatible request body mapping
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Context:\n{context_string}\n\nQuestion: {question}"}
            ],
            "temperature": 0.0
        }

        try:
            # Forward context transaction directly onto Groq cloud servers
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
            # Grab raw metadatas for all entries in the collection
            results = self.collection.get(include=['metadatas'])
            if not results or 'metadatas' not in results or not results['metadatas']:
                return []

            # Extract both source name and clearance using a dictionary to keep unique records
            source_dict = {}
            for meta in results['metadatas']:
                if meta and 'source' in meta:
                    filename = meta['source']
                    clearance = meta.get('clearance', 'public')  # Fallback default to public
                    source_dict[filename] = clearance

            # Convert to a sorted list of dictionaries for easier rendering in Jinja
            unique_sources = [
                {"filename": fname, "clearance": tier}
                for fname, tier in sorted(source_dict.items())
            ]
            return unique_sources

        except Exception as e:
            print(f"Error fetching ChromaDB sources with clearance: {str(e)}")
            return []

    def delete_source(self, filename):
        """Completely purges all vector chunks associated with a specific file."""
        try:
            self.collection.delete(where={"source": filename})
            return True
        except Exception as e:
            print(f"Error deleting from ChromaDB: {str(e)}")
            return False