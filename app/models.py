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


class Student(db.Model, UserMixin):
    __tablename__ = 'student'
    id: Mapped[int] = mapped_column(db.Integer, primary_key=True)
    username: Mapped[str] = mapped_column(db.String, index=True, unique=True)
    email: Mapped[str] = mapped_column(db.String, index=True, unique=True)
    student_id: Mapped[str] = mapped_column(db.String, index=True, unique=True)
    password_hash: Mapped[str] = mapped_column(db.String)

    # Dynamic Access and Authorization Tiers
    role: Mapped[str] = mapped_column(db.String, default='student')  # student, faculty, admin
    is_approved: Mapped[bool] = mapped_column(db.Boolean, default=False)

    # Cascading relational relationship linking users to historical logs
    chat_messages: Mapped[list['ChatMessage']] = relationship(
        "ChatMessage",
        back_populates="user",
        cascade="all, delete-orphan"
    )

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class ChatMessage(db.Model):
    __tablename__ = 'chat_message'
    id: Mapped[int] = mapped_column(db.Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(db.Integer, db.ForeignKey('student.id'))
    sender: Mapped[str] = mapped_column(db.String)  # 'user' or 'bot'
    message: Mapped[str] = mapped_column(db.Text)
    timestamp: Mapped[datetime] = mapped_column(db.DateTime, default=datetime.utcnow)

    user = relationship("Student", back_populates="chat_messages")


@login.user_loader
def load_user(id):
    return db.session.get(Student, int(id))


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
        Queries ChromaDB checking for content records matching authorized metadata,
        packages context results, and routes safely to the Groq Cloud API Gateway.
        """
        db_results = self.collection.query(
            query_texts=[question],
            n_results=3,
            where={"clearance": {"$in": allowed_clearances}}
        )

        # Guard clause: ensure valid baseline vectors exist in the return payload
        if not db_results or not db_results['documents'] or not db_results['documents'][0]:
            return "I am sorry, but I cannot locate relevant documentation parameters in my verified database context.", "None"

        context_string = "\n".join(db_results['documents'][0])
        meta = db_results['metadatas'][0][0]
        citation = f"{meta['source']} (Page {meta['page']})"

        system_instruction = (
            "You are an academic regulations assistant. Answer the user's question relying strictly on the provided context. "
            "If the answer cannot be found within the background context, reply that you cannot locate the information."
        )

        # 🚀 Fetch Groq cloud environment token credentials
        api_key = os.environ.get('GROQ_API_KEY')
        if not api_key:
            return "Configuration Error: GROQ_API_KEY missing from system .env file.", "None"

        # 🚀 Set up Groq HTTP Headers & OpenAI compatible request body mapping
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
            # 🚀 Forward context transaction directly onto Groq cloud servers
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=30
            )

            if response.status_code == 200:
                # 🚀 Parse out message choices matching OpenAI standard responses
                return response.json()['choices'][0]['message']['content'], citation
            else:
                return f"Cloud Ingestion Warning: Groq API returned status code {response.status_code}. Detail: {response.text}", "None"
        except Exception as e:
            return f"Cloud Connection Outage: Unable to connect to Groq endpoints. Detail: {str(e)}", "None"