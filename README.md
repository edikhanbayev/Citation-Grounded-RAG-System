
A high-performance, multi-turn Retrieval-Augmented Generation (RAG) platform built with Flask, ChromaDB, BM25, PyMuPDF, and Groq LLMs. Features a 3-Tier Optimization Architecture, Selective Cache Invalidation, Sliding-Window Document Chunking, and an LLM-as-a-Judge Evaluation Suite.

### Retrieval Layer Architecture and Retrieval Pipeline
The engine uses a 3-tier hierarchy to optimize query latency, ensure retrieval precision, and minimize LLM API overhead:

┌───────────────────────────────┐
                  │    User Query (Standalone)    │
                  └───────────────┬───────────────┘
                                  │
      ┌───────────────────────────┴───────────────────────────┐
      ▼                                                       ▼
[Layer 1: Semantic Cache]                             [Layer 2: RAM BM25 Buffer]
Distance threshold <= 0.05                             In-memory tokenized corpus
Tagged for Selective Invalidation                      Sub-millisecond keyword lookup
      │ (Hit)                                                 │
      ├──> Return Cached Answer                               │
      │                                                       ▼
      └──> (Miss) ───────────────────────────► [Layer 3: Two-Tier Vector Store]
                                               Hot Tier (High-frequency docs)
                                               Archive Tier (Full repository)
                                                              │
                                                              ▼
                                               [Reciprocal Rank Fusion (RRF)]
                                               Merges Dense + Sparse Ranks
                                                              │
                                                              ▼
                                               [Groq LLaMA 3.3-70B Generation]


### Retrieval Layers

* Layer 1 (Semantic Cache): Caches vector embeddings of query-response pairs (<= 0.05 distance threshold). Cached entries are tagged with source file IDs (`source_files`) to allow 'selective cache invalidation' without flushing unrelated cached queries when a document changes.
* Layer 2 (In-Memory RAM Buffer):Loads document text chunks into RAM at startup for sub-millisecond BM25 lexical search without disk I/O bottlenecks.
* Layer 3 (Two-Tier Vector Store): Segregates high-frequency documents into a 'Hot Tier' for rapid access, automatically promoting documents when hit thresholds are reached.
* Reciprocal Rank Fusion (RRF): Combines dense vector distance ranks and sparse BM25 keyword scores to prevent hallucinations and improve precision.

---

## 📄 Document Chunking Strategy

Documents are processed and converted into vector embeddings using a page-aware 'Sliding Window Chunking Strategy' designed to preserve semantic context across chunk boundaries:

* Extraction Engine: Page-by-page text extraction using `PyMuPDF` (`fitz`).
* Window Size: `150 words` (~200 tokens) per chunk.
* Window Overlap: `30 words` overlap between consecutive chunks to ensure context (e.g., split sentences or definitions) isn't lost across chunk boundaries.
* Noise Filter: Chunks shorter than 30 characters are filtered out to exclude header/footer artifacts and empty lines.
* Metadata Association: Each generated chunk is tagged with crucial metadata attributes:
  * `source`: Original PDF filename.
  * `page`: Exact page number for precise UI citation mapping.
  * `clearance`: Security access tier (`public`, `internal`, `restricted`).

---

## 🛠 Tech Stack

* Web Framework: Flask, Flask-SQLAlchemy (SQLite), Flask-Login, Flask-WTF
* Vector Store & Embeddings: ChromaDB (`DefaultEmbeddingFunction`)
* Lexical Search: `rank_bm25` (BM25Okapi)
* Document Engine: PyMuPDF (`fitz`), Python `ThreadPoolExecutor` (Async background processing)
* LLM Integration & Judge: Groq API (`llama-3.3-70b-versatile`)
* Evaluation & Data Processing: Pandas, Requests, python-dotenv

---

## 📂 Project Structure

├── app/
│   ├── __init__.py           # Application factory & extensions setup
│   ├── models.py             # Database models & RagEngine (3-Tier RAG Engine)
│   ├── forms.py              # Login, Registration, and Document Upload forms
│   ├── templates/            # directory for HTML templates
│   ├── routes/
│   │   ├── admin.py          # Admin dashboard & source management
│   │   ├── auth.py           # Registration, approval, & authentication
│   │   └── chat.py           # Query execution, session memory, & citation handling
│   └── services/
│       └── document_service.py # Background async PDF processing & file cleanup
├── tests/                    # Evaluation & integration testing scripts
│   ├── eval_project.py       # Groq LLM-as-a-Judge dataset evaluator, custom made test
│   ├── eval_ragas.py         # RAGAS test for evaluating on 4 criteria
│   ├── test_ablation.py      # Hybrid search test VS dense, sparse tests
│   ├── test_multiturn.py     # HTTP multi-turn session and cache integration test
│   └── test_hot_tier.py      # Hot-tier promotion load
├── app.db                    # SQLAlchemy database storage (app.db)
├── uploads/                  # Raw uploaded PDF documents
├── chroma_db/                # Persistent vector database store
├── requirements.txt          # Project dependencies
├── config.py                 # Project configuration
└── setup.py                  # Application initial setup

## 🔐 Security & Access Clearances
The platform enforces strict Role-Based Access Control (RBAC) across document retrieval:

Role        Approval Required           Document Clearance Levels Accessible
Student     Auto-approved               public
Faculty     Requires Admin Verification public, internal
Admin       Manual System Setup         public, internal, restricted


## 🧪 Testing & Evaluation Suite

Rather than using basic unit testing frameworks, the project uses a three-part custom automated evaluation and integration harness:

1. LLM-as-a-Judge Evaluation (tests/test_eval.py)
Runs 16 benchmark handbook queries directly through the RagEngine pipeline and submits the output to llama-3.3-70b-versatile acting as an impartial evaluator. It grades system performance on a 0.0 to 1.0 scale across four core metrics:

Faithfulness: Groundedness of response strictly within retrieved context.

Answer Relevancy: Directness of the generated answer to the prompt.

Context Precision: Signal-to-noise ratio of retrieved context chunks.

Context Recall: Coverage of ground-truth facts within retrieved chunks.

Output: Generates a detailed breakdown saved to rag_evaluation_results.csv.

2. Multi-Turn Session & Cache Integration Test (tests/test_multiturn.py)
Simulates authenticated user sessions over HTTP to verify:

CSRF token extraction and session login flow.

Multi-turn query rewriting (e.g., resolving follow-ups like "Which floor is it on?" to "What floor is the Education Support Office located on?").

Cross-session Layer 1 semantic cache hits.

3. Hot-Tier Promotion Load Test (tests/test_load.py)
Fires sequential queries against specific document categories over HTTP to test access counter tracking, monitor response latency in milliseconds, and verify automatic document promotion into the Layer 3 Hot Collection.

## 🚀 Setup, Testing & Execution
1. Environment Setup
Create a .env file in the root directory:

SECRET_KEY=your-super-secret-key
GROQ_API_KEY=your-groq-api-key

2. Installation

# Clone repository and navigate to root
git clone <repository-url>
cd <repository-folder>

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

3. Run the Application

flask run
Access the web application at http://127.0.0.1:5000.

4. Running the Test Suite
Make sure the Flask application is running before executing HTTP integration tests.

 1. Run LLM-as-a-Judge Evaluation (generates rag_evaluation_results.csv)
python tests/test_eval.py

 2. Run Multi-Turn Context & Cache Test
python tests/test_multiturn.py

 3. Run Hot-Tier Promotion Load Test
python tests/test_load.py


