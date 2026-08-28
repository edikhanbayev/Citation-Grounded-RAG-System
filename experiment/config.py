import os
from pathlib import Path

from dotenv import load_dotenv


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# LOAD .env FROM PROJECT ROOT
# ============================================================

ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "openai/gpt-oss-120b"

GROQ_API_URL = (
    "https://api.groq.com/openai/v1/chat/completions"
)

TEMPERATURE = 0.0


# ============================================================
# API KEYS
# ============================================================

RAG_API_KEY = os.getenv(
    "GROQ_RAG_API_KEY"
)

PLAIN_LLM_API_KEY = os.getenv(
    "GROQ_PLAIN_API_KEY"
)


# ============================================================
# EXPERIMENT
# ============================================================

MAX_QUESTIONS = 33


# ============================================================
# GROQ LIMITS
# ============================================================

RPM_LIMIT = 30
RPD_LIMIT = 1000

TPM_LIMIT = 8000
TPD_LIMIT = 200000


# ============================================================
# SAFETY
# ============================================================

MIN_REQUEST_INTERVAL = 4.0

MAX_RETRIES = 4

REQUEST_TIMEOUT = 60


# ============================================================
# RESULTS
# ============================================================

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "results"
)