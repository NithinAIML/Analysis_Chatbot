## config.py

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI API configuration
OPENAI_API_KEY = ""
EMBEDDING_MODEL = "text-embedding-3-large"
GENERATION_MODEL = "gpt-4o"

# Document processing configuration
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 8000
CHUNK_SIZE_STEP = 100
MIN_CHUNK_OVERLAP = 0
MAX_CHUNK_OVERLAP = 500
CHUNK_OVERLAP_STEP = 50

# Retrieval configuration
DEFAULT_TOP_K = 5
MAX_TOP_K = 20
SIMILARITY_THRESHOLD = 0.7
RERANKING_ENABLED = True

# Memory configuration
MEMORY_K = 10
MAX_TOKENS_LIMIT = 4000

# File storage paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")

# Evaluation configuration
RETRIEVAL_METRICS = ["precision", "recall", "f1", "mrr"]
GENERATION_METRICS = ["bleu", "rouge", "bertscore"]

# UI configuration
THEME_COLOR = "#7E56C2"
MAX_CHAT_HISTORY = 20