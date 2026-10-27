"""
Central Configuration for Advanced RAG System
Supports:
1. Hybrid Search + Reranking
2. GraphRAG with Neo4j
3. Evaluation with RAGAS
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if available
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

class Settings:
    # Base paths
    PROJECT_ROOT: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    SAMPLE_CORPUS_DIR: Path = DATA_DIR / "sample_corpus"
    CHROMA_PERSIST_DIR: Path = BASE_DIR / "storage" / "chromadb"
    EVAL_DATASET_PATH: Path = DATA_DIR / "evaluation_dataset.json"

    # Embedding and Reranker Models
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    RERANKER_MODEL_NAME: str = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # Chunking Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "400"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "80"))

    # Hybrid Retrieval Defaults
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "5"))
    HYBRID_ALPHA: float = float(os.getenv("HYBRID_ALPHA", "0.5"))  # 0.0 = Pure BM25, 1.0 = Pure Dense
    RRF_K: int = int(os.getenv("RRF_K", "60"))                      # Smoothing factor for RRF
    RERANK_TOP_N: int = int(os.getenv("RERANK_TOP_N", "3"))         # Top passages after reranking

    # Neo4j Graph Database Configuration
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME: str = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password123")
    NEO4J_DATABASE: str = os.getenv("NEO4J_DATABASE", "neo4j")
    USE_IN_MEMORY_GRAPH_FALLBACK: bool = os.getenv("USE_IN_MEMORY_GRAPH_FALLBACK", "true").lower() == "true"

    # LLM API Keys (optional; local intelligent fallback available)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    DEFAULT_LLM_PROVIDER: str = os.getenv("DEFAULT_LLM_PROVIDER", "auto")

    # RAGAS Configuration
    RAGAS_METRICS: list = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall"
    ]

settings = Settings()
