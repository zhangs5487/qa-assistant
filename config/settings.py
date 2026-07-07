"""Application configuration — single source of truth for all settings.

Uses Pydantic Settings v2. Values are loaded from environment variables,
.env file, and YAML overrides (in that order of precedence).
"""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Top-level settings for the QA Assistant project."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Environment ----
    environment: Literal["dev", "prod", "test"] = "dev"
    debug: bool = False

    # ---- Paths ----
    data_dir: str = "./data"
    raw_data_dir: str = "./data/raw"
    clean_data_dir: str = "./data/clean"
    models_cache_dir: str = "./models"  # Local embedding model cache

    # ---- Crawler ----
    crawl_delay: float = 1.0  # seconds between requests to the same domain
    crawl_max_pages: int = 1000
    crawl_user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    )
    proxy_list: list[str] = []
    obey_robots_txt: bool = True
    concurrent_requests: int = 4

    # ---- Storage (MongoDB) ----
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "qa_assistant"
    file_store_type: Literal["local", "minio"] = "local"
    # MinIO (only used when file_store_type == "minio")
    minio_endpoint: str = ""
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_bucket: str = "qa-raw-docs"

    # ---- Milvus (Vector DB) ----
    # Dev: local file path for Milvus Lite (NOT set as MILVUS_URI env var or pymilvus breaks!)
    # Prod: "http://milvus:19530" for standalone server
    milvus_db_path: str = "./data/milvus.db"
    milvus_qa_collection: str = "qa_pairs"
    milvus_chunk_collection: str = "doc_chunks"

    # ---- Chunking ----
    chunk_size: int = 500  # approximate token count per chunk
    chunk_overlap: int = 50

    # ---- Retrieval ----
    retrieval_top_k: int = 10       # Top-K candidates per retrieval path
    rrf_k: int = 60                 # RRF fusion constant
    qa_match_threshold: float = 0.85  # Min similarity for QA direct match

    # ---- Embedding ----
    embedding_provider: Literal["local_bge", "local_qwen3", "openai", "api"] = "local_bge"
    embedding_model: str = "BAAI/bge-m3"     # local → model path; api → model name (e.g. "text-embedding-3-small")
    embedding_dimensions: int = 1024         # BGE-M3: 1024; embedding-2: 1024; text-embedding-3-small: 1536

    # ---- Reranker (cross-encoder re-ranking) ----
    reranker_provider: Literal["local_bge", "none"] = "local_bge"  # "none" to disable
    reranker_model: str = "./models/bge-reranker-v2-m3"
    reranker_top_k: int = 5       # Final number of results after re-ranking
    reranker_candidates: int = 15 # Number of candidates fed into the reranker

    # ---- LLM / Chat (for RAG generation & QA augmentation) ----
    # 通用 API 模式: provider=api, 搭配 api_base_url + api_key + chat_model
    # 本地模式: provider=local_llm, 搭配 local_llm_path
    # 保留兼容: openai
    chat_provider: Literal["api", "openai", "local_llm"] = "api"
    chat_model: str = "deepseek-v4-flash"
    # 通用 API (OpenAI 兼容格式)
    chat_api_key: str = ""
    chat_api_base_url: str = ""
    # Local LLM path (used when chat_provider == "local_llm")
    local_llm_path: str = "./models/qwen3-1.7b"
    # API keys (loaded from .env — never commit)
    openai_api_key: str = ""
    openai_base_url: str = ""  # For proxies / Azure / compatible endpoints

    # ---- QA Augmentation ----
    qa_augmentation_enabled: bool = True
    qa_augmentation_variants: int = 5  # Number of paraphrase variants per QA pair

    # ---- API Fetcher ----
    api_base_url: str = "https://cqaip.cn"
    casdoor_session_id: str = ""  # Casdoor SSO session cookie for authenticated endpoints
    request_timeout: int = 30  # seconds
    request_retries: int = 3

    # ---- Logging ----
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


# Singleton — import this everywhere
settings = Settings()
