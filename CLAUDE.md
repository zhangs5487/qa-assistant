# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: QA Assistant (问答助手)

Building an intelligent QA system with data sourced from `https://cqaip.cn/`. Two answer modes:
- **QA mode (primary)**: Semantic Q-to-Q matching -> return paired A. Fast, no LLM cost.
- **RAG mode (fallback)**: Retrieve document chunks -> LLM generates answer. Handles unseen questions.

**Current state**: Phase 1 COMPLETE. Two vector knowledge bases built in **Milvus** (pymilvus 2.6 + Milvus Lite):
- **FAQ知识库** (`qa_pairs`): 395 QA pairs (policies + nav + platform usage + LLM-augmented)
- **文档知识库** (`doc_chunks`): 3,510 document chunks (from 11 API endpoints)
- **Test script**: `python scripts/test_qa.py` — interactive QA+RAG, queries Milvus directly
- Phase 2 pending: FastAPI service, BM25+RRF, Streamlit UI, local BGE-M3 model

**Authority**: `问答助手开发.md` is the definitive development manual. All architectural decisions are documented there.

## Project Structure

```
qa-assistant/
├── config/                    # Pydantic Settings v2 + .env.template
├── libs/
│   ├── shared/                # Data contracts (RawDocument, CleanDocument, Chunk, QAPair)
│   ├── crawler/               # REST API fetcher (replaces Scrapy -- all data is API-based)
│   │   ├── api_fetcher.py     # Generic paginated API fetcher with auth support
│   │   ├── sources.py         # Data source definitions (11 cqaip.cn API endpoints)
│   │   └── cli.py             # CLI entry: python -m crawler.cli --list
│   ├── cleaner/               # Parser + chunker (LangChain) + pipeline orchestrator
│   │   ├── parser.py          # Per-source parsers: policies, news, datasets, models, etc.
│   │   ├── chunker.py         # Chinese-aware text chunking (500tk/50ov)
│   │   └── pipeline.py        # Orchestrate parse -> clean -> chunk
│   ├── storage/               # Abstract repos + MongoDB/Milvus/local-FS (pending Phase 2)
│   ├── llm/                   # EmbeddingProvider/ChatProvider ABCs + factory (ready)
│   └── pipeline/              # (future) service orchestration
├── scripts/                   # run_pipeline.py: fetch + clean + chunk in one command
├── data/                      # Git-ignored: raw/, clean/ (5.4MB), exports/
├── models/                    # Local embedding model cache
└── docs/                      # cqaip-mapping.md, architecture docs
```

## Tech Stack

| Module | Choice |
|--------|--------|
| Data source | REST APIs (11 endpoints discovered via reverse engineering) |
| API fetcher | `httpx` with pagination + retry + Casdoor SSO auth |
| RAG orchestration | LangChain (planned for Phase 2) |
| Vector DB | **Milvus** (pymilvus 2.6 + Milvus Lite, embedded) |
| Retrieval | Dense vector + BM25 + RRF (planned for Phase 2) |
| Embedding | **BGE-M3** target (ModelScope, 1024-dim); current: ZhipuAI API stand-in |
| LLM generation | Pluggable: ZhipuAI / OpenAI / Ollama |
| Storage (meta) | MongoDB (planned for Phase 2) |
| Backend | FastAPI (planned for Phase 2) |

## Data Sources (cqaip.cn)

| Source | Items | Auth | Paginated |
|--------|-------|------|-----------|
| policies | 16 | No | No |
| industry_news | 34 | No | Yes |
| competitions | 12 | No | No |
| datasets | 2,176 | Yes (Casdoor) | Yes |
| marketplace | 158 | Yes (Casdoor) | Yes |
| models | 371 | No | Yes |
| apps, communities, site_config, news | ~8 | Mixed | No |

## Common Commands

### Data Pipeline
```bash
python scripts/run_pipeline.py                        # Full: fetch + clean + chunk
python scripts/run_pipeline.py --fetch-only            # Fetch only
python scripts/run_pipeline.py --clean-only            # Clean existing raw data
python scripts/run_pipeline.py --sources policies      # Specific sources only

# Module CLI (advanced):
PYTHONPATH="libs/crawler/src;libs/shared/src;config" python -m crawler.cli --list
PYTHONPATH="libs/crawler/src;libs/shared/src;config" python -m crawler.cli --all
```

### Setup
```bash
pip install -e ".[dev,crawler,cleaner,storage]"
pytest -v
ruff check . && ruff format .
```

## Architecture

### Data Pipeline (Current - Phase 1)
```
REST APIs (11 sources) -> APIFetcher (paginated, auth) -> Raw JSON (data/raw/)
  -> Parser (per-source) -> CleanDocument -> Clean JSON (data/clean/documents/)
  -> Chunker (LangChain, 500tk/50ov, Chinese separators) -> Chunk JSON (data/clean/chunks/)
  -> (future) Embed -> Milvus insert
```

### QA Mode (Planned - Phase 2)
```
User question -> Embed -> Milvus QA collection search
  -> similarity > 0.85? -> return matched answer
  -> otherwise -> RAG fallback
```

### RAG Mode (Planned - Phase 2)
```
User question -> Embed
  -> Dense vector retrieval (Milvus) + Sparse keyword retrieval (BM25)
  -> RRF fusion (k=60) -> Prompt assembly -> LLM generation (streaming SSE)
```

### LLM Abstraction (Ready)
- `EmbeddingProvider` / `ChatProvider` ABCs in `libs/llm/base.py`
- Factory in `libs/llm/factory.py`: reads config, instantiates correct provider
- Providers: `openai.py`, `ollama.py`, `local_qwen.py` + `local_bge.py` (for BGE-M3)
- **Rule**: no code outside `libs/llm/` imports a provider SDK directly

## Configuration

Single source: `config/settings.py` (Pydantic Settings v2). Secrets via `.env` (git-ignored). Template: `config/.env.template`.

Key settings:

| Variable | Purpose | Default |
|----------|---------|---------|
| `CASDOOR_SESSION_ID` | Auth for protected APIs | (from .env) |
| `EMBEDDING_PROVIDER` | Embedding source | `local_bge` |
| `EMBEDDING_MODEL` | Model name/path | `./models/bge-m3` |
| `CHAT_PROVIDER` | LLM for generation | `api` |
| `MILVUS_URI` | Milvus connection | `lite://./data/milvus.db` |
| `QA_MATCH_THRESHOLD` | QA-vs-RAG cutoff | `0.85` |
| `RRF_K` | RRF fusion constant | `60` |

## Key Design Decisions

- **REST API fetcher instead of Scrapy** -- after reverse engineering, all content is served through structured JSON APIs. No scraping needed.
- **Casdoor SSO for auth** -- datasets, marketplace, and skills require login. Session cookie is stored in `.env`.
- **QA-primary, RAG-fallback** -- most questions hit the fast/cheap QA path; RAG only for edge cases
- **BGE-M3 local embedding** -- zero API cost, no rate limits, data stays local; supports dense+sparse dual output
- **Hybrid retrieval (vector + BM25 + RRF)** -- production best practice, matches Alibaba/ZhipuAI patterns
- **Provider-agnostic LLM layer** -- config-driven; swap between ZhipuAI/OpenAI/Ollama without code changes
