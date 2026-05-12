# Anthropic Docs RAG Agent

A production-quality AI agent that answers questions over Anthropic's documentation with source citations.

Built as a qualification project for the Senior AI Builder role at Ovidius AI.

---

## What This Does

- Scrapes and ingests docs.anthropic.com into a Supabase pgvector database
- Answers natural language questions with citations back to source URLs
- Uses a Claude Managed Agent with a custom `search_docs` tool
- Exposes a FastAPI endpoint for single-turn and multi-turn Q&A
- Includes an LLM-as-judge evaluation suite with 12 golden Q&A pairs

---

## Architecture
User Question
↓
FastAPI endpoint
↓
Claude Managed Agent
↓
search_docs tool → Supabase pgvector similarity search
↓
Claude synthesizes answer with citations
↓
Response with source URLs

---

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| LLM | Claude Sonnet 4 | Anthropic-native, best tool calling |
| Embeddings | OpenAI text-embedding-3-small | Fast, cheap, accurate |
| Vector Store | Supabase + pgvector | Production stack |
| API | FastAPI | Clean Python, auto docs |
| Eval | LLM-as-judge | Measures actual answer quality |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/sanjayghimire/anthropic-docs-agent.git
cd anthropic-docs-agent
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set up environment variables

```bash
cp .env.example .env
```

Fill in your keys in `.env`:
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=eyJ...

### 4. Set up Supabase

- Create a free project at supabase.com
- Go to SQL Editor and run the contents of `supabase_schema.sql`

### 5. Run ingestion

```bash
python ingest.py
```

This scrapes docs.anthropic.com, chunks the text, embeds it, and stores it in Supabase. Takes about 10 minutes.

### 6. Start the API

```bash
uvicorn main:app --reload
```

### 7. Ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Claude Sonnet and what can it do?"}'
```

---

## Running the Evaluation

```bash
python eval.py
```

Output looks like:
Running evaluation on 12 golden Q&A pairs...
Q1: What is Claude's context window?        Score: 5/5 ✅
Q2: How do I enable streaming?              Score: 4/5 ✅
...
─────────────────────────────────────────
Average Score:  4.3 / 5.0
Pass Rate:      11 / 12 (91.6%)
Weakest Q:      Q7 — What are rate limits?

---

## Tradeoffs & What I'd Do Next

| What I Built | Production Approach |
|---|---|
| In-memory session store | Redis for multi-instance scaling |
| Fixed character chunking | Semantic chunking for better context |
| Vector-only search | Hybrid BM25 + vector with RRF fusion |
| No re-ranking | Cross-encoder re-ranker for precision |
| No auth on API | API key header + rate limiting |

**If this were day one of the job I would:**
1. Add hybrid search — pgvector + PostgreSQL full-text with Reciprocal Rank Fusion
2. Add streaming responses via SSE so answers appear token by token
3. Build a re-ingestion webhook triggered when Anthropic updates their docs
4. Move sessions to Redis for horizontal scaling