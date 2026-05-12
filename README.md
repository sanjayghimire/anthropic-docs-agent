# Anthropic Docs RAG Agent

A production-quality AI agent that answers questions over Anthropic's
documentation with source citations.

Built as a qualification project for the Senior AI Builder role at Ovidius AI.

---

## What This Does

- Ingests Anthropic's official llms-full.txt into a Supabase pgvector database
- Answers natural language questions with citations back to source URLs
- Uses a Claude Managed Agent with a custom `search_docs` tool
- Exposes a FastAPI endpoint for single-turn and multi-turn Q&A
- Includes an LLM-as-judge evaluation suite with 12 golden Q&A pairs

---

## Architecture
User Question
↓
FastAPI endpoint (/ask or /chat)
↓
Claude Managed Agent (agent.py)
↓
Claude calls search_docs tool
↓
search_docs embeds query → Supabase pgvector similarity search
↓
Top 5 most relevant chunks returned
↓
Claude reads chunks → synthesizes answer with citations
↓
Response with answer + source URLs

---

## File Structure
anthropic-docs-agent/
├── ingest.py            → downloads llms-full.txt, chunks, embeds, stores in Supabase
├── agent.py             → Claude Managed Agent with search_docs tool + session management
├── main.py              → FastAPI server with /ask and /chat endpoints
├── eval.py              → LLM-as-judge evaluation suite
├── eval_dataset.json    → 12 golden Q&A pairs
├── eval_report.json     → full eval results with scores and reasoning
├── supabase_schema.sql  → run this in Supabase SQL editor to set up database
├── requirements.txt     → all Python dependencies
└── .env.example         → template for environment variables

---

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| LLM Agent | Claude Haiku | Higher token rate limits for demo |
| LLM Judge | Claude Sonnet 4 | Better reasoning for eval scoring |
| Embeddings | OpenAI text-embedding-3-small | Fast, cheap, works everywhere |
| Vector Store | Supabase + pgvector | Your production stack |
| API | FastAPI | Clean Python, auto-generates docs |
| Eval | LLM-as-judge | Measures actual answer quality |

---

## How the Agent Works

This is Option A from the brief — a Claude Managed Agent with a custom tool.

User sends question to /ask or /chat endpoint
FastAPI passes question to run_agent() in agent.py
Claude receives question + search_docs tool definition
Claude decides to call search_docs tool
search_docs embeds the question with OpenAI
Supabase finds top 5 most similar chunks
Claude reads those chunks
Claude writes answer with source citations
Response returned to user


Key point — Claude decides WHEN to call the tool based on the tool
description. It only searches when it needs information. Simple
greetings get answered without hitting the database at all.

### Session Management
Multi-turn conversations work by storing message history in a
dictionary keyed by session_id. Each follow-up question includes
the full conversation history so Claude remembers context across turns.

```python
# In-memory session store
sessions = {}  # session_id → list of messages
```

In production this would be Redis for persistence and horizontal scaling.

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

> **Note for Windows users:** If pip install fails with compiler errors
> run these two commands first:
> ```bash
> pip install supabase==2.3.4
> pip install httpx==0.27.0
> ```
> Then run `pip install -r requirements.txt` again.

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
- Go to SQL Editor
- Paste and run the full contents of `supabase_schema.sql`
- This creates the documents table and match_documents search function

### 5. Run ingestion

```bash
python ingest.py
```

Downloads Anthropic's official llms-full.txt, chunks the first 200 pages,
embeds with OpenAI, and stores in Supabase.

- Time: ~8 minutes
- Cost: under $0.50
- Result: ~391 chunks stored

### 6. Start the API

```bash
uvicorn main:app --reload
```

Server runs at `http://127.0.0.1:8000`

Auto-generated API docs at `http://127.0.0.1:8000/docs`

### 7. Ask a single question

```powershell
# Windows PowerShell
$r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/ask" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"question": "How do I enable streaming in Claude API?"}' `
  -UseBasicParsing
$r.Content
```

### 8. Multi-turn conversation

```powershell
# First question
$r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/chat" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"question": "What is Claude Opus?"}' `
  -UseBasicParsing
$r.Content

# Copy session_id from response, then ask follow up
$r2 = Invoke-WebRequest -Uri "http://127.0.0.1:8000/chat" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"question": "How does it compare to Haiku?", "session_id": "session_1"}' `
  -UseBasicParsing
$r2.Content
```

---

## Running the Evaluation

```bash
python eval.py
```

Actual results from this submission:
=======================================================
Anthropic Docs Agent — Evaluation Suite
Q1: What are the main Claude model families?    Score: 4/5 ✅
Q2: What is the purpose of the system prompt?   Score: 4/5 ✅
Q3: How do you enable streaming responses?      Score: 5/5 ✅
Q4: What is tool use in Claude?                 Score: 5/5 ✅
Q5: What is the Messages API?                   Score: 5/5 ✅
Average Score : 4.6 / 5.0
Pass Rate     : 5 / 5 (100%)
Weakest Q     : Q1 — version numbers may go stale

> **Note:** Dataset has 12 questions in eval_dataset.json.
> Running 5 due to rate limits on low API tier — Sonnet has
> 8,000 output tokens per minute. With a higher tier all 12
> run in under 60 seconds using async parallel calls.

Full results saved to `eval_report.json` after each run.

---

## Key Decisions & Why

### Why llms-full.txt instead of scraping?
docs.anthropic.com is JavaScript-rendered. A simple HTTP scraper only
gets empty HTML with "Loading..." text. Anthropic officially publishes
llms-full.txt for AI tools — 76MB of clean content across 1,400 pages.
More reliable than scraping and officially supported.

### Why Supabase + pgvector?
It is your production stack. pgvector stores and searches 1,536-dimension
embeddings directly inside Postgres with a single SQL function — no
separate vector database needed.

### Why Claude Managed Agent with a custom tool?
The search_docs tool lets Claude decide when to search rather than always
searching. This is the correct agent pattern — Claude only hits the
database when it actually needs information.

### Why LLM-as-judge for eval?
It scales. Hand-checking answers is slow and subjective. Using Claude as
judge is fast, consistent, and mirrors how production teams measure
quality at scale.

### Why OpenAI embeddings?
Tried free local embedding libraries first but they all required Rust or
C++ compiler tools that fail on Windows — setup friction for anyone
trying to run this. OpenAI embeddings cost under $0.50 for full ingestion
and work everywhere with just an API key.

---

## Tradeoffs & What I'd Do Next

| What I Built | Production Approach |
|---|---|
| In-memory session store | Redis for persistence + scaling |
| Fixed character chunking | Semantic chunking at topic boundaries |
| Vector-only search | Hybrid BM25 + vector with RRF fusion |
| No re-ranking | Cross-encoder re-ranker for precision |
| No auth on API | API key header + rate limiting |
| OpenAI embeddings | Cohere embeddings for stronger retrieval |
| Messy citation URLs | Filter to docs.anthropic.com only |
| Haiku for agent | Sonnet throughout with higher API tier |
| 5 eval questions | All 12 async with higher API tier |

**If this were day one of the job:**

1. **Hybrid search** — pgvector + PostgreSQL full-text with Reciprocal
   Rank Fusion so exact keyword queries don't get missed
2. **Streaming responses** — SSE so answers appear token by token
3. **Re-ingestion webhook** — triggered when Anthropic updates their
   docs so the database never goes stale
4. **Redis + async eval** — sessions that persist across restarts,
   all 12 eval questions running in parallel