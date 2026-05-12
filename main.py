import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import run_agent
from dotenv import load_dotenv

load_dotenv()

# ── Create FastAPI app ────────────────────────────────
app = FastAPI(
    title="Anthropic Docs Agent",
    description="AI agent that answers questions over Anthropic documentation",
    version="1.0.0"
)

# ── Request and Response Models ───────────────────────
# These define the shape of data coming IN and going OUT
# Pydantic validates automatically — wrong data = clean error

class AskRequest(BaseModel):
    question: str

class ChatRequest(BaseModel):
    question:   str
    session_id: str = None   # optional — if None we create a new session

class Source(BaseModel):
    title:      str
    url:        str
    content:    str = ""

class AskResponse(BaseModel):
    answer:     str
    sources:    list[Source]

class ChatResponse(BaseModel):
    answer:     str
    sources:    list[Source]
    session_id: str


# ── Endpoints ─────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    Single turn Q&A.
    Send a question, get an answer with sources.
    No conversation history.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result = run_agent(request.question)

    sources = [
        Source(
            title=s.get("title", ""),
            url=s.get("url", ""),
            content=s.get("content", "")[:200]
        )
        for s in result["sources"]
    ]

    return AskResponse(
        answer=result["answer"],
        sources=sources
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Multi-turn conversation.
    Pass session_id to continue an existing conversation.
    If no session_id, a new conversation starts.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result = run_agent(request.question, session_id=request.session_id)

    sources = [
        Source(
            title=s.get("title", ""),
            url=s.get("url", ""),
            content=s.get("content", "")[:200]
        )
        for s in result["sources"]
    ]

    return ChatResponse(
        answer=result["answer"],
        sources=sources,
        session_id=result["session_id"]
    )