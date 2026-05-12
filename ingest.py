import os
import re
import time
from openai import OpenAI
from supabase import create_client
from dotenv import load_dotenv
import httpx

# ── Load keys from .env ───────────────────────────────
load_dotenv()

# ── Clients ───────────────────────────────────────────
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# ── Settings ──────────────────────────────────────────
CHUNK_SIZE    = 2000
CHUNK_OVERLAP = 200
EMBED_MODEL   = "text-embedding-3-small"
LLMS_TXT_URL  = "https://platform.claude.com/llms-full.txt"


# ── STEP 1: Download llms.txt ─────────────────────────
def download_llms_txt() -> str:
    print("Downloading llms.txt from Anthropic...")
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        response = client.get(LLMS_TXT_URL)
        print(f"Downloaded {len(response.text)} characters")
        return response.text


# ── STEP 2: Split into pages ──────────────────────────
def parse_pages(content: str) -> list[dict]:
    """
    llms.txt organizes content like this:

    # Page Title
    URL: https://docs.anthropic.com/en/docs/something
    
    Page content here...

    # Next Page Title
    URL: https://...

    We split on lines starting with # to get individual pages.
    Each page becomes a dict with title, url, and content.
    """
    pages  = []
    blocks = re.split(r'\n(?=# )', content)

    for block in blocks:
        if not block.strip():
            continue

        lines   = block.strip().split('\n')
        title   = lines[0].replace('#', '').strip()

        # Find URL in the block
        url     = ""
        url_match = re.search(r'https?://[^\s]+', block)
        if url_match:
            url = url_match.group(0)

        # Content is everything after the first two lines
        content_text = '\n'.join(lines[2:]).strip()

        if content_text and len(content_text) > 50:
            pages.append({
                "title":   title,
                "url":     url or LLMS_TXT_URL,
                "content": content_text
            })

    print(f"Parsed {len(pages)} pages")
    return pages


# ── STEP 3: Chunk text ────────────────────────────────
def chunk_text(text: str) -> list[str]:
    if not text.strip():
        return []

    if len(text) <= CHUNK_SIZE:
        return [text]

    chunks = []
    start  = 0

    while start < len(text):
        end = start + CHUNK_SIZE

        if end < len(text):
            break_point = text.rfind(". ", start, end)
            if break_point > start + (CHUNK_SIZE // 2):
                end = break_point + 1

        chunks.append(text[start:end].strip())
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return [c for c in chunks if c]


# ── STEP 4: Embed ─────────────────────────────────────
def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    response = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )

    return [item.embedding for item in response.data]


# ── STEP 5: Store in Supabase ─────────────────────────
def upsert_chunks(chunks: list[str], embeddings: list[list[float]], url: str, title: str):
    rows = []

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        rows.append({
            "content":     chunk,
            "url":         url,
            "title":       title,
            "chunk_index": i,
            "embedding":   embedding
        })

    if not rows:
        return

    supabase.table("documents").upsert(
        rows,
        on_conflict="url,chunk_index"
    ).execute()


# ── MAIN ──────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  Anthropic Docs Ingestion Pipeline")
    print("=" * 50)

    # Step 1: Download
    content = download_llms_txt()

    # Step 2: Parse into pages
    pages = parse_pages(content)[:200]

    total_chunks = 0

    for i, page in enumerate(pages):
        print(f"\n[{i+1}/{len(pages)}] {page['title']}")

        # Step 3: Chunk
        chunks = chunk_text(page["content"])
        print(f"  ↳ {len(chunks)} chunks")

        if not chunks:
            continue

        # Step 4: Embed
        embeddings = embed_texts(chunks)

        # Step 5: Store
        upsert_chunks(chunks, embeddings, page["url"], page["title"])

        total_chunks += len(chunks)
        time.sleep(0.1)

    print(f"\n{'='*50}")
    print(f"  Done! {total_chunks} chunks stored from {len(pages)} pages")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()