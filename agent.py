import os
from anthropic import Anthropic
from supabase import create_client
from openai import OpenAI
from dotenv import load_dotenv

# ── Load keys ─────────────────────────────────────────
load_dotenv()

# ── Clients ───────────────────────────────────────────
anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# ── Settings ──────────────────────────────────────────
MODEL         = "claude-sonnet-4-5"
MAX_TOKENS    = 1024
EMBED_MODEL   = "text-embedding-3-small"

# ── Session storage ───────────────────────────────────
# This is a simple dictionary that stores conversation history
# Key   = session_id (a unique string per conversation)
# Value = list of messages in that conversation
sessions = {}

# ── Tool Definition ───────────────────────────────────
# This is what we hand to Claude so it knows what tools exist
# Claude reads this and decides when to call search_docs
TOOLS = [
    {
        "name": "search_docs",
        "description": "Search the Anthropic documentation to find relevant information. Use this tool whenever you need to answer questions about Claude, the Anthropic API, models, pricing, or any Anthropic-related topic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant documentation chunks"
                }
            },
            "required": ["query"]
        }
    }
]

# ── System Prompt ─────────────────────────────────────
SYSTEM_PROMPT = """You are an expert on Anthropic's documentation and APIs.

When answering questions:
1. Always use the search_docs tool to find relevant information
2. Base your answers on the documentation you find
3. Always cite your sources by including the URL at the end of your answer
4. Be concise and accurate
5. If you cannot find relevant information, say so clearly

Format citations like this:
Source: [title](url)"""

# ── Search Function ───────────────────────────────────
def search_docs(query: str) -> list[dict]:
    """
    This is the actual function that runs when Claude calls the tool.
    
    Steps:
    1. Convert the query to a vector (embed it)
    2. Search Supabase for the most similar vectors
    3. Return the top 5 matching chunks
    """
    # Convert query text to numbers (same model we used for ingestion)
    response = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=query
    )
    query_embedding = response.data[0].embedding

    # Search Supabase using our match_documents function
    # This finds the 5 most similar chunks to the query
    results = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_count": 5
        }
    ).execute()

    return results.data

# ── Agent Function ────────────────────────────────────
def run_agent(question: str, session_id: str = None) -> dict:
    """
    The main agent loop.
    
    1. Load conversation history for this session
    2. Add the new question to history
    3. Send to Claude with tools
    4. If Claude calls search_docs → run it → send results back
    5. Get final answer
    6. Save updated history
    7. Return answer + sources
    """

    # ── Load or create session ────────────────────────
    if session_id and session_id in sessions:
        messages = sessions[session_id].copy()
    else:
        messages = []
        if not session_id:
            session_id = f"session_{len(sessions) + 1}"

    # ── Add user question to history ──────────────────
    messages.append({
        "role": "user",
        "content": question
    })

    sources = []

    # ── Agent loop ────────────────────────────────────
    # We loop because Claude might call the tool multiple times
    while True:
        # Send messages to Claude
        response = anthropic.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages
        )

        # ── Check why Claude stopped ──────────────────
        # end_turn    → Claude finished answering
        # tool_use    → Claude wants to call a tool
        if response.stop_reason == "end_turn":
            # Extract the text answer
            answer = ""
            for block in response.content:
                if hasattr(block, "text"):
                    answer = block.text
                    break

            # Save updated history for next turn
            messages.append({
                "role": "assistant",
                "content": response.content
            })
            sessions[session_id] = messages

            return {
                "answer":     answer,
                "sources":    sources,
                "session_id": session_id
            }

        elif response.stop_reason == "tool_use":
            # Find which tool Claude wants to call
            tool_use_block = None
            for block in response.content:
                if block.type == "tool_use":
                    tool_use_block = block
                    break

            # Add Claude's tool request to message history
            messages.append({
                "role":    "assistant",
                "content": response.content
            })

            # ── Run the actual tool ───────────────────
            if tool_use_block.name == "search_docs":
                query   = tool_use_block.input["query"]
                results = search_docs(query)
                sources = results

                # Format results for Claude to read
                tool_result = "\n\n".join([
                    f"Title: {r['title']}\nURL: {r['url']}\nContent: {r['content']}"
                    for r in results
                ])

                # Send tool results back to Claude
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type":        "tool_result",
                            "tool_use_id": tool_use_block.id,
                            "content":     tool_result
                        }
                    ]
                })


# ── Quick test ────────────────────────────────────────
if __name__ == "__main__":
    print("Testing agent...")
    result = run_agent("What is Claude Sonnet and what can it do?")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources found: {len(result['sources'])}")
    for s in result['sources']:
        print(f"  - {s['title']} | {s['url']}")