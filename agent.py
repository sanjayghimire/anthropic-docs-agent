import os
import uuid
from anthropic import Anthropic
from supabase import create_client
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

MODEL       = "claude-haiku-4-5"
MAX_TOKENS  = 1024
EMBED_MODEL = "text-embedding-3-small"

sessions = {}

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

SYSTEM_PROMPT = """You are an expert on Anthropic's documentation and APIs.

When answering questions:
1. Always use the search_docs tool to find relevant information
2. Base your answers on the documentation you find
3. Always cite your sources by including the URL at the end of your answer
4. Be concise and accurate
5. If you cannot find relevant information, say so clearly

Format citations like this:
Source: [title](url)"""


def search_docs(query: str) -> list[dict]:
    response = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=query
    )
    query_embedding = response.data[0].embedding

    results = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_count": 5
        }
    ).execute()

    return results.data


def run_agent(question: str, session_id: str = None) -> dict:

    if session_id and session_id in sessions:
        messages = sessions[session_id].copy()
    else:
        messages = []
        session_id = f"session_{str(uuid.uuid4())[:8]}"

    messages.append({
        "role": "user",
        "content": question
    })

    sources = []

    while True:
        response = anthropic.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages
        )

        if response.stop_reason == "end_turn":
            answer = ""
            for block in response.content:
                if hasattr(block, "text"):
                    answer = block.text
                    break

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
            # Get ALL tool calls in this response
            tool_use_blocks = [
                block for block in response.content
                if block.type == "tool_use"
            ]

            if not tool_use_blocks:
                break

            # Add Claude's response to history
            messages.append({
                "role":    "assistant",
                "content": response.content
            })

            # Handle ALL tool calls and collect ALL results
            tool_results = []
            for tool_use_block in tool_use_blocks:
                if tool_use_block.name == "search_docs":
                    query   = tool_use_block.input["query"]
                    results = search_docs(query)
                    sources = results

                    tool_result = "\n\n".join([
                        f"Title: {r['title']}\nURL: {r['url']}\nContent: {r['content']}"
                        for r in results
                    ])

                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": tool_use_block.id,
                        "content":     tool_result
                    })

            # Send ALL tool results back in one message
            messages.append({
                "role":    "user",
                "content": tool_results
            })

        else:
            break

    return {
        "answer":     "I could not find a good answer. Please try again.",
        "sources":    [],
        "session_id": session_id
    }


if __name__ == "__main__":
    print("Testing agent...")
    result = run_agent("What is Claude Sonnet and what can it do?")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources found: {len(result['sources'])}")
    for s in result['sources']:
        print(f"  - {s['title']} | {s['url']}")