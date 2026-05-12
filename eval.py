import json
import os
import time
from anthropic import Anthropic
from agent import run_agent
from dotenv import load_dotenv

load_dotenv()

anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── Load golden dataset ───────────────────────────────
def load_dataset() -> list[dict]:
    with open("eval_dataset.json", "r") as f:
        return json.load(f)


# ── LLM as Judge ──────────────────────────────────────
def judge_answer(question: str, expected: str, actual: str) -> dict:
    """
    Ask Claude to score our agent's answer.
    
    We give Claude:
    - The original question
    - The correct expected answer
    - What our agent actually said
    
    Claude scores it 1-5 and explains why.
    """
    prompt = f"""You are evaluating an AI agent's answer quality.

Question: {question}

Expected Answer: {expected}

Agent's Answer: {actual}

Score the agent's answer from 1 to 5:
1 = Completely wrong or missing
2 = Mostly wrong with some correct elements  
3 = Partially correct but missing key information
4 = Mostly correct with minor gaps
5 = Fully correct and complete

Respond in this exact JSON format:
{{
    "score": <number 1-5>,
    "reasoning": "<one sentence explanation>"
}}

Return only the JSON, nothing else."""

    response = anthropic.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )

    # Parse the JSON response from Claude
    result = json.loads(response.content[0].text)
    return result


# ── Run Evaluation ────────────────────────────────────
def run_eval():
    print("=" * 55)
    print("  Anthropic Docs Agent — Evaluation Suite")
    print("=" * 55)

    dataset     = load_dataset()[:5]
    results     = []
    total_score = 0
    passed      = 0
    PASS_THRESHOLD = 3

    for item in dataset:
        print(f"\n{item['id']}: {item['question'][:50]}...")

        # Run our agent
        agent_result = run_agent(item["question"])
        actual_answer = agent_result["answer"]

        # Judge the answer
        judgment = judge_answer(
            item["question"],
            item["expected"],
            actual_answer
        )

        score  = judgment["score"]
        passed_this = score >= PASS_THRESHOLD

        total_score += score
        if passed_this:
            passed += 1

        status = "✅" if passed_this else "❌"
        print(f"  Score: {score}/5 {status}")
        print(f"  Reason: {judgment['reasoning']}")

        time.sleep(20)

        results.append({
            "id":        item["id"],
            "question":  item["question"],
            "expected":  item["expected"],
            "actual":    actual_answer,
            "score":     score,
            "passed":    passed_this,
            "reasoning": judgment["reasoning"]
        })

    # ── Summary ───────────────────────────────────────
    avg_score  = total_score / len(dataset)
    pass_rate  = (passed / len(dataset)) * 100

    # Find weakest question
    weakest = min(results, key=lambda x: x["score"])

    print(f"\n{'=' * 55}")
    print(f"  RESULTS")
    print(f"{'=' * 55}")
    print(f"  Average Score : {avg_score:.1f} / 5.0")
    print(f"  Pass Rate     : {passed}/{len(dataset)} ({pass_rate:.1f}%)")
    print(f"  Weakest Q     : {weakest['id']} — {weakest['question'][:40]}")
    print(f"{'=' * 55}")

    # Save full report
    with open("eval_report.json", "w") as f:
        json.dump({
            "summary": {
                "average_score": avg_score,
                "pass_rate":     f"{pass_rate:.1f}%",
                "passed":        passed,
                "total":         len(dataset),
                "weakest_question": weakest["id"]
            },
            "results": results
        }, f, indent=2)

    print(f"\n  Full report saved to eval_report.json")


if __name__ == "__main__":
    run_eval()