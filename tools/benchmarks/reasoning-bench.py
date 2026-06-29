#!/usr/bin/env python3
"""
Reasoning benchmark — evaluates model intelligence across 5 dimensions.

Speed doesn't matter if the output is garbage. This runs a quick 5-question
battery and scores each response for correctness. The final score is a
0-100 composite that combines with tok/s for the "real" ranking.

Dimensions tested:
  1. Math reasoning (multi-step word problem)
  2. Logic puzzle (deductive reasoning)
  3. Code correctness (find the bug)
  4. Instruction following (specific format constraints)
  5. Common sense reasoning (physical world understanding)

Scoring: automated keyword/pattern matching + numerical answer extraction.
No LLM judge needed — these answers are objectively right or wrong.
"""

import json
import time
import re
import sys
import os
import argparse
from urllib.request import urlopen, Request
from urllib.error import URLError

# ── Reasoning Battery ──────────────────────────────────────────────────────────

QUESTIONS = [

    # 1. Math: multi-step word problem (tests chained arithmetic reasoning)
    {
        "category": "math",
        "weight": 1.2,
        "prompt": (
            "A store sells apples at $1.50 each and oranges at $2.00 each. "
            "Sarah buys 3 more apples than oranges. She spends $25.50 total. "
            "How many oranges did she buy? Show your work and give the final "
            "answer as a single number at the end."
        ),
        "max_tokens": 256,
    },

    # 2. Logic: deductive reasoning (tests if-then chains)
    {
        "category": "logic",
        "weight": 1.0,
        "prompt": (
            "Five people sit in a row: Alice, Bob, Carol, Dave, Eve. "
            "Alice is not next to Bob. Carol is next to Eve. Dave is at "
            "one end of the row. Bob is not at any end. Who is sitting "
            "in the middle seat? Explain your reasoning, then state the "
            "answer clearly."
        ),
        "max_tokens": 384,
    },

    # 3. Code correctness: find the bug (tests code comprehension)
    {
        "category": "code",
        "weight": 1.2,
        "prompt": (
            "This Python function should return the sum of all even numbers "
            "in a list, but it has a bug. What is the bug and what's the "
            "corrected version?\n\n"
            "```python\n"
            "def sum_evens(numbers):\n"
            "    total = 0\n"
            "    for n in numbers:\n"
            "        if n % 2 == 0:\n"
            "            total =+ n\n"
            "    return total\n"
            "```\n\n"
            "Explain the bug, then provide the fixed code."
        ),
        "max_tokens": 256,
    },

    # 4. Instruction following: specific format (tests compliance)
    {
        "category": "instruction_following",
        "weight": 0.8,
        "prompt": (
            "List exactly 3 countries in Europe whose names start with the "
            "letter 'S'. Format your answer as a numbered list (1, 2, 3) "
            "with only the country names, no explanations. Include nothing "
            "else in your response."
        ),
        "max_tokens": 128,
    },

    # 5. Common sense: physical reasoning (tests world knowledge)
    {
        "category": "common_sense",
        "weight": 1.0,
        "prompt": (
            "I place a glass of water on a table, then put a book on top "
            "of the glass. What happens? Describe what actually occurs in "
            "the real world, not what you think I want to hear."
        ),
        "max_tokens": 256,
    },
]


# ── Scoring Functions ──────────────────────────────────────────────────────────

def score_math(response: str) -> float:
    """Correct answer: oranges = 6.
    1.5*(o+3) + 2*o = 25.5 → 1.5o + 4.5 + 2o = 25.5 → 3.5o = 21 → o = 6.
    """
    text = response.lower().strip()
    score = 0.0

    # Method credit: set up an equation
    has_equation = any(kw in text for kw in [
        "1.5", "oranges", "apples", "equation", "variable", "let ", "=", "3 +", "+ 3"
    ])
    if has_equation:
        score += 0.2

    # Work shown
    has_work = any(kw in text for kw in [
        "3.5", "21", "multiply", "subtract", "solve", "therefore", "divide"
    ])
    if has_work:
        score += 0.2

    # Check for correct numerical answer: 6
    numbers = re.findall(r'\b(\d+)\b', text)
    found_six = False
    for n in numbers:
        if n == "6":
            found_six = True
            break
    # Also check "answer: 6" or "answer is 6" patterns
    if re.search(r'(?:answer|result|oranges?)[\s:=]*(?:is|:|=)*\s*6\b', text):
        found_six = True

    if found_six:
        score += 0.5
    elif "21" in text and "3.5" in text:
        # Correct intermediate but wrong final
        score += 0.3
    elif "24" in text:
        # Used $28.50 instead of $25.50 (old version of question)
        score += 0.1

    # Shows chain-of-thought reasoning
    if any(kw in text for kw in ["let o", "let x", "let n", "let the number"]):
        score += 0.1

    return min(score, 1.0)


def score_logic(response: str) -> float:
    """Correct answer: Carol is in the middle seat (position 3).
    Valid arrangement: Dave(1), Bob(2), Carol(3), Eve(4), Alice(5)
    or Dave(1), Alice(2), Carol(3), Eve(4), Bob(5)—but Bob at end ✗
    Dave(1), Bob(2), Carol(3), Eve(4), Alice(5): Bob not at end ✓, Alice(5) not next to Bob(2) ✓, Carol(3) next to Eve(4) ✓.
    Middle = Carol.
    """
    text = response.lower().strip()
    score = 0.0

    # Shows structured reasoning
    has_reasoning = any(kw in text for kw in [
        "position", "seat", "constraint", "since", "therefore",
        "because", "must", "can't", "cannot", "let's", "if "
    ])
    if has_reasoning:
        score += 0.2

    # Correct answer: Carol in the middle
    if re.search(r'\bcarol\b.*\bmiddle\b', text) or re.search(r'\bmiddle\b.*\bcarol\b', text):
        score += 0.5
    elif re.search(r'(?:seat\s*[3#]|third|3rd|position\s*3).*\bcarol\b', text):
        score += 0.5
    elif "carol" in text:
        # Mentioned Carol but didn't explicitly say middle
        score += 0.15

    # Checks all constraints
    constraint_checks = sum(1 for kw in [
        "not next to", "adjacent", "next to", "at the end", "end of", "not at"
    ] if kw in text)
    score += min(constraint_checks * 0.05, 0.2)

    # Shows elimination process
    if any(kw in text for kw in ["eliminat", "rules out", "doesn't work", "won't work", "invalid"]):
        score += 0.1

    return min(score, 1.0)


def score_code(response: str) -> float:
    """The bug is `=+` instead of `+=`. Model should identify this."""
    text = response.lower().strip()
    score = 0.0

    # Identifies the core bug: =+ vs +=
    if "=+" in text or re.search(r'=\s*\+.*instead|=\+.*bug|=\+.*error|=\+.*wrong', text):
        score += 0.4

    if "+= n" in text or "+=n" in text or "+= n" in response:
        score += 0.3  # provides the fix

    # Explains WHY it's wrong (= +n sets total to n each iteration instead of adding)
    if any(kw in text for kw in [
        "set total to", "reassigns", "overwrites", "instead of adding",
        "assigns positive", "each iteration", "replaces total",
        "not incrementing", "assignment operator", "augmented"
    ]):
        score += 0.2

    # Provides complete corrected function
    if "def sum_evens" in text or "def sum_evens" in response.lower():
        score += 0.1

    return min(score, 1.0)


def score_instruction_following(response: str) -> float:
    """Must be exactly 3 items, numbered, European countries starting with S, no extra text."""
    text = response.strip()
    score = 0.0

    # Valid European countries starting with S
    valid_countries = {
        "spain", "sweden", "switzerland", "slovakia", "slovenia",
        "san marino", "serbia"
    }

    # Check numbered format
    has_numbered = bool(re.search(r'^\s*1[\.\)]', text, re.MULTILINE))
    if has_numbered:
        score += 0.15

    # Count list items
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    numbered_items = [l for l in lines if re.match(r'^\d+[\.\)]', l)]

    if len(numbered_items) == 3:
        score += 0.25  # exactly 3
    elif len(numbered_items) in (2, 4):
        score += 0.1

    # Check if countries are actually European S-countries
    found_valid = set()
    text_lower = text.lower()
    for c in valid_countries:
        if c in text_lower:
            found_valid.add(c)

    if len(found_valid) >= 3:
        score += 0.4
    elif len(found_valid) >= 2:
        score += 0.25
    elif len(found_valid) >= 1:
        score += 0.1

    # Penalize extra explanation text (should be ONLY the list)
    non_list_lines = [l for l in lines if not re.match(r'^\d+[\.\)]', l)]
    if non_list_lines:
        score -= 0.1

    # Check for bogus entries (non-European or not starting with S)
    false_countries = ["singapore", "south africa", "south korea", "sri lanka",
                       "syria", "sudan", "somalia", "saudi", "south sudan",
                       "senegal", "sierra leone"]
    has_false = any(fc in text_lower for fc in false_countries)
    if has_false:
        score -= 0.15

    return max(min(score, 1.0), 0.0)


def score_common_sense(response: str) -> float:
    """The book won't balance on a glass — it falls, the glass tips/breaks, water spills."""
    text = response.lower().strip()
    score = 0.0

    # Penalize sycophantic agreement with impossible premise
    if any(kw in text for kw in [
        "sits on top", "rests on", "stays on", "stable", "balances",
        "nothing happens", "it's fine", "no problem", "book sits"
    ]):
        score -= 0.3

    # Correct: recognizes instability
    if any(kw in text for kw in [
        "fall", "tip", "slide", "unstable", "wobble",
        "won't balance", "won't stay", "can't balance", "not stable", "unlikely",
        "wouldn't", "would not", "impractical"
    ]):
        score += 0.4

    # Correct: water spills
    if any(kw in text for kw in ["spill", "splash", "pour out", "water goes"]):
        score += 0.2

    # Correct: glass might break or tip over
    if any(kw in text for kw in [
        "break", "crack", "shatter", "tip over", "topple", "knock"
    ]):
        score += 0.2

    # Has nuanced real-world reasoning
    if any(kw in text for kw in [
        "surface area", "center of gravity", "weight", "narrow", "wider",
        "base", "support", "physics"
    ]):
        score += 0.1

    # Doesn't sycophantically agree with the premise
    if any(kw in text for kw in [
        "actually", "in reality", "real world", "unlikely", "impractical"
    ]):
        score += 0.1

    return max(min(score, 1.0), 0.0)


SCORE_FNS = {
    "math": score_math,
    "logic": score_logic,
    "code": score_code,
    "instruction_following": score_instruction_following,
    "common_sense": score_common_sense,
}


# ── Execution ──────────────────────────────────────────────────────────────────

def ask_model(port, prompt, max_tokens, timeout=120):
    """Send a question to the model, return (response_text, tok_s, elapsed)."""
    data = json.dumps({
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "stream": False
    }).encode()
    req = Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    start = time.time()
    try:
        resp = urlopen(req, timeout=timeout)
        elapsed = time.time() - start
        result = json.loads(resp.read())
        text = result.get("choices", [{}])[0].get("text", "")
        usage = result.get("usage", {})
        tokens = usage.get("completion_tokens", 0)
        tok_s = tokens / elapsed if elapsed > 0 else 0
        return text, tok_s, elapsed
    except Exception as e:
        return f"ERROR: {e}", 0, 0


def run_reasoning_bench(port):
    """Run all 5 questions against a model on the given port.
    Returns dict with scores per category + composite + speed stats."""
    results = []
    total_weighted_score = 0.0
    total_weight = 0.0
    total_time = 0.0

    for q in QUESTIONS:
        response, tok_s, elapsed = ask_model(port, q["prompt"], q["max_tokens"])

        if response.startswith("ERROR:"):
            results.append({
                "category": q["category"],
                "score": 0.0,
                "weight": q["weight"],
                "response": response,
                "status": "error"
            })
            total_weight += q["weight"]
            continue

        scorer = SCORE_FNS[q["category"]]
        score = scorer(response)

        results.append({
            "category": q["category"],
            "score": round(score, 2),
            "weight": q["weight"],
            "tok_s": round(tok_s, 1),
            "time_s": round(elapsed, 2),
            "status": "ok",
            "response_preview": response[:200] if len(response) > 200 else response
        })

        total_weighted_score += score * q["weight"]
        total_weight += q["weight"]
        total_time += elapsed

    # Composite score: weighted average, scaled to 0-100
    composite = (total_weighted_score / total_weight * 100) if total_weight > 0 else 0

    return {
        "composite": round(composite, 1),
        "questions": results,
        "total_time_s": round(total_time, 1),
        "avg_tok_s": round(sum(r.get("tok_s", 0) for r in results) / max(len(results), 1), 1),
    }


def print_results(name, bench):
    """Pretty-print reasoning benchmark results."""
    print(f"\n{'Category':<25} {'Score':>6} {'Weight':>7} {'tok/s':>7}")
    print("-" * 50)
    for q in bench["questions"]:
        if q["status"] == "ok":
            bar = "█" * int(q["score"] * 10) + "░" * (10 - int(q["score"] * 10))
            print(f"{q['category']:<25} {q['score']:>5.0%}  {q['weight']:>6.1f}x  {q.get('tok_s', 0):>6.1f}  {bar}")
        else:
            print(f"{q['category']:<25} {'ERR':>6}  {q['weight']:>6.1f}x")

    grade = "F"
    if bench["composite"] >= 90: grade = "A+"
    elif bench["composite"] >= 80: grade = "A"
    elif bench["composite"] >= 70: grade = "B"
    elif bench["composite"] >= 60: grade = "C"
    elif bench["composite"] >= 50: grade = "D"

    print(f"\n{'='*60}")
    print(f"COMPOSITE: {bench['composite']}/100 [{grade}]")
    print(f"Avg speed: {bench['avg_tok_s']} tok/s | Total: {bench['total_time_s']}s")


def main():
    parser = argparse.ArgumentParser(description="Reasoning benchmark")
    parser.add_argument("--port", type=int, required=True, help="Port of running model")
    parser.add_argument("--name", type=str, default="unknown", help="Model name")
    parser.add_argument("--output", type=str, help="Optional: save results to JSON")
    args = parser.parse_args()

    print(f"Reasoning benchmark: {args.name} on port {args.port}")
    print(f"Questions: {len(QUESTIONS)} (math, logic, code, instruction-following, common-sense)")
    print("=" * 60)

    bench = run_reasoning_bench(args.port)
    print_results(args.name, bench)

    if args.output:
        output = {"name": args.name, "date": time.strftime("%Y-%m-%d"), **bench}
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved: {args.output}")

    return bench


if __name__ == "__main__":
    main()
