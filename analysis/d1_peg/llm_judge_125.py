import json
import requests

with open("peg_answers_125.json") as f:
    peg = json.load(f)

with open("../../egs/world_history_full/exp/World_History_Volume_1/results/results.json") as f:
    mega = json.load(f)["results"]

mega_map = {x["question"]: x["answer"] for x in mega}

results = []

for i, item in enumerate(peg):

    q = item["question"]

    peg_answer = item["answer"]
    mega_answer = mega_map[q]

    prompt = f"""
Question:
{q}

Answer A (PEG):
{peg_answer}

Answer B (MegaRAG):
{mega_answer}

Evaluate:

1. Semantic Similarity (1-5)
2. Completeness of PEG relative to MegaRAG (1-5)
3. Factual Consistency (1-5)

Winner:
PEG / MegaRAG / Tie

Return ONLY JSON:

{{
  "similarity": 0,
  "completeness": 0,
  "factual_consistency": 0,
  "winner": ""
}}
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5:7b",
            "prompt": prompt,
            "stream": False
        }
    )

    text = response.json()["response"]

    print(f"[{i+1}/{len(peg)}] done")

    results.append({
        "question": q,
        "judge_output": text
    })

with open("llm_judge_results_125.json", "w") as f:
    json.dump(results, f, indent=2)

print("saved -> llm_judge_results_125.json")
