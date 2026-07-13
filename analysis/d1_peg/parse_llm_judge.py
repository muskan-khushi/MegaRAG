import json
import re

with open("llm_judge_results.json") as f:
    data = json.load(f)

sim_scores = []
comp_scores = []
fact_scores = []

wins = {
    "PEG": 0,
    "MegaRAG": 0,
    "Tie": 0,
    "Unknown": 0
}

for item in data:

    text = item["judge_output"]

    try:
        # extract first JSON block
        m = re.search(r"\{.*\}", text, re.DOTALL)

        if not m:
            raise ValueError()

        obj = json.loads(m.group())

        sim = int(obj["similarity"])
        comp = int(obj["completeness"])
        fact = int(obj["factual_consistency"])

        winner = obj["winner"].strip()

        sim_scores.append(sim)
        comp_scores.append(comp)
        fact_scores.append(fact)

        if winner in wins:
            wins[winner] += 1
        else:
            wins["Unknown"] += 1

    except Exception:

        print("\nFAILED PARSE:")
        print(text[:500])

def avg(x):
    return sum(x)/len(x) if x else 0

print("\n" + "="*60)
print("LLM JUDGE SUMMARY")
print("="*60)

print(f"Questions judged : {len(sim_scores)}")

print()
print(f"Mean Similarity          : {avg(sim_scores):.2f}/5")
print(f"Mean Completeness        : {avg(comp_scores):.2f}/5")
print(f"Mean Factual Consistency : {avg(fact_scores):.2f}/5")

print()
print("WIN COUNTS")
for k,v in wins.items():
    print(f"{k:10s} : {v}")

print()
print("WIN RATES")

total = sum(wins.values())

for k,v in wins.items():
    if total:
        print(f"{k:10s} : {100*v/total:.1f}%")
