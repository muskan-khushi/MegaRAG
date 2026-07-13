import json

with open("peg_answers_25.json") as f:
    peg = json.load(f)

with open("../../egs/world_history_full/exp/World_History_Volume_1/results/results.json") as f:
    mega = json.load(f)["results"]

mega_map = {
    x["question"]: x["answer"]
    for x in mega
}

print("PEG answers:", len(peg))
print("MegaRAG answers:", len(mega))

matches = 0

for item in peg:

    q = item["question"]

    if q in mega_map:
        matches += 1

print("Question matches:", matches)

for item in peg[:5]:

    q = item["question"]

    print("\n" + "="*100)
    print("QUESTION:")
    print(q)

    print("\nPEG ANSWER:")
    print(item["answer"][:1000])

    print("\nMEGARAG ANSWER:")
    print(mega_map[q][:1000])
