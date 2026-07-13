import json

with open("peg_answers_25.json") as f:
    peg = json.load(f)

with open("../../egs/world_history_full/exp/World_History_Volume_1/results/results.json") as f:
    mega = json.load(f)["results"]

mega_map = {x["question"]: x["answer"] for x in mega}

with open("all_comparisons.txt", "w", encoding="utf-8") as out:

    for i, item in enumerate(peg, 1):

        q = item["question"]

        out.write("=" * 120 + "\n")
        out.write(f"QUESTION {i}\n")
        out.write("=" * 120 + "\n\n")

        out.write("QUESTION:\n")
        out.write(q + "\n\n")

        out.write("PEG ANSWER:\n")
        out.write(item["answer"] + "\n\n")

        out.write("MEGARAG ANSWER:\n")
        out.write(mega_map.get(q, "NOT FOUND") + "\n\n")

        out.write("\n\n")
