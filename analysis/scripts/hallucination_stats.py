import json
import re

f = "egs/world_history_tiny/exp/World_History_Volume_1/intermediate_answers.jsonl"

for i,line in enumerate(open(f),1):
    d = json.loads(line)

    text = d["fusion_answer"]

    urls = len(re.findall(r'https?://', text))

    paths = len(re.findall(
        r'/(?:[^ \n]+)\.(?:pdf|json|md|docx|txt|png|jpg)',
        text
    ))

    docs = len(re.findall(
        r'\b[\w\-]+\.(?:pdf|json|md|docx|txt|png|jpg)\b',
        text
    ))

    placeholders = len(re.findall(
        r'<[^>]+>',
        text
    ))

    total = urls + paths + docs + placeholders

    print(
        f"Q{i}: "
        f"URLs={urls} "
        f"Paths={paths} "
        f"Docs={docs} "
        f"Placeholders={placeholders} "
        f"Total={total}"
    )
