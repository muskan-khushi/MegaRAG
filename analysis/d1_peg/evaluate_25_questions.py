import json
import re
import asyncio
import torch
import numpy as np

from transformers import AutoModel
from megarag.llms.hf import hf_gme_embed

QUERY_FILE = "../../egs/world_history_full/data/queries.txt"

# ------------------------
# Extract questions
# ------------------------
with open(QUERY_FILE, "r", encoding="utf-8") as f:
    text = f.read()

questions = re.findall(r"- Question \d+:\s+(.+)", text)

questions = questions[:25]

print("Questions loaded:", len(questions))

# ------------------------
# Load metadata
# ------------------------
with open("chunk_metadata.json") as f:
    meta = json.load(f)

# ------------------------
# Load chunk embeddings
# ------------------------
X = np.load("chunk_embeddings.npy")

X = torch.tensor(X, dtype=torch.float32)
X = torch.nn.functional.normalize(X, dim=1)

# ------------------------
# Load GME
# ------------------------
print("Loading GME model...")

model = AutoModel.from_pretrained(
    "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
    torch_dtype=torch.float16,
    device_map="cuda",
    trust_remote_code=True,
).eval()

results = []

# ------------------------
# Retrieval loop
# ------------------------
for i, qtext in enumerate(questions):

    q = asyncio.run(
        hf_gme_embed(
            embed_model=model,
            texts=[qtext],
            is_query=True
        )
    )

    q = torch.tensor(q[0], dtype=torch.float32)
    q = torch.nn.functional.normalize(q, dim=0)

    scores = X @ q

    best = torch.argmax(scores).item()

    results.append({
        "question_id": i + 1,
        "question": qtext,
        "node": best,
        "score": float(scores[best]),
        "chunk_text": meta[best]["content"][:1500]
    })

    print(f"{i+1}/25 complete")

# ------------------------
# Save
# ------------------------
with open("retrieval_results_25.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved retrieval_results_25.json")
