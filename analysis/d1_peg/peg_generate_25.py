import json
import re
import asyncio
import subprocess
import torch
import numpy as np

from transformers import AutoModel
from megarag.llms.hf import hf_gme_embed

QUERY_FILE = "../../egs/world_history_full/data/queries.txt"

# -----------------------
# Load questions
# -----------------------
with open(QUERY_FILE, "r", encoding="utf-8") as f:
    text = f.read()

questions = re.findall(r"- Question \d+:\s+(.+)", text)
questions = questions[:25]

print("Questions:", len(questions))

# -----------------------
# Load metadata
# -----------------------
with open("chunk_metadata.json") as f:
    metadata = json.load(f)

# -----------------------
# Load chunk embeddings
# -----------------------
X = np.load("chunk_embeddings.npy")

X = torch.tensor(X, dtype=torch.float32)
X = torch.nn.functional.normalize(X, dim=1)

# -----------------------
# Load GME model
# -----------------------
print("Loading GME...")

model = AutoModel.from_pretrained(
    "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
    trust_remote_code=True,
    device_map="cuda",
    torch_dtype=torch.float16
).eval()

results = []

for i, question in enumerate(questions):

    print(f"\n[{i+1}/25] {question[:60]}...")

    q_emb = asyncio.run(
        hf_gme_embed(
            embed_model=model,
            texts=[question],
            is_query=True
        )
    )

    q_tensor = torch.tensor(q_emb[0], dtype=torch.float32)
    q_tensor = torch.nn.functional.normalize(q_tensor, dim=0)

    scores = X @ q_tensor

    top5 = torch.topk(scores, k=5)

    node_ids = top5.indices.tolist()

    context = "\n\n".join(
        metadata[idx]["content"][:1500]
        for idx in node_ids
    )

    prompt = f"""
Use ONLY the supplied context.

Question:
{question}

Context:
{context}

Answer:
"""

    response = subprocess.run(
        [
            "/scratch/data/divyasaxena_rs/Muskan_internship/ollama_local/bin/ollama",
            "run",
            "qwen2.5:7b"
        ],
        input=prompt,
        text=True,
        capture_output=True
    )

    answer = response.stdout.strip()

    results.append({
        "question": question,
        "top5_nodes": node_ids,
        "answer": answer
    })

    with open("peg_answers_25.json", "w") as f:
        json.dump(results, f, indent=2)

print("\nSaved peg_answers_25.json")
