import json
import asyncio
import torch
import numpy as np

from transformers import AutoModel
from megarag.llms.hf import hf_gme_embed

QUESTIONS = [
    "How did trade networks facilitate cultural exchanges between Ancient Mesopotamia and the Indus Valley Civilization?",
    "In what ways did Ancient Egypt's social structure influence its development and interaction with neighboring civilizations?",
    "How can understanding the governance systems of the Persian Empire enhance students’ understanding of ancient political structures?",
    "What lessons about empire-building can students learn from examining the rise and fall of the Kingdom of Kush?",
    "How did the rise of Islam influence governance and culture in regions it dominated?"
]

with open("chunk_metadata.json") as f:
    meta = json.load(f)

X = np.load("chunk_embeddings.npy")
X = torch.tensor(X, dtype=torch.float32)
X = torch.nn.functional.normalize(X, dim=1)

print("Loading model...")

model = AutoModel.from_pretrained(
    "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
    torch_dtype=torch.float16,
    device_map="cuda",
    trust_remote_code=True,
).eval()

for qtext in QUESTIONS:

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

    print("\n" + "="*100)
    print("QUESTION:")
    print(qtext)

    print("\nBEST NODE:", best)
    print("SCORE:", float(scores[best]))

    print("\nTOP CHUNK:")
    print(meta[best]["content"][:1200])
