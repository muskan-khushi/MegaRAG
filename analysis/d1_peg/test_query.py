import json
import asyncio
import torch
import numpy as np

from transformers import AutoModel
from megarag.llms.hf import hf_gme_embed

QUESTION = """
How did trade networks facilitate cultural exchanges between Ancient Mesopotamia and the Indus Valley Civilization?
"""

with open("chunk_metadata.json") as f:
    meta = json.load(f)

X = np.load("chunk_embeddings.npy")

X = torch.tensor(X, dtype=torch.float32)
X = torch.nn.functional.normalize(X, dim=1)

print("Loading GME model...")

model = AutoModel.from_pretrained(
    "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
    torch_dtype=torch.float16,
    device_map="cuda",
    trust_remote_code=True,
).eval()

q = asyncio.run(
    hf_gme_embed(
        embed_model=model,
        texts=[QUESTION],
        is_query=True
    )
)

q = torch.tensor(q[0], dtype=torch.float32)
q = torch.nn.functional.normalize(q, dim=0)

scores = X @ q

topk = torch.topk(scores, k=5)

print("\nQUESTION:")
print(QUESTION)

for rank, idx in enumerate(topk.indices.tolist(), start=1):
    print("\n" + "=" * 80)
    print(f"RANK {rank}")
    print("NODE:", idx)
    print("SCORE:", float(scores[idx]))
    print(meta[idx]["content"][:1000])
