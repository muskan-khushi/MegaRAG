import json
import torch
import numpy as np

with open("chunk_metadata.json") as f:
    meta = json.load(f)

X = np.load("chunk_embeddings.npy")

X = torch.tensor(X, dtype=torch.float32)
X = torch.nn.functional.normalize(X, dim=1)

node = 107

sim = X @ X.T
vals, idx = torch.topk(sim, k=6, dim=1)

print("SOURCE NODE:", node)
print(meta[node]["content"][:1200])

for j in range(1,6):
    nbr = idx[node][j].item()

    print("\n" + "="*80)
    print("NEIGHBOR:", nbr)
    print("SIMILARITY:", vals[node][j].item())
    print(meta[nbr]["content"][:1200])
