import json
import torch
import numpy as np
from transformers import AutoModel

print("Loading model...")

model = AutoModel.from_pretrained(
    "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
    trust_remote_code=True,
    device_map="cuda"
).eval()

chunk_embeddings = np.load("chunk_embeddings.npy")

with open("chunk_metadata.json") as f:
    metadata = json.load(f)

# Normalize chunk embeddings
chunk_tensor = torch.tensor(chunk_embeddings, dtype=torch.float32)
chunk_tensor = torch.nn.functional.normalize(chunk_tensor, dim=1)

questions = [
    "How did trade networks facilitate cultural exchanges between Ancient Mesopotamia and the Indus Valley Civilization?",
    "In what ways did Ancient Egypt's social structure influence its development and interaction with neighboring civilizations?",
    "How can understanding the governance systems of the Persian Empire enhance students’ understanding of ancient political structures?",
    "What lessons about empire-building can students learn from examining the rise and fall of the Kingdom of Kush?",
    "How did the rise of Islam influence governance and culture in regions it dominated?"
]

for q in questions:

    print("\n" + "=" * 100)
    print("QUESTION:")
    print(q)

    with torch.no_grad():
        q_emb = model.get_text_embeddings(
            texts=[q],
            instruction="Find an image that matches the given text.",
            is_query=True
        )

    q_tensor = q_emb.detach().cpu().float()[0]
    q_tensor = torch.nn.functional.normalize(q_tensor, dim=0)

    sims = (chunk_tensor @ q_tensor).numpy()

    top5 = np.argsort(sims)[::-1][:5]

    print("\nTOP 5 RESULTS")

    for rank, node in enumerate(top5, 1):

        print("\n" + "-" * 80)
        print("RANK:", rank)
        print("NODE:", int(node))
        print("SCORE:", float(sims[node]))

        print(metadata[node]["content"][:500])
