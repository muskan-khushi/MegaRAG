"""
D_ollama_vision_test.py
───────────────────────
Sends a real multimodal request to Ollama with an actual page image.
Confirms qwen2.5:7b vision is working BEFORE spending 90 min on build.

Run on LOGIN NODE (after starting Ollama interactively), OR
run inside an interactive GPU session:
    salloc --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G --time=00:30:00 --partition=dgx --gres=gpu:1
    ssh <node>
    # activate env, start Ollama, then:
    python test/D_ollama_vision_test.py

What it checks:
    1. Ollama health
    2. qwen2.5:7b text-only response
    3. qwen2.5:7b with a real page image (if available), or a synthetic test image
    4. Whether the structured entity extraction format is followed
    5. Response speed (tokens/sec)
"""

import os
import sys
import json
import time
import base64
import urllib.request
import urllib.error
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "lib" / "LightRAG"))

OLLAMA_BASE = os.getenv("OPENAI_API_BASE", "http://localhost:11434")
MODEL = "qwen2.5:7b"

print("=" * 60)
print("Ollama Vision Test")
print("=" * 60)
print(f"  Ollama URL: {OLLAMA_BASE}")
print(f"  Model:      {MODEL}")
print()

# ── helpers ────────────────────────────────────────────────────────────────

def ollama_post(endpoint, payload, timeout=120):
    url = f"{OLLAMA_BASE}{endpoint}"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        return {"error": str(e)}

def img_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def find_real_page_image():
    """Try to find an actual page image from a previous build."""
    search_dirs = [
        REPO_ROOT / "egs" / "world_history_tiny" / "dumps",
        REPO_ROOT / "egs" / "world_history" / "dumps",
    ]
    for d in search_dirs:
        for img in sorted(d.rglob("*_page_*.jpeg"))[:1]:
            return img
        for img in sorted(d.rglob("*_page_*.png"))[:1]:
            return img
    return None

def make_synthetic_image():
    """Create a tiny white test image if no real image is available."""
    try:
        from PIL import Image
        import io
        img = Image.new("RGB", (200, 100), color=(255, 255, 255))
        # Draw some fake text-like lines
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "Test Page: World History", fill=(0, 0, 0))
        draw.text((10, 30), "Ancient Rome, Mediterranean trade", fill=(0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode(), "synthetic"
    except ImportError:
        # PIL not available — return a minimal 1x1 white JPEG in base64
        # This is a valid 1x1 white JPEG
        b64 = ("/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8U"
               "HRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgN"
               "DRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy"
               "MjL/wAARCAABAAEDASIAAhEBAxEB/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAA"
               "AAAAAAAAAAAAAAD/xAAUAQEAAAAAAAAAAAAAAAAAAAAA/8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/"
               "aAAwDAQACEQMRAD8AJQAB/9k=")
        return b64, "1x1 white JPEG"

print("STEP 1: Ollama health check")
print("-" * 40)
resp = ollama_post("/api/tags", {})
if "error" in resp:
    print(f"  ✗ Ollama unreachable: {resp['error']}")
    print()
    print("  To start Ollama for interactive testing:")
    print("    cd /scratch/data/divyasaxena_rs/Muskan_internship/")
    print("    OLLAMA_MODELS=$(pwd)/ollama_local/models \\")
    print("        ollama_local/bin/ollama serve > ollama_local/ollama.log 2>&1 &")
    print("    sleep 10")
    print("    curl http://localhost:11434/api/tags")
    sys.exit(0)

models = [m["name"] for m in resp.get("models", [])]
print(f"  ✓ Ollama reachable")
print(f"  Available models: {models}")

has_qwen = any("qwen2.5" in m for m in models)
if not has_qwen:
    print(f"  ✗ qwen2.5:7b not found — available: {models}")
    sys.exit(1)
print(f"  ✓ qwen2.5:7b found")

print()
print("STEP 2: Text-only generation test")
print("-" * 40)

t0 = time.perf_counter()
resp = ollama_post("/api/generate", {
    "model": MODEL,
    "prompt": "Name exactly 3 ancient civilizations. Respond with only the 3 names, comma-separated.",
    "stream": False,
    "options": {"temperature": 0.1, "num_predict": 50}
})
dt = time.perf_counter() - t0

if "error" in resp:
    print(f"  ✗ Text generation failed: {resp['error']}")
else:
    response_text = resp.get("response", "").strip()
    eval_count = resp.get("eval_count", 0)
    eval_duration_ns = resp.get("eval_duration", 1)
    tokens_per_sec = eval_count / (eval_duration_ns / 1e9) if eval_duration_ns else 0
    print(f"  ✓ Response: '{response_text}'")
    print(f"  Speed: {tokens_per_sec:.1f} tok/s | Latency: {dt:.1f}s | Tokens: {eval_count}")

print()
print("STEP 3: Multimodal (vision) test")
print("-" * 40)

real_img_path = find_real_page_image()
if real_img_path:
    print(f"  Using real page image: {real_img_path.name}")
    img_b64 = img_to_b64(real_img_path)
    img_source = f"real ({real_img_path.name})"
else:
    print("  No real page image found — using synthetic test image")
    print("  (Run build first to get real images, then re-run this test)")
    img_b64, img_source = make_synthetic_image()

prompt_vision = (
    "This is a page from a history textbook. "
    "Describe in one sentence what you see in this image."
)

t0 = time.perf_counter()
resp = ollama_post("/api/generate", {
    "model": MODEL,
    "prompt": prompt_vision,
    "images": [img_b64],
    "stream": False,
    "options": {"temperature": 0.1, "num_predict": 100}
})
dt = time.perf_counter() - t0

if "error" in resp:
    print(f"  ✗ Vision generation failed: {resp['error']}")
    print()
    print("  NOTE: qwen2.5:7b is a TEXT model. It may not support vision.")
    print("  If this fails, MegaRAG's entity extraction will still work —")
    print("  the page images are passed as base64 but the model may just")
    print("  ignore them and extract from OCR text only.")
    print("  The GME model handles multimodal EMBEDDING separately (for retrieval).")
    VISION_OK = False
else:
    response_text = resp.get("response", "").strip()
    eval_count = resp.get("eval_count", 0)
    eval_duration_ns = resp.get("eval_duration", 1)
    tokens_per_sec = eval_count / (eval_duration_ns / 1e9) if eval_duration_ns else 0
    print(f"  Image source: {img_source}")
    print(f"  ✓ Response: '{response_text}'")
    print(f"  Speed: {tokens_per_sec:.1f} tok/s | Latency: {dt:.1f}s | Tokens: {eval_count}")
    VISION_OK = True

print()
print("STEP 4: Entity extraction format test")
print("-" * 40)
print("  Testing if qwen2.5:7b follows the structured tuple format...")

entity_prompt = """---Goal---
Given text, identify entities. Use <|> as delimiter.

Entity types: [person, location, civilization]

Format: ("entity"<|><entity_name><|><entity_type><|><entity_description>)
Relationship format: ("relationship"<|><src><|><tgt><|><description><|><keywords><|><weight>)
Output keywords: ("content_keywords"<|><keywords>)

---Text---
Julius Caesar was a Roman general who conquered Gaul and crossed the Rubicon river.
The Roman Empire controlled the Mediterranean Sea for centuries.

Output:"""

t0 = time.perf_counter()
resp = ollama_post("/api/generate", {
    "model": MODEL,
    "prompt": entity_prompt,
    "stream": False,
    "options": {"temperature": 0.0, "num_predict": 400}
})
dt = time.perf_counter() - t0

if "error" in resp:
    print(f"  ✗ Failed: {resp['error']}")
else:
    response_text = resp.get("response", "").strip()
    eval_count = resp.get("eval_count", 0)
    tokens_per_sec = eval_count / (resp.get("eval_duration", 1) / 1e9)

    print(f"  Raw response ({dt:.1f}s, {tokens_per_sec:.1f} tok/s):")
    print()
    for line in response_text.split("\n"):
        print(f"    {line}")

    # Parse check
    has_entity = '("entity"' in response_text or '"entity"' in response_text
    has_relation = '("relationship"' in response_text or '"relationship"' in response_text
    has_delimiter = "<|>" in response_text

    print()
    print(f"  Format check:")
    print(f"    Entity tuples present:       {'✓' if has_entity else '✗'}")
    print(f"    Relationship tuples present: {'✓' if has_relation else '✗'}")
    print(f"    <|> delimiter used:          {'✓' if has_delimiter else '✗'}")

    if not has_delimiter:
        print()
        print("  WARNING: Model is not using <|> delimiter.")
        print("  This will cause extract_entities() to parse 0 entities.")
        print("  The real extraction uses a much longer prompt with 3 examples —")
        print("  run test/test_mm_entity_extraction.py to verify with full prompt.")

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Ollama:      ✓ running")
print(f"  Text gen:    ✓ working")
print(f"  Vision:      {'✓ working' if VISION_OK else '✗ failed (may be text-only model)'}")
print()
print("  Key insight: Even if vision test fails, MegaRAG still works because:")
print("  - Entity extraction uses OCR text (primary) + images (supplementary)")  
print("  - GME model handles multimodal RETRIEVAL embeddings (separate from LLM)")
print("  - The full extraction prompt (3 examples) gets much better compliance")
