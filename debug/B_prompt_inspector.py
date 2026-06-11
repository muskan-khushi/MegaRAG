"""
B_prompt_inspector.py
─────────────────────
Prints EXACTLY what prompt + image list gets sent to Ollama for page 0
of World History, without actually calling the LLM.

Run on LOGIN NODE:
    python test/B_prompt_inspector.py

What it checks:
    1. pages_content.json loads correctly (if dumps/ exists from a previous run)
    2. The chunking function produces the right structure
    3. The entity extraction prompt is assembled correctly
    4. Image paths exist on disk
    5. Token count of the prompt (so you know if it fits in qwen2.5:7b context)
    6. Prints a truncated preview of what the LLM would actually receive
"""

import os
import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "lib" / "LightRAG"))

print("=" * 60)
print("STEP 1: Locate pages_content.json")
print("=" * 60)

# Try to find a real pages_content.json from any previous build attempt
search_paths = [
    REPO_ROOT / "egs" / "world_history_tiny" / "dumps" / "World_History_Volume_1" / "pages_content.json",
    REPO_ROOT / "egs" / "world_history" / "dumps" / "World_History_Volume_1" / "pages_content.json",
]

pages_content_path = None
for p in search_paths:
    if p.exists():
        pages_content_path = p
        print(f"  Found: {p}")
        break

if pages_content_path is None:
    print("  No pages_content.json found — using synthetic data for prompt inspection")
    print("  (Run the build first to get real data, then re-run this script)")
    # Build a minimal synthetic pages_content to still test prompt assembly
    pages_content = {
        "0": {
            "text": "Section Summaries\nSection summaries distill the information in each section. Key Terms are bold and followed by definitions. Assessments include Review Questions and Application Questions.\nAbout the Authors: Ann Kordas (Johnson & Wales University), Ryan J. Lynch (Columbus State University).",
            "page_image": "",   # no real image available
            "figure_images": [],
        }
    }
    using_synthetic = True
else:
    with open(pages_content_path) as f:
        pages_content = json.load(f)
    using_synthetic = False

print(f"  Pages available: {len(pages_content)}")
for idx in list(pages_content.keys())[:3]:
    p = pages_content[idx]
    print(f"  Page {idx}: text={len(p.get('text',''))} chars, "
          f"page_image={'EXISTS' if p.get('page_image') and Path(p['page_image']).exists() else 'MISSING/EMPTY'}, "
          f"fig_imgs={len(p.get('figure_images', []))}")

print()
print("=" * 60)
print("STEP 2: Run chunking function on pages_content")
print("=" * 60)

try:
    from lightrag.utils import Tokenizer
    from megarag.operate import chunking_by_token_or_page

    # Tokenizer is instantiated internally by LightRAG — replicate here
    tokenizer = Tokenizer(model_name="gpt-4o")

    chunks = chunking_by_token_or_page(
        tokenizer=tokenizer,
        content=json.dumps(pages_content),
        split_by_page=True,
    )
    print(f"  Total chunks produced: {len(chunks)}")
    for i, c in enumerate(chunks[:5]):
        print(f"  Chunk {i}: tokens={c['tokens']}, "
              f"page_img={'set' if c.get('page_img') else 'empty'}, "
              f"fig_imgs={len(c.get('fig_imgs', []))}, "
              f"text_preview='{c['content'][:60].strip()}...'")
except Exception as e:
    import traceback
    print(f"  FAILED: {e}")
    print(traceback.format_exc())
    chunks = []

print()
print("=" * 60)
print("STEP 3: Assemble entity extraction prompt for page 0")
print("=" * 60)

try:
    from megarag.prompt import PROMPTS

    language = "English"
    entity_types = [
        "time_period", "geographical_location", "civilization_or_empire",
        "historical_concept_or_event", "historical_figure",
        "cultural_or_religious_movement", "source_or_artifact", "textbook_structure"
    ]
    example_number = 1
    examples_raw = PROMPTS["multimodal_entity_extraction_examples"]
    examples = "\n".join(examples_raw[:example_number])

    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=", ".join(entity_types),
        language=language,
    )
    examples_formatted = examples.format(**context_base)

    # Use chunk 0 if available, else synthetic
    if chunks:
        chunk = chunks[0]
    else:
        chunk = {
            "content": pages_content["0"]["text"],
            "page_img": pages_content["0"].get("page_image", ""),
            "fig_imgs": pages_content["0"].get("figure_images", []),
        }

    prompt = PROMPTS["multimodal_entity_extraction_init"].format(
        **{
            **context_base,
            "examples": examples_formatted,
            "input_text": chunk["content"],
        }
    )

    # Check images
    images = []
    if chunk.get("page_img"):
        images.append(chunk["page_img"])
    images += [img for img in chunk.get("fig_imgs", []) if img]

    print(f"  Prompt length: {len(prompt)} characters")
    print(f"  Images to send: {len(images)}")
    for img in images:
        exists = Path(img).exists()
        print(f"    {'✓' if exists else '✗ MISSING'} {img}")

    # Token count
    try:
        from lightrag.utils import Tokenizer
        tok = Tokenizer(model_name="gpt-4o")
        token_count = len(tok.encode(prompt))
        print(f"  Prompt token count: {token_count}")
        if token_count > 6000:
            print(f"  WARNING: prompt is large — qwen2.5:7b context is 32k but "
                  f"long prompts slow extraction significantly")
        else:
            print(f"  Token count OK ✓")
    except Exception:
        print(f"  (Token count skipped — tokenizer not available)")

    print()
    print("  ── PROMPT PREVIEW (first 1500 chars) ──────────────────")
    print(prompt[:1500])
    print("  ... [truncated] ...")
    print("  ── END PREVIEW ─────────────────────────────────────────")

except Exception as e:
    import traceback
    print(f"  FAILED: {e}")
    print(traceback.format_exc())

print()
print("=" * 60)
print("STEP 4: Image path audit across ALL pages")
print("=" * 60)

if not using_synthetic:
    missing_page_imgs = []
    missing_fig_imgs = []
    empty_page_imgs = []

    for idx, page in pages_content.items():
        pi = page.get("page_image", "")
        if not pi:
            empty_page_imgs.append(idx)
        elif not Path(pi).exists():
            missing_page_imgs.append((idx, pi))

        for fi in page.get("figure_images", []):
            if fi and not Path(fi).exists():
                missing_fig_imgs.append((idx, fi))

    print(f"  Pages with empty page_image:   {len(empty_page_imgs)}")
    print(f"  Pages with MISSING page_image: {len(missing_page_imgs)}")
    print(f"  Missing figure images:          {len(missing_fig_imgs)}")

    if missing_page_imgs:
        print("  First 5 missing page images:")
        for idx, path in missing_page_imgs[:5]:
            print(f"    Page {idx}: {path}")

    if not missing_page_imgs and not missing_fig_imgs:
        print("  All image paths valid ✓")
else:
    print("  SKIPPED — using synthetic data")

print()
print("=" * 60)
print("STEP 5: Refinement prompt assembly check")
print("=" * 60)

try:
    refine_prompt_template = PROMPTS["multimodal_entity_extraction_refine"]
    # Check it has all required format keys
    required_keys = [
        "{language}", "{entity_types}", "{examples}", "{input_text}", "{kg_context}",
        "{tuple_delimiter}", "{record_delimiter}", "{completion_delimiter}"
    ]
    missing_keys = [k for k in required_keys if k not in refine_prompt_template]
    if missing_keys:
        print(f"  WARNING: Refinement prompt missing keys: {missing_keys}")
    else:
        print(f"  Refinement prompt template: OK ✓ (all required keys present)")

    # Try formatting it
    test_refine = refine_prompt_template.format(
        **{**context_base, "examples": examples_formatted,
           "input_text": "test text", "kg_context": "test kg context"}
    )
    print(f"  Refinement prompt renders without error ✓")
    print(f"  Refinement prompt length: {len(test_refine)} chars")

except Exception as e:
    print(f"  FAILED: {e}")

print()
print("Done — run this again after build to audit real image paths.")
