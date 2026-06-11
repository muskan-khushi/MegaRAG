"""
A_query_smoke_test.py
─────────────────────
Tests the query pipeline without needing a real built MMKG.
Creates a minimal dummy exp/ directory, then runs aquery() against it.
Catches import errors, storage init errors, and LLM connection errors early.

Run on LOGIN NODE (no GPU needed — GME model not loaded here):
    python test/A_query_smoke_test.py

What it checks:
    1. All imports resolve correctly
    2. MegaRAG initialises storages without crashing
    3. Ollama is reachable at localhost:11434
    4. A raw LLM call to qwen2.5:7b works (text-only, no GPU)
    5. Keyword extraction from a query works
    6. The query pipeline runs end-to-end on empty stores (expected: fail_response)
"""

import os
import sys
import json
import asyncio
import tempfile
from pathlib import Path

# ── make sure we're running from repo root ─────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "lib" / "LightRAG"))

print("=" * 60)
print("STEP 1: Import check")
print("=" * 60)

try:
    import torch
    print(f"  torch           OK  ({torch.__version__})")
except ImportError as e:
    print(f"  torch           FAIL  {e}")

try:
    import transformers
    print(f"  transformers    OK  ({transformers.__version__})")
except ImportError as e:
    print(f"  transformers    FAIL  {e}")

try:
    from megarag import MegaRAG
    print(f"  megarag         OK")
except ImportError as e:
    print(f"  megarag         FAIL  {e}")
    sys.exit(1)

try:
    from lightrag.base import QueryParam
    from lightrag.kg.shared_storage import initialize_pipeline_status
    from lightrag.utils import TokenTracker, wrap_embedding_func_with_attrs
    print(f"  lightrag        OK")
except ImportError as e:
    print(f"  lightrag        FAIL  {e}")
    sys.exit(1)

try:
    from megarag.llms.openai import gpt_4o_mini_complete
    print(f"  megarag.llms    OK")
except ImportError as e:
    print(f"  megarag.llms    FAIL  {e}")
    sys.exit(1)

print()
print("=" * 60)
print("STEP 2: Ollama connectivity check")
print("=" * 60)

import urllib.request
import urllib.error

def check_ollama():
    try:
        req = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
        data = json.loads(req.read())
        models = [m["name"] for m in data.get("models", [])]
        print(f"  Ollama reachable: YES")
        print(f"  Available models: {models}")
        has_qwen = any("qwen" in m for m in models)
        print(f"  qwen2.5:7b present: {'YES ✓' if has_qwen else 'NO ✗ — build job may fail'}")
        return has_qwen
    except urllib.error.URLError as e:
        print(f"  Ollama reachable: NO — {e}")
        print(f"  (This is expected on the login node if Ollama isn't started yet)")
        print(f"  The SLURM job starts Ollama automatically — this check is for interactive use")
        return False

ollama_ok = check_ollama()

print()
print("=" * 60)
print("STEP 3: Raw LLM call (text-only, no GPU)")
print("=" * 60)

async def test_raw_llm():
    if not ollama_ok:
        print("  SKIPPED — Ollama not reachable")
        return False
    try:
        token_tracker = TokenTracker()
        result = await gpt_4o_mini_complete(
            prompt="Reply with exactly one word: hello",
            system_prompt="You are a test assistant. Follow instructions exactly.",
            token_tracker=token_tracker,
        )
        print(f"  LLM response: '{result.strip()}'")
        print(f"  Tokens used:  {token_tracker}")
        return True
    except Exception as e:
        print(f"  LLM call FAILED: {e}")
        return False

llm_ok = asyncio.run(test_raw_llm())

print()
print("=" * 60)
print("STEP 4: MegaRAG storage initialisation (dummy working_dir)")
print("=" * 60)

async def test_storage_init():
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"  Using temp dir: {tmpdir}")
        try:
            @wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=32768)
            async def dummy_embed(texts=[], images=[], is_query=False):
                import numpy as np
                n = max(len(texts), len(images), 1)
                return np.random.randn(n, 1536).astype("float32")

            token_tracker = TokenTracker()

            async def dummy_llm(prompt, input_images=None, system_prompt=None,
                                history_messages=None, keyword_extraction=False, **kwargs):
                if keyword_extraction:
                    # Must return valid JSON for keyword extraction
                    return json.dumps({
                        "high_level_keywords": ["test keyword"],
                        "low_level_keywords": ["entity"]
                    })
                return "This is a dummy LLM response for testing."

            addon_params = {
                "example_number": 1,
                "language": "English",
                "entity_types": ["person", "location", "event"],
                "entity_extract_max_gleaning": 0,
                "entity_refine_max_times": 0,
                "refine_subgraph_top_k": 10,
                "refine_subgraph_max_token_for_global_context": 1000,
                "refine_subgraph_max_token_for_local_context": 1000,
                "refine_subgraph_max_token_for_text_unit": 1000,
                "chunk_top_k": 3,
                "embed_parallel_limit": 1,
            }

            rag = MegaRAG(
                working_dir=tmpdir,
                llm_model_func=dummy_llm,
                embedding_func=dummy_embed,
                addon_params=addon_params,
            )
            await rag.initialize_storages()
            await initialize_pipeline_status()
            print(f"  Storage init:   OK ✓")

            # Try a query — should return fail_response since stores are empty
            param = QueryParam(mode="naive", chunk_top_k=3, enable_rerank=False)
            response = await rag.aquery("What is the capital of France?", param=param)
            print(f"  Query response: '{str(response)[:80]}...'")
            print(f"  (Expected: fail_response or empty — stores are empty)")
            return True

        except Exception as e:
            import traceback
            print(f"  FAILED: {e}")
            print(traceback.format_exc())
            return False

storage_ok = asyncio.run(test_storage_init())

print()
print("=" * 60)
print("STEP 5: Keyword extraction logic check")
print("=" * 60)

async def test_keyword_extraction():
    """Test that the keyword extraction prompt produces parseable output."""
    if not ollama_ok:
        print("  SKIPPED — Ollama not reachable")
        return False
    try:
        from lightrag.operate import get_keywords_from_query
        from lightrag.base import QueryParam
        from lightrag.utils import wrap_embedding_func_with_attrs
        import numpy as np

        @wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=32768)
        async def dummy_embed(texts=[], images=[], is_query=False):
            n = max(len(texts), len(images), 1)
            return np.random.randn(n, 1536).astype("float32")

        token_tracker = TokenTracker()

        async def real_llm(prompt, input_images=None, system_prompt=None,
                           history_messages=None, keyword_extraction=False, **kwargs):
            return await gpt_4o_mini_complete(
                prompt=prompt,
                input_images=input_images,
                system_prompt=system_prompt,
                history_messages=history_messages,
                keyword_extraction=keyword_extraction,
                token_tracker=token_tracker,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            rag = MegaRAG(
                working_dir=tmpdir,
                llm_model_func=real_llm,
                embedding_func=dummy_embed,
                addon_params={
                    "example_number": 1,
                    "language": "English",
                    "entity_types": ["person", "location"],
                    "entity_extract_max_gleaning": 0,
                    "entity_refine_max_times": 0,
                    "refine_subgraph_top_k": 10,
                    "refine_subgraph_max_token_for_global_context": 1000,
                    "refine_subgraph_max_token_for_local_context": 1000,
                    "refine_subgraph_max_token_for_text_unit": 1000,
                    "chunk_top_k": 3,
                    "embed_parallel_limit": 1,
                },
            )
            await rag.initialize_storages()
            await initialize_pipeline_status()

            from dataclasses import asdict
            param = QueryParam(mode="hybrid", chunk_top_k=3, enable_rerank=False)
            query = "How did ancient Roman trade networks connect the Mediterranean world?"

            hl_kw, ll_kw = await get_keywords_from_query(
                query, param, asdict(rag), rag.llm_response_cache
            )
            print(f"  Query: '{query}'")
            print(f"  High-level keywords: {hl_kw}")
            print(f"  Low-level keywords:  {ll_kw}")
            print(f"  Tokens used: {token_tracker}")
            kw_ok = len(hl_kw) > 0 or len(ll_kw) > 0
            print(f"  Result: {'OK ✓' if kw_ok else 'WARNING — empty keywords, query mode may fall back'}")
            return kw_ok

    except Exception as e:
        import traceback
        print(f"  FAILED: {e}")
        print(traceback.format_exc())
        return False

kw_ok = asyncio.run(test_keyword_extraction())

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
results = {
    "Imports":          True,   # if we got here, imports passed
    "Ollama reachable": ollama_ok,
    "LLM call":         llm_ok,
    "Storage init":     storage_ok,
    "Keyword extract":  kw_ok,
}
for k, v in results.items():
    status = "✓ PASS" if v else "✗ FAIL/SKIP"
    print(f"  {k:25s} {status}")

all_critical = results["Imports"] and results["Storage init"]
print()
if all_critical:
    print("Critical path: OK — query pipeline should work once build completes.")
else:
    print("Critical path: ISSUES FOUND — fix before running query job.")
