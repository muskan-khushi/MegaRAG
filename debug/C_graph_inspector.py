"""
C_graph_inspector.py
────────────────────
Run immediately after build completes to validate the output in ~10 seconds.
Gives you entity count, relationship count, graph connectivity, isolated nodes,
embedding dimension check, and a sample of what was extracted.

Run after build:
    python test/C_graph_inspector.py

Or point at a specific exp dir:
    python test/C_graph_inspector.py --exp-dir egs/world_history_tiny/exp/World_History_Volume_1
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--exp-dir",
        default="egs/world_history_tiny/exp/World_History_Volume_1",
        help="Path to exp directory relative to repo root"
    )
    return p.parse_args()

def load_json(path):
    with open(path) as f:
        return json.load(f)

def check_file(path, label):
    if not path.exists():
        print(f"  ✗ MISSING  {label}: {path}")
        return False
    size_kb = path.stat().st_size / 1024
    print(f"  ✓          {label}: {size_kb:.1f} KB")
    return True

args = parse_args()
exp_dir = REPO_ROOT / args.exp_dir

print("=" * 60)
print(f"Graph Inspector — {exp_dir.name}")
print("=" * 60)
print(f"  Path: {exp_dir}")
print()

print("STEP 1: Output file presence check")
print("-" * 40)
files = {
    "vdb_entities.json":            "Entity vectors",
    "vdb_relationships.json":       "Relationship vectors",
    "vdb_chunks.json":              "Chunk/page image vectors",
    "graph_chunk_entity_relation.graphml": "Knowledge graph",
    "kv_store_text_chunks.json":    "Text chunk KV store",
    "kv_store_full_docs.json":      "Full docs KV store",
    "kv_store_doc_status.json":     "Doc status KV store",
    "llm_response_cache.json":      "LLM cache",
}
all_present = True
for fname, label in files.items():
    ok = check_file(exp_dir / fname, label)
    all_present = all_present and ok

if not all_present:
    print()
    print("Some files missing — build may have failed or is still running.")
    print("Check: grep 'Exit code' logs/build_*.out")
    sys.exit(1)

print()
print("STEP 2: Entity analysis")
print("-" * 40)

try:
    ent_db = load_json(exp_dir / "vdb_entities.json")
    entities = ent_db.get("data", [])
    print(f"  Total entities: {len(entities)}")

    if entities:
        # Type distribution
        type_counts = Counter(e.get("entity_type", "unknown") for e in entities)
        print(f"  Entity types:")
        for etype, count in type_counts.most_common():
            bar = "█" * min(count, 30)
            print(f"    {etype:35s} {count:3d}  {bar}")

        # Embedding dimension check
        sample_vec = entities[0].get("__vector__")
        if sample_vec:
            dim = len(sample_vec)
            print(f"  Embedding dim: {dim} {'✓ (expected 1536)' if dim == 1536 else '✗ UNEXPECTED'}")
        else:
            print(f"  Embedding dim: NOT FOUND in first record")

        # Sample entities
        print(f"  Sample entities (first 8):")
        for e in entities[:8]:
            name = e.get("entity_name", "?")
            etype = e.get("entity_type", "?")
            desc = e.get("description", "")[:60]
            print(f"    [{etype}] {name}: {desc}...")

        # Description length distribution
        desc_lens = [len(e.get("description", "")) for e in entities]
        print(f"  Description lengths: min={min(desc_lens)}, "
              f"avg={sum(desc_lens)//len(desc_lens)}, max={max(desc_lens)}")

        # Check for suspiciously short descriptions (LLM hallucination indicator)
        short = [e for e in entities if len(e.get("description", "")) < 20]
        if short:
            print(f"  WARNING: {len(short)} entities with very short descriptions (<20 chars):")
            for e in short[:3]:
                print(f"    '{e.get('entity_name')}': '{e.get('description')}'")

except Exception as e:
    print(f"  FAILED: {e}")

print()
print("STEP 3: Relationship analysis")
print("-" * 40)

try:
    rel_db = load_json(exp_dir / "vdb_relationships.json")
    rels = rel_db.get("data", [])
    print(f"  Total relationships: {len(rels)}")

    if rels:
        # Weight distribution
        weights = [r.get("weight", 1.0) for r in rels]
        print(f"  Weight: min={min(weights):.1f}, avg={sum(weights)/len(weights):.1f}, max={max(weights):.1f}")

        # Most connected entities (by relationship count)
        entity_degree = Counter()
        for r in rels:
            entity_degree[r.get("src_id", "")] += 1
            entity_degree[r.get("tgt_id", "")] += 1
        print(f"  Most connected entities:")
        for entity, degree in entity_degree.most_common(5):
            print(f"    {entity}: {degree} connections")

        # Sample relationships
        print(f"  Sample relationships (first 5):")
        for r in rels[:5]:
            src = r.get("src_id", "?")
            tgt = r.get("tgt_id", "?")
            kw  = r.get("keywords", "?")[:40]
            print(f"    {src} → {tgt}  [{kw}]")

except Exception as e:
    print(f"  FAILED: {e}")

print()
print("STEP 4: Chunk/page image vector analysis")
print("-" * 40)

try:
    chunk_db = load_json(exp_dir / "vdb_chunks.json")
    chunks = chunk_db.get("data", [])
    print(f"  Total chunk vectors: {len(chunks)}")
    print(f"  (Should equal number of pages processed — 10 for world_history_tiny)")

    if chunks:
        sample_vec = chunks[0].get("__vector__")
        if sample_vec:
            arr = np.array(sample_vec)
            print(f"  Embedding dim: {len(arr)}")
            print(f"  Vector norm (page 0): {np.linalg.norm(arr):.3f} "
                  f"(GME uses dot-product, not normalized — norm != 1 is expected)")

except Exception as e:
    print(f"  FAILED: {e}")

print()
print("STEP 5: Doc status check")
print("-" * 40)

try:
    doc_status_raw = load_json(exp_dir / "kv_store_doc_status.json")
    # LightRAG stores as {doc_id: {status: ..., ...}}
    docs = doc_status_raw
    status_counts = Counter()
    for doc_id, info in docs.items():
        if isinstance(info, dict):
            status_counts[info.get("status", "unknown")] += 1

    print(f"  Document statuses:")
    for status, count in status_counts.items():
        icon = "✓" if status == "PROCESSED" else "✗"
        print(f"    {icon} {status}: {count}")

    failed_docs = {k: v for k, v in docs.items()
                   if isinstance(v, dict) and v.get("status") == "FAILED"}
    if failed_docs:
        print(f"  FAILED documents:")
        for doc_id, info in failed_docs.items():
            print(f"    {doc_id}: {info.get('error', 'no error message')[:100]}")

except Exception as e:
    print(f"  FAILED: {e}")

print()
print("STEP 6: GraphML sanity check")
print("-" * 40)

try:
    graphml_path = exp_dir / "graph_chunk_entity_relation.graphml"
    content = graphml_path.read_text()
    node_count = content.count("<node ")
    edge_count = content.count("<edge ")
    print(f"  GraphML nodes: {node_count}")
    print(f"  GraphML edges: {edge_count}")

    # Isolated node check (nodes with no edges)
    import re
    node_ids = set(re.findall(r'<node id="([^"]+)"', content))
    edge_nodes = set(re.findall(r'source="([^"]+)"', content))
    edge_nodes |= set(re.findall(r'target="([^"]+)"', content))
    isolated = node_ids - edge_nodes
    print(f"  Isolated nodes (no edges): {len(isolated)}")
    if isolated and len(isolated) <= 10:
        print(f"  Isolated node IDs: {list(isolated)[:5]}")

    ratio = edge_count / max(node_count, 1)
    print(f"  Edge/node ratio: {ratio:.2f} "
          f"({'healthy' if ratio > 0.5 else 'LOW — LLM may not have extracted relationships'})")

except Exception as e:
    print(f"  FAILED: {e}")

print()
print("STEP 7: LLM cache statistics")
print("-" * 40)

try:
    cache = load_json(exp_dir / "llm_response_cache.json")
    total_cached = len(cache)
    print(f"  Cached LLM calls: {total_cached}")
    print(f"  (Higher = faster reruns; 10-page build typically caches 20-30 calls)")
    cache_types = Counter()
    for key in cache.keys():
        # cache keys are like "default:extract:hash" or "default:query:hash"
        parts = key.split(":")
        if len(parts) >= 2:
            cache_types[parts[1]] += 1
    if cache_types:
        print(f"  Cache type breakdown:")
        for ctype, count in cache_types.most_common():
            print(f"    {ctype}: {count}")
except Exception as e:
    print(f"  FAILED: {e}")

print()
print("=" * 60)
print("FINAL VERDICT")
print("=" * 60)

try:
    n_entities = len(load_json(exp_dir / "vdb_entities.json").get("data", []))
    n_rels = len(load_json(exp_dir / "vdb_relationships.json").get("data", []))
    n_chunks = len(load_json(exp_dir / "vdb_chunks.json").get("data", []))

    health = "HEALTHY ✓" if n_entities > 10 and n_rels > 5 and n_chunks > 0 else "NEEDS ATTENTION ✗"
    print(f"  {health}")
    print(f"  {n_entities} entities | {n_rels} relationships | {n_chunks} page vectors")

    if n_entities < 10:
        print()
        print("  SUGGESTION: Very few entities extracted. Try:")
        print("    1. Increase example_number to 3 in addon_params.yaml")
        print("    2. Check build log for Ollama errors or timeouts")
        print("    3. Verify Ollama was running during build (search 'Ollama' in .out log)")

    if n_rels == 0:
        print()
        print("  SUGGESTION: Zero relationships. The LLM extracted entities but")
        print("  didn't generate relationship tuples. Check the build log for")
        print("  'Chunk X extracted N Ent + 0 Rel' — if all chunks show 0 Rel,")
        print("  the model isn't following the format. Try example_number: 3.")

except Exception as e:
    print(f"  Could not compute verdict: {e}")
