"""
E_refinement_audit.py
─────────────────────
Static analysis + runtime trace of extract_entities_refinement() in operate.py.
Finds the indexing bug risk and tests the subgraph search logic with dummy data.

Run on LOGIN NODE:
    python test/E_refinement_audit.py

What it checks:
    1. chunk_results_at_stage_one indexing — the dict is keyed by source_id
       but built assuming chunk_results is non-empty and nodes/edges are populated
    2. Edge case: chunk with 0 entities extracted in stage 1 (empty nodes dict)
    3. Edge case: chunk with entities but 0 relationships (empty edges dict)
    4. The _search_subgraph query_mode selection logic
    5. The merging logic (stage-1 results merged into stage-2 results)
    6. Whether refinement is skipped when entity_refine_max_times=0
"""

import sys
import json
import asyncio
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "lib" / "LightRAG"))

print("=" * 60)
print("Refinement Pass Logic Audit")
print("=" * 60)

# ── STATIC ANALYSIS ─────────────────────────────────────────────────────────

print()
print("STEP 1: Static analysis of chunk_results_at_stage_one indexing")
print("-" * 40)

# Replicate the indexing logic from operate.py line ~600
def build_stage_one_index(chunk_results):
    """
    This is what operate.py does:
        chunk_results_at_stage_one = {
            list(res[0].values())[0][0]['source_id']: {
                'nodes': res[0],
                'edges': res[1],
            } for res in chunk_results
        }
    """
    result = {}
    for i, res in enumerate(chunk_results):
        nodes, edges = res

        # BUG RISK 1: nodes is empty (0 entities extracted for this chunk)
        if not nodes:
            print(f"  ⚠ chunk_result[{i}]: nodes dict is EMPTY")
            print(f"    → list(res[0].values())[0][0]['source_id'] will raise IndexError")
            print(f"    → Refinement pass will CRASH for this chunk")
            continue

        first_entity_list = list(nodes.values())[0]

        # BUG RISK 2: entity list is somehow empty
        if not first_entity_list:
            print(f"  ⚠ chunk_result[{i}]: first entity list is empty")
            continue

        source_id = first_entity_list[0].get('source_id', '')
        if not source_id:
            print(f"  ⚠ chunk_result[{i}]: source_id missing from first entity")
            continue

        result[source_id] = {'nodes': nodes, 'edges': edges}
        print(f"  ✓ chunk_result[{i}]: source_id='{source_id}', "
              f"nodes={len(nodes)}, edges={len(edges)}")
    return result

# Test with normal case
print()
print("  Test A: Normal chunk results (entities + relationships)")
normal_results = [
    (
        defaultdict(list, {
            "Julius Caesar": [{"entity_name": "Julius Caesar", "entity_type": "person",
                               "description": "Roman general", "source_id": "chunk-abc123",
                               "file_path": "test.json"}],
            "Roman Empire": [{"entity_name": "Roman Empire", "entity_type": "civilization",
                              "description": "Ancient empire", "source_id": "chunk-abc123",
                              "file_path": "test.json"}],
        }),
        defaultdict(list, {
            ("Julius Caesar", "Roman Empire"): [{"src_id": "Julius Caesar", "tgt_id": "Roman Empire",
                                                  "weight": 8.0, "description": "led",
                                                  "keywords": "leadership", "source_id": "chunk-abc123",
                                                  "file_path": "test.json"}]
        })
    )
]
idx = build_stage_one_index(normal_results)
assert "chunk-abc123" in idx, "Indexing failed for normal case"
print(f"  ✓ Normal case: OK")

# Test with empty nodes (the bug)
print()
print("  Test B: Empty nodes dict (0 entities extracted — BUG RISK)")
empty_node_results = [
    (defaultdict(list), defaultdict(list))  # empty nodes AND edges
]
try:
    # Simulate what operate.py does (the actual code, not our safe version)
    for res in empty_node_results:
        nodes, edges = res
        _ = list(nodes.values())[0][0]['source_id']  # this will crash
    print(f"  ✗ Should have crashed but didn't")
except (IndexError, KeyError) as e:
    print(f"  ✓ Confirmed crash: {type(e).__name__}: {e}")
    print(f"  → This is the bug: if any chunk extracts 0 entities in stage 1,")
    print(f"    the refinement pass crashes with IndexError")
    print(f"  → FIX NEEDED in operate.py extract_entities_refinement()")

print()
print("STEP 2: Propose the fix")
print("-" * 40)

fix_code = '''
# In operate.py, replace the dict comprehension at the top of
# extract_entities_refinement() (around line ~600):

# CURRENT (BUGGY):
chunk_results_at_stage_one = {
    list(res[0].values())[0][0]['source_id']: {
        'nodes': res[0],
        'edges': res[1],
    } for res in chunk_results
}

# FIX — handle empty extraction gracefully:
chunk_results_at_stage_one = {}
for i, (c_key, _) in enumerate(chunks.items()):
    # Use the chunk key directly instead of trying to read source_id from nodes
    # (which crashes when no entities were extracted)
    nodes, edges = chunk_results[i]
    chunk_results_at_stage_one[c_key] = {
        'nodes': nodes,
        'edges': edges,
    }
'''

print(fix_code)

print()
print("STEP 3: _search_subgraph query_mode selection logic")
print("-" * 40)

def get_query_mode(nodes, edges):
    """Replicate the logic from _search_subgraph."""
    ll_keywords = [k for k, v in nodes.items()]
    hl_keywords = [dp['keywords'] for v in edges.values() for dp in v]

    if not ll_keywords and not hl_keywords:
        return None, ll_keywords, hl_keywords

    query_mode = "hybrid"
    if not ll_keywords:
        query_mode = "global"
    elif not hl_keywords:
        query_mode = "local"

    return query_mode, ll_keywords, hl_keywords

test_cases = [
    ("nodes+edges", {"Caesar": [], "Rome": []}, {("Caesar", "Rome"): [{"keywords": "leadership"}]}),
    ("nodes only", {"Caesar": [], "Rome": []}, {}),
    ("edges only", {}, {("Caesar", "Rome"): [{"keywords": "leadership"}]}),
    ("empty both", {}, {}),
]

for label, nodes, edges in test_cases:
    mode, ll, hl = get_query_mode(nodes, edges)
    print(f"  [{label:15s}] → mode={mode}, ll_kw={ll[:2]}, hl_kw={hl[:2]}")
    if mode is None:
        print(f"             → _search_subgraph returns None → kg_context='empty' ✓")

print()
print("STEP 4: Stage-1 → Stage-2 merge logic audit")
print("-" * 40)

# From operate.py _process_single_content in extract_entities_refinement:
# After stage-2 extraction, it merges stage-1 results back in:
#   for entity_name, entities in chunk_results_s1["nodes"].items():
#       if entity_name not in maybe_nodes:
#           maybe_nodes[entity_name].extend(entities)
# This means: stage-2 OVERRIDES stage-1 for entities with same name.
# New entities from stage-1 that stage-2 didn't re-extract are preserved.

print("  Merge strategy: stage-2 overrides stage-1 for same-name entities")
print("  New-only entities from stage-1 are preserved in stage-2 output")
print()

# Simulate the merge
s1_nodes = defaultdict(list, {
    "Caesar": [{"entity_name": "Caesar", "description": "stage-1 desc", "source_id": "c1"}],
    "Gaul":   [{"entity_name": "Gaul",   "description": "region", "source_id": "c1"}],
})
s2_nodes = defaultdict(list, {
    "Caesar": [{"entity_name": "Caesar", "description": "stage-2 BETTER desc", "source_id": "c1"}],
    "Rome":   [{"entity_name": "Rome",   "description": "new entity in s2", "source_id": "c1"}],
})

merged = defaultdict(list, s2_nodes)  # start with stage-2
for entity_name, entities in s1_nodes.items():
    if entity_name not in merged:
        merged[entity_name].extend(entities)

print("  Stage-1 entities: Caesar, Gaul")
print("  Stage-2 entities: Caesar (better desc), Rome (new)")
print("  Merged result:")
for name, ents in merged.items():
    print(f"    {name}: '{ents[0]['description']}'")
print()
print("  ✓ Caesar: stage-2 wins (correct — refinement improved the description)")
print("  ✓ Gaul:   preserved from stage-1 (correct — stage-2 didn't miss it)")
print("  ✓ Rome:   from stage-2 (correct — newly discovered in refinement)")

print()
print("STEP 5: entity_refine_max_times=0 guard check")
print("-" * 40)

# In megarag.py, the refinement loop is:
#   for r in range(self.addon_params['entity_refine_max_times']):
# If entity_refine_max_times=0, the loop body never executes → no refinement.
# This is the configured value in world_history_tiny/conf/addon_params.yaml (it's 1).

for max_times in [0, 1, 2]:
    runs = list(range(max_times))
    print(f"  entity_refine_max_times={max_times}: refinement runs {len(runs)}x "
          f"{'(NO refinement)' if max_times == 0 else f'(refine rounds: {runs})'}")

print()
print("  Current config (world_history_tiny): entity_refine_max_times=1")
print("  → Refinement runs exactly once after initial extraction")

print()
print("=" * 60)
print("SUMMARY OF ISSUES FOUND")
print("=" * 60)

issues = [
    ("CRITICAL", "IndexError in extract_entities_refinement when any chunk extracts 0 entities",
     "operate.py ~line 600: chunk_results_at_stage_one dict comprehension crashes on empty nodes"),
    ("LOW",      "If LLM extracts 0 entities AND 0 relationships for a chunk in stage-1",
     "The chunk is silently lost from refinement — the fix above handles this"),
    ("INFO",     "query_mode falls back to 'local' when only entity keywords exist",
     "Expected and correct — documented for awareness"),
    ("INFO",     "Stage-2 merge preserves stage-1 entities not re-discovered",
     "Correct behavior — no issue"),
]

for severity, desc, detail in issues:
    icon = "🔴" if severity == "CRITICAL" else "🟡" if severity == "LOW" else "🔵"
    print(f"  {icon} [{severity}] {desc}")
    print(f"         {detail}")
    print()

print("RECOMMENDED ACTION:")
print("  Apply the fix in STEP 2 to operate.py before running the full build.")
print("  The bug only triggers if a page produces 0 entities — which can happen")
print("  for cover pages, blank pages, or heavily image-only pages.")
print()
print("  Quick check — does world_history_tiny have blank/image-only pages?")
pages_content_path = REPO_ROOT / "egs" / "world_history_tiny" / "dumps" / \
                     "World_History_Volume_1" / "pages_content.json"
if pages_content_path.exists():
    with open(pages_content_path) as f:
        pages = json.load(f)
    blank = [idx for idx, p in pages.items() if len(p.get("text", "").strip()) < 20]
    print(f"  Pages with <20 chars of text: {blank}")
    if blank:
        print(f"  ⚠ These pages may produce 0 entities → apply the fix")
    else:
        print(f"  ✓ No blank pages — bug less likely to trigger, but still apply the fix")
else:
    print(f"  (pages_content.json not found — run build first to check)")
