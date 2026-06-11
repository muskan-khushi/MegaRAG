#!/usr/bin/env python3
"""
apply_refinement_fix.py
───────────────────────
Applies the fix for the IndexError bug in extract_entities_refinement().

Run ONCE before resubmitting the build:
    cd /scratch/data/divyasaxena_rs/Muskan_internship/MegaRAG/
    python test/apply_refinement_fix.py

Then verify:
    git diff megarag/operate.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
TARGET = REPO_ROOT / "megarag" / "operate.py"

print(f"Patching: {TARGET}")

content = TARGET.read_text(encoding="utf-8")

# The buggy code we want to replace
OLD = '''    chunk_results_at_stage_one = {
        list(res[0].values())[0][0]['source_id']: {
            'nodes': res[0],
            'edges': res[1],
        } for res in chunk_results
    }'''

# The safe replacement — keyed by chunk key directly, handles empty nodes
NEW = '''    # Build stage-1 index keyed by chunk key (not source_id from nodes,
    # which crashes when a chunk produces 0 entities).
    chunk_results_at_stage_one = {}
    for (c_key, _chunk_dp), (s1_nodes, s1_edges) in zip(
        list(chunks.items()), chunk_results
    ):
        chunk_results_at_stage_one[c_key] = {
            'nodes': s1_nodes,
            'edges': s1_edges,
        }'''

if OLD not in content:
    print()
    print("OLD pattern not found in operate.py.")
    print("Either the fix was already applied, or the source has changed.")
    print()
    print("Manual check — search for this in megarag/operate.py:")
    print("    chunk_results_at_stage_one = {")
    print("        list(res[0].values())[0][0]['source_id']:")
    sys.exit(1)

patched = content.replace(OLD, NEW, 1)
assert patched != content, "Replace had no effect"

# Also fix the downstream lookup which uses chunk_key correctly now
# The _process_single_content call passes chunk_results_s1 = chunk_results_at_stage_one[c[0]]
# where c[0] is the chunk key — this already matches our new indexing, no change needed there.

TARGET.write_text(patched, encoding="utf-8")
print("✓ Fix applied successfully.")
print()
print("Verify with:")
print("  git diff megarag/operate.py")
print()
print("Then commit:")
print("  git add megarag/operate.py")
print('  git commit -m "fix: handle empty entity extraction in refinement pass"')
print("  git push origin master")
