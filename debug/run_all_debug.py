#!/usr/bin/env python3
"""
run_all_debug.py
────────────────
Runs all 5 debug scripts in order and produces a final summary report.
Safe to run on the login node while your build job is queued.

Usage:
    cd /scratch/data/divyasaxena_rs/Muskan_internship/MegaRAG/
    python test/run_all_debug.py

    # After build completes, run again to include graph inspection:
    python test/run_all_debug.py --include-graph-check
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
TEST_DIR  = REPO_ROOT / "test"

def run_script(name, path, timeout=120):
    print(f"\n{'='*60}")
    print(f"  RUNNING: {name}")
    print(f"{'='*60}\n")
    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(REPO_ROOT),
        capture_output=False,   # let output flow to terminal in real time
        timeout=timeout,
    )
    dt = time.perf_counter() - t0
    return result.returncode, dt

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--include-graph-check", action="store_true",
                   help="Run C_graph_inspector.py (only useful after build completes)")
    p.add_argument("--skip-ollama", action="store_true",
                   help="Skip tests that require Ollama to be running")
    return p.parse_args()

args = parse_args()

scripts = [
    ("E: Refinement audit (static, always run first)", "E_refinement_audit.py"),
    ("B: Prompt inspector", "B_prompt_inspector.py"),
    ("A: Query smoke test", "A_query_smoke_test.py"),
]

if not args.skip_ollama:
    scripts.append(("D: Ollama vision test", "D_ollama_vision_test.py"))

if args.include_graph_check:
    scripts.append(("C: Graph inspector", "C_graph_inspector.py"))

print()
print("╔══════════════════════════════════════════════════════════╗")
print("║           MegaRAG Debug Suite — All Checks              ║")
print("╚══════════════════════════════════════════════════════════╝")
print()
print("Running while build job is queued/running.")
print(f"Repo root: {REPO_ROOT}")
print()

results = {}
for label, script_name in scripts:
    script_path = TEST_DIR / script_name
    if not script_path.exists():
        print(f"\n⚠ SKIPPED (file not found): {script_path}")
        results[label] = ("SKIPPED", 0)
        continue
    try:
        rc, dt = run_script(label, script_path)
        results[label] = ("OK" if rc == 0 else "FAILED", dt)
    except subprocess.TimeoutExpired:
        results[label] = ("TIMEOUT", 120)
    except Exception as e:
        results[label] = (f"ERROR: {e}", 0)

print()
print("╔══════════════════════════════════════════════════════════╗")
print("║                    FINAL SUMMARY                        ║")
print("╚══════════════════════════════════════════════════════════╝")
print()
for label, (status, dt) in results.items():
    icon = "✓" if status == "OK" else ("⚠" if status == "SKIPPED" else "✗")
    print(f"  {icon}  {label}")
    print(f"     Status: {status}  |  Time: {dt:.1f}s")
    print()

print()
print("NEXT STEPS:")
print("  1. Apply the operate.py fix from E_refinement_audit.py output")
print("  2. Check if Ollama vision test (D) passed — affects extraction quality")
print("  3. Once build finishes: python test/run_all_debug.py --include-graph-check")
print("  4. If graph check shows <10 entities: increase example_number to 3 in addon_params.yaml")
print()
print("Monitor your build job:")
print("  squeue -u divyasaxena_rs")
print("  tail -f /scratch/data/divyasaxena_rs/Muskan_internship/MegaRAG/logs/build_<JOBID>.out")
