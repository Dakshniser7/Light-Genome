#!/usr/bin/env python3
"""One-command pipeline:  python run_all.py            (validate -> experiments -> figures -> summary)
Options: --only validate|experiments|figures|summary|verify   run a single stage
         --skip-validate                               skip the validation stage
Full run takes ~10 min on 2 cores; seeds are fixed so results are reproducible."""
import argparse, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
os.makedirs("results", exist_ok=True)
os.makedirs("figs", exist_ok=True)
STAGES = {
    "validate": [sys.executable, "validate.py"],
    "experiments": [sys.executable, "experiments.py"],
    "figures": [sys.executable, "figures.py"],
    "summary": [sys.executable, "summarize.py"],
    "verify": [sys.executable, "verify.py"],
}

ap = argparse.ArgumentParser()
ap.add_argument("--only", choices=STAGES)
ap.add_argument("--skip-validate", action="store_true")
a = ap.parse_args()
todo = [a.only] if a.only else [s for s in STAGES if not (s == "validate" and a.skip_validate)]
for s in todo:
    t = time.time()
    print(f"\n=== {s} ===", flush=True)
    if subprocess.call(STAGES[s]) != 0:
        sys.exit(f"stage '{s}' failed")
    print(f"=== {s} done in {time.time()-t:.0f}s ===", flush=True)
print("\nAll done. Results: results/*.json, figures: figs/*.png")
