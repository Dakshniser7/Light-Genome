#!/usr/bin/env python3
"""Compare freshly generated results/*.json with the shipped reference/*.json.
Exit code 0 = reproduced.  Default tolerance is tiny (seeds are fixed); different
numpy/scipy versions can change the last digits, hence a small tolerance."""
import json, os, sys
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
TOL = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-6

def flat(o):
    if isinstance(o, dict):
        for k in sorted(o):
            yield from flat(o[k])
    elif isinstance(o, list):
        for v in o:
            yield from flat(v)
    elif isinstance(o, (int, float)):
        yield float(o)

bad = 0
for n in ["e1", "e2", "e3a", "e3b", "e4", "e5", "e6"]:
    try:
        a = np.array(list(flat(json.load(open(f"results/{n}.json")))))
        b = np.array(list(flat(json.load(open(f"reference/{n}.json")))))
    except FileNotFoundError:
        print(f"{n}: MISSING (run the pipeline first)"); bad += 1; continue
    if a.shape != b.shape:
        print(f"{n}: structure differs"); bad += 1; continue
    d = float(np.nanmax(np.abs(a - b)))
    ok = d <= TOL
    bad += not ok
    print(f"{n}: max abs diff {d:.2e}  {'OK' if ok else 'DIFFERS'}")
print("REPRODUCED" if bad == 0 else "NOT REPRODUCED")
sys.exit(bad != 0)
