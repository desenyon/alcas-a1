"""
Run this on Brev FIRST before running any figure scripts.
Prints the actual structure of every result JSON so figures can be fixed.

Usage: python3 inspect_json_structure.py
"""
import json, os

BASE = os.path.expanduser("~/alcas")
RES  = os.path.join(BASE, "results")
DATA = os.path.join(BASE, "data", "petase")

def show(path, max_keys=3, depth=0):
    indent = "  " * depth
    try:
        with open(path) as f:
            obj = json.load(f)
    except Exception as e:
        print(f"{indent}ERROR loading {path}: {e}")
        return

    if isinstance(obj, dict):
        keys = list(obj.keys())
        print(f"{indent}dict  ({len(keys)} keys)")
        for k in keys[:max_keys]:
            v = obj[k]
            print(f"{indent}  key={repr(k)!s:40s}  val_type={type(v).__name__}", end="")
            if isinstance(v, (int, float, str, bool)):
                print(f"  val={repr(v)}")
            elif isinstance(v, dict):
                sub_keys = list(v.keys())
                print(f"  sub_keys={sub_keys[:5]}")
            elif isinstance(v, list):
                print(f"  len={len(v)}  first={repr(v[0]) if v else 'EMPTY'}")
            else:
                print()
        if len(keys) > max_keys:
            print(f"{indent}  ... and {len(keys)-max_keys} more keys")
    elif isinstance(obj, list):
        print(f"{indent}list  (len={len(obj)})")
        if obj:
            ex = obj[0]
            print(f"{indent}  first element type={type(ex).__name__}", end="")
            if isinstance(ex, dict):
                print(f"  keys={list(ex.keys())[:8]}")
            elif isinstance(ex, (int, float, str)):
                print(f"  val={repr(ex)}")
            else:
                print()
    else:
        print(f"{indent}{type(obj).__name__}  val={repr(obj)[:80]}")

FILES = {
    "foldx/allosteric_foldx.json":        os.path.join(RES, "foldx", "allosteric_foldx.json"),
    "foldx/active_site_foldx.json":       os.path.join(RES, "foldx", "active_site_foldx.json"),
    "foldx/stability_allosteric.json":    os.path.join(RES, "foldx", "stability_allosteric.json"),
    "foldx/stability_active_site.json":   os.path.join(RES, "foldx", "stability_active_site.json"),
    "ablation/ablation_summary.json":     os.path.join(RES, "ablation", "ablation_summary.json"),
    "esmfold/esmfold_analysis.json":      os.path.join(RES, "esmfold", "esmfold_analysis.json"),
    "docking/docking_summary.json":       os.path.join(RES, "docking", "docking_summary.json"),
    "conservation/conservation_analysis.json": os.path.join(RES, "conservation", "conservation_analysis.json"),
    "models/ensemble_test_metrics.json":  os.path.join(RES, "models", "ensemble_test_metrics.json"),
    "petase/masks_allosteric.json":       os.path.join(DATA, "masks_allosteric.json"),
    "petase/masks_active.json":           os.path.join(DATA, "masks_active.json"),
    "petase/coupling/coupling_to_triad.json": os.path.join(DATA, "coupling", "coupling_to_triad.json"),
}

# Also find any candidate files
CAND_DIR = os.path.join(RES, "candidates")
if os.path.isdir(CAND_DIR):
    for fname in os.listdir(CAND_DIR):
        if fname.endswith(".json"):
            FILES[f"candidates/{fname}"] = os.path.join(CAND_DIR, fname)

# Also list what's actually in key directories
for label, dirpath in [
    ("results/foldx/",        os.path.join(RES, "foldx")),
    ("results/esmfold/",      os.path.join(RES, "esmfold")),
    ("results/docking/",      os.path.join(RES, "docking")),
    ("results/ablation/",     os.path.join(RES, "ablation")),
    ("results/conservation/", os.path.join(RES, "conservation")),
    ("results/candidates/",   os.path.join(RES, "candidates")),
    ("results/models/",       os.path.join(RES, "models")),
    ("data/petase/",          DATA),
    ("data/petase/coupling/", os.path.join(DATA, "coupling")),
]:
    if os.path.isdir(dirpath):
        files = os.listdir(dirpath)
        print(f"\nDIR {label}: {files}")
    else:
        print(f"\nDIR {label}: NOT FOUND")

print("\n" + "="*60 + "\nJSON STRUCTURES\n" + "="*60)
for label, path in FILES.items():
    print(f"\n── {label}")
    if os.path.exists(path):
        show(path)
    else:
        print(f"  FILE NOT FOUND: {path}")