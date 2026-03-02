#!/usr/bin/env python3
"""
ALCAS MD Analysis - Step 1: Prerequisite Check
Run: python3 step1_check_md.py
Paste md_step1_manifest.json back to Claude when done.
"""
import os, sys, json, subprocess
from pathlib import Path

BASE      = Path(os.path.expanduser("~/alcas"))
MD_DIR    = BASE / "data/petase/md"
RES_DIR   = BASE / "results"

REPS = ["rep1", "rep2", "rep3"]

print("=" * 60)
print("ALCAS MD — Step 1: Prerequisite Check")
print("=" * 60)

manifest = {}
all_ok = True

# 1. Trajectory files
print("\n[1] Trajectory files:")
manifest["trajectories"] = {}
for rep in REPS:
    top  = MD_DIR / f"{rep}_topology.pdb"
    traj = MD_DIR / f"{rep}_traj.dcd"
    top_ok  = top.exists()
    traj_ok = traj.exists()
    top_mb  = round(top.stat().st_size / 1e6, 2)  if top_ok  else 0
    traj_mb = round(traj.stat().st_size / 1e6, 2) if traj_ok else 0
    status = "OK" if (top_ok and traj_ok) else "MISSING"
    if not (top_ok and traj_ok):
        all_ok = False
    print(f"  {status}  {rep}: topology={top_mb}MB  traj={traj_mb}MB")
    manifest["trajectories"][rep] = {
        "topology_exists": top_ok, "topology_mb": top_mb,
        "traj_exists": traj_ok,    "traj_mb": traj_mb
    }

# 2. MDAnalysis
print("\n[2] MDAnalysis:")
try:
    import MDAnalysis as mda
    print(f"  OK  {mda.__version__}")
    manifest["mdanalysis"] = mda.__version__

    # Quick load of rep1 to count frames and atoms
    top  = MD_DIR / "rep1_topology.pdb"
    traj = MD_DIR / "rep1_traj.dcd"
    if top.exists() and traj.exists():
        u = mda.Universe(str(top), str(traj))
        n_frames = len(u.trajectory)
        n_atoms  = len(u.atoms)
        print(f"  OK  rep1: {n_frames} frames, {n_atoms} atoms")
        manifest["rep1_frames"] = n_frames
        manifest["rep1_atoms"]  = n_atoms
        # Print residue list for triad confirmation
        try:
            triad = u.select_atoms("resname SER and resid 160") | \
                    u.select_atoms("resname HIS and resid 237") | \
                    u.select_atoms("resname ASP and resid 206")
            print(f"  Triad atoms found: {len(triad)}")
            manifest["triad_atoms_found"] = len(triad)
        except Exception as e:
            print(f"  WARN triad select: {e}")
except ImportError:
    print("  MISSING — run: pip install MDAnalysis")
    manifest["mdanalysis"] = None
    all_ok = False

# 3. Numpy/Scipy
print("\n[3] Scientific stack:")
for pkg in ["numpy", "scipy", "matplotlib"]:
    try:
        m = __import__(pkg)
        v = getattr(m, "__version__", "ok")
        print(f"  OK  {pkg} {v}")
    except ImportError:
        print(f"  MISSING  {pkg}")
        all_ok = False

# 4. FoldX results (need candidate list)
print("\n[4] FoldX results (for candidate list):")
foldx_allo = RES_DIR / "foldx/allosteric_foldx.json"
foldx_as   = RES_DIR / "foldx/active_site_foldx.json"
for label, fp in [("allosteric", foldx_allo), ("active_site", foldx_as)]:
    if fp.exists():
        with open(fp) as f:
            d = json.load(f)
        n = len(d) if isinstance(d, list) else len(d.get("candidates", d))
        print(f"  OK  {label}: {n} entries")
        manifest[f"foldx_{label}_n"] = n
    else:
        print(f"  MISSING  {fp}")
        manifest[f"foldx_{label}_n"] = 0

# 5. md_manifest.json from Brev run
print("\n[5] Existing md_manifest.json:")
mm = MD_DIR / "md_manifest.json"
if mm.exists():
    with open(mm) as f:
        existing = json.load(f)
    print(f"  FOUND — keys: {list(existing.keys())}")
    manifest["existing_manifest"] = existing
else:
    print("  not found (normal if first run)")
    manifest["existing_manifest"] = None

# Summary
manifest["all_ok"] = all_ok
print("\n" + "=" * 60)
print("READY" if all_ok else "NOT READY — fix issues above")
print("=" * 60)

out = Path("md_step1_manifest.json")
with open(out, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"\nSaved: {out.resolve()}")
print(">>> Paste the contents of md_step1_manifest.json to Claude <<<")