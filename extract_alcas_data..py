#!/usr/bin/env python3
"""
ALCAS Data Extraction Script
Run this on your Brev instance: python3 extract_alcas_data.py
It will produce alcas_data_dump.json in the current directory.
Upload that file to Claude.
"""

import json
import os
import numpy as np
import traceback

BASE = os.path.expanduser("~/alcas")
out = {}

def safe_read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return {"_error": str(e), "_path": path}

def safe_read_csv(path):
    try:
        with open(path) as f:
            return f.read()
    except Exception as e:
        return f"ERROR: {e}"

def safe_read_npy(path):
    try:
        arr = np.load(path)
        if arr.ndim == 1:
            return {
                "shape": list(arr.shape),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "values": arr.tolist()
            }
        elif arr.ndim == 2:
            return {
                "shape": list(arr.shape),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "first_row": arr[0].tolist(),
                "diagonal": np.diag(arr).tolist() if arr.shape[0] == arr.shape[1] else None,
                "full_matrix": arr.tolist() if arr.shape[0] <= 300 else "TOO_LARGE"
            }
        else:
            return {"shape": list(arr.shape), "note": "3D+ array, skipped"}
    except Exception as e:
        return {"_error": str(e), "_path": path}

print("Extracting ALCAS data...")

# ── FoldX ──────────────────────────────────────────────────────────────────
print("  foldx...")
out["foldx_summary"]          = safe_read_json(f"{BASE}/results/foldx/foldx_summary.json")
out["stability_summary"]      = safe_read_json(f"{BASE}/results/foldx/stability_summary.json")
out["active_site_foldx"]      = safe_read_json(f"{BASE}/results/foldx/active_site_foldx.json")
out["allosteric_foldx"]       = safe_read_json(f"{BASE}/results/foldx/allosteric_foldx.json")
out["stability_active_site"]  = safe_read_json(f"{BASE}/results/foldx/stability_active_site.json")
out["stability_allosteric"]   = safe_read_json(f"{BASE}/results/foldx/stability_allosteric.json")

# ── Ablation ───────────────────────────────────────────────────────────────
print("  ablation...")
out["ablation_summary"]       = safe_read_json(f"{BASE}/results/ablation/ablation_summary.json")
out["distance_only_results"]  = safe_read_json(f"{BASE}/results/ablation/distance_only_results.json")

# ── ESMFold ────────────────────────────────────────────────────────────────
print("  esmfold...")
out["esmfold_analysis"]       = safe_read_json(f"{BASE}/results/esmfold/esmfold_analysis.json")

# ── Docking ────────────────────────────────────────────────────────────────
print("  docking...")
out["docking_summary"]        = safe_read_json(f"{BASE}/results/docking/docking_summary.json")

# ── Conservation ───────────────────────────────────────────────────────────
print("  conservation...")
out["conservation_analysis"]  = safe_read_json(f"{BASE}/results/conservation/conservation_analysis.json")

# ── Composite / Candidates ─────────────────────────────────────────────────
print("  composite...")
out["composite_summary"]      = safe_read_json(f"{BASE}/results/composite/composite_summary.json")
out["final_candidate_table"]  = safe_read_csv(f"{BASE}/results/composite/final_candidate_table.csv")
out["top_candidates_txt"]     = safe_read_csv(f"{BASE}/results/composite/top_candidates.txt")

# ── MD / MM-GBSA ──────────────────────────────────────────────────────────
print("  md / mmgbsa...")
out["holo_mmgbsa_results"]    = safe_read_json(f"{BASE}/results/md/holo_mmgbsa_results.json")
out["md_step3_rmsf"]          = safe_read_json(f"{BASE}/md_step3_rmsf_fixed.json")
out["md_manifest"]            = safe_read_json(f"{BASE}/data/petase/md/md_manifest.json")

# ── MD log previews (first 100 lines each) ─────────────────────────────────
for rep in [1, 2, 3]:
    path = f"{BASE}/data/petase/md/rep{rep}_md.log"
    try:
        with open(path) as f:
            lines = f.readlines()[:100]
        out[f"md_rep{rep}_log_head"] = "".join(lines)
    except Exception as e:
        out[f"md_rep{rep}_log_head"] = f"ERROR: {e}"

# ── Coupling numpy arrays ──────────────────────────────────────────────────
print("  coupling arrays...")
coupling_dir = f"{BASE}/data/petase/coupling"
out["coupling_cross_corr"]       = safe_read_npy(f"{coupling_dir}/cross_corr_matrix.npy")
out["coupling_score"]            = safe_read_npy(f"{coupling_dir}/coupling_score.npy")
out["coupling_to_triad"]         = safe_read_npy(f"{coupling_dir}/coupling_to_triad.npy")
out["betweenness_centrality"]    = safe_read_npy(f"{coupling_dir}/betweenness_centrality.npy")
out["msf"]                       = safe_read_npy(f"{coupling_dir}/msf.npy")
out["all_residue_scores"]        = safe_read_json(f"{coupling_dir}/all_residue_scores.json")

# ── Masks ──────────────────────────────────────────────────────────────────
print("  masks...")
out["masks_allosteric"]              = safe_read_json(f"{BASE}/data/petase/masks_allosteric.json")
out["mask_active_site"]              = safe_read_json(f"{BASE}/data/petase/mask_active_site.json")
out["mask_allosteric_distance_only"] = safe_read_json(f"{BASE}/data/petase/mask_allosteric_distance_only.json")
out["mask_allosteric_pre_md"]        = safe_read_json(f"{BASE}/data/petase/mask_allosteric_pre_md.json")

# ── FAST-PETase validation ─────────────────────────────────────────────────
print("  fast-petase...")
out["fast_petase_validation"]    = safe_read_json(f"{BASE}/data/petase/fast_petase_validation.json")

# ── GNN model metrics ─────────────────────────────────────────────────────
print("  gnn metrics...")
out["ensemble_random_metrics"]   = safe_read_json(f"{BASE}/results/models/ensemble_random_test_metrics.json")
out["ensemble_temporal_metrics"] = safe_read_json(f"{BASE}/results/models/ensemble_temporal_test_metrics.json")
out["target_stats"]              = safe_read_json(f"{BASE}/results/models/target_stats.json")

# Per-seed histories
for seed in [42, 7, 13, 99, 2024, 314]:
    path = f"{BASE}/results/models/seed_{seed}/history.json"
    out[f"seed_{seed}_history"] = safe_read_json(path)

# GNN predictions (arrays)
print("  gnn prediction arrays...")
out["gnn_random_preds"]    = safe_read_npy(f"{BASE}/results/models/ensemble_random_test_preds.npy")
out["gnn_random_targets"]  = safe_read_npy(f"{BASE}/results/models/ensemble_random_test_targets.npy")
out["gnn_random_std"]      = safe_read_npy(f"{BASE}/results/models/ensemble_random_test_std.npy")
out["gnn_temporal_preds"]  = safe_read_npy(f"{BASE}/results/models/ensemble_temporal_test_preds.npy")
out["gnn_temporal_targets"]= safe_read_npy(f"{BASE}/results/models/ensemble_temporal_test_targets.npy")
out["gnn_temporal_std"]    = safe_read_npy(f"{BASE}/results/models/ensemble_temporal_test_std.npy")

# ── Candidates & scores ────────────────────────────────────────────────────
print("  candidates...")
out["active_site_candidates"]  = safe_read_json(f"{BASE}/results/candidates/active_site_candidates.json")
out["allosteric_candidates"]   = safe_read_json(f"{BASE}/results/candidates/allosteric_candidates.json")
out["active_site_scored"]      = safe_read_json(f"{BASE}/results/scores/active_site_scored.json")
out["allosteric_scored"]       = safe_read_json(f"{BASE}/results/scores/allosteric_scored.json")
out["comparison_summary"]      = safe_read_json(f"{BASE}/results/scores/comparison_summary.json")

# ── Mechanistic verification (if exists) ───────────────────────────────────
print("  mechanistic...")
mech_candidates = ["Q182I", "G79P_M262L", "G79P_S169A", "G79P_A179V"]
for cand in mech_candidates:
    path = f"{BASE}/results/mechanism/{cand}_coupling_delta.json"
    if os.path.exists(path):
        out[f"mech_{cand}"] = safe_read_json(path)

# ── Meta CSV (first 20 rows) ───────────────────────────────────────────────
print("  meta csv...")
try:
    with open(f"{BASE}/data/processed/meta.csv") as f:
        lines = f.readlines()
    out["meta_csv_head"] = "".join(lines[:21])
    out["meta_csv_total_rows"] = len(lines) - 1
except Exception as e:
    out["meta_csv_head"] = f"ERROR: {e}"

# ── Write output ───────────────────────────────────────────────────────────
out_path = os.path.join(os.path.expanduser("~"), "alcas_data_dump.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2, default=str)

size_mb = os.path.getsize(out_path) / 1e6
print(f"\nDone. Output: {out_path}  ({size_mb:.1f} MB)")
print("Download with:  scp <your-brev-host>:~/alcas_data_dump.json .")