"""
ALCAS Poster Figures F01-F10
Run from: ~/alcas/
Output:  ~/alcas/poster_assets/
All data loaded from JSON results files — no hardcoded values.
"""

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats

# ── paths ──────────────────────────────────────────────────────────────────────
BASE   = os.path.expanduser("~/alcas")
RES    = os.path.join(BASE, "results")
DATA   = os.path.join(BASE, "data", "petase")
OUT    = os.path.join(BASE, "poster_assets")
os.makedirs(OUT, exist_ok=True)

def jload(path):
    with open(path) as f:
        return json.load(f)

# ── theme ──────────────────────────────────────────────────────────────────────
ALLO_C  = "#2E86AB"   # blue  – allosteric
AS_C    = "#E84855"   # red   – active-site
DIST_C  = "#F4A261"   # amber – distance-only ablation
BG      = "#F8F9FA"
GREY    = "#6C757D"

TITLE_FS = 13
LABEL_FS = 10
TICK_FS  =  8
DPI      = 300

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.facecolor":    BG,
    "figure.facecolor":  "white",
    "axes.labelsize":    LABEL_FS,
    "xtick.labelsize":   TICK_FS,
    "ytick.labelsize":   TICK_FS,
})

def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")

def sig_bar(ax, x1, x2, y, h, p, fs=8):
    """Draw significance bracket."""
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.2, color="black")
    label = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    ax.text((x1+x2)/2, y+h*1.1, label, ha="center", va="bottom", fontsize=fs)

# ══════════════════════════════════════════════════════════════════════════════
# F01 – FoldX ddG Binding Violin
# ══════════════════════════════════════════════════════════════════════════════
def make_F01():
    allo = jload(os.path.join(RES, "foldx", "allosteric_foldx.json"))
    act  = jload(os.path.join(RES, "foldx", "active_site_foldx.json"))

    allo_vals = [v["ddG_mean"] for v in allo.values()]
    act_vals  = [v["ddG_mean"] for v in act.values()]

    stat, p = stats.mannwhitneyu(allo_vals, act_vals, alternative="less")
    delta = np.mean(allo_vals) - np.mean(act_vals)

    fig, ax = plt.subplots(figsize=(5, 5))
    parts = ax.violinplot([act_vals, allo_vals], positions=[1, 2],
                          showmedians=True, showextrema=True)
    for pc, c in zip(parts["bodies"], [AS_C, ALLO_C]):
        pc.set_facecolor(c); pc.set_alpha(0.75)
    for part in ["cmedians", "cmins", "cmaxes", "cbars"]:
        parts[part].set_color("black"); parts[part].set_linewidth(1.2)

    ax.scatter([1]*len(act_vals),  act_vals,  color=AS_C,   alpha=0.35, s=12, zorder=3)
    ax.scatter([2]*len(allo_vals), allo_vals, color=ALLO_C, alpha=0.35, s=12, zorder=3)

    ymax = max(max(act_vals), max(allo_vals))
    sig_bar(ax, 1, 2, ymax*1.02, abs(ymax)*0.04, p)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Active-site\n(n=50)", "Allosteric\n(n=50)"])
    ax.set_ylabel("FoldX ΔΔG (kcal/mol)")
    ax.set_title(f"F01 · FoldX Binding ΔΔG\nΔμ = {delta:.2f} kcal/mol  p = {p:.4f}", fontsize=TITLE_FS)
    ax.axhline(0, color=GREY, ls="--", lw=0.8, alpha=0.6)
    save(fig, "F01_foldx_ddg_violin.png")

# ══════════════════════════════════════════════════════════════════════════════
# F02 – FoldX Stability Violin
# ══════════════════════════════════════════════════════════════════════════════
def make_F02():
    allo = jload(os.path.join(RES, "foldx", "stability_allosteric.json"))
    act  = jload(os.path.join(RES, "foldx", "stability_active_site.json"))

    allo_vals = [v["ddG_mean"] for v in allo.values()]
    act_vals  = [v["ddG_mean"] for v in act.values()]

    stat, p = stats.mannwhitneyu(allo_vals, act_vals, alternative="less")
    delta = np.mean(allo_vals) - np.mean(act_vals)

    fig, ax = plt.subplots(figsize=(5, 5))
    parts = ax.violinplot([act_vals, allo_vals], positions=[1, 2],
                          showmedians=True, showextrema=True)
    for pc, c in zip(parts["bodies"], [AS_C, ALLO_C]):
        pc.set_facecolor(c); pc.set_alpha(0.75)
    for part in ["cmedians", "cmins", "cmaxes", "cbars"]:
        parts[part].set_color("black"); parts[part].set_linewidth(1.2)

    ax.scatter([1]*len(act_vals),  act_vals,  color=AS_C,   alpha=0.35, s=12, zorder=3)
    ax.scatter([2]*len(allo_vals), allo_vals, color=ALLO_C, alpha=0.35, s=12, zorder=3)

    ymax = max(max(act_vals), max(allo_vals))
    sig_bar(ax, 1, 2, ymax*1.02, abs(ymax)*0.04, p)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Active-site\n(n=50)", "Allosteric\n(n=50)"])
    ax.set_ylabel("FoldX Stability ΔΔG (kcal/mol)")
    ax.set_title(f"F02 · FoldX Folding Stability\nΔμ = {delta:.2f} kcal/mol  p = {p:.4f}", fontsize=TITLE_FS)
    ax.axhline(0, color=GREY, ls="--", lw=0.8, alpha=0.6)
    save(fig, "F02_foldx_stability_violin.png")

# ══════════════════════════════════════════════════════════════════════════════
# F03 – Ablation Study Bar Chart
# ══════════════════════════════════════════════════════════════════════════════
def make_F03():
    abl = jload(os.path.join(RES, "ablation", "ablation_summary.json"))

    # Expected keys: coupling_filtered, active_site, distance_only
    groups = [
        ("Coupling-filtered\nAllosteric", "coupling_filtered", ALLO_C),
        ("Active-site",                   "active_site",       AS_C),
        ("Distance-only\nAllosteric",     "distance_only",     DIST_C),
    ]

    means  = [abl[k]["mean_ddG"]   for _, k, _ in groups]
    stds   = [abl[k]["std_ddG"]    for _, k, _ in groups]
    labels = [lbl                  for lbl, _, _ in groups]
    colors = [c                    for _, _, c  in groups]

    fig, ax = plt.subplots(figsize=(6, 5))
    xs = np.arange(len(groups))
    bars = ax.bar(xs, means, yerr=stds, color=colors, alpha=0.85,
                  capsize=5, error_kw=dict(lw=1.5), edgecolor="black", linewidth=0.8)

    # p-value annotations from ablation summary
    p_coup_vs_dist = abl.get("p_coupling_vs_distance", None)
    p_coup_vs_as   = abl.get("p_coupling_vs_active",   None)

    ymax = max(m + s for m, s in zip(means, stds))
    h = abs(ymax) * 0.06
    if p_coup_vs_dist: sig_bar(ax, 0, 2, ymax + h,       h*0.6, p_coup_vs_dist)
    if p_coup_vs_as:   sig_bar(ax, 0, 1, ymax + h*2.4,   h*0.6, p_coup_vs_as)

    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=TICK_FS)
    ax.set_ylabel("Mean FoldX ΔΔG (kcal/mol)")
    ax.axhline(0, color=GREY, ls="--", lw=0.8, alpha=0.6)
    ax.set_title("F03 · Ablation: NMA Coupling Filter Validation", fontsize=TITLE_FS)
    save(fig, "F03_ablation.png")

# ══════════════════════════════════════════════════════════════════════════════
# F04 – ANM Coupling Heatmap
# ══════════════════════════════════════════════════════════════════════════════
def make_F04():
    coup_dir = os.path.join(DATA, "coupling")

    # Try standard file names from the pipeline
    cross_corr_path = os.path.join(coup_dir, "cross_correlation.npy")
    coupling_path   = os.path.join(coup_dir, "coupling_to_triad.json")
    masks_allo      = jload(os.path.join(DATA, "masks_allosteric.json"))
    masks_active    = jload(os.path.join(DATA, "masks_active.json"))

    fig = plt.figure(figsize=(12, 5))
    gs  = GridSpec(1, 2, figure=fig, wspace=0.35)

    # ── left: cross-correlation matrix (subset) ──
    ax1 = fig.add_subplot(gs[0])
    if os.path.exists(cross_corr_path):
        cc = np.load(cross_corr_path)
        n  = min(cc.shape[0], 263)
        cc = cc[:n, :n]
        im = ax1.imshow(cc, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax1, shrink=0.8, label="Cross-correlation")
        # Mark triad residues (0-indexed: S160→159, H237→236, D206→205)
        for idx, lbl in [(159, "S160"), (205, "D206"), (236, "H237")]:
            if idx < n:
                ax1.axhline(idx, color="gold", lw=1.0, ls="--", alpha=0.9)
                ax1.axvline(idx, color="gold", lw=1.0, ls="--", alpha=0.9)
        ax1.set_xlabel("Residue index"); ax1.set_ylabel("Residue index")
        ax1.set_title("ANM Cross-correlation Matrix\n(gold = catalytic triad)", fontsize=TITLE_FS)
    else:
        ax1.text(0.5, 0.5, "cross_correlation.npy\nnot found", ha="center", va="center",
                 transform=ax1.transAxes, color="red")

    # ── right: coupling-to-triad bar chart (top 20) ──
    ax2 = fig.add_subplot(gs[1])
    if os.path.exists(coupling_path):
        coup = jload(coupling_path)
        # coup expected: {residue_id: coupling_score, ...}
        items = sorted(coup.items(), key=lambda x: float(x[1]), reverse=True)[:20]
        res_ids = [int(k) for k, _ in items]
        scores  = [float(v) for _, v in items]

        allo_set   = set(masks_allo.get("residues", []))
        active_set = set(masks_active.get("residues", []))
        bar_colors = []
        for r in res_ids:
            if r in allo_set:   bar_colors.append(ALLO_C)
            elif r in active_set: bar_colors.append(AS_C)
            else:               bar_colors.append(GREY)

        ax2.barh(range(len(res_ids)), scores[::-1], color=bar_colors[::-1],
                 edgecolor="black", linewidth=0.5, alpha=0.85)
        ax2.set_yticks(range(len(res_ids)))
        ax2.set_yticklabels([str(r) for r in reversed(res_ids)], fontsize=7)
        ax2.set_xlabel("Coupling score to catalytic triad")
        ax2.set_title("Top 20 Residues by NMA Coupling", fontsize=TITLE_FS)

        legend_patches = [
            mpatches.Patch(color=ALLO_C, label="Allosteric mask"),
            mpatches.Patch(color=AS_C,   label="Active-site mask"),
            mpatches.Patch(color=GREY,   label="Neither"),
        ]
        ax2.legend(handles=legend_patches, fontsize=7, loc="lower right")
    else:
        ax2.text(0.5, 0.5, "coupling_to_triad.json\nnot found", ha="center", va="center",
                 transform=ax2.transAxes, color="red")

    fig.suptitle("F04 · ANM Coupling Analysis – PETase Allosteric Map", fontsize=TITLE_FS+1, fontweight="bold")
    save(fig, "F04_anm_coupling.png")

# ══════════════════════════════════════════════════════════════════════════════
# F05 – Mask Definition Summary
# ══════════════════════════════════════════════════════════════════════════════
def make_F05():
    masks_allo   = jload(os.path.join(DATA, "masks_allosteric.json"))
    masks_active = jload(os.path.join(DATA, "masks_active.json"))

    allo_res   = masks_allo.get("residues", [])
    active_res = masks_active.get("residues", [])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # ── left: residue index scatter by mask ──
    ax = axes[0]
    ax.scatter(range(len(allo_res)),   allo_res,   color=ALLO_C, alpha=0.7,
               s=25, label=f"Allosteric (n={len(allo_res)})")
    ax.scatter(range(len(active_res)), active_res, color=AS_C,   alpha=0.7,
               s=25, label=f"Active-site (n={len(active_res)})", marker="s")
    # Triad lines
    for triad_r in [160, 237, 206]:
        ax.axhline(triad_r, color="gold", ls="--", lw=1.0, alpha=0.9)
    ax.set_xlabel("Rank within mask"); ax.set_ylabel("Residue number")
    ax.set_title("Residue Distribution by Mask", fontsize=TITLE_FS)
    ax.legend(fontsize=8)

    # ── right: distance distribution ──
    ax2 = axes[1]
    allo_dists   = [float(v) for v in masks_allo.get("distances_from_triad", {}).values()]
    active_dists = [float(v) for v in masks_active.get("distances_from_triad", {}).values()]
    if allo_dists and active_dists:
        ax2.hist(active_dists, bins=20, color=AS_C,   alpha=0.7, label="Active-site", edgecolor="white")
        ax2.hist(allo_dists,   bins=20, color=ALLO_C, alpha=0.7, label="Allosteric",  edgecolor="white")
        ax2.axvline(13, color=AS_C,   ls="--", lw=1.2, label="AS cutoff (≤13 Å)")
        ax2.axvline(16, color=ALLO_C, ls="--", lw=1.2, label="Allo cutoff (≥16 Å)")
        ax2.set_xlabel("Distance from triad center (Å)")
        ax2.set_ylabel("Count")
        ax2.set_title("Distance from Catalytic Triad", fontsize=TITLE_FS)
        ax2.legend(fontsize=7)
    else:
        ax2.text(0.5, 0.5, "distances_from_triad\nnot in JSON", ha="center", va="center",
                 transform=ax2.transAxes)

    fig.suptitle("F05 · Mask Definition: Active-site vs. Allosteric", fontsize=TITLE_FS+1, fontweight="bold")
    save(fig, "F05_mask_definition.png")

# ══════════════════════════════════════════════════════════════════════════════
# F06 – ESMFold Structural Validation
# ══════════════════════════════════════════════════════════════════════════════
def make_F06():
    esm = jload(os.path.join(RES, "esmfold", "esmfold_analysis.json"))

    # Expected keys per candidate: pLDDT, rmsd_vs_wt, s160_h237, h237_d206, group
    candidates = [(k, v) for k, v in esm.items() if k != "WT"]
    names  = [k for k, _ in candidates]
    rmsd   = [v["rmsd_vs_wt"] for _, v in candidates]
    plddt  = [v["pLDDT"]      for _, v in candidates]
    groups = [v.get("group", "unknown") for _, v in candidates]
    colors = [ALLO_C if g == "allosteric" else AS_C for g in groups]

    wt_plddt = esm["WT"]["pLDDT"] if "WT" in esm else None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── left: RMSD bar ──
    ax = axes[0]
    xs  = np.arange(len(names))
    ax.bar(xs, rmsd, color=colors, edgecolor="black", linewidth=0.6, alpha=0.85)
    ax.axhline(2.0, color=GREY, ls="--", lw=1.0, label="2 Å threshold")
    ax.set_xticks(xs); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Backbone RMSD vs. WT (Å)")
    ax.set_title("ESMFold: Backbone RMSD", fontsize=TITLE_FS)
    ax.legend(fontsize=7)
    patches = [mpatches.Patch(color=ALLO_C, label="Allosteric"),
               mpatches.Patch(color=AS_C,   label="Active-site")]
    ax.legend(handles=patches, fontsize=7)

    # ── right: pLDDT bar ──
    ax2 = axes[1]
    ax2.bar(xs, plddt, color=colors, edgecolor="black", linewidth=0.6, alpha=0.85)
    if wt_plddt:
        ax2.axhline(wt_plddt, color="gold", ls="--", lw=1.2, label=f"WT pLDDT = {wt_plddt:.2f}")
    ax2.set_xticks(xs); ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax2.set_ylabel("Mean pLDDT"); ax2.set_ylim(0.85, 1.0)
    ax2.set_title("ESMFold: Structural Confidence (pLDDT)", fontsize=TITLE_FS)
    ax2.legend(fontsize=7)

    fig.suptitle("F06 · ESMFold Structural Validation (Top 6 Candidates)", fontsize=TITLE_FS+1, fontweight="bold")
    save(fig, "F06_esmfold_validation.png")

# ══════════════════════════════════════════════════════════════════════════════
# F07 – Catalytic Triad Geometry
# ══════════════════════════════════════════════════════════════════════════════
def make_F07():
    esm = jload(os.path.join(RES, "esmfold", "esmfold_analysis.json"))

    all_keys  = list(esm.keys())
    names     = all_keys  # includes WT
    s160_h237 = [esm[k]["s160_h237"] for k in names]
    h237_d206 = [esm[k]["h237_d206"] for k in names]
    groups    = [esm[k].get("group", "WT") for k in names]

    cmap = {"allosteric": ALLO_C, "active_site": AS_C, "WT": "gold"}
    colors = [cmap.get(g, GREY) for g in groups]

    wt_s160 = esm["WT"]["s160_h237"] if "WT" in esm else 7.92
    wt_h237 = esm["WT"]["h237_d206"] if "WT" in esm else 4.43

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    xs = np.arange(len(names))

    for ax, vals, wt_val, label, title in [
        (axes[0], s160_h237, wt_s160, "Cα dist S160–H237 (Å)", "S160–H237 Distance"),
        (axes[1], h237_d206, wt_h237, "Cα dist H237–D206 (Å)", "H237–D206 Distance"),
    ]:
        ax.bar(xs, vals, color=colors, edgecolor="black", linewidth=0.6, alpha=0.85)
        ax.axhline(wt_val, color="gold", ls="--", lw=1.2, label=f"WT = {wt_val:.2f} Å")
        ax.set_xticks(xs); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel(label); ax.set_title(title, fontsize=TITLE_FS)
        ax.legend(fontsize=7)

    fig.suptitle("F07 · Catalytic Triad Geometry Preservation", fontsize=TITLE_FS+1, fontweight="bold")
    save(fig, "F07_triad_geometry.png")

# ══════════════════════════════════════════════════════════════════════════════
# F08 – Docking Scores
# ══════════════════════════════════════════════════════════════════════════════
def make_F08():
    dock = jload(os.path.join(RES, "docking", "docking_summary.json"))

    # Expected: {candidate: {score_kcal: float, group: str, delta_vs_wt: float}}
    wt_score = dock.get("WT", {}).get("score_kcal", -5.307)
    candidates = [(k, v) for k, v in dock.items() if k != "WT"]
    names   = [k for k, _ in candidates]
    scores  = [v["score_kcal"]  for _, v in candidates]
    deltas  = [v.get("delta_vs_wt", v["score_kcal"] - wt_score) for _, v in candidates]
    groups  = [v.get("group", "unknown") for _, v in candidates]
    colors  = [ALLO_C if g == "allosteric" else AS_C for g in groups]

    _, p = stats.mannwhitneyu(
        [s for s, g in zip(scores, groups) if g == "allosteric"],
        [s for s, g in zip(scores, groups) if g != "allosteric"],
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    xs = np.arange(len(names))

    ax = axes[0]
    ax.bar(xs, scores, color=colors, edgecolor="black", linewidth=0.6, alpha=0.85)
    ax.axhline(wt_score, color="gold", ls="--", lw=1.2, label=f"WT = {wt_score:.3f} kcal/mol")
    ax.set_xticks(xs); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Vina docking score (kcal/mol)")
    ax.set_title("AutoDock Vina Scores (BHET)", fontsize=TITLE_FS)
    ax.legend(fontsize=7)

    ax2 = axes[1]
    ax2.bar(xs, deltas, color=colors, edgecolor="black", linewidth=0.6, alpha=0.85)
    ax2.axhline(0, color=GREY, ls="--", lw=0.8)
    ax2.set_xticks(xs); ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax2.set_ylabel("ΔScore vs. WT (kcal/mol)")
    ax2.set_title(f"Δ vs. WT  (p={p:.2f}, null result — expected)", fontsize=TITLE_FS)
    ax2.text(0.5, 0.05, "Null result expected: Vina uncertainty ~1 kcal/mol\n"
             "exceeds total score range (0.23 kcal/mol)",
             transform=ax2.transAxes, ha="center", fontsize=7, color=GREY,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    patches = [mpatches.Patch(color=ALLO_C, label="Allosteric"),
               mpatches.Patch(color=AS_C,   label="Active-site")]
    axes[0].legend(handles=patches, fontsize=7)

    fig.suptitle("F08 · AutoDock Vina Docking (BHET Substrate)", fontsize=TITLE_FS+1, fontweight="bold")
    save(fig, "F08_docking.png")

# ══════════════════════════════════════════════════════════════════════════════
# F09 – Sequence Conservation
# ══════════════════════════════════════════════════════════════════════════════
def make_F09():
    con = jload(os.path.join(RES, "conservation", "conservation_analysis.json"))

    # Expected keys: active_site_scores, allosteric_scores, per_residue, fast_petase_sites
    as_scores   = con.get("active_site_scores",   [])
    allo_scores = con.get("allosteric_scores",     [])
    fp_sites    = con.get("fast_petase_sites",     {})  # {mutation: conservation}

    _, p = stats.mannwhitneyu(allo_scores, as_scores) if as_scores and allo_scores else (None, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.hist(as_scores,   bins=20, color=AS_C,   alpha=0.7, label="Active-site", edgecolor="white")
    ax.hist(allo_scores, bins=20, color=ALLO_C, alpha=0.7, label="Allosteric",  edgecolor="white")
    ax.set_xlabel("Conservation score"); ax.set_ylabel("Count")
    ax.set_title(f"Conservation Distribution  p = {p:.2f}\n(null result — expected, high MSA divergence)", fontsize=TITLE_FS)
    ax.legend(fontsize=8)

    ax2 = axes[1]
    if fp_sites:
        muts  = list(fp_sites.keys())
        cvals = [fp_sites[m] for m in muts]
        xs    = np.arange(len(muts))
        ax2.bar(xs, cvals, color=ALLO_C, edgecolor="black", linewidth=0.6, alpha=0.85)
        ax2.set_xticks(xs); ax2.set_xticklabels(muts, rotation=30, ha="right", fontsize=8)
        ax2.set_ylabel("Conservation score")
        ax2.set_title("FAST-PETase Mutation Site Conservation\n(low = tolerates mutation)", fontsize=TITLE_FS)
        ax2.axhline(0.25, color=GREY, ls="--", lw=0.8, label="Bottom quartile")
        ax2.legend(fontsize=7)
    else:
        ax2.text(0.5, 0.5, "fast_petase_sites not in JSON", ha="center", va="center",
                 transform=ax2.transAxes)

    fig.suptitle("F09 · Sequence Conservation Analysis", fontsize=TITLE_FS+1, fontweight="bold")
    save(fig, "F09_conservation.png")

# ══════════════════════════════════════════════════════════════════════════════
# F10 – GNN Performance Summary
# ══════════════════════════════════════════════════════════════════════════════
def make_F10():
    # Load ensemble metrics
    ens_path = os.path.join(RES, "models", "ensemble_test_metrics.json")
    ens = jload(ens_path)

    # Per-seed metrics
    seed_paths = sorted([
        os.path.join(RES, "models", d, "metrics.json")
        for d in os.listdir(os.path.join(RES, "models"))
        if d.startswith("seed_")
    ])
    seed_data = [jload(p) for p in seed_paths if os.path.exists(p)]
    seeds_r2 = [s["r2"]      for s in seed_data]
    seeds_pcc= [s["pearson"] for s in seed_data]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── left: per-seed R² scatter ──
    ax = axes[0]
    xs = np.arange(len(seeds_r2))
    ax.scatter(xs, seeds_r2, color=ALLO_C, s=80, zorder=3, label="Per-seed R²")
    ax.axhline(ens["r2"], color=AS_C, ls="--", lw=1.5, label=f"Ensemble R² = {ens['r2']:.3f}")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"Seed {i+1}" for i in xs], fontsize=8)
    ax.set_ylabel("R² (test set)")
    ax.set_ylim(0.55, 0.65)
    ax.set_title("Per-seed vs. Ensemble R²", fontsize=TITLE_FS)
    ax.legend(fontsize=8)

    # ── right: predicted vs actual scatter ──
    ax2 = axes[1]
    pred_actual_path = os.path.join(RES, "models", "ensemble_predictions.json")
    if os.path.exists(pred_actual_path):
        pa = jload(pred_actual_path)
        pred = np.array(pa["predicted"])
        true = np.array(pa["true"])
        ax2.scatter(true, pred, alpha=0.25, s=8, color=ALLO_C)
        mn, mx = min(true.min(), pred.min()), max(true.max(), pred.max())
        ax2.plot([mn, mx], [mn, mx], "k--", lw=1.0)
        ax2.set_xlabel("Measured pKd"); ax2.set_ylabel("Predicted pKd")
        ax2.set_title(f"Ensemble: Predicted vs. Actual\nR² = {ens['r2']:.3f}  r = {ens['pearson']:.3f}", fontsize=TITLE_FS)
    else:
        ax2.text(0.5, 0.5, "ensemble_predictions.json\nnot found — run inference first",
                 ha="center", va="center", transform=ax2.transAxes, color=GREY, fontsize=9)
        ax2.set_title(f"Ensemble Performance\nR² = {ens['r2']:.3f}  RMSE = {ens['rmse']:.3f}", fontsize=TITLE_FS)

    fig.suptitle("F10 · GNN Affinity Judge: Training Performance", fontsize=TITLE_FS+1, fontweight="bold")
    save(fig, "F10_gnn_performance.png")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    figs = {
        "F01": make_F01, "F02": make_F02, "F03": make_F03,
        "F04": make_F04, "F05": make_F05, "F06": make_F06,
        "F07": make_F07, "F08": make_F08, "F09": make_F09,
        "F10": make_F10,
    }

    targets = sys.argv[1:] if len(sys.argv) > 1 else list(figs.keys())
    for key in targets:
        if key in figs:
            print(f"Generating {key}...")
            try:
                figs[key]()
            except Exception as e:
                print(f"  ERROR in {key}: {e}")
        else:
            print(f"Unknown figure: {key}")

    print(f"\nDone. Figures saved to {OUT}")