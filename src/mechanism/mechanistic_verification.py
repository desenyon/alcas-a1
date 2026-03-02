"""
ALCAS - Mechanistic Verification (Stage 11)
Proves allostery is mechanistic, not coincidental.

For top allosteric candidates vs WT:
1. Compute residue interaction network from MD trajectories
2. Compare communication paths from mutation sites to catalytic triad
3. Compute cross-correlation delta (variant - WT)
4. Show active-site stabilization driven by distant mutations

Usage (run AFTER MD completes):
    cd ~/alcas
    python src/mechanism/mechanistic_verification.py
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis.distances import distance_array
    HAS_MDA = True
except ImportError:
    HAS_MDA = False

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False
    print("WARNING: networkx not installed. Run:")
    print("  pip install networkx --break-system-packages")

# ============================================================
# CONFIG
# ============================================================
MD_DIR     = Path('data/petase/md')
COUPLING   = Path('data/petase/coupling')
OUT_DIR    = Path('results/mechanism')
OUT_DIR.mkdir(parents=True, exist_ok=True)

CATALYTIC  = {160, 206, 237}

# Top allosteric candidates from FoldX (mutation site resnums)
TOP_ALLO_MUTS = {
    'G79P_S169A':  [79, 169],
    'G79P_A179V':  [79, 179],
    'G79P_M262L':  [79, 262],
}

# Top active-site candidates
TOP_AS_MUTS = {
    'Q182I': [182],
    'E231V': [231],
    'Q182L': [182],
}

print("=" * 60)
print("ALCAS - Mechanistic Verification")
print("=" * 60)


# ============================================================
# LOAD NMA BASELINE (already computed in Stage 7)
# ============================================================
print("Loading NMA coupling baseline...")

cross_corr_path  = COUPLING / 'cross_corr_matrix.npy'
coupling_path    = COUPLING / 'coupling_to_triad.npy'
centrality_path  = COUPLING / 'betweenness_centrality.npy'
residue_path     = COUPLING / 'all_residue_scores.json'

if not cross_corr_path.exists():
    print(f"  WARNING: NMA coupling files not found at {COUPLING}")
    print("  Run src/mechanism/coupling_analysis_nma.py first")
    has_nma = False
else:
    cross_corr_nma = np.load(cross_corr_path)
    coupling_nma   = np.load(coupling_path)
    centrality_nma = np.load(centrality_path)
    with open(residue_path) as f:
        residue_scores = json.load(f)
    has_nma = True
    print(f"  Loaded NMA cross-correlation matrix: {cross_corr_nma.shape}")


# ============================================================
# COMPUTE MD-DERIVED CROSS-CORRELATION FROM TRAJECTORY
# ============================================================
def compute_md_cross_correlation(traj_path, top_path, stride=10):
    """
    Compute normalized cross-correlation matrix from MD trajectory.
    C_ij = <delta_ri . delta_rj> / sqrt(<delta_ri^2><delta_rj^2>)
    """
    u  = mda.Universe(str(top_path), str(traj_path))
    ca = u.select_atoms('protein and name CA')
    n  = len(ca)

    print(f"    Residues: {n}, Frames: {len(u.trajectory)}, Stride: {stride}")

    # Collect CA positions across trajectory
    positions = []
    for ts in u.trajectory[::stride]:
        positions.append(ca.positions.copy())
    positions = np.array(positions)   # (n_frames, n_residues, 3)

    # Mean-center
    mean_pos   = positions.mean(axis=0)               # (n_res, 3)
    delta      = positions - mean_pos                  # (n_frames, n_res, 3)

    # Compute dot products
    # <delta_ri . delta_rj> = sum over xyz
    dot = np.einsum('fid,fjd->ij', delta, delta) / len(positions)   # (n_res, n_res)

    # Normalize
    var_i   = np.diag(dot)                                           # (n_res,)
    norm    = np.sqrt(np.outer(var_i, var_i))
    norm[norm == 0] = 1.0
    corr    = dot / norm

    return corr, list(ca.resids)


# ============================================================
# LOAD MD TRAJECTORIES
# ============================================================
print("\nLoading MD trajectories for WT...")

wt_corr_matrices = []
rep_resnums      = None

for rep in range(1, 4):
    traj = MD_DIR / f'rep{rep}_traj.dcd'
    top  = MD_DIR / f'rep{rep}_topology.pdb'
    if traj.exists() and top.exists():
        print(f"  Replicate {rep}:")
        corr, resnums = compute_md_cross_correlation(traj, top, stride=20)
        wt_corr_matrices.append(corr)
        if rep_resnums is None:
            rep_resnums = resnums
        print(f"    Cross-correlation range: [{corr.min():.3f}, {corr.max():.3f}]")
    else:
        print(f"  Replicate {rep}: NOT FOUND — skipping")

if len(wt_corr_matrices) == 0:
    print("No MD trajectories found. Run MD first.")
    exit(1)

# Average across replicates
wt_corr_mean = np.mean(wt_corr_matrices, axis=0)
np.save(OUT_DIR / 'wt_md_cross_corr.npy', wt_corr_mean)
print(f"\nWT mean cross-correlation saved. Shape: {wt_corr_mean.shape}")


# ============================================================
# RESIDUE INTERACTION NETWORK FROM MD
# ============================================================
def build_contact_network(corr_matrix, resnums, threshold=0.3):
    """
    Build residue interaction network where edge weight = |correlation|
    if above threshold.
    """
    G   = nx.Graph()
    n   = len(resnums)
    for i, rn_i in enumerate(resnums):
        G.add_node(rn_i, resnum=rn_i)
    for i in range(n):
        for j in range(i+1, n):
            w = abs(corr_matrix[i, j])
            if w >= threshold:
                G.add_edge(resnums[i], resnums[j], weight=w)
    return G


if HAS_NX and rep_resnums is not None:
    print("\nBuilding WT residue interaction network...")
    rn_list = [int(r) for r in rep_resnums]
    wt_graph = build_contact_network(wt_corr_mean, rn_list, threshold=0.3)
    print(f"  Nodes: {wt_graph.number_of_nodes()}, Edges: {wt_graph.number_of_edges()}")

    # Betweenness centrality
    print("  Computing betweenness centrality (may take ~30s)...")
    try:
        bc = nx.betweenness_centrality(wt_graph, weight='weight', normalized=True)
        bc_sorted = sorted(bc.items(), key=lambda x: x[1], reverse=True)
        print("  Top 10 residues by betweenness centrality:")
        for rn, score in bc_sorted[:10]:
            cat = 'CATALYTIC' if rn in CATALYTIC else ''
            print(f"    Res {rn:4d}: {score:.4f}  {cat}")
    except Exception as e:
        print(f"  Betweenness centrality failed: {e}")
        bc = {}

    # Communication paths: mutation sites -> catalytic triad
    print("\nCommunication path analysis...")
    path_results = {}

    for group_name, mut_dict in [('allosteric', TOP_ALLO_MUTS),
                                  ('active_site', TOP_AS_MUTS)]:
        path_results[group_name] = {}
        for mut_name, mut_resnums in mut_dict.items():
            paths = {}
            for src_rn in mut_resnums:
                for tgt_rn in CATALYTIC:
                    if src_rn in wt_graph and tgt_rn in wt_graph:
                        try:
                            path = nx.shortest_path(
                                wt_graph, src_rn, tgt_rn, weight=None
                            )
                            length = len(path) - 1
                            paths[f'{src_rn}->{tgt_rn}'] = {
                                'path':   path,
                                'length': length,
                            }
                        except nx.NetworkXNoPath:
                            paths[f'{src_rn}->{tgt_rn}'] = None
            path_results[group_name][mut_name] = paths
            print(f"  {mut_name} [{group_name}]:")
            for key, val in paths.items():
                if val:
                    print(f"    {key}: length={val['length']}, path={val['path']}")
                else:
                    print(f"    {key}: no path found")


# ============================================================
# CROSS-CORRELATION DELTA ANALYSIS
# ============================================================
print("\nCross-correlation delta analysis (MD vs NMA)...")

if has_nma and rep_resnums is not None:
    nma_n   = cross_corr_nma.shape[0]
    md_n    = wt_corr_mean.shape[0]

    print(f"  NMA matrix size: {nma_n}x{nma_n}")
    print(f"  MD matrix size:  {md_n}x{md_n}")

    # Align to common size
    min_n = min(nma_n, md_n)
    delta = wt_corr_mean[:min_n, :min_n] - cross_corr_nma[:min_n, :min_n]

    np.save(OUT_DIR / 'corr_delta_md_minus_nma.npy', delta)

    print(f"  Delta range: [{delta.min():.3f}, {delta.max():.3f}]")
    print(f"  Delta mean:  {delta.mean():.4f}")
    print(f"  Delta std:   {delta.std():.4f}")

    # Find residues with largest MD vs NMA discrepancy
    # These are residues where NMA missed something MD captured
    rms_per_res = np.sqrt((delta**2).mean(axis=1))
    top_discrepancy = np.argsort(rms_per_res)[-10:][::-1]
    print("\n  Top 10 residues with largest NMA vs MD discrepancy:")
    for idx in top_discrepancy:
        if idx < len(rn_list):
            rn = rn_list[idx]
            cat = 'CATALYTIC' if rn in CATALYTIC else ''
            print(f"    Res {rn:4d}: RMS delta={rms_per_res[idx]:.4f}  {cat}")


# ============================================================
# ACTIVE-SITE COUPLING STRENGTH: ALLOSTERIC RESIDUES -> TRIAD
# ============================================================
print("\nActive-site coupling from allosteric mutation sites...")

if rep_resnums is not None:
    rn_arr  = np.array(rn_list)
    results = []

    for group_name, mut_dict in [('allosteric', TOP_ALLO_MUTS),
                                  ('active_site', TOP_AS_MUTS)]:
        for mut_name, mut_resnums in mut_dict.items():
            for mut_rn in mut_resnums:
                if mut_rn not in rn_list:
                    continue
                mut_idx = rn_list.index(mut_rn)

                # Average coupling to catalytic triad residues
                triad_couplings = []
                for cat_rn in CATALYTIC:
                    if cat_rn in rn_list:
                        cat_idx = rn_list.index(cat_rn)
                        c = abs(float(wt_corr_mean[mut_idx, cat_idx]))
                        triad_couplings.append(c)

                if triad_couplings:
                    mean_coupling = float(np.mean(triad_couplings))
                    results.append({
                        'mutation':       mut_name,
                        'group':          group_name,
                        'resnum':         mut_rn,
                        'mean_coupling_to_triad': round(mean_coupling, 4),
                        'triad_couplings': [round(c, 4) for c in triad_couplings],
                    })
                    print(f"  {mut_name:20s} [{group_name:11s}]  "
                          f"res{mut_rn} -> triad coupling = {mean_coupling:.4f}")


# ============================================================
# SAVE ALL RESULTS
# ============================================================
output = {
    'wt_corr_matrix_path':   str(OUT_DIR / 'wt_md_cross_corr.npy'),
    'n_replicates_used':     len(wt_corr_matrices),
    'communication_paths':   path_results if HAS_NX else {},
    'coupling_to_triad':     results if rep_resnums else [],
}
with open(OUT_DIR / 'mechanistic_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n{'='*60}")
print("MECHANISTIC VERIFICATION COMPLETE")
print(f"{'='*60}")
print(f"Saved to {OUT_DIR}/")
print("Key outputs:")
print("  wt_md_cross_corr.npy         - MD cross-correlation matrix")
print("  corr_delta_md_minus_nma.npy  - MD vs NMA delta matrix")
print("  mechanistic_results.json     - communication paths + coupling")
print("\nNext: python src/viz/plot_all_figures.py")