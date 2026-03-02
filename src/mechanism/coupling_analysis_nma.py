"""
ALCAS - Allosteric Coupling Analysis via Normal Mode Analysis
Uses ProDy elastic network model (ANM) on WT PETase crystal structure.

Computes:
  1. Residue cross-correlations (C_ij matrix)
  2. Contact map and residue interaction network
  3. Betweenness centrality of each residue in the network
  4. Mean square fluctuations (B-factors proxy)
  5. Coupling score to catalytic triad residues
  6. Final allosteric mask: distance + top coupling score

Outputs:
  data/petase/coupling/correlation_matrix.npy
  data/petase/coupling/coupling_scores.json
  data/petase/masks_allosteric.json  (FINAL locked mask)
  data/petase/masks_active.json      (confirmed)
"""

import numpy as np
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import prody
prody.confProDy(verbosity='none')

# ============================================================
# CONFIG
# ============================================================
PDB_PATH     = Path('data/petase/5XJH.pdb')
OUT_DIR      = Path('data/petase/coupling')
OUT_DIR.mkdir(parents=True, exist_ok=True)

MASK_DIR     = Path('data/petase')

# Catalytic triad
CATALYTIC_TRIAD = [160, 206, 237]

# Mask thresholds
ACTIVE_SITE_DIST   = 8.0   # A from triad atoms
ALLOSTERIC_MIN_DIST= 16.0  # A from active-site residues
TOP_COUPLING_PCT   = 50    # top 30% by coupling score = allosteric candidates
N_MODES            = 20    # ANM modes for correlation (20 is standard)
CONTACT_CUTOFF     = 7.5   # A for residue contact network


# ============================================================
# LOAD STRUCTURE
# ============================================================
print("=" * 60)
print("ALCAS - NMA Coupling Analysis")
print("=" * 60)

print("Loading PETase structure...")
struct  = prody.parsePDB(str(PDB_PATH))
protein = struct.select('protein and chain A')
ca      = protein.select('calpha')

residues = list(ca.getResnums())
n_res    = len(residues)
print(f"  {n_res} residues, {len(residues)} CA atoms")
print(f"  Catalytic triad: {CATALYTIC_TRIAD}")


# ============================================================
# ANM NORMAL MODE ANALYSIS
# ============================================================
print(f"\nBuilding ANM with {N_MODES} modes...")
anm = prody.ANM('PETase')
anm.buildHessian(ca, cutoff=CONTACT_CUTOFF)
anm.calcModes(n_modes=N_MODES)
print(f"  ANM built. Variance explained by first mode: "
      f"{anm[0].getVariance()/anm.getVariances().sum()*100:.1f}%")

# Cross-correlation matrix [n_res x n_res]
print("Computing cross-correlation matrix...")
cross_corr = prody.calcCrossCorr(anm[:N_MODES])  # values in [-1, 1]
np.save(OUT_DIR / 'cross_corr_matrix.npy', cross_corr)
print(f"  Cross-corr matrix shape: {cross_corr.shape}")
print(f"  Range: [{cross_corr.min():.3f}, {cross_corr.max():.3f}]")

# Mean square fluctuations
msf = prody.calcSqFlucts(anm[:N_MODES])
np.save(OUT_DIR / 'msf.npy', msf)


# ============================================================
# RESIDUE INTERACTION NETWORK
# ============================================================
print("\nBuilding residue interaction network...")
coords  = ca.getCoords()   # [n_res, 3]
resnum  = ca.getResnums()

# Contact matrix: residues within CONTACT_CUTOFF
dist_matrix = np.linalg.norm(
    coords[:, None, :] - coords[None, :, :], axis=2
)
contact_matrix = (dist_matrix < CONTACT_CUTOFF).astype(float)
np.fill_diagonal(contact_matrix, 0)

# Weight contacts by absolute cross-correlation
weighted_adj = contact_matrix * np.abs(cross_corr)
np.save(OUT_DIR / 'weighted_adjacency.npy', weighted_adj)

# Betweenness centrality via networkx
print("Computing betweenness centrality...")
import networkx as nx
G = nx.from_numpy_array(weighted_adj)
bc = nx.betweenness_centrality(G, weight='weight', normalized=True)
bc_array = np.array([bc[i] for i in range(n_res)])
np.save(OUT_DIR / 'betweenness_centrality.npy', bc_array)
print(f"  Top 5 residues by centrality: "
      f"{[resnum[i] for i in bc_array.argsort()[-5:][::-1]]}")


# ============================================================
# COUPLING TO CATALYTIC TRIAD
# ============================================================
print("\nComputing coupling to catalytic triad...")

# Map catalytic triad resnums to indices
triad_indices = []
for rn in CATALYTIC_TRIAD:
    idx = np.where(resnum == rn)[0]
    if len(idx) > 0:
        triad_indices.append(int(idx[0]))
        print(f"  Residue {rn}: index {idx[0]}")
    else:
        print(f"  WARNING: residue {rn} not found in CA selection")

# Mean absolute cross-correlation to triad residues
coupling_to_triad = np.mean(
    np.abs(cross_corr[:, triad_indices]), axis=1
)
np.save(OUT_DIR / 'coupling_to_triad.npy', coupling_to_triad)
print(f"  Coupling range: [{coupling_to_triad.min():.4f}, {coupling_to_triad.max():.4f}]")

# Combined coupling score: 0.5 * coupling_to_triad + 0.5 * betweenness
coupling_score = 0.5 * (coupling_to_triad / coupling_to_triad.max()) + \
                 0.5 * (bc_array / bc_array.max() if bc_array.max() > 0 else bc_array)
np.save(OUT_DIR / 'coupling_score.npy', coupling_score)


# ============================================================
# LOAD ACTIVE SITE MASK
# ============================================================
print("\nLoading active site mask...")
with open(MASK_DIR / 'mask_active_site.json') as f:
    active_mask_data = json.load(f)
active_site_resnums = set(active_mask_data['residues'])
print(f"  Active site: {len(active_site_resnums)} residues")

# Active-site CA coordinates
# Use triad CENTER as reference point for all distance calculations
triad_indices_list = [i for i, rn in enumerate(resnum) if rn in CATALYTIC_TRIAD]
triad_center       = coords[triad_indices_list].mean(axis=0)
as_indices         = [i for i, rn in enumerate(resnum) if rn in active_site_resnums]
as_coords          = coords[as_indices]


# ============================================================
# DEFINE FINAL ALLOSTERIC MASK
# ============================================================
print("\nDefining allosteric mask...")

coupling_threshold = np.percentile(coupling_score, 100 - TOP_COUPLING_PCT)
print(f"  Coupling threshold (top {TOP_COUPLING_PCT}%): {coupling_threshold:.4f}")

allosteric_residues = []
excluded_too_close  = []
excluded_low_coupling = []

for i, rn in enumerate(resnum):
    # Skip active site
    if rn in active_site_resnums:
        continue
    # Skip catalytic triad
    if rn in CATALYTIC_TRIAD:
        continue

    # Distance constraint: >16A from catalytic triad center
    min_dist = float(np.linalg.norm(coords[i] - triad_center))
    if min_dist < ALLOSTERIC_MIN_DIST:
        excluded_too_close.append(rn)
        continue

    # Coupling constraint: top 30% by combined score
    if coupling_score[i] < coupling_threshold:
        excluded_low_coupling.append(rn)
        continue

    allosteric_residues.append({
        'resnum':              int(rn),
        'min_dist_from_as':   float(round(min_dist, 2)),
        'coupling_to_triad':  float(round(float(coupling_to_triad[i]), 4)),
        'betweenness':        float(round(float(bc_array[i]), 6)),
        'coupling_score':     float(round(float(coupling_score[i]), 4)),
    })

allosteric_residues.sort(key=lambda x: x['coupling_score'], reverse=True)
allosteric_resnums = [r['resnum'] for r in allosteric_residues]

print(f"\n  Total residues:          {n_res}")
print(f"  Active-site excluded:    {len(active_site_resnums)}")
print(f"  Too close (<12A):        {len(excluded_too_close)}")
print(f"  Low coupling excluded:   {len(excluded_low_coupling)}")
print(f"  ALLOSTERIC MASK SIZE:    {len(allosteric_residues)}")
print(f"\n  Top 10 allosteric residues by coupling score:")
for r in allosteric_residues[:10]:
    print(f"    Res {r['resnum']:4d}: dist={r['min_dist_from_as']:5.1f}A "
          f"coupling={r['coupling_to_triad']:.4f} "
          f"centrality={r['betweenness']:.5f} "
          f"score={r['coupling_score']:.4f}")


# ============================================================
# SAVE MASKS
# ============================================================
allosteric_mask = {
    'residues':            allosteric_resnums,
    'n_residues':          len(allosteric_resnums),
    'method':              'ANM cross-correlation + betweenness centrality',
    'n_modes':             N_MODES,
    'contact_cutoff_A':    CONTACT_CUTOFF,
    'min_dist_from_as_A':  ALLOSTERIC_MIN_DIST,
    'top_coupling_pct':    TOP_COUPLING_PCT,
    'coupling_threshold':  float(coupling_threshold),
    'details':             allosteric_residues,
    'status':              'FINAL - locked before search',
}
with open(MASK_DIR / 'masks_allosteric.json', 'w') as f:
    json.dump(allosteric_mask, f, indent=2)

# Save coupling scores for all residues
all_scores = {
    int(rn): {
        'coupling_to_triad': float(coupling_to_triad[i]),
        'betweenness':       float(bc_array[i]),
        'coupling_score':    float(coupling_score[i]),
        'in_active_site':    bool(rn in active_site_resnums),
        'in_allosteric':     bool(rn in allosteric_resnums),
    }
    for i, rn in enumerate(resnum)
}
with open(OUT_DIR / 'all_residue_scores.json', 'w') as f:
    json.dump(all_scores, f, indent=2)

print(f"\nSaved:")
print(f"  {MASK_DIR}/masks_allosteric.json  (FINAL allosteric mask)")
print(f"  {OUT_DIR}/cross_corr_matrix.npy")
print(f"  {OUT_DIR}/coupling_to_triad.npy")
print(f"  {OUT_DIR}/coupling_score.npy")
print(f"  {OUT_DIR}/all_residue_scores.json")

# ============================================================
# FAST-PETASE VALIDATION
# ============================================================
print("\n" + "="*60)
print("FAST-PETase Mutation Validation")
print("="*60)
FAST_MUTATIONS = {121: 'S121E', 186: 'D186H', 224: 'R224Q', 233: 'N233K', 280: 'R280A'}

for rn, mut in FAST_MUTATIONS.items():
    idx = np.where(resnum == rn)[0]
    if len(idx) == 0:
        print(f"  {mut}: NOT FOUND in structure")
        continue
    i   = int(idx[0])
    in_as   = rn in active_site_resnums
    in_allo = rn in allosteric_resnums
    dist    = float(np.linalg.norm(coords[i] - triad_center))
    score   = float(coupling_score[i])
    cat     = 'ACTIVE-SITE' if in_as else ('ALLOSTERIC' if in_allo else 'INTERMEDIATE')
    print(f"  {mut}: dist={dist:.1f}A | score={score:.4f} | {cat}")

print("\n" + "="*60)
print("COUPLING ANALYSIS COMPLETE")
print(f"Allosteric mask: {len(allosteric_resnums)} residues")
print(f"Active-site mask: {len(active_site_resnums)} residues")
print("Both masks LOCKED. Ready for ALCAS search.")
print("="*60)
