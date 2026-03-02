"""
ALCAS - Affinity Scoring of Mutant Candidates
Scores all active-site and allosteric candidates using the trained
ensemble GNN affinity model. Builds graphs in the exact same format
as the training data (plain dicts with ligand_x, protein_x, etc.).

Feature dims (must match training):
  ligand_x: 43, ligand_edge_attr: 11 (5 bond + 6 RBF)
  protein_x: 35, protein_edge_attr: 22 (16 RBF + 6 structural)
  cross_edge_attr: 18 (16 RBF + 2)
"""

import json
import numpy as np
import torch
import pickle
import sys
from pathlib import Path
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.affinity_model import AffinityModel
from data.dataloader import collate_fn

# ============================================================
# CONFIG
# ============================================================
CANDIDATE_DIR = Path('results/candidates')
MODEL_DIR     = Path('results/models')
OUT_DIR       = Path('results/scores')
OUT_DIR.mkdir(parents=True, exist_ok=True)

PDB_PATH    = Path('data/petase/5XJH.pdb')
SEEDS       = [42, 7, 13, 99, 2024, 314]
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BHET_SMILES = 'OC(=O)c1ccc(cc1)C(=O)OCCO'

AA3TO1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
    'HSD':'H','HSE':'H','HSP':'H','HIE':'H','HID':'H',
}


# ============================================================
# GRAPH BUILDING — dims locked to match training data
# ============================================================

def rbf_encode(dist, n_rbf, max_dist):
    centers = np.linspace(0, max_dist, n_rbf)
    sigma   = max_dist / n_rbf
    return np.exp(-((dist - centers) ** 2) / (2 * sigma ** 2))


def atom_features(atom):
    """43-dim ligand atom features."""
    from rdkit import Chem
    ATOM_TYPES = ['C','N','O','S','F','P','Cl','Br','I','H','B','Si','Se','Te','other']
    HYBRID = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]
    sym       = atom.GetSymbol()
    atom_type = [int(sym == t) for t in ATOM_TYPES[:-1]] + [int(sym not in ATOM_TYPES[:-1])]  # 15
    hyb       = [int(atom.GetHybridization() == h) for h in HYBRID]                            # 5
    feats = (
        atom_type +
        hyb +
        [atom.GetDegree() / 6.0] +
        [atom.GetFormalCharge() / 4.0] +
        [atom.GetNumImplicitHs() / 4.0] +
        [int(atom.GetIsAromatic())] +
        [int(atom.IsInRing())] +
        [atom.GetMass() / 100.0] +
        [atom.GetTotalValence() / 6.0] +
        [int(atom.IsInRingSize(3)), int(atom.IsInRingSize(4)),
         int(atom.IsInRingSize(5)), int(atom.IsInRingSize(6)),
         int(atom.IsInRingSize(7))] +
        [atom.GetNumRadicalElectrons()] +
        [int(atom.GetNoImplicit())] +
        [0.0] * 9
    )
    return feats[:43]


def build_ligand_graph(smiles):
    """Build ligand graph. ligand_edge_attr = 11 dims (5 bond + 6 RBF)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    mol  = Chem.RemoveHs(mol)
    conf = mol.GetConformer()
    pos  = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
    x    = np.array([atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)

    from rdkit import Chem as _Chem
    BT = _Chem.rdchem.BondType

    src, dst, ea = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf   = [
            int(bond.GetBondType() == BT.SINGLE),
            int(bond.GetBondType() == BT.DOUBLE),
            int(bond.GetBondType() == BT.TRIPLE),
            int(bond.GetBondType() == BT.AROMATIC),
            int(bond.GetIsConjugated()),
        ]                                                         # 5
        d    = float(np.linalg.norm(pos[i] - pos[j]))
        rbf  = rbf_encode(d, n_rbf=6, max_dist=5.0).tolist()    # 6
        feat = bf + rbf                                           # 11
        src += [i, j]; dst += [j, i]; ea += [feat, feat]

    return {
        'x':          torch.tensor(x,          dtype=torch.float),
        'pos':        torch.tensor(pos,        dtype=torch.float),
        'edge_index': torch.tensor([src, dst], dtype=torch.long),
        'edge_attr':  torch.tensor(ea,         dtype=torch.float),
    }


def residue_features(aa, idx, n_res):
    """35-dim residue features."""
    AA_LIST     = list('ACDEFGHIKLMNPQRSTVWY')
    HYDROPHOBIC = set('AVILMFYW')
    POLAR       = set('STNQ')
    CHARGED_POS = set('KRH')
    CHARGED_NEG = set('DE')
    AROMATIC    = set('FYW')
    one_hot = [int(aa == a) for a in AA_LIST]                    # 20
    props   = [
        int(aa in HYDROPHOBIC),
        int(aa in POLAR),
        int(aa in CHARGED_POS),
        int(aa in CHARGED_NEG),
        int(aa in AROMATIC),
        idx / max(n_res - 1, 1),
    ]                                                             # 6
    feat = one_hot + props + [0.0] * 9                           # 35
    return feat[:35]


def build_protein_graph(coords, sequence, cutoff=8.0):
    """Build protein graph. protein_edge_attr = 22 dims (16 RBF + 6 structural)."""
    n   = len(sequence)
    pos = np.array(coords, dtype=np.float32)
    x   = np.array([residue_features(aa, i, n) for i, aa in enumerate(sequence)],
                   dtype=np.float32)

    dist_mat = np.linalg.norm(pos[:, None] - pos[None, :], axis=2)
    src, dst, ea = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j: continue
            d = dist_mat[i, j]
            if d < cutoff:
                rbf  = rbf_encode(d, n_rbf=16, max_dist=cutoff).tolist()   # 16
                feat = rbf + [
                    abs(i - j) / max(n - 1, 1),
                    d / cutoff,
                    int(abs(i - j) == 1),
                    int(abs(i - j) == 2),
                    int(abs(i - j) < 5),
                    int(abs(i - j) >= 5),
                ]                                                            # 22
                src.append(i); dst.append(j); ea.append(feat)

    return {
        'x':          torch.tensor(x,          dtype=torch.float),
        'pos':        torch.tensor(pos,        dtype=torch.float),
        'edge_index': torch.tensor([src, dst], dtype=torch.long)  if src else torch.zeros(2, 0, dtype=torch.long),
        'edge_attr':  torch.tensor(ea,         dtype=torch.float) if ea  else torch.zeros(0, 22, dtype=torch.float),
    }


def build_cross_edges(pro_pos, lig_pos, cutoff=4.5):
    """Cross edges. cross_edge_attr = 18 dims (16 RBF + 2)."""
    src, dst, ea = [], [], []
    for pi, pc in enumerate(pro_pos):
        for li, lc in enumerate(lig_pos):
            d = float(np.linalg.norm(pc - lc))
            if d < cutoff:
                rbf  = rbf_encode(d, n_rbf=16, max_dist=cutoff).tolist()   # 16
                feat = rbf + [d / cutoff, float(np.exp(-d))]                # 18
                src.append(pi); dst.append(li); ea.append(feat)

    if not src:
        lig_cent = lig_pos.mean(axis=0)
        dists    = np.linalg.norm(pro_pos - lig_cent, axis=1)
        for pi in np.argsort(dists)[:5]:
            for li in range(len(lig_pos)):
                d    = float(np.linalg.norm(pro_pos[pi] - lig_pos[li]))
                rbf  = rbf_encode(d, n_rbf=16, max_dist=max(d + 1, cutoff)).tolist()
                feat = rbf + [min(d / cutoff, 3.0), float(np.exp(-d))]
                src.append(pi); dst.append(li); ea.append(feat)

    return {
        'edge_index': torch.tensor([src, dst], dtype=torch.long),
        'edge_attr':  torch.tensor(ea,         dtype=torch.float),
    }


def build_graph(sequence, pro_coords, lig_graph):
    pro   = build_protein_graph(pro_coords, sequence)
    cross = build_cross_edges(pro_coords, lig_graph['pos'].numpy())
    return {
        'ligand_x':           lig_graph['x'],
        'ligand_edge_index':  lig_graph['edge_index'],
        'ligand_edge_attr':   lig_graph['edge_attr'],
        'ligand_pos':         lig_graph['pos'],
        'protein_x':          pro['x'],
        'protein_edge_index': pro['edge_index'],
        'protein_edge_attr':  pro['edge_attr'],
        'protein_pos':        pro['pos'],
        'cross_edge_index':   cross['edge_index'],
        'cross_edge_attr':    cross['edge_attr'],
        'y':                  torch.tensor([0.0]),
    }


# ============================================================
# VALIDATE DIMS
# ============================================================
print("Validating feature dimensions against training data...")
with open('data/processed/graphs/test.pkl', 'rb') as f:
    ref_graphs = pickle.load(f)
ref = ref_graphs[0]

print(f"  Training: lig_x={ref['ligand_x'].shape[1]} "
      f"pro_x={ref['protein_x'].shape[1]} "
      f"lig_ea={ref['ligand_edge_attr'].shape[1]} "
      f"pro_ea={ref['protein_edge_attr'].shape[1]} "
      f"cross_ea={ref['cross_edge_attr'].shape[1]}")

import prody as _pdy; _pdy.confProDy(verbosity='none')
_s      = _pdy.parsePDB(str(PDB_PATH))
_ca     = _s.select('protein and chain A and calpha')
_seq    = list(''.join(AA3TO1.get(r, 'X') for r in _ca.getResnames()))
_coords = _ca.getCoords()
_lig    = build_ligand_graph(BHET_SMILES)
_g      = build_graph(_seq, _coords, _lig)

print(f"  Ours:     lig_x={_g['ligand_x'].shape[1]} "
      f"pro_x={_g['protein_x'].shape[1]} "
      f"lig_ea={_g['ligand_edge_attr'].shape[1]} "
      f"pro_ea={_g['protein_edge_attr'].shape[1]} "
      f"cross_ea={_g['cross_edge_attr'].shape[1]}")

assert _g['ligand_x'].shape[1]          == ref['ligand_x'].shape[1],          "ligand_x mismatch"
assert _g['protein_x'].shape[1]         == ref['protein_x'].shape[1],         "protein_x mismatch"
assert _g['ligand_edge_attr'].shape[1]  == ref['ligand_edge_attr'].shape[1],  "ligand_edge_attr mismatch"
assert _g['protein_edge_attr'].shape[1] == ref['protein_edge_attr'].shape[1], "protein_edge_attr mismatch"
assert _g['cross_edge_attr'].shape[1]   == ref['cross_edge_attr'].shape[1],   "cross_edge_attr mismatch"
print("  All dims match.")


# ============================================================
# LOAD ENSEMBLE
# ============================================================
print(f"\nLoading ensemble (device={DEVICE})...")

_ckpt0 = torch.load(MODEL_DIR / 'seed_42' / 'best_model.pt',
                    map_location='cpu', weights_only=True)
y_mean = float(_ckpt0['y_mean'])
y_std  = float(_ckpt0['y_std'])
print(f"Target stats: mean={y_mean:.3f} std={y_std:.3f}")

models = []
for seed in SEEDS:
    ckpt_path = MODEL_DIR / f'seed_{seed}' / 'best_model.pt'
    if not ckpt_path.exists():
        print(f"  WARNING: {ckpt_path} not found"); continue
    ckpt       = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    state      = ckpt['model_state']
    hidden_dim = ckpt['config']['hidden_dim']
    model = AffinityModel(
        ligand_node_dim  = ref['ligand_x'].shape[1],
        ligand_edge_dim  = ref['ligand_edge_attr'].shape[1],
        protein_node_dim = ref['protein_x'].shape[1],
        protein_edge_dim = ref['protein_edge_attr'].shape[1],
        cross_edge_dim   = ref['cross_edge_attr'].shape[1],
        hidden_dim       = hidden_dim,
        dropout          = 0.15,
    ).to(DEVICE)
    model.load_state_dict(state)
    model.eval()
    models.append(model)
    print(f"  Loaded seed {seed} (hidden_dim={hidden_dim})")

print(f"Ensemble: {len(models)} models")


# ============================================================
# SCORING
# ============================================================
def score_graph(graph_dict):
    batch = collate_fn([graph_dict])
    batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}
    preds = []
    with torch.no_grad():
        for m in models:
            z    = m(batch)
            pred = float(z.cpu().item()) * y_std + y_mean
            preds.append(pred)
    return float(np.mean(preds)), float(np.std(preds))


# ============================================================
# LOAD WT STRUCTURE
# ============================================================
print("\nLoading WT PETase...")
import prody
prody.confProDy(verbosity='none')
struct     = prody.parsePDB(str(PDB_PATH))
ca         = struct.select('protein and chain A and calpha')
resnums    = ca.getResnums()
resnames   = ca.getResnames()
wt_seq     = list(''.join(AA3TO1.get(r, 'X') for r in resnames))
rn_to_idx  = {int(rn): i for i, rn in enumerate(resnums)}
pro_coords = ca.getCoords()

print("Building BHET ligand graph...")
lig_graph = build_ligand_graph(BHET_SMILES)
print(f"  BHET: {lig_graph['x'].shape[0]} heavy atoms")

print("Scoring WT baseline...")
wt_graph         = build_graph(wt_seq, pro_coords, lig_graph)
wt_score, wt_std = score_graph(wt_graph)
print(f"  WT pAffinity: {wt_score:.4f} +/- {wt_std:.4f}")


# ============================================================
# SCORE ALL CANDIDATES
# ============================================================
def apply_mutations(candidate):
    seq = wt_seq.copy()
    if candidate['n_muts'] == 1:
        idx = rn_to_idx.get(candidate['resnum'])
        if idx is not None:
            seq[idx] = candidate['mut_aa']
    else:
        for rn, aa in zip(candidate['resnum'], candidate['mut_aa']):
            idx = rn_to_idx.get(rn)
            if idx is not None:
                seq[idx] = aa
    return seq


def score_group(candidates_path, group_name):
    with open(candidates_path) as f:
        data = json.load(f)
    candidates = data['candidates']
    print(f"\nScoring {group_name} ({len(candidates)} candidates)...")
    results = []
    for i, cand in enumerate(candidates):
        seq        = apply_mutations(cand)
        graph      = build_graph(seq, pro_coords, lig_graph)
        score, std = score_graph(graph)
        delta      = score - wt_score
        results.append({
            **cand,
            'predicted_pAff': round(score, 4),
            'uncertainty':    round(std,   4),
            'delta_vs_wt':    round(delta, 4),
            'wt_pAff':        round(wt_score, 4),
        })
        print(f"  [{i+1:3d}/{len(candidates)}] {cand['mutation']:25s} "
              f"pAff={score:.4f}  d={delta:+.4f}  unc={std:.4f}")
    results.sort(key=lambda x: x['predicted_pAff'], reverse=True)
    return results


active_results = score_group(CANDIDATE_DIR / 'active_site_candidates.json', 'Active-Site')
allo_results   = score_group(CANDIDATE_DIR / 'allosteric_candidates.json',  'Allosteric')


# ============================================================
# STATISTICS
# ============================================================
active_arr = np.array([r['predicted_pAff'] for r in active_results])
allo_arr   = np.array([r['predicted_pAff'] for r in allo_results])

u_stat, p_val = scipy_stats.mannwhitneyu(allo_arr, active_arr, alternative='greater')
_,      p_two = scipy_stats.mannwhitneyu(allo_arr, active_arr, alternative='two-sided')

np.random.seed(42)
boot = [np.random.choice(allo_arr,   len(allo_arr),   replace=True).mean() -
        np.random.choice(active_arr, len(active_arr), replace=True).mean()
        for _ in range(10000)]
ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])


# ============================================================
# SAVE
# ============================================================
with open(OUT_DIR / 'active_site_scored.json', 'w') as f:
    json.dump({'group': 'active_site', 'wt_pAff': wt_score,
               'n': len(active_results), 'results': active_results}, f, indent=2)
with open(OUT_DIR / 'allosteric_scored.json', 'w') as f:
    json.dump({'group': 'allosteric', 'wt_pAff': wt_score,
               'n': len(allo_results), 'results': allo_results}, f, indent=2)

summary = {
    'wt_pAff':              round(wt_score, 4),
    'active_site_mean':     round(float(active_arr.mean()), 4),
    'active_site_std':      round(float(active_arr.std()),  4),
    'active_site_top1':     active_results[0]['predicted_pAff'],
    'active_site_top1_mut': active_results[0]['mutation'],
    'allosteric_mean':      round(float(allo_arr.mean()), 4),
    'allosteric_std':       round(float(allo_arr.std()),  4),
    'allosteric_top1':      allo_results[0]['predicted_pAff'],
    'allosteric_top1_mut':  allo_results[0]['mutation'],
    'delta_mean':           round(float(allo_arr.mean() - active_arr.mean()), 4),
    'ci_95_lo':             round(float(ci_lo), 4),
    'ci_95_hi':             round(float(ci_hi), 4),
    'mann_whitney_u':       round(float(u_stat), 2),
    'p_value_one_sided':    round(float(p_val),  6),
    'p_value_two_sided':    round(float(p_two),  6),
}
with open(OUT_DIR / 'comparison_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*60}")
print("ALCAS SCORING SUMMARY")
print(f"{'='*60}")
print(f"WT baseline:                   pAff = {wt_score:.4f} +/- {wt_std:.4f}")
print(f"Active-site top-1: {active_results[0]['mutation']:15s}   pAff = {active_results[0]['predicted_pAff']:.4f}")
print(f"Active-site mean:              pAff = {active_arr.mean():.4f} +/- {active_arr.std():.4f}")
print(f"Allosteric top-1:  {allo_results[0]['mutation']:15s}   pAff = {allo_results[0]['predicted_pAff']:.4f}")
print(f"Allosteric mean:               pAff = {allo_arr.mean():.4f} +/- {allo_arr.std():.4f}")
print(f"Mean difference (allo-active):        {allo_arr.mean()-active_arr.mean():+.4f}")
print(f"95% bootstrap CI:                     [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"Mann-Whitney p (one-sided):           {p_val:.4f}")
print(f"Result: {'SIGNIFICANT' if p_val < 0.05 else 'not significant'} at alpha=0.05")
print(f"{'='*60}")
print(f"Saved to {OUT_DIR}/")
print("Next: python src/validate/dock_candidates.py")