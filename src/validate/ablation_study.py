"""
ALCAS - Ablation Study
Compares coupling-filtered allosteric mask vs distance-only allosteric mask.

This validates the NMA coupling step: if coupling-filtered candidates
outperform distance-only candidates, the allostery identification adds value
beyond simple geometric distance filtering.

Pipeline:
1. Generate distance-only allosteric mask (>=16A, no coupling filter)
2. Propose mutations via ESM-2
3. Score with FoldX ddG
4. Compare distributions: coupling-filtered vs distance-only vs active-site

Usage:
    cd ~/alcas
    python src/validate/ablation_study.py
"""

import json
import re
import subprocess
import numpy as np
import torch
import prody
import esm
from pathlib import Path
from scipy import stats as scipy_stats

# ============================================================
# CONFIG
# ============================================================
PDB_PATH      = Path('data/petase/5XJH.pdb')
MASK_ACTIVE   = Path('data/petase/mask_active_site.json')
MASK_ALLO_CF  = Path('data/petase/masks_allosteric.json')          # coupling-filtered
FOLDX         = Path('/home/ubuntu/foldx5/foldx_20270131')
FOLDX_DIR     = Path('/home/ubuntu/alcas/foldx_work')
REPAIRED_PDB  = '5XJH_Repair.pdb'
OUT_DIR       = Path('/home/ubuntu/alcas/results/ablation')
OUT_DIR.mkdir(parents=True, exist_ok=True)

CATALYTIC     = {160, 206, 237}
AA_LIST       = list('ADEFGHIKLMNPQRSTVWY')   # no C
N_SINGLE      = 20   # matched budget per group
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'

AA3TO1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
    'HSD':'H','HSE':'H','HSP':'H',
}

print("=" * 60)
print("ALCAS - Ablation Study")
print("Coupling-filtered vs Distance-only allosteric mask")
print("=" * 60)


# ============================================================
# LOAD STRUCTURE
# ============================================================
prody.confProDy(verbosity='none')
struct   = prody.parsePDB(str(PDB_PATH))
ca       = struct.select('protein and chain A and calpha')
resnums  = list(ca.getResnums())
resnames = list(ca.getResnames())
coords   = ca.getCoords()
wt_seq   = list(''.join(AA3TO1.get(r, 'X') for r in resnames))
rn2idx   = {int(rn): i for i, rn in enumerate(resnums)}
rn2wt    = {int(rn): AA3TO1.get(res, 'X') for rn, res in zip(resnums, resnames)}

# Triad center for distance calculations
triad_center = np.mean([
    coords[rn2idx[160]],
    coords[rn2idx[206]],
    coords[rn2idx[237]],
], axis=0)


# ============================================================
# BUILD DISTANCE-ONLY ALLOSTERIC MASK
# ============================================================
with open(MASK_ACTIVE) as f:
    active_resnums = set(json.load(f)['residues'])
with open(MASK_ALLO_CF) as f:
    coupling_filtered_resnums = set(json.load(f)['residues'])

distance_only = []
for rn, coord in zip(resnums, coords):
    rn_int = int(rn)
    if rn_int in active_resnums or rn_int in CATALYTIC:
        continue
    dist = float(np.linalg.norm(coord - triad_center))
    if dist >= 16.0:
        distance_only.append(rn_int)

distance_only_set = set(distance_only)

print(f"Active-site mask:          {len(active_resnums)} residues")
print(f"Coupling-filtered mask:    {len(coupling_filtered_resnums)} residues")
print(f"Distance-only mask:        {len(distance_only)} residues")
print(f"In distance-only but NOT coupling-filtered (low coupling, excluded by NMA): "
      f"{len(distance_only_set - coupling_filtered_resnums)}")
print(f"In coupling-filtered only: {len(coupling_filtered_resnums - distance_only_set)}")

# Save distance-only mask
with open('data/petase/mask_allosteric_distance_only.json', 'w') as f:
    json.dump({
        'residues':       distance_only,
        'n':              len(distance_only),
        'method':         'distance_only_geq16A_from_triad_center',
        'coupling_filter': False,
    }, f, indent=2)
print("Saved mask_allosteric_distance_only.json")


# ============================================================
# ESM-2 MUTATION SCORING
# ============================================================
print(f"\nLoading ESM-2 on {DEVICE}...")
model_esm, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
model_esm = model_esm.to(DEVICE).eval()
batch_converter = alphabet.get_batch_converter()


def score_mutations(positions):
    """Score all single mutations at given positions via ESM-2 masked marginals."""
    scores = {}
    data   = [("wt", ''.join(wt_seq))]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(DEVICE)

    with torch.no_grad():
        for pos_rn in positions:
            idx    = rn2idx.get(pos_rn)
            wt_aa  = rn2wt.get(pos_rn, 'X')
            if idx is None or wt_aa == 'X':
                continue
            tok_masked         = tokens.clone()
            tok_masked[0, idx+1] = alphabet.mask_idx
            logits   = model_esm(tok_masked, repr_layers=[])['logits']
            log_prob = torch.log_softmax(logits[0, idx+1], dim=-1)
            for aa in AA_LIST:
                if aa == wt_aa:
                    continue
                tok_id = alphabet.get_idx(aa)
                scores[(pos_rn, aa)] = float(log_prob[tok_id].cpu())
    return scores


def top_singles(scores, n):
    ranked  = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for (rn, aa), score in ranked:
        wt = rn2wt.get(rn, 'X')
        results.append({
            'mutation':  f'{wt}{rn}{aa}',
            'resnum':    int(rn),
            'wt_aa':     wt,
            'mut_aa':    aa,
            'esm_score': round(score, 4),
            'n_muts':    1,
        })
        if len(results) >= n:
            break
    return results


print("Scoring distance-only allosteric positions...")
do_mutable = [rn for rn in distance_only if rn not in CATALYTIC]
do_scores  = score_mutations(do_mutable)
do_singles = top_singles(do_scores, N_SINGLE)

print(f"  Distance-only candidates: {len(do_singles)}")
print("  Top 5:")
for c in do_singles[:5]:
    print(f"    {c['mutation']:10s}  ESM={c['esm_score']:.4f}")


# ============================================================
# FOLDX ddG SCORING
# ============================================================
import os
os.chdir(FOLDX_DIR)


def format_mutation(cand):
    return f"{cand['wt_aa']}A{cand['resnum']}{cand['mut_aa']};"


def run_foldx_ddg(mutation_str):
    with open(FOLDX_DIR / 'individual_list.txt', 'w') as f:
        f.write(mutation_str + '\n')
    cmd = [
        str(FOLDX), '--command=BuildModel',
        f'--pdb={REPAIRED_PDB}',
        '--mutant-file=individual_list.txt',
        '--numberOfRuns=3',
        '--temperature=298', '--ionStrength=0.05', '--pH=7',
        '--out-pdb=false', '--output-file=abl_out',
    ]
    subprocess.run(cmd, capture_output=True, cwd=str(FOLDX_DIR))

    dif_file = FOLDX_DIR / 'Dif_abl_out.fxout'
    if not dif_file.exists():
        for fp in FOLDX_DIR.glob('Dif_*.fxout'):
            dif_file = fp; break

    ddg = None
    if dif_file.exists():
        stem = REPAIRED_PDB.replace('.pdb', '')
        for line in open(dif_file):
            if line.startswith(stem):
                parts = line.strip().split('\t')
                if len(parts) > 1:
                    try: ddg = float(parts[1])
                    except: pass

    for pat in ['Dif_*.fxout','Raw_*.fxout','PdbList_*.fxout',
                'Average_*.fxout','individual_list.txt']:
        for fp in FOLDX_DIR.glob(pat):
            try: fp.unlink()
            except: pass

    return ddg


def score_group_foldx(candidates, group_name):
    print(f"\nFoldX scoring: {group_name} ({len(candidates)} candidates)...")
    results = []
    for i, cand in enumerate(candidates):
        mut_str = format_mutation(cand)
        ddg     = run_foldx_ddg(mut_str)
        ddg_str = f'{ddg:+.3f}' if ddg is not None else 'FAILED'
        flag    = ' ***' if (ddg is not None and ddg < -0.5) else ''
        print(f"  [{i+1:3d}/{len(candidates)}] {cand['mutation']:15s}  ddG={ddg_str}{flag}")
        results.append({**cand, 'foldx_ddg': round(ddg, 4) if ddg is not None else None})
    results.sort(key=lambda x: x['foldx_ddg'] if x['foldx_ddg'] is not None else 999)
    return results


do_results = score_group_foldx(do_singles, 'Distance-Only Allosteric')

# Load existing coupling-filtered and active-site results for comparison
with open('/home/ubuntu/alcas/results/foldx/active_site_foldx.json') as f:
    as_results = json.load(f)['results']
with open('/home/ubuntu/alcas/results/foldx/allosteric_foldx.json') as f:
    cf_results = json.load(f)['results']

# Take matched N_SINGLE from each for fair comparison
as_arr = np.array([r['foldx_ddg'] for r in sorted(
    as_results, key=lambda x: x['foldx_ddg'] if x['foldx_ddg'] else 999
)[:N_SINGLE] if r['foldx_ddg'] is not None])

cf_arr = np.array([r['foldx_ddg'] for r in sorted(
    cf_results, key=lambda x: x['foldx_ddg'] if x['foldx_ddg'] else 999
)[:N_SINGLE] if r['foldx_ddg'] is not None])

do_arr = np.array([r['foldx_ddg'] for r in do_results if r['foldx_ddg'] is not None])


# ============================================================
# STATISTICS
# ============================================================
# Key ablation test: coupling-filtered < distance-only (CF more stabilizing)
u_cf_do, p_cf_do = scipy_stats.mannwhitneyu(cf_arr, do_arr, alternative='less')
u_cf_as, p_cf_as = scipy_stats.mannwhitneyu(cf_arr, as_arr, alternative='less')
u_do_as, p_do_as = scipy_stats.mannwhitneyu(do_arr, as_arr, alternative='less')

np.random.seed(42)
boot_cf_do = [
    np.random.choice(cf_arr, len(cf_arr), replace=True).mean() -
    np.random.choice(do_arr, len(do_arr), replace=True).mean()
    for _ in range(10000)
]
ci_lo, ci_hi = np.percentile(boot_cf_do, [2.5, 97.5])


# ============================================================
# SAVE
# ============================================================
with open(OUT_DIR / 'distance_only_results.json', 'w') as f:
    json.dump({'group': 'distance_only', 'n': len(do_results),
               'results': do_results}, f, indent=2)

summary = {
    'active_site_mean_ddg':       round(float(as_arr.mean()), 4),
    'active_site_std_ddg':        round(float(as_arr.std()),  4),
    'distance_only_mean_ddg':     round(float(do_arr.mean()), 4),
    'distance_only_std_ddg':      round(float(do_arr.std()),  4),
    'coupling_filtered_mean_ddg': round(float(cf_arr.mean()), 4),
    'coupling_filtered_std_ddg':  round(float(cf_arr.std()),  4),
    'delta_cf_vs_do':             round(float(cf_arr.mean() - do_arr.mean()), 4),
    'ci_95_lo':                   round(float(ci_lo), 4),
    'ci_95_hi':                   round(float(ci_hi), 4),
    'p_cf_vs_do':                 round(float(p_cf_do), 6),
    'p_cf_vs_as':                 round(float(p_cf_as), 6),
    'p_do_vs_as':                 round(float(p_do_as), 6),
}
with open(OUT_DIR / 'ablation_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)


# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*60}")
print("ABLATION STUDY SUMMARY")
print(f"{'='*60}")
print(f"Active-site          mean ddG = {as_arr.mean():+.4f} +/- {as_arr.std():.4f}")
print(f"Distance-only allo   mean ddG = {do_arr.mean():+.4f} +/- {do_arr.std():.4f}")
print(f"Coupling-filtered    mean ddG = {cf_arr.mean():+.4f} +/- {cf_arr.std():.4f}")
print(f"")
print(f"Coupling-filtered vs Distance-only:")
print(f"  Delta:   {cf_arr.mean()-do_arr.mean():+.4f} kcal/mol")
print(f"  95% CI:  [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"  p-value: {p_cf_do:.4f}  ({'SIGNIFICANT' if p_cf_do < 0.05 else 'not significant'})")
print(f"")
print(f"Coupling-filtered vs Active-site:")
print(f"  p-value: {p_cf_as:.4f}  ({'SIGNIFICANT' if p_cf_as < 0.05 else 'not significant'})")
print(f"")
print(f"Distance-only vs Active-site:")
print(f"  p-value: {p_do_as:.4f}  ({'SIGNIFICANT' if p_do_as < 0.05 else 'not significant'})")
print(f"{'='*60}")
print(f"Saved to {OUT_DIR}/")
print()
print("Interpretation:")
print("  If coupling-filtered < distance-only: NMA coupling adds value beyond distance")
print("  If distance-only < active-site: distance alone is sufficient baseline")
print("  If coupling-filtered < active-site: full ALCAS pipeline is validated")