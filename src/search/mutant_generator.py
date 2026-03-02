"""
ALCAS - Mutant Generator
Proposes single and double mutants for both active-site and allosteric masks.
Uses ESM-2 log-likelihoods to score mutation plausibility.
Outputs ranked candidate tables for both groups.
"""

import json
import numpy as np
from pathlib import Path
import torch
import esm

# ============================================================
# CONFIG
# ============================================================
MASK_DIR     = Path('data/petase')
OUT_DIR      = Path('results/candidates')
OUT_DIR.mkdir(parents=True, exist_ok=True)

PDB_PATH     = Path('data/petase/5XJH.pdb')

# Mutation budget: matched between groups
N_SINGLE     = 30   # top 30 single mutants per group
N_DOUBLE     = 20   # top 20 double mutants per group

# Amino acids to consider (exclude C to avoid disulfide disruption)
AA_LIST = list('ADEFGHIKLMNPQRSTVWY')

# Catalytic triad - never mutate
CATALYTIC    = {160, 206, 237}

# ============================================================
# LOAD MASKS
# ============================================================
print("="*60)
print("ALCAS - Mutant Generator")
print("="*60)

with open(MASK_DIR / 'mask_active_site.json') as f:
    active_data = json.load(f)
active_resnums = [r for r in active_data['residues'] if r not in CATALYTIC]

with open(MASK_DIR / 'masks_allosteric.json') as f:
    allo_data = json.load(f)
allo_resnums = [r for r in allo_data['residues'] if r not in CATALYTIC]

print(f"Active-site mask:  {len(active_resnums)} mutable residues")
print(f"Allosteric mask:   {len(allo_resnums)} mutable residues")

# ============================================================
# LOAD WT PETASE SEQUENCE
# ============================================================
print("\nExtracting WT PETase sequence...")
import prody
prody.confProDy(verbosity='none')
struct  = prody.parsePDB(str(PDB_PATH))
protein = struct.select('protein and chain A and calpha')
resnum  = protein.getResnums()
resname = protein.getResnames()

# 3-letter to 1-letter
AA3TO1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
    'HSD':'H','HSE':'H','HSP':'H','HIE':'H','HID':'H',
}

wt_seq    = ''.join(AA3TO1.get(r, 'X') for r in resname)
rn_to_idx = {int(rn): i for i, rn in enumerate(resnum)}
rn_to_wt  = {int(rn): AA3TO1.get(res, 'X')
              for rn, res in zip(resnum, resname)}

print(f"WT sequence length: {len(wt_seq)}")
print(f"First 20 AA: {wt_seq[:20]}")

# ============================================================
# LOAD ESM-2 FOR MUTATION SCORING
# ============================================================
print("\nLoading ESM-2 (150M)...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
model = model.to(device).eval()
batch_converter = alphabet.get_batch_converter()
print(f"ESM-2 loaded on {device}")


def score_mutations(seq, positions, wt_map):
    """
    Score all single mutations at given positions using ESM-2
    masked marginal log-likelihoods.
    Returns dict: {(resnum, mut_aa): score}
    """
    scores = {}
    data   = [("wt", seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    with torch.no_grad():
        # Get logits for each masked position
        for pos_rn in positions:
            idx = rn_to_idx.get(pos_rn)
            if idx is None:
                continue
            wt_aa = wt_map.get(pos_rn, 'X')
            if wt_aa == 'X':
                continue

            # Mask this position
            tok_masked         = tokens.clone()
            tok_masked[0, idx+1] = alphabet.mask_idx  # +1 for BOS token

            logits = model(tok_masked, repr_layers=[])['logits']
            log_probs = torch.log_softmax(logits[0, idx+1], dim=-1)

            for aa in AA_LIST:
                if aa == wt_aa:
                    continue
                tok_id = alphabet.get_idx(aa)
                score  = float(log_probs[tok_id].cpu())
                scores[(pos_rn, aa)] = score

    return scores


# ============================================================
# SCORE SINGLE MUTANTS
# ============================================================
print("\nScoring single mutants...")
print(f"  Active-site: {len(active_resnums)} positions x {len(AA_LIST)-1} mutations...")
active_scores = score_mutations(wt_seq, active_resnums, rn_to_wt)
print(f"  Allosteric:  {len(allo_resnums)} positions x {len(AA_LIST)-1} mutations...")
allo_scores   = score_mutations(wt_seq, allo_resnums, rn_to_wt)

print(f"  Active-site scored: {len(active_scores)} mutations")
print(f"  Allosteric scored:  {len(allo_scores)} mutations")


def top_singles(scores, wt_map, n):
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for (rn, aa), score in ranked[:n*3]:   # oversample then filter
        wt = wt_map.get(rn, 'X')
        results.append({
            'mutation':   f'{wt}{rn}{aa}',
            'resnum':     int(rn),
            'wt_aa':      wt,
            'mut_aa':     aa,
            'esm_score':  float(round(score, 4)),
            'n_muts':     1,
        })
        if len(results) >= n:
            break
    return results


# ============================================================
# SCORE DOUBLE MUTANTS (top singles x top singles, same group)
# ============================================================
def top_doubles(scores, single_list, wt_map, seq, n):
    """Combine top singles into doubles, score additively."""
    top = single_list[:15]   # top 15 singles as base
    doubles = []
    seen = set()
    for i, s1 in enumerate(top):
        for j, s2 in enumerate(top):
            if j <= i:
                continue
            if s1['resnum'] == s2['resnum']:
                continue
            key = (min(s1['resnum'], s2['resnum']), max(s1['resnum'], s2['resnum']))
            if key in seen:
                continue
            seen.add(key)
            combined_score = s1['esm_score'] + s2['esm_score']
            doubles.append({
                'mutation':   f"{s1['mutation']}+{s2['mutation']}",
                'resnum':     [s1['resnum'], s2['resnum']],
                'mut_aa':     [s1['mut_aa'], s2['mut_aa']],
                'esm_score':  float(round(combined_score, 4)),
                'n_muts':     2,
            })
    doubles.sort(key=lambda x: x['esm_score'], reverse=True)
    return doubles[:n]


print("\nRanking top candidates...")
active_singles = top_singles(active_scores, rn_to_wt, N_SINGLE)
allo_singles   = top_singles(allo_scores,   rn_to_wt, N_SINGLE)
active_doubles = top_doubles(active_scores, active_singles, rn_to_wt, wt_seq, N_DOUBLE)
allo_doubles   = top_doubles(allo_scores,   allo_singles,   rn_to_wt, wt_seq, N_DOUBLE)

active_candidates = active_singles + active_doubles
allo_candidates   = allo_singles   + allo_doubles

print(f"  Active-site candidates: {len(active_candidates)} "
      f"({len(active_singles)} single + {len(active_doubles)} double)")
print(f"  Allosteric candidates:  {len(allo_candidates)} "
      f"({len(allo_singles)} single + {len(allo_doubles)} double)")

# ============================================================
# SAVE
# ============================================================
with open(OUT_DIR / 'active_site_candidates.json', 'w') as f:
    json.dump({
        'group':      'active_site',
        'n_total':    len(active_candidates),
        'candidates': active_candidates,
    }, f, indent=2)

with open(OUT_DIR / 'allosteric_candidates.json', 'w') as f:
    json.dump({
        'group':      'allosteric',
        'n_total':    len(allo_candidates),
        'candidates': allo_candidates,
    }, f, indent=2)

# Print top 10 each
print("\n--- Top 10 Active-Site Candidates ---")
for c in active_singles[:10]:
    print(f"  {c['mutation']:10s}  ESM={c['esm_score']:.4f}")

print("\n--- Top 10 Allosteric Candidates ---")
for c in allo_singles[:10]:
    print(f"  {c['mutation']:10s}  ESM={c['esm_score']:.4f}")

print(f"\nSaved to {OUT_DIR}/")
print("Next: python src/search/alcas_score.py")
