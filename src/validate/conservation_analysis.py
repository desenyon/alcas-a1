"""
ALCAS - Sequence Conservation Analysis
Aligns PETase homologs, computes per-residue conservation scores,
then compares conservation between active-site and allosteric mask residues.

Hypothesis: allosteric residues are less conserved than active-site residues,
explaining why they tolerate mutation better and validating ALCAS mask design.

Usage:
    cd ~/alcas
    python src/validate/conservation_analysis.py
"""

import json
import numpy as np
import subprocess
import tempfile
from pathlib import Path
from scipy import stats as scipy_stats
from collections import Counter

# ============================================================
# CONFIG
# ============================================================
HOMOLOGS_FASTA = Path('data/petase/petase_homologs.fasta')
MASK_ACTIVE    = Path('data/petase/mask_active_site.json')
MASK_ALLO      = Path('data/petase/masks_allosteric.json')
PDB_PATH       = Path('data/petase/5XJH.pdb')
OUT_DIR        = Path('results/conservation')
OUT_DIR.mkdir(parents=True, exist_ok=True)

AA3TO1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
    'HSD':'H','HSE':'H','HSP':'H',
}

print("=" * 60)
print("ALCAS - Sequence Conservation Analysis")
print("=" * 60)


# ============================================================
# LOAD WT SEQUENCE AND MASKS
# ============================================================
import prody
prody.confProDy(verbosity='none')

struct   = prody.parsePDB(str(PDB_PATH))
ca       = struct.select('protein and chain A and calpha')
resnums  = list(ca.getResnums())
resnames = list(ca.getResnames())
wt_seq   = ''.join(AA3TO1.get(r, 'X') for r in resnames)
rn2idx   = {int(rn): i for i, rn in enumerate(resnums)}

with open(MASK_ACTIVE) as f:
    active_resnums = set(json.load(f)['residues'])
with open(MASK_ALLO) as f:
    allo_resnums = set(json.load(f)['residues'])

print(f"WT sequence length: {len(wt_seq)}")
print(f"Active-site mask:   {len(active_resnums)} residues")
print(f"Allosteric mask:    {len(allo_resnums)} residues")


# ============================================================
# ADD WT TO HOMOLOGS FASTA AND BUILD MSA
# ============================================================
# Prepend WT sequence to homologs file
wt_fasta = f'>WT_5XJH_PETase\n{wt_seq}\n'
with open(HOMOLOGS_FASTA) as f:
    homolog_text = f.read()

# Remove any sequences that are too short or too long
lines      = homolog_text.split('\n')
sequences  = {}
current_id = None
current_seq = []
for line in lines:
    if line.startswith('>'):
        if current_id and current_seq:
            seq = ''.join(current_seq)
            if 150 <= len(seq) <= 600:   # reasonable PETase length
                sequences[current_id] = seq
        current_id  = line[1:].split()[0]
        current_seq = []
    else:
        current_seq.append(line.strip())
if current_id and current_seq:
    seq = ''.join(current_seq)
    if 150 <= len(seq) <= 600:
        sequences[current_id] = seq

print(f"Homologs after length filter: {len(sequences)}")

# Write combined FASTA
combined_fasta = OUT_DIR / 'combined.fasta'
with open(combined_fasta, 'w') as f:
    f.write(wt_fasta)
    for sid, seq in sequences.items():
        f.write(f'>{sid}\n{seq}\n')

total_seqs = 1 + len(sequences)
print(f"Total sequences for MSA: {total_seqs}")


# ============================================================
# RUN MUSCLE MSA
# ============================================================
aligned_fasta = OUT_DIR / 'aligned.fasta'
print("\nRunning MUSCLE alignment...")

# Try muscle v5 syntax first, fall back to v3
result = subprocess.run(
    ['muscle', '-align', str(combined_fasta), '-output', str(aligned_fasta)],
    capture_output=True, text=True
)
if result.returncode != 0:
    # muscle v3 syntax
    result = subprocess.run(
        ['muscle', '-in', str(combined_fasta), '-out', str(aligned_fasta)],
        capture_output=True, text=True
    )

if result.returncode != 0 or not aligned_fasta.exists():
    print(f"MUSCLE failed: {result.stderr[:200]}")
    print("Falling back to pairwise alignment with Biopython...")
    # Use Biopython pairwise aligner as fallback
    from Bio import pairwise2
    from Bio.pairwise2 import format_alignment

    # Simple approach: align each homolog to WT
    aligned_seqs = {'WT_5XJH_PETase': wt_seq}
    for sid, seq in sequences.items():
        alns = pairwise2.align.globalms(wt_seq, seq, 2, -1, -10, -0.5)
        if alns:
            aligned_seqs[sid] = alns[0].seqB
        else:
            aligned_seqs[sid] = seq

    with open(aligned_fasta, 'w') as f:
        for sid, seq in aligned_seqs.items():
            f.write(f'>{sid}\n{seq}\n')
    print(f"Pairwise alignment complete: {len(aligned_seqs)} sequences")
else:
    print(f"MUSCLE alignment complete")


# ============================================================
# PARSE MSA
# ============================================================
def parse_fasta(path):
    seqs = {}
    current_id = None
    current_seq = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    seqs[current_id] = ''.join(current_seq)
                current_id  = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
    if current_id:
        seqs[current_id] = ''.join(current_seq)
    return seqs

msa = parse_fasta(aligned_fasta)
print(f"Parsed MSA: {len(msa)} sequences, alignment length={len(next(iter(msa.values())))}")


# ============================================================
# FIND WT ROW AND MAP ALIGNMENT POSITIONS TO RESNUMS
# ============================================================
wt_key = 'WT_5XJH_PETase'
if wt_key not in msa:
    wt_key = next(k for k in msa if '5XJH' in k or 'WT' in k)

wt_aligned = msa[wt_key]
aln_len    = len(wt_aligned)

# Map alignment column -> original residue index (0-based)
col2residx = {}
res_idx = 0
for col, aa in enumerate(wt_aligned):
    if aa != '-':
        col2residx[col] = res_idx
        res_idx += 1

print(f"WT residues mapped: {len(col2residx)}/{len(wt_seq)}")


# ============================================================
# COMPUTE PER-COLUMN CONSERVATION SCORE
# Using Shannon entropy: H = -sum(p_i * log2(p_i))
# Conservation = 1 - H/log2(20)
# ============================================================
AA_ALPHABET = set('ACDEFGHIKLMNPQRSTVWY')
MAX_ENTROPY = np.log2(20)

def column_conservation(col_idx, msa_dict):
    """Compute conservation score for one alignment column."""
    residues = []
    for seq in msa_dict.values():
        if col_idx < len(seq):
            aa = seq[col_idx].upper()
            if aa in AA_ALPHABET:
                residues.append(aa)
    if len(residues) < 2:
        return None
    counts = Counter(residues)
    total  = sum(counts.values())
    probs  = [c / total for c in counts.values()]
    entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    return 1.0 - entropy / MAX_ENTROPY   # 1=fully conserved, 0=random


print("\nComputing per-residue conservation scores...")
conservation = {}   # resnum -> conservation score

for col, res_idx in col2residx.items():
    if res_idx < len(resnums):
        resnum = resnums[res_idx]
        score  = column_conservation(col, msa)
        if score is not None:
            conservation[int(resnum)] = round(score, 4)

print(f"Conservation scores computed for {len(conservation)} residues")
print(f"Range: [{min(conservation.values()):.3f}, {max(conservation.values()):.3f}]")


# ============================================================
# COMPARE ACTIVE-SITE VS ALLOSTERIC CONSERVATION
# ============================================================
active_scores = [conservation[rn] for rn in active_resnums if rn in conservation]
allo_scores   = [conservation[rn] for rn in allo_resnums   if rn in conservation]

active_arr = np.array(active_scores)
allo_arr   = np.array(allo_scores)

# Test: active-site more conserved than allosteric (active > allo)
u_stat, p_val = scipy_stats.mannwhitneyu(active_arr, allo_arr, alternative='greater')
_, p_two      = scipy_stats.mannwhitneyu(active_arr, allo_arr, alternative='two-sided')

np.random.seed(42)
boot = [
    np.random.choice(active_arr, len(active_arr), replace=True).mean() -
    np.random.choice(allo_arr,   len(allo_arr),   replace=True).mean()
    for _ in range(10000)
]
ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])


# ============================================================
# FAST-PETASE MUTATION CONSERVATION
# ============================================================
FAST_PETASE_MUTS = {
    'S121E': 121, 'D186H': 186, 'R224Q': 224, 'N233K': 233, 'R280A': 280
}
print("\nFAST-PETase mutation site conservation:")
for mut, rn in FAST_PETASE_MUTS.items():
    score = conservation.get(rn, None)
    print(f"  {mut}: resnum={rn}  conservation={score}")


# ============================================================
# TOP AND BOTTOM CONSERVED RESIDUES
# ============================================================
sorted_cons = sorted(conservation.items(), key=lambda x: x[1], reverse=True)
print("\nTop 10 most conserved residues:")
for rn, score in sorted_cons[:10]:
    group = 'active' if rn in active_resnums else ('allosteric' if rn in allo_resnums else 'other')
    print(f"  Res {rn:4d}: {score:.3f}  [{group}]")

print("\nTop 10 least conserved residues:")
for rn, score in sorted_cons[-10:]:
    group = 'active' if rn in active_resnums else ('allosteric' if rn in allo_resnums else 'other')
    print(f"  Res {rn:4d}: {score:.3f}  [{group}]")


# ============================================================
# SAVE
# ============================================================
output = {
    'n_sequences':          total_seqs,
    'n_residues_scored':    len(conservation),
    'active_site_n':        len(active_scores),
    'active_site_mean':     round(float(active_arr.mean()), 4),
    'active_site_std':      round(float(active_arr.std()),  4),
    'allosteric_n':         len(allo_scores),
    'allosteric_mean':      round(float(allo_arr.mean()),   4),
    'allosteric_std':       round(float(allo_arr.std()),    4),
    'delta_conservation':   round(float(active_arr.mean() - allo_arr.mean()), 4),
    'ci_95_lo':             round(float(ci_lo), 4),
    'ci_95_hi':             round(float(ci_hi), 4),
    'mann_whitney_u':       round(float(u_stat), 2),
    'p_value_one_sided':    round(float(p_val),  6),
    'p_value_two_sided':    round(float(p_two),  6),
    'per_residue_conservation': conservation,
    'fast_petase_conservation': {
        mut: conservation.get(rn) for mut, rn in FAST_PETASE_MUTS.items()
    },
}
with open(OUT_DIR / 'conservation_analysis.json', 'w') as f:
    json.dump(output, f, indent=2)


# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*60}")
print("CONSERVATION ANALYSIS SUMMARY")
print(f"{'='*60}")
print(f"Sequences in MSA:         {total_seqs}")
print(f"Active-site mean conservation: {active_arr.mean():.4f} +/- {active_arr.std():.4f}")
print(f"Allosteric mean conservation:  {allo_arr.mean():.4f} +/- {allo_arr.std():.4f}")
print(f"Delta (active - allo):         {active_arr.mean()-allo_arr.mean():+.4f}")
print(f"95% bootstrap CI:              [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"Mann-Whitney p (active>allo):  {p_val:.4f}")
print(f"Result: {'SIGNIFICANT' if p_val < 0.05 else 'not significant'} at alpha=0.05")
print(f"{'='*60}")
print(f"Saved to {OUT_DIR}/conservation_analysis.json")
print("Interpretation: if active-site > allosteric conservation, allosteric")
print("residues are less evolutionarily constrained and tolerate mutation better.")