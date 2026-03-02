"""
ALCAS - FoldX ddG Scoring
Scores all active-site and allosteric mutants using FoldX BuildModel.
ddG = dG(mutant) - dG(WT). Negative = stabilizing, positive = destabilizing.

Usage:
    cd ~/alcas/foldx_work
    python run_foldx_ddg.py
"""

import json
import subprocess
import numpy as np
import os
import re
from pathlib import Path
from scipy import stats as scipy_stats

# ============================================================
# CONFIG - adjust paths if needed
# ============================================================
FOLDX         = Path('/home/ubuntu/foldx5/foldx_20270131')
WORK_DIR      = Path('/home/ubuntu/alcas/foldx_work')
REPAIRED_PDB  = '5XJH_Repair.pdb'
CANDIDATE_DIR = Path('/home/ubuntu/alcas/results/candidates')
OUT_DIR       = Path('/home/ubuntu/alcas/results/foldx')
OUT_DIR.mkdir(parents=True, exist_ok=True)

os.chdir(WORK_DIR)


# ============================================================
# LOAD CANDIDATES
# ============================================================
with open(CANDIDATE_DIR / 'active_site_candidates.json') as f:
    active_cands = json.load(f)['candidates']
with open(CANDIDATE_DIR / 'allosteric_candidates.json') as f:
    allo_cands = json.load(f)['candidates']

print("=" * 60)
print("ALCAS - FoldX ddG Scoring Pipeline")
print("=" * 60)
print(f"Active-site candidates: {len(active_cands)}")
print(f"Allosteric candidates:  {len(allo_cands)}")
print(f"FoldX binary:          {FOLDX}")
print(f"Repaired PDB:          {REPAIRED_PDB}")
print(f"Output dir:            {OUT_DIR}")
print()


# ============================================================
# FORMAT MUTATIONS FOR FOLDX individual_list.txt
# FoldX format: {WT_AA}{Chain}{ResNum}{MUT_AA};
# e.g. WA159H; means Trp->His at chain A residue 159
# Multiple mutations on same line = double mutant
# ============================================================
def format_mutation(cand):
    if cand['n_muts'] == 1:
        return f"{cand['wt_aa']}A{cand['resnum']}{cand['mut_aa']};"
    else:
        # Parse WT residues from mutation string e.g. "W159H+C203G"
        parts = []
        muts  = cand['mutation'].split('+')
        for m in muts:
            wt_aa  = m[0]
            mut_aa = m[-1]
            resnum = int(re.search(r'\d+', m).group())
            parts.append(f"{wt_aa}A{resnum}{mut_aa};")
        return ''.join(parts)


# ============================================================
# RUN FOLDX BuildModel FOR ONE MUTATION
# Returns ddG in kcal/mol, or None on failure
# ============================================================
def run_foldx_ddg(mutation_str, label):
    # Write individual_list.txt
    ind_file = WORK_DIR / 'individual_list.txt'
    with open(ind_file, 'w') as f:
        f.write(mutation_str + '\n')

    cmd = [
        str(FOLDX), '--command=BuildModel',
        f'--pdb={REPAIRED_PDB}',
        '--mutant-file=individual_list.txt',
        '--numberOfRuns=3',
        '--temperature=298',
        '--ionStrength=0.05',
        '--pH=7',
        '--out-pdb=false',
        '--output-file=foldx_out',
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(WORK_DIR)
    )

    # Find Dif output file
    dif_file = WORK_DIR / 'Dif_foldx_out.fxout'
    if not dif_file.exists():
        for f in WORK_DIR.glob('Dif_*.fxout'):
            dif_file = f
            break

    if not dif_file.exists():
        print(f"    WARNING: No Dif file for {label}")
        print(f"    stderr: {result.stderr[-300:]}")
        _cleanup()
        return None

    # Parse ddG values (one per run, take mean)
    pdb_stem  = REPAIRED_PDB.replace('.pdb', '')
    ddg_values = []
    with open(dif_file) as f:
        for line in f:
            if line.startswith(pdb_stem):
                parts = line.strip().split('\t')
                if len(parts) > 1:
                    try:
                        ddg_values.append(float(parts[1]))
                    except ValueError:
                        pass

    _cleanup()

    if ddg_values:
        return float(np.mean(ddg_values))

    print(f"    WARNING: Could not parse ddG for {label}")
    print(f"    Dif file contents: {open(dif_file).read()[:300] if dif_file.exists() else 'missing'}")
    return None


def _cleanup():
    for pattern in ['Dif_*.fxout', 'Raw_*.fxout', 'PdbList_*.fxout',
                    'Average_*.fxout', 'individual_list.txt',
                    'WT_foldx_out_*.pdb', 'foldx_out_*.pdb']:
        for fp in WORK_DIR.glob(pattern):
            try:
                fp.unlink()
            except Exception:
                pass


# ============================================================
# SCORE ONE GROUP
# ============================================================
def score_group(candidates, group_name):
    print(f"Scoring {group_name} ({len(candidates)} candidates)...")
    results  = []
    failures = 0

    for i, cand in enumerate(candidates):
        mut_str = format_mutation(cand)
        ddg     = run_foldx_ddg(mut_str, cand['mutation'])

        if ddg is None:
            ddg = float('nan')
            failures += 1

        flag = ' ***' if (not np.isnan(ddg) and ddg < -0.5) else ''
        print(f"  [{i+1:3d}/{len(candidates)}] {cand['mutation']:28s} "
              f"mut={mut_str:20s} ddG={ddg:+.3f} kcal/mol{flag}")

        results.append({
            **cand,
            'foldx_ddg':    round(ddg, 4) if not np.isnan(ddg) else None,
            'mutation_str': mut_str,
        })

    # Sort by ddG ascending (most stabilizing first), NaN last
    results.sort(key=lambda x: x['foldx_ddg'] if x['foldx_ddg'] is not None else 999)
    print(f"  Failures: {failures}/{len(candidates)}")
    return results


# ============================================================
# RUN SCORING
# ============================================================
active_results = score_group(active_cands, 'Active-Site')
print()
allo_results   = score_group(allo_cands,   'Allosteric')

# ============================================================
# STATISTICS
# ============================================================
# Filter out failed runs
active_ddg = np.array([r['foldx_ddg'] for r in active_results
                        if r['foldx_ddg'] is not None], dtype=float)
allo_ddg   = np.array([r['foldx_ddg'] for r in allo_results
                        if r['foldx_ddg'] is not None], dtype=float)

# Lower ddG = more stabilizing = better
# One-sided test: allosteric < active-site (allosteric more stabilizing)
u_stat, p_val = scipy_stats.mannwhitneyu(allo_ddg, active_ddg, alternative='less')
_,      p_two = scipy_stats.mannwhitneyu(allo_ddg, active_ddg, alternative='two-sided')

np.random.seed(42)
boot = [
    np.random.choice(allo_ddg,   len(allo_ddg),   replace=True).mean() -
    np.random.choice(active_ddg, len(active_ddg), replace=True).mean()
    for _ in range(10000)
]
ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

# ============================================================
# SAVE
# ============================================================
with open(OUT_DIR / 'active_site_foldx.json', 'w') as f:
    json.dump({'group': 'active_site', 'n': len(active_results),
               'results': active_results}, f, indent=2)

with open(OUT_DIR / 'allosteric_foldx.json', 'w') as f:
    json.dump({'group': 'allosteric', 'n': len(allo_results),
               'results': allo_results}, f, indent=2)

top_active = next(r for r in active_results if r['foldx_ddg'] is not None)
top_allo   = next(r for r in allo_results   if r['foldx_ddg'] is not None)

summary = {
    'active_site_n':        len(active_ddg),
    'active_site_mean_ddg': round(float(active_ddg.mean()), 4),
    'active_site_std_ddg':  round(float(active_ddg.std()),  4),
    'active_site_top1_mut': top_active['mutation'],
    'active_site_top1_ddg': top_active['foldx_ddg'],
    'allosteric_n':         len(allo_ddg),
    'allosteric_mean_ddg':  round(float(allo_ddg.mean()), 4),
    'allosteric_std_ddg':   round(float(allo_ddg.std()),  4),
    'allosteric_top1_mut':  top_allo['mutation'],
    'allosteric_top1_ddg':  top_allo['foldx_ddg'],
    'delta_mean_ddg':       round(float(allo_ddg.mean() - active_ddg.mean()), 4),
    'ci_95_lo':             round(float(ci_lo), 4),
    'ci_95_hi':             round(float(ci_hi), 4),
    'mann_whitney_u':       round(float(u_stat), 2),
    'p_value_one_sided':    round(float(p_val),  6),
    'p_value_two_sided':    round(float(p_two),  6),
}
with open(OUT_DIR / 'foldx_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

# ============================================================
# PRINT SUMMARY
# ============================================================
print(f"\n{'='*60}")
print("FOLDX ddG SUMMARY")
print(f"{'='*60}")
print(f"Active-site top-1:  {top_active['mutation']:22s}  ddG = {top_active['foldx_ddg']:+.3f} kcal/mol")
print(f"Active-site mean:   ddG = {active_ddg.mean():+.4f} +/- {active_ddg.std():.4f} kcal/mol")
print(f"Allosteric top-1:   {top_allo['mutation']:22s}  ddG = {top_allo['foldx_ddg']:+.3f} kcal/mol")
print(f"Allosteric mean:    ddG = {allo_ddg.mean():+.4f} +/- {allo_ddg.std():.4f} kcal/mol")
print(f"Delta (allo-active):      {allo_ddg.mean()-active_ddg.mean():+.4f} kcal/mol")
print(f"95% bootstrap CI:         [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"Mann-Whitney p (one-sided, allo<active): {p_val:.4f}")
print(f"Result: {'SIGNIFICANT' if p_val < 0.05 else 'not significant'} at alpha=0.05")
print(f"{'='*60}")
print(f"Saved to {OUT_DIR}/")