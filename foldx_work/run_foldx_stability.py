"""
ALCAS - FoldX Stability Analysis
Computes folding ddG for all active-site and allosteric candidates.
ddG < 0 = mutation stabilizes the fold vs WT.
ddG > 0 = mutation destabilizes the fold vs WT.

This is separate from the earlier ddG interaction run.
Here we measure thermodynamic stability of the fold itself,
not binding affinity to BHET.

Usage:
    cd ~/alcas/foldx_work
    python run_foldx_stability.py
"""

import json
import subprocess
import numpy as np
import os
import re
from pathlib import Path
from scipy import stats as scipy_stats

# ============================================================
# CONFIG
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
print("ALCAS - FoldX Stability Analysis")
print("=" * 60)
print(f"Active-site candidates: {len(active_cands)}")
print(f"Allosteric candidates:  {len(allo_cands)}")


# ============================================================
# FORMAT MUTATIONS
# ============================================================
def format_mutation(cand):
    if cand['n_muts'] == 1:
        return f"{cand['wt_aa']}A{cand['resnum']}{cand['mut_aa']};"
    parts = []
    for m in cand['mutation'].split('+'):
        wt_aa  = m[0]
        mut_aa = m[-1]
        resnum = int(re.search(r'\d+', m).group())
        parts.append(f"{wt_aa}A{resnum}{mut_aa};")
    return ''.join(parts)


# ============================================================
# RUN FOLDX BuildModel (stability mode)
# ============================================================
def run_stability(mutation_str, label):
    with open(WORK_DIR / 'individual_list.txt', 'w') as f:
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
        '--output-file=stab_out',
    ]
    subprocess.run(cmd, capture_output=True, text=True, cwd=str(WORK_DIR))

    # Find output file
    dif_file = WORK_DIR / 'Dif_stab_out.fxout'
    if not dif_file.exists():
        for fp in WORK_DIR.glob('Dif_*.fxout'):
            dif_file = fp
            break

    ddg = None
    if dif_file.exists():
        stem = REPAIRED_PDB.replace('.pdb', '')
        with open(dif_file) as f:
            for line in f:
                if line.startswith(stem):
                    parts = line.strip().split('\t')
                    if len(parts) > 1:
                        try:
                            ddg = float(parts[1])
                        except ValueError:
                            pass

    # Cleanup
    for pattern in ['Dif_*.fxout', 'Raw_*.fxout', 'PdbList_*.fxout',
                    'Average_*.fxout', 'individual_list.txt']:
        for fp in WORK_DIR.glob(pattern):
            try:
                fp.unlink()
            except Exception:
                pass

    return ddg


# ============================================================
# SCORE ALL CANDIDATES
# ============================================================
def score_group(candidates, group_name):
    print(f"\nScoring {group_name} ({len(candidates)} candidates)...")
    results  = []
    failures = 0

    for i, cand in enumerate(candidates):
        mut_str = format_mutation(cand)
        ddg     = run_stability(mut_str, cand['mutation'])

        if ddg is None:
            failures += 1

        ddg_str = f'{ddg:+.3f}' if ddg is not None else 'FAILED'
        flag    = ' ***' if (ddg is not None and ddg < -1.0) else ''
        print(f"  [{i+1:3d}/{len(candidates)}] {cand['mutation']:28s} "
              f"mut={mut_str:20s} ddG={ddg_str} kcal/mol{flag}")

        results.append({
            **cand,
            'stability_ddg': round(ddg, 4) if ddg is not None else None,
            'mutation_str':  mut_str,
        })

    results.sort(key=lambda x: x['stability_ddg'] if x['stability_ddg'] is not None else 999)
    print(f"  Failures: {failures}/{len(candidates)}")
    return results


active_results = score_group(active_cands, 'Active-Site')
allo_results   = score_group(allo_cands,   'Allosteric')


# ============================================================
# STATISTICS
# ============================================================
active_arr = np.array([r['stability_ddg'] for r in active_results
                       if r['stability_ddg'] is not None], dtype=float)
allo_arr   = np.array([r['stability_ddg'] for r in allo_results
                       if r['stability_ddg'] is not None], dtype=float)

# Lower ddG = more stable = better
# One-sided: allosteric < active-site
u_stat, p_val = scipy_stats.mannwhitneyu(allo_arr, active_arr, alternative='less')
_,      p_two = scipy_stats.mannwhitneyu(allo_arr, active_arr, alternative='two-sided')

np.random.seed(42)
boot = [
    np.random.choice(allo_arr,   len(allo_arr),   replace=True).mean() -
    np.random.choice(active_arr, len(active_arr), replace=True).mean()
    for _ in range(10000)
]
ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

top_active = next(r for r in active_results if r['stability_ddg'] is not None)
top_allo   = next(r for r in allo_results   if r['stability_ddg'] is not None)


# ============================================================
# SAVE
# ============================================================
with open(OUT_DIR / 'stability_active_site.json', 'w') as f:
    json.dump({'group': 'active_site', 'n': len(active_results),
               'results': active_results}, f, indent=2)

with open(OUT_DIR / 'stability_allosteric.json', 'w') as f:
    json.dump({'group': 'allosteric', 'n': len(allo_results),
               'results': allo_results}, f, indent=2)

summary = {
    'active_site_n':           len(active_arr),
    'active_site_mean_ddg':    round(float(active_arr.mean()), 4),
    'active_site_std_ddg':     round(float(active_arr.std()),  4),
    'active_site_top1_mut':    top_active['mutation'],
    'active_site_top1_ddg':    top_active['stability_ddg'],
    'allosteric_n':            len(allo_arr),
    'allosteric_mean_ddg':     round(float(allo_arr.mean()), 4),
    'allosteric_std_ddg':      round(float(allo_arr.std()),  4),
    'allosteric_top1_mut':     top_allo['mutation'],
    'allosteric_top1_ddg':     top_allo['stability_ddg'],
    'delta_mean_ddg':          round(float(allo_arr.mean() - active_arr.mean()), 4),
    'ci_95_lo':                round(float(ci_lo), 4),
    'ci_95_hi':                round(float(ci_hi), 4),
    'mann_whitney_u':          round(float(u_stat), 2),
    'p_value_one_sided':       round(float(p_val),  6),
    'p_value_two_sided':       round(float(p_two),  6),
}
with open(OUT_DIR / 'stability_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)


# ============================================================
# PRINT SUMMARY
# ============================================================
print(f"\n{'='*60}")
print("FOLDX STABILITY SUMMARY")
print(f"{'='*60}")
print(f"Active-site top-1:  {top_active['mutation']:22s}  ddG = {top_active['stability_ddg']:+.3f} kcal/mol")
print(f"Active-site mean:   ddG = {active_arr.mean():+.4f} +/- {active_arr.std():.4f} kcal/mol")
print(f"Allosteric top-1:   {top_allo['mutation']:22s}  ddG = {top_allo['stability_ddg']:+.3f} kcal/mol")
print(f"Allosteric mean:    ddG = {allo_arr.mean():+.4f} +/- {allo_arr.std():.4f} kcal/mol")
print(f"Delta (allo-active):      {allo_arr.mean()-active_arr.mean():+.4f} kcal/mol")
print(f"95% bootstrap CI:         [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"Mann-Whitney p (one-sided, allo<active): {p_val:.4f}")
print(f"Result: {'SIGNIFICANT' if p_val < 0.05 else 'not significant'} at alpha=0.05")
print(f"{'='*60}")
print(f"Saved to {OUT_DIR}/")