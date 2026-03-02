"""
ALCAS — Stage C: Composite Scoring + final_candidate_table.csv
Run from ~/alcas:  python src/validate/composite_scoring.py

Composite rank score per candidate:
    composite = w_ddg   * norm(FoldX_ddG,   lower=better)
              + w_stab  * norm(FoldX_stab,  lower=better)
              + w_rmsd  * norm(RMSD_vs_WT,  lower=better)

Weights: ddg=0.50, stab=0.30, rmsd=0.20
(ddg is the pre-registered primary endpoint; heaviest weight)

Normalization: min-max over all 100 candidates per metric.
Lower raw value = better = lower normalized score.
Final composite: lower = better (same direction as ddG).

Outputs:
    results/composite/final_candidate_table.csv
    results/composite/composite_summary.json
    results/composite/top_candidates.txt   (human-readable summary)
"""

import json
import csv
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path('.')
OUT  = ROOT / 'results' / 'composite'
OUT.mkdir(parents=True, exist_ok=True)

# ── load raw results ─────────────────────────────────────────────────────────
print("Loading data...")

with open(ROOT / 'results/foldx/active_site_foldx.json')      as f: as_fx  = json.load(f)['results']
with open(ROOT / 'results/foldx/allosteric_foldx.json')        as f: al_fx  = json.load(f)['results']
with open(ROOT / 'results/foldx/stability_active_site.json')   as f: as_st  = json.load(f)['results']
with open(ROOT / 'results/foldx/stability_allosteric.json')    as f: al_st  = json.load(f)['results']
with open(ROOT / 'results/esmfold/esmfold_analysis.json')      as f: esm_d  = json.load(f)
with open(ROOT / 'results/foldx/foldx_summary.json')           as f: fx_s   = json.load(f)
with open(ROOT / 'results/foldx/stability_summary.json')       as f: st_s   = json.load(f)
with open(ROOT / 'results/ablation/ablation_summary.json')     as f: ab_s   = json.load(f)

# Build lookup dicts
stab_as_m  = {r['mutation']: r['stability_ddg'] for r in as_st}
stab_al_m  = {r['mutation']: r['stability_ddg'] for r in al_st}
esm_rmsd_m = {d['name']: d['rmsd_vs_wt'] for d in esm_d}

# ESM-2 score if present in foldx results
esm_score_m = {}
for r in as_fx + al_fx:
    if 'esm_score' in r and r['esm_score'] is not None:
        esm_score_m[r['mutation']] = r['esm_score']

print(f"  Active-site candidates:  {len(as_fx)}")
print(f"  Allosteric candidates:   {len(al_fx)}")
print(f"  ESMFold structures:      {len(esm_d)}")
print(f"  ESM-2 scores available:  {len(esm_score_m)}")

# ── build unified candidate table ────────────────────────────────────────────
candidates = []

for r in as_fx:
    mut = r['mutation']
    s   = mut.replace('+', '_')
    candidates.append({
        'mutation':    mut,
        'group':       'active_site',
        'foldx_ddg':   r['foldx_ddg'],
        'foldx_stab':  stab_as_m.get(mut),
        'esmfold_rmsd':esm_rmsd_m.get(mut) or esm_rmsd_m.get(s),
        'esm_score':   esm_score_m.get(mut),
    })

for r in al_fx:
    mut = r['mutation']
    s   = mut.replace('+', '_')
    candidates.append({
        'mutation':    mut,
        'group':       'allosteric',
        'foldx_ddg':   r['foldx_ddg'],
        'foldx_stab':  stab_al_m.get(mut),
        'esmfold_rmsd':esm_rmsd_m.get(mut) or esm_rmsd_m.get(s),
        'esm_score':   esm_score_m.get(mut),
    })

print(f"\nTotal candidates: {len(candidates)}")

# ── min-max normalization (lower raw = lower norm = better) ──────────────────
def minmax_norm(values):
    """Normalize list to [0,1]. None stays None. Lower = better preserved."""
    valid = [v for v in values if v is not None]
    if not valid or max(valid) == min(valid):
        return [0.0 if v is not None else None for v in values]
    lo, hi = min(valid), max(valid)
    return [((v - lo) / (hi - lo)) if v is not None else None for v in values]

ddg_raw  = [c['foldx_ddg']    for c in candidates]
stab_raw = [c['foldx_stab']   for c in candidates]
rmsd_raw = [c['esmfold_rmsd'] for c in candidates]

ddg_norm  = minmax_norm(ddg_raw)
stab_norm = minmax_norm(stab_raw)
rmsd_norm = minmax_norm(rmsd_raw)

# Weights
W_DDG  = 0.50
W_STAB = 0.30
W_RMSD = 0.20

for i, c in enumerate(candidates):
    d  = ddg_norm[i]
    s  = stab_norm[i]
    r  = rmsd_norm[i]

    # Composite uses available metrics; re-weight proportionally if some missing
    parts = [(d, W_DDG), (s, W_STAB), (r, W_RMSD)]
    avail = [(v, w) for v, w in parts if v is not None]
    if avail:
        total_w = sum(w for _, w in avail)
        composite = sum(v * w for v, w in avail) / total_w
    else:
        composite = None

    c['ddg_norm']   = round(d,  4) if d  is not None else None
    c['stab_norm']  = round(s,  4) if s  is not None else None
    c['rmsd_norm']  = round(r,  4) if r  is not None else None
    c['composite']  = round(composite, 4) if composite is not None else None

# ── rank all candidates by composite (lower = better) ────────────────────────
ranked = sorted(
    [c for c in candidates if c['composite'] is not None],
    key=lambda x: x['composite']
)
# Assign ranks
for rank_i, c in enumerate(ranked, 1):
    c['composite_rank'] = rank_i
# Candidates without composite get no rank
for c in candidates:
    if 'composite_rank' not in c:
        c['composite_rank'] = None

# ── correlation analysis ─────────────────────────────────────────────────────
print("\n── Internal Consistency ────────────────────────────────────────")

# 1. FoldX ddG vs FoldX Stability
ddg_stab_pairs = [(c['foldx_ddg'], c['foldx_stab'])
                   for c in candidates
                   if c['foldx_ddg'] is not None and c['foldx_stab'] is not None]
if len(ddg_stab_pairs) > 3:
    x1, y1 = zip(*ddg_stab_pairs)
    r1, p1 = scipy_stats.pearsonr(x1, y1)
    print(f"FoldX ddG vs FoldX Stability:  r = {r1:.3f},  p = {p1:.4f}")
    ddg_stab_r, ddg_stab_p = r1, p1
else:
    print("FoldX ddG vs FoldX Stability: insufficient data")
    ddg_stab_r, ddg_stab_p = None, None

# 2. ESM-2 score vs FoldX ddG
esm_ddg_pairs = [(c['esm_score'], c['foldx_ddg'])
                  for c in candidates
                  if c['esm_score'] is not None and c['foldx_ddg'] is not None]
if len(esm_ddg_pairs) > 3:
    x2, y2 = zip(*esm_ddg_pairs)
    r2, p2 = scipy_stats.pearsonr(x2, y2)
    print(f"ESM-2 score   vs FoldX ddG:    r = {r2:.3f},  p = {p2:.4f}")
    esm_ddg_r, esm_ddg_p = r2, p2
else:
    print("ESM-2 score vs FoldX ddG: not available in FoldX output")
    esm_ddg_r, esm_ddg_p = None, None

# ── top candidate summaries ───────────────────────────────────────────────────
print("\n── Top 5 Allosteric Candidates (composite rank) ────────────────")
top_allo = [c for c in ranked if c['group'] == 'allosteric'][:5]
for c in top_allo:
    ddg  = f"{c['foldx_ddg']:+.3f}"  if c['foldx_ddg']    is not None else "N/A"
    stab = f"{c['foldx_stab']:+.3f}" if c['foldx_stab']   is not None else "N/A"
    rmsd = f"{c['esmfold_rmsd']:.3f}" if c['esmfold_rmsd'] is not None else "N/A"
    print(f"  #{c['composite_rank']:3d}  {c['mutation']:<20s}  "
          f"ddG={ddg}  stab={stab}  RMSD={rmsd}  "
          f"composite={c['composite']:.4f}")

print("\n── Top 5 Active-Site Candidates (composite rank) ───────────────")
top_as = [c for c in ranked if c['group'] == 'active_site'][:5]
for c in top_as:
    ddg  = f"{c['foldx_ddg']:+.3f}"  if c['foldx_ddg']    is not None else "N/A"
    stab = f"{c['foldx_stab']:+.3f}" if c['foldx_stab']   is not None else "N/A"
    rmsd = f"{c['esmfold_rmsd']:.3f}" if c['esmfold_rmsd'] is not None else "N/A"
    print(f"  #{c['composite_rank']:3d}  {c['mutation']:<20s}  "
          f"ddG={ddg}  stab={stab}  RMSD={rmsd}  "
          f"composite={c['composite']:.4f}")

print("\n── Overall top-ranked allosteric candidate ─────────────────────")
best_allo = top_allo[0] if top_allo else None
if best_allo:
    print(f"  {best_allo['mutation']}  (rank #{best_allo['composite_rank']} overall)")
    print(f"  Recommended for experimental follow-up.")

# ── Bonferroni-corrected multi-endpoint summary ───────────────────────────────
print("\n── Multiple Comparison Correction ──────────────────────────────")
endpoints = [
    ('FoldX ddG (binding)',     fx_s['p_value_one_sided']),
    ('FoldX stability',         st_s['p_value_one_sided']),
    ('Ablation: CF vs DO',      ab_s['p_cf_vs_do']),
    ('Ablation: ALCAS vs AS',   ab_s['p_cf_vs_as']),
]
n_tests      = len(endpoints)
bonf_alpha   = 0.05 / n_tests
print(f"  n tests = {n_tests},  Bonferroni alpha = {bonf_alpha:.4f}")
bonf_results = []
for name, p in endpoints:
    survives = p < bonf_alpha
    bonf_results.append({'endpoint': name, 'p': p, 'bonferroni': survives})
    marker = '✓ SURVIVES' if survives else '✗ does not survive'
    print(f"  {name:<35s}  p={p:.4f}  {marker}")

# BH correction
p_vals_sorted = sorted(enumerate([e[1] for e in endpoints]), key=lambda x: x[1])
bh_results = {}
for rank_j, (orig_i, p) in enumerate(p_vals_sorted, 1):
    bh_threshold = (rank_j / n_tests) * 0.05
    bh_results[orig_i] = p <= bh_threshold
print(f"\n  Benjamini-Hochberg (FDR) correction:")
for j, (name, p) in enumerate(endpoints):
    marker = '✓ SURVIVES' if bh_results[j] else '✗ does not survive'
    print(f"  {name:<35s}  p={p:.4f}  {marker}")

# ── write CSV ─────────────────────────────────────────────────────────────────
csv_path = OUT / 'final_candidate_table.csv'
fieldnames = [
    'composite_rank', 'mutation', 'group',
    'foldx_ddg', 'foldx_stab', 'esmfold_rmsd', 'esm_score',
    'ddg_norm', 'stab_norm', 'rmsd_norm', 'composite',
]

def fmt(v):
    if v is None: return ''
    if isinstance(v, float): return f'{v:.4f}'
    return str(v)

rows_sorted = sorted(candidates,
                     key=lambda x: (x['composite_rank'] or 9999, x['mutation']))

with open(csv_path, 'w', newline='') as csvf:
    writer = csv.DictWriter(csvf, fieldnames=fieldnames)
    writer.writeheader()
    for c in rows_sorted:
        writer.writerow({k: fmt(c.get(k)) for k in fieldnames})

print(f"\n✓ CSV written: {csv_path}  ({len(rows_sorted)} rows)")

# ── write JSON summary ────────────────────────────────────────────────────────
summary = {
    'n_candidates':          len(candidates),
    'n_allosteric':          sum(1 for c in candidates if c['group'] == 'allosteric'),
    'n_active_site':         sum(1 for c in candidates if c['group'] == 'active_site'),
    'weights': {
        'foldx_ddg':  W_DDG,
        'foldx_stab': W_STAB,
        'esmfold_rmsd': W_RMSD,
    },
    'top_allosteric': [
        {k: c[k] for k in ['mutation','composite_rank','foldx_ddg',
                             'foldx_stab','esmfold_rmsd','composite']}
        for c in top_allo
    ],
    'top_active_site': [
        {k: c[k] for k in ['mutation','composite_rank','foldx_ddg',
                             'foldx_stab','esmfold_rmsd','composite']}
        for c in top_as
    ],
    'best_allosteric_recommendation': best_allo['mutation'] if best_allo else None,
    'correlation': {
        'foldx_ddg_vs_stability': {'r': ddg_stab_r, 'p': ddg_stab_p},
        'esm2_vs_foldx_ddg':      {'r': esm_ddg_r,  'p': esm_ddg_p},
    },
    'bonferroni_alpha': bonf_alpha,
    'bonferroni_results': bonf_results,
}

json_path = OUT / 'composite_summary.json'
with open(json_path, 'w') as jf:
    json.dump(summary, jf, indent=2, default=str)

print(f"✓ JSON written: {json_path}")

# ── write human-readable summary ─────────────────────────────────────────────
txt_path = OUT / 'top_candidates.txt'
with open(txt_path, 'w') as tf:
    tf.write("ALCAS — Top Candidates for Experimental Follow-Up\n")
    tf.write("=" * 55 + "\n\n")
    tf.write("Composite score = 0.50×norm(FoldX_ddG) + 0.30×norm(Stability) + 0.20×norm(RMSD)\n")
    tf.write("Lower composite = better (all metrics: lower raw value = better)\n\n")

    tf.write("TOP 5 ALLOSTERIC CANDIDATES\n")
    tf.write("-" * 45 + "\n")
    for c in top_allo:
        tf.write(f"#{c['composite_rank']:3d}  {c['mutation']}\n")
        tf.write(f"      FoldX ddG:   {c['foldx_ddg']:+.3f} kcal/mol\n"
                 if c['foldx_ddg'] is not None else "      FoldX ddG:   N/A\n")
        tf.write(f"      Stability:   {c['foldx_stab']:+.3f} kcal/mol\n"
                 if c['foldx_stab'] is not None else "      Stability:   N/A\n")
        tf.write(f"      RMSD vs WT:  {c['esmfold_rmsd']:.3f} Å\n"
                 if c['esmfold_rmsd'] is not None else "      RMSD vs WT:  N/A\n")
        tf.write(f"      Composite:   {c['composite']:.4f}\n\n")

    tf.write("TOP 5 ACTIVE-SITE CANDIDATES\n")
    tf.write("-" * 45 + "\n")
    for c in top_as:
        tf.write(f"#{c['composite_rank']:3d}  {c['mutation']}\n")
        tf.write(f"      FoldX ddG:   {c['foldx_ddg']:+.3f} kcal/mol\n"
                 if c['foldx_ddg'] is not None else "      FoldX ddG:   N/A\n")
        tf.write(f"      Stability:   {c['foldx_stab']:+.3f} kcal/mol\n"
                 if c['foldx_stab'] is not None else "      Stability:   N/A\n")
        tf.write(f"      RMSD vs WT:  {c['esmfold_rmsd']:.3f} Å\n"
                 if c['esmfold_rmsd'] is not None else "      RMSD vs WT:  N/A\n")
        tf.write(f"      Composite:   {c['composite']:.4f}\n\n")

    tf.write("MULTIPLE COMPARISON CORRECTION\n")
    tf.write("-" * 45 + "\n")
    tf.write(f"Bonferroni alpha = 0.05 / {n_tests} = {bonf_alpha:.4f}\n\n")
    for br in bonf_results:
        mark = "SURVIVES" if br['bonferroni'] else "does not survive"
        tf.write(f"  {br['endpoint']:<35s}  p={br['p']:.4f}  {mark}\n")

print(f"✓ Text summary: {txt_path}")
print(f"\n{'='*55}")
print("COMPOSITE SCORING COMPLETE")
print(f"{'='*55}")
print(f"  Output directory: {OUT}")
print(f"  Files: final_candidate_table.csv, composite_summary.json, top_candidates.txt")
print(f"\n  Best allosteric candidate: {best_allo['mutation'] if best_allo else 'N/A'}")
print(f"  Add to poster as Stage C deliverable.")