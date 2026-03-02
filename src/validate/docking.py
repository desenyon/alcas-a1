"""
ALCAS - AutoDock Vina Docking Pipeline
Docks BHET and MHET against WT PETase and top candidates.
Uses OpenBabel for all PDBQT preparation (no ADFRsuite required).

Usage:
    cd ~/alcas
    python src/validate/docking.py
"""

import json
import subprocess
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats

# ============================================================
# CONFIG
# ============================================================
ESMFOLD_DIR  = Path('/home/ubuntu/alcas/results/esmfold')
FOLDX_AS     = Path('/home/ubuntu/alcas/results/foldx/active_site_foldx.json')
FOLDX_AL     = Path('/home/ubuntu/alcas/results/foldx/allosteric_foldx.json')
OUT_DIR      = Path('/home/ubuntu/alcas/results/docking')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Box center: catalytic triad centroid from ESMFold WT structure
BOX_CENTER     = (-5.249, 2.270, -13.751)
BOX_SIZE       = (25.0, 25.0, 25.0)
EXHAUSTIVENESS = 16
NUM_MODES      = 5

BHET_SMILES  = 'OCCOC(=O)c1ccc(cc1)C(=O)OCCO'
MHET_SMILES  = 'OC(=O)c1ccc(cc1)C(=O)OCCO'
LIGANDS      = {'BHET': BHET_SMILES, 'MHET': MHET_SMILES}

print("=" * 60)
print("ALCAS - AutoDock Vina Docking Pipeline")
print("=" * 60)
print(f"Box center: {BOX_CENTER}")
print(f"Box size:   {BOX_SIZE}")


# ============================================================
# PREPARE LIGANDS
# ============================================================
lig_dir = OUT_DIR / 'ligands'
lig_dir.mkdir(exist_ok=True)

def smiles_to_pdbqt(smiles, name, out_dir):
    smi_file   = out_dir / f'{name}.smi'
    mol2_file  = out_dir / f'{name}.mol2'
    pdbqt_file = out_dir / f'{name}.pdbqt'
    with open(smi_file, 'w') as f:
        f.write(f'{smiles} {name}\n')
    subprocess.run(
        ['obabel', str(smi_file), '-O', str(mol2_file), '--gen3d', '--best', '-h'],
        capture_output=True
    )
    subprocess.run(
        ['obabel', str(mol2_file), '-O', str(pdbqt_file), '-xh'],
        capture_output=True
    )
    if pdbqt_file.exists():
        print(f"  Ligand prepared: {name}")
        return pdbqt_file
    print(f"  WARNING: ligand prep failed for {name}")
    return None

print("\nPreparing ligands...")
lig_pdbqt = {}
for name, smiles in LIGANDS.items():
    p = smiles_to_pdbqt(smiles, name, lig_dir)
    if p:
        lig_pdbqt[name] = p


# ============================================================
# PREPARE RECEPTORS (OpenBabel only, no ADFRsuite)
# ============================================================
rec_dir = OUT_DIR / 'receptors'
rec_dir.mkdir(exist_ok=True)

def prepare_receptor_pdbqt(pdb_path, name, out_dir):
    clean_pdb  = out_dir / f'{name}_clean.pdb'
    pdbqt_file = out_dir / f'{name}_proper.pdbqt'

    # Strip non-ATOM records
    with open(pdb_path) as f:
        lines = [l for l in f if l.startswith('ATOM')]
    with open(clean_pdb, 'w') as f:
        f.writelines(lines)
        f.write('END\n')

    subprocess.run(
        ['obabel', str(clean_pdb), '-O', str(pdbqt_file),
         '-xr', '-h', '--partialcharge', 'gasteiger'],
        capture_output=True, text=True
    )

    if pdbqt_file.exists() and pdbqt_file.stat().st_size > 1000:
        print(f"  Receptor prepared: {name} ({pdbqt_file.stat().st_size} bytes)")
        return pdbqt_file
    print(f"  WARNING: receptor prep failed for {name}")
    return None


with open(FOLDX_AS) as f:
    as_results = sorted(json.load(f)['results'],
                        key=lambda x: x['foldx_ddg'] if x['foldx_ddg'] else 999)[:3]
with open(FOLDX_AL) as f:
    al_results = sorted(json.load(f)['results'],
                        key=lambda x: x['foldx_ddg'] if x['foldx_ddg'] else 999)[:3]

candidates = [('WT', 'wt')]
for r in as_results:
    candidates.append((r['mutation'], 'active_site'))
for r in al_results:
    candidates.append((r['mutation'], 'allosteric'))

print(f"\nPreparing {len(candidates)} receptors...")
rec_pdbqt = {}
for name, group in candidates:
    safe_name = name.replace('+', '_')
    esm_pdb   = ESMFOLD_DIR / f'{safe_name}.pdb'
    if not esm_pdb.exists():
        print(f"  MISSING: {esm_pdb}")
        continue
    pdbqt = prepare_receptor_pdbqt(esm_pdb, safe_name, rec_dir)
    if pdbqt:
        rec_pdbqt[name] = (pdbqt, group)

print(f"Receptors ready: {len(rec_pdbqt)}/{len(candidates)}")


# ============================================================
# RUN VINA DOCKING
# ============================================================
def run_vina(receptor_pdbqt, ligand_pdbqt, out_pdbqt):
    cmd = [
        'vina',
        '--receptor',       str(receptor_pdbqt),
        '--ligand',         str(ligand_pdbqt),
        '--out',            str(out_pdbqt),
        '--center_x',       str(BOX_CENTER[0]),
        '--center_y',       str(BOX_CENTER[1]),
        '--center_z',       str(BOX_CENTER[2]),
        '--size_x',         str(BOX_SIZE[0]),
        '--size_y',         str(BOX_SIZE[1]),
        '--size_z',         str(BOX_SIZE[2]),
        '--exhaustiveness', str(EXHAUSTIVENESS),
        '--num_modes',      str(NUM_MODES),
        '--energy_range',   '3',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    best_score = None
    in_table   = False
    for line in result.stdout.split('\n'):
        if '-----+' in line:
            in_table = True
            continue
        if in_table:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    mode  = int(parts[0])
                    score = float(parts[1])
                    if mode == 1:
                        best_score = score
                        break
                except ValueError:
                    pass

    return best_score


print("\nRunning docking...")
docking_results = []

for rec_name, (rec_pdbqt_path, group) in rec_pdbqt.items():
    for lig_name, lig_pdbqt_path in lig_pdbqt.items():
        safe_rec = rec_name.replace('+', '_')
        out_file = OUT_DIR / f'{safe_rec}_{lig_name}_out.pdbqt'
        score    = run_vina(rec_pdbqt_path, lig_pdbqt_path, out_file)
        score_str = f'{score:.3f}' if score is not None else 'FAILED'
        flag      = ' ***' if (score is not None and score < -6.0) else ''
        print(f"  {rec_name:28s} + {lig_name:5s}  {score_str} kcal/mol{flag}")
        docking_results.append({
            'receptor':   rec_name,
            'group':      group,
            'ligand':     lig_name,
            'vina_score': score,
        })


# ============================================================
# STATISTICS
# ============================================================
bhet     = [r for r in docking_results if r['ligand'] == 'BHET']
wt_row   = next((r for r in bhet if r['receptor'] == 'WT'), None)
wt_score = wt_row['vina_score'] if wt_row else None

as_scores = np.array([r['vina_score'] for r in bhet
                      if r['group'] == 'active_site' and r['vina_score'] is not None])
al_scores = np.array([r['vina_score'] for r in bhet
                      if r['group'] == 'allosteric' and r['vina_score'] is not None])

p_val = None
if len(as_scores) > 1 and len(al_scores) > 1:
    _, p_val = scipy_stats.mannwhitneyu(al_scores, as_scores, alternative='less')

for r in docking_results:
    if wt_score and r['vina_score'] and r['ligand'] == 'BHET':
        r['delta_vs_wt'] = round(r['vina_score'] - wt_score, 3)


# ============================================================
# SAVE
# ============================================================
summary = {
    'box_center':            list(BOX_CENTER),
    'box_size':              list(BOX_SIZE),
    'exhaustiveness':        EXHAUSTIVENESS,
    'n_receptors':           len(rec_pdbqt),
    'n_ligands':             len(lig_pdbqt),
    'wt_bhet_score':         wt_score,
    'active_site_bhet_mean': round(float(as_scores.mean()), 3) if len(as_scores) > 0 else None,
    'allosteric_bhet_mean':  round(float(al_scores.mean()), 3) if len(al_scores) > 0 else None,
    'delta_allo_vs_active':  round(float(al_scores.mean() - as_scores.mean()), 3) if len(as_scores) > 0 and len(al_scores) > 0 else None,
    'p_value_one_sided':     round(float(p_val), 6) if p_val is not None else None,
    'all_results':           docking_results,
}
with open(OUT_DIR / 'docking_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)


# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*60}")
print("DOCKING SUMMARY (BHET — primary substrate)")
print(f"{'='*60}")
if wt_score:
    print(f"WT baseline:             {wt_score:.3f} kcal/mol")
if len(as_scores) > 0:
    print(f"Active-site mean:        {as_scores.mean():.3f} kcal/mol")
if len(al_scores) > 0:
    print(f"Allosteric mean:         {al_scores.mean():.3f} kcal/mol")
if len(as_scores) > 0 and len(al_scores) > 0:
    print(f"Delta (allo - active):   {al_scores.mean()-as_scores.mean():.3f} kcal/mol")
if p_val is not None:
    print(f"Mann-Whitney p:          {p_val:.4f} ({'SIGNIFICANT' if p_val < 0.05 else 'not significant'})")
print(f"{'='*60}")
print("Note: lower Vina score = stronger predicted binding")
print(f"Saved to {OUT_DIR}/")

print("\nAll BHET results (sorted by score):")
for r in sorted(bhet, key=lambda x: x['vina_score'] if x['vina_score'] else 0):
    delta = f"  delta={r.get('delta_vs_wt',0):+.3f}" if 'delta_vs_wt' in r else ''
    print(f"  {r['receptor']:28s} [{r['group']:11s}]  {r['vina_score']:.3f} kcal/mol{delta}")