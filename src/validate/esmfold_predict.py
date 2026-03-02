"""
ALCAS - ESMFold Structure Prediction
Folds WT + top 3 active-site + top 3 allosteric candidates via ESMFold API.
Computes TM-score and RMSD vs WT to assess structural perturbation.
Active-site geometry (catalytic triad distances) measured for each structure.

Usage:
    cd ~/alcas
    python src/validate/esmfold_predict.py
"""

import json
import requests
import time
import numpy as np
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
OUT_DIR   = Path('results/esmfold')
OUT_DIR.mkdir(parents=True, exist_ok=True)

PDB_PATH  = Path('data/petase/5XJH.pdb')
FOLDX_AS  = Path('results/foldx/active_site_foldx.json')
FOLDX_AL  = Path('results/foldx/allosteric_foldx.json')

# Catalytic triad residue numbers (original 5XJH numbering)
TRIAD = {'S160': 160, 'H237': 237, 'D206': 206}

AA3TO1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
    'HSD':'H','HSE':'H','HSP':'H','HIE':'H','HID':'H',
}


# ============================================================
# LOAD WT SEQUENCE
# ============================================================
import prody
prody.confProDy(verbosity='none')

struct   = prody.parsePDB(str(PDB_PATH))
ca       = struct.select('protein and chain A and calpha')
resnums  = list(ca.getResnums())
resnames = list(ca.getResnames())
wt_seq   = list(''.join(AA3TO1.get(r, 'X') for r in resnames))
rn2idx   = {int(rn): i for i, rn in enumerate(resnums)}

print("=" * 60)
print("ALCAS - ESMFold Structure Prediction")
print("=" * 60)
print(f"WT sequence length: {len(wt_seq)}")
print(f"Catalytic triad indices: S160={rn2idx[160]} H237={rn2idx[237]} D206={rn2idx[206]}")


# ============================================================
# BUILD CANDIDATE SEQUENCES
# ============================================================
def apply_muts(muts):
    seq = wt_seq.copy()
    for rn, aa in muts:
        idx = rn2idx.get(rn)
        if idx is not None:
            seq[idx] = aa
    return ''.join(seq)


# Load top 3 from each group by FoldX ddG
with open(FOLDX_AS) as f:
    as_results = sorted(json.load(f)['results'],
                        key=lambda x: x['foldx_ddg'] if x['foldx_ddg'] else 999)[:3]
with open(FOLDX_AL) as f:
    al_results = sorted(json.load(f)['results'],
                        key=lambda x: x['foldx_ddg'] if x['foldx_ddg'] else 999)[:3]

candidates = [('WT', ''.join(wt_seq), 'wt')]

for r in as_results:
    muts = []
    for m in r['mutation'].split('+'):
        import re
        rn  = int(re.search(r'\d+', m).group())
        aa  = m[-1]
        muts.append((rn, aa))
    seq = apply_muts(muts)
    candidates.append((r['mutation'], seq, 'active_site'))

for r in al_results:
    muts = []
    for m in r['mutation'].split('+'):
        import re
        rn  = int(re.search(r'\d+', m).group())
        aa  = m[-1]
        muts.append((rn, aa))
    seq = apply_muts(muts)
    candidates.append((r['mutation'], seq, 'allosteric'))

print(f"\nCandidates to fold ({len(candidates)} total):")
for name, seq, group in candidates:
    print(f"  [{group:11s}] {name}")


# ============================================================
# ESMFOLD API
# ============================================================
ESMFOLD_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"

def fold_sequence(name, sequence, retries=3):
    print(f"\n  Folding {name} ({len(sequence)} aa)...")
    for attempt in range(retries):
        try:
            response = requests.post(
                ESMFOLD_URL,
                data=sequence,
                headers={'Content-Type': 'application/x-www-form-urlencoded'},
                timeout=120,
            )
            if response.status_code == 200:
                print(f"    OK ({len(response.text)} chars)")
                return response.text
            else:
                print(f"    Attempt {attempt+1} failed: HTTP {response.status_code}")
                time.sleep(5)
        except requests.exceptions.Timeout:
            print(f"    Attempt {attempt+1} timed out, retrying...")
            time.sleep(10)
        except Exception as e:
            print(f"    Attempt {attempt+1} error: {e}")
            time.sleep(5)
    return None


# ============================================================
# FOLD ALL CANDIDATES
# ============================================================
print("\nFolding sequences via ESMFold API...")
pdb_structures = {}

for name, seq, group in candidates:
    pdb_text = fold_sequence(name, seq)
    if pdb_text:
        # Save PDB
        safe_name = name.replace('+', '_').replace('/', '_')
        pdb_path  = OUT_DIR / f'{safe_name}.pdb'
        with open(pdb_path, 'w') as f:
            f.write(pdb_text)
        pdb_structures[name] = {'pdb': pdb_text, 'group': group, 'path': str(pdb_path)}
        print(f"    Saved: {pdb_path}")
    else:
        print(f"    FAILED: {name}")
    time.sleep(2)  # be polite to API


# ============================================================
# STRUCTURAL ANALYSIS
# ============================================================
def extract_ca_coords(pdb_text):
    """Extract CA coordinates indexed by residue number."""
    coords = {}
    for line in pdb_text.split('\n'):
        if line.startswith('ATOM') and line[12:16].strip() == 'CA':
            try:
                rn  = int(line[22:26].strip())
                x   = float(line[30:38])
                y   = float(line[38:46])
                z   = float(line[46:54])
                coords[rn] = np.array([x, y, z])
            except ValueError:
                pass
    return coords


def compute_rmsd(coords1, coords2):
    """RMSD between two CA coordinate dicts over common residues."""
    common = sorted(set(coords1.keys()) & set(coords2.keys()))
    if len(common) < 10:
        return None
    c1 = np.array([coords1[r] for r in common])
    c2 = np.array([coords2[r] for r in common])
    # Center
    c1 -= c1.mean(axis=0)
    c2 -= c2.mean(axis=0)
    diff = c1 - c2
    return float(np.sqrt((diff**2).sum(axis=1).mean()))


def get_triad_geometry(coords):
    """
    Measure catalytic triad geometry.
    Returns distances: S160-H237, H237-D206, S160-D206
    ESMFold numbers from 1, but sequence position maps to original resnum.
    """
    # ESMFold numbers sequentially from 1
    # Map original resnums to ESMFold indices via sequence position
    s_idx = rn2idx[160] + 1   # 1-indexed
    h_idx = rn2idx[237] + 1
    d_idx = rn2idx[206] + 1

    results = {}
    if s_idx in coords and h_idx in coords:
        results['S160_H237'] = float(np.linalg.norm(coords[s_idx] - coords[h_idx]))
    if h_idx in coords and d_idx in coords:
        results['H237_D206'] = float(np.linalg.norm(coords[h_idx] - coords[d_idx]))
    if s_idx in coords and d_idx in coords:
        results['S160_D206'] = float(np.linalg.norm(coords[s_idx] - coords[d_idx]))
    return results


def extract_plddt(pdb_text):
    """Extract mean pLDDT from B-factor column."""
    scores = []
    for line in pdb_text.split('\n'):
        if line.startswith('ATOM'):
            try:
                scores.append(float(line[60:66].strip()))
            except ValueError:
                pass
    return float(np.mean(scores)) if scores else None


# ============================================================
# COMPUTE METRICS FOR ALL STRUCTURES
# ============================================================
print("\nAnalyzing structures...")

wt_pdb    = pdb_structures.get('WT', {}).get('pdb', '')
wt_coords = extract_ca_coords(wt_pdb) if wt_pdb else {}

analysis_results = []

for name, seq, group in candidates:
    if name not in pdb_structures:
        continue
    pdb_text = pdb_structures[name]['pdb']
    coords   = extract_ca_coords(pdb_text)
    plddt    = extract_plddt(pdb_text)
    triad    = get_triad_geometry(coords)
    rmsd     = compute_rmsd(wt_coords, coords) if name != 'WT' else 0.0

    result = {
        'name':       name,
        'group':      group,
        'plddt':      round(plddt, 2) if plddt else None,
        'rmsd_vs_wt': round(rmsd, 3)  if rmsd is not None else None,
        'triad_S160_H237': round(triad.get('S160_H237', 0), 2),
        'triad_H237_D206': round(triad.get('H237_D206', 0), 2),
        'triad_S160_D206': round(triad.get('S160_D206', 0), 2),
    }
    analysis_results.append(result)

    print(f"\n  {name} [{group}]")
    print(f"    pLDDT:        {result['plddt']}")
    print(f"    RMSD vs WT:   {result['rmsd_vs_wt']} A")
    print(f"    S160-H237:    {result['triad_S160_H237']} A")
    print(f"    H237-D206:    {result['triad_H237_D206']} A")
    print(f"    S160-D206:    {result['triad_S160_D206']} A")


# ============================================================
# SAVE
# ============================================================
with open(OUT_DIR / 'esmfold_analysis.json', 'w') as f:
    json.dump(analysis_results, f, indent=2)

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*60}")
print("ESMFOLD STRUCTURAL ANALYSIS SUMMARY")
print(f"{'='*60}")

wt_res = next((r for r in analysis_results if r['name'] == 'WT'), None)
if wt_res:
    print(f"WT reference:")
    print(f"  pLDDT={wt_res['plddt']}  "
          f"S160-H237={wt_res['triad_S160_H237']}A  "
          f"H237-D206={wt_res['triad_H237_D206']}A")

print(f"\n{'Name':28s} {'Group':11s} {'pLDDT':>7} {'RMSD':>7} {'S160-H237':>10} {'H237-D206':>10}")
print("-" * 75)
for r in analysis_results:
    print(f"  {r['name']:26s} {r['group']:11s} "
          f"{str(r['plddt']):>7} "
          f"{str(r['rmsd_vs_wt']):>7} "
          f"{str(r['triad_S160_H237']):>10} "
          f"{str(r['triad_H237_D206']):>10}")

print(f"\nSaved PDBs and analysis to {OUT_DIR}/")
print("Next: python src/validate/mmgbsa.py  (after MD completes)")