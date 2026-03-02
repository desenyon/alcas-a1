#!/usr/bin/env python3
"""
ALCAS - Holo MD + MM/GBSA v2
Uses RDKit + AMBER14 + GAFF2 via openmmforcefields (no openff.toolkit).
Runs 2ns OpenMM simulation for G79P+M262L, Q182I, WT with BHET.

Place at: ~/alcas/src/validate/holo_md_mmgbsa.py
Run from: ~/alcas/
Command:  python src/validate/holo_md_mmgbsa.py
"""
import json, warnings, sys, time
import numpy as np
from pathlib import Path
warnings.filterwarnings("ignore")

import openmm as mm
import openmm.app as app
import openmm.unit as unit
from pdbfixer import PDBFixer
from openmmforcefields.generators import SystemGenerator
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
import MDAnalysis as mda
from MDAnalysis.transformations import center_in_box
from MDAnalysis.lib.distances import distance_array

BASE     = Path.home() / "alcas"
ESM_DIR  = BASE / "results/esmfold"
DOCK_DIR = BASE / "results/docking"
OUT_DIR  = BASE / "results/md"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATES = [
    {"name": "G79P_M262L", "group": "allosteric"},
    {"name": "Q182I",       "group": "active_site"},
    {"name": "WT",          "group": "wt"},
]

TEMPERATURE      = 298.15 * unit.kelvin
PRESSURE         = 1.0    * unit.bar
TIMESTEP         = 2.0    * unit.femtoseconds
EQUIL_STEPS      = 50_000    # 100 ps
PROD_STEPS       = 1_000_000 # 2 ns
REPORT_EVERY     = 2_000     # frame every 4 ps -> 500 frames
BHET_SMILES      = "OCCOC(=O)c1ccc(cc1)C(=O)OCCO"

print("=" * 60)
print("ALCAS — Holo MD + MM/GBSA v2 (RDKit backend)")
print("=" * 60)


def extract_best_pose(pdbqt_path):
    """Extract MODEL 1 atom lines and Vina score from pdbqt."""
    lines = Path(pdbqt_path).read_text().split("\n")
    score, atoms, in_m1 = None, [], False
    for line in lines:
        if line.strip() == "MODEL 1":
            in_m1 = True
        elif line.startswith("ENDMDL") and in_m1:
            break
        elif in_m1:
            if "VINA RESULT" in line:
                try: score = float(line.split()[3])
                except: pass
            if line.startswith(("ATOM","HETATM")):
                atoms.append(line)
    return atoms, score


def pose_to_sdf(atom_lines, smiles, out_path):
    """Convert pdbqt atom lines to SDF using RDKit for correct bond orders."""
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol)

    # Replace conformer coords with docking pose coords
    if len(atom_lines) > 0:
        conf = mol.GetConformer()
        heavy = [a for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
        heavy_pos = []
        for line in atom_lines:
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                heavy_pos.append((x, y, z))
            except: pass
        # Map heavy atoms to docked positions
        for i, atom in enumerate(heavy):
            if i < len(heavy_pos):
                conf.SetAtomPosition(atom.GetIdx(), heavy_pos[i])

    writer = Chem.SDWriter(str(out_path))
    writer.write(mol)
    writer.close()
    return out_path


def fix_protein(pdb_in, pdb_out):
    fixer = PDBFixer(filename=str(pdb_in))
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(True)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    with open(pdb_out, "w") as f:
        app.PDBFile.writeFile(fixer.topology, fixer.positions, f)


def run_candidate(cand):
    name  = cand["name"]
    group = cand["group"]
    print(f"\n{'='*60}")
    print(f"  {name}  ({group})")
    print(f"{'='*60}")

    work_dir = OUT_DIR / name
    work_dir.mkdir(exist_ok=True)

    # 1. Extract docking pose
    pose_atoms, vina_score = extract_best_pose(DOCK_DIR / f"{name}_BHET_out.pdbqt")
    print(f"  Vina best pose: {vina_score} kcal/mol  |  {len(pose_atoms)} heavy atoms")

    # 2. Build ligand SDF from RDKit
    sdf_path = work_dir / "BHET.sdf"
    pose_to_sdf(pose_atoms, BHET_SMILES, sdf_path)
    print(f"  Ligand SDF: {sdf_path}")

    # 3. Fix protein
    fixed_pdb = work_dir / f"{name}_fixed.pdb"
    print(f"  Fixing protein...")
    fix_protein(ESM_DIR / f"{name}.pdb", fixed_pdb)

    # 4. Load protein
    pdb = app.PDBFile(str(fixed_pdb))

    # 5. Load ligand via RDKit -> OpenMM
    rdmol = Chem.SDMolSupplier(str(sdf_path), removeHs=False)[0]
    if rdmol is None:
        raise ValueError(f"Could not load ligand SDF for {name}")

    # 6. SystemGenerator with GAFF2 for ligand, AMBER14 for protein
    system_generator = SystemGenerator(
        forcefields=["amber/ff14SB.xml", "amber/tip3p_standard.xml"],
        small_molecule_forcefield="gaff-2.11",
        molecules=[rdmol],
        forcefield_kwargs={
            "constraints":    app.HBonds,
            "rigidWater":     True,
            "removeCMMotion": True,
            "hydrogenMass":   1.5 * unit.amu,
        },
    )

    # 7. Get ligand topology + positions from RDKit mol
    from openmmforcefields.generators import GAFFTemplateGenerator
    from openmm.app import Element
    lig_top  = app.Topology()
    lig_chain = lig_top.addChain()
    lig_res   = lig_top.addResidue("LIG", lig_chain)
    conf = rdmol.GetConformer()
    lig_positions = []
    for atom in rdmol.GetAtoms():
        el = Element.getBySymbol(atom.GetSymbol())
        lig_top.addAtom(atom.GetSymbol(), el, lig_res)
        pos = conf.GetAtomPosition(atom.GetIdx())
        lig_positions.append(mm.Vec3(pos.x, pos.y, pos.z) * 0.1)  # A->nm
    lig_positions = lig_positions * unit.nanometers

    # 8. Combine protein + ligand
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.add(lig_top, lig_positions)

    # 9. Add solvent
    print(f"  Adding solvent...")
    modeller.addSolvent(
        system_generator.forcefield,
        model="tip3p",
        padding=1.0 * unit.nanometers,
        ionicStrength=0.15 * unit.molar,
    )
    print(f"  System size: {modeller.topology.getNumAtoms()} atoms")

    # 10. Create system
    system = system_generator.create_system(
        modeller.topology,
        molecules=[rdmol],
    )
    system.addForce(mm.MonteCarloBarostat(PRESSURE, TEMPERATURE, 25))

    # 11. Integrator + simulation
    integrator = mm.LangevinMiddleIntegrator(TEMPERATURE, 1.0/unit.picoseconds, TIMESTEP)
    try:
        platform = mm.Platform.getPlatformByName("CUDA")
        props    = {"CudaPrecision": "mixed"}
        print(f"  Platform: CUDA")
    except Exception:
        platform = mm.Platform.getPlatformByName("CPU")
        props    = {}
        print(f"  Platform: CPU")

    sim = app.Simulation(modeller.topology, system, integrator, platform, props)
    sim.context.setPositions(modeller.positions)

    # 12. Minimize
    print(f"  Minimizing...")
    sim.minimizeEnergy(maxIterations=2000)

    # 13. Equilibrate
    print(f"  Equilibrating 100 ps...")
    sim.context.setVelocitiesToTemperature(TEMPERATURE)
    sim.step(EQUIL_STEPS)

    # 14. Production
    traj_file = str(work_dir / f"{name}_holo.dcd")
    log_file  = str(work_dir / f"{name}_holo.log")
    sim.reporters.append(app.DCDReporter(traj_file, REPORT_EVERY))
    sim.reporters.append(app.StateDataReporter(
        log_file, REPORT_EVERY,
        step=True, time=True, potentialEnergy=True,
        temperature=True, progress=True, totalSteps=PROD_STEPS,
    ))
    print(f"  Running 2 ns production...")
    t0 = time.time()
    sim.step(PROD_STEPS)
    elapsed = (time.time() - t0) / 60
    print(f"  Done: {elapsed:.1f} min")

    # 15. MM/GBSA from trajectory
    mmgbsa = compute_mmgbsa(traj_file, str(fixed_pdb))

    result = {
        "name": name, "group": group,
        "vina_score": vina_score,
        "n_atoms": modeller.topology.getNumAtoms(),
        "elapsed_min": round(elapsed, 2),
        **mmgbsa,
    }
    json.dump(result, open(work_dir / f"{name}_result.json","w"), indent=2)
    return result


def compute_mmgbsa(traj_dcd, top_pdb):
    """LJ interaction energy proxy between protein and ligand over trajectory."""
    EPSILON = {"C":0.086,"N":0.170,"O":0.210,"S":0.250,"H":0.015}
    RMIN2   = {"C":1.908,"N":1.824,"O":1.661,"S":2.000,"H":0.600}

    def lj(name):
        e = name[0].upper()
        return EPSILON.get(e, 0.086), RMIN2.get(e, 1.908)

    u = mda.Universe(top_pdb, traj_dcd)
    protein = u.select_atoms("protein")
    ligand  = u.select_atoms("resname LIG MOL UNL BHET")
    if len(ligand) == 0:
        ligand = u.select_atoms("not protein and not resname HOH WAT SOL and not name NA CL and mass > 2")

    print(f"  MM/GBSA: protein={len(protein)} ligand={len(ligand)} atoms")
    if len(ligand) == 0:
        return {"mmgbsa_mean": None, "mmgbsa_std": None, "n_frames": 0}

    u.trajectory.add_transformations(center_in_box(protein, wrap=True))

    energies = []
    skip = max(1, len(u.trajectory) // 200)
    for ts in u.trajectory[::skip]:
        dmat = distance_array(protein.positions, ligand.positions)
        mask = dmat < 6.0
        if not mask.any(): continue
        e = 0.0
        for pi, li in zip(*np.where(mask)):
            r = dmat[pi, li]
            if r < 0.5: continue
            ep, rp = lj(protein.atoms[pi].name)
            el, rl = lj(ligand.atoms[li].name)
            rmin = rp + rl
            ratio = (rmin / r) ** 6
            e += np.sqrt(ep * el) * (ratio**2 - 2*ratio)
        energies.append(e)

    arr = np.array(energies)
    print(f"  MM/GBSA: {arr.mean():.3f} +/- {arr.std():.3f} kcal/mol ({len(arr)} frames)")
    return {
        "mmgbsa_mean": round(float(arr.mean()), 4),
        "mmgbsa_std":  round(float(arr.std()),  4),
        "n_frames":    len(arr),
    }


# ── Main ──────────────────────────────────────────────────────
all_results = []
for cand in CANDIDATES:
    try:
        r = run_candidate(cand)
        all_results.append(r)
    except Exception as e:
        print(f"  ERROR {cand['name']}: {e}")
        import traceback; traceback.print_exc()

allo = [r["mmgbsa_mean"] for r in all_results if r["group"]=="allosteric" and r["mmgbsa_mean"]]
asite= [r["mmgbsa_mean"] for r in all_results if r["group"]=="active_site" and r["mmgbsa_mean"]]
wt   = [r["mmgbsa_mean"] for r in all_results if r["group"]=="wt"          and r["mmgbsa_mean"]]

summary = {
    "candidates":         all_results,
    "allosteric_mmgbsa":  allo[0]  if allo  else None,
    "active_site_mmgbsa": asite[0] if asite else None,
    "wt_mmgbsa":          wt[0]    if wt    else None,
    "delta_allo_vs_as":   round(allo[0]-asite[0],4) if (allo and asite) else None,
    "delta_allo_vs_wt":   round(allo[0]-wt[0],  4) if (allo and wt)    else None,
    "method": "LJ interaction energy proxy, 2ns holo OpenMM, AMBER14+GAFF2, CUDA",
}

out = OUT_DIR / "holo_mmgbsa_results.json"
json.dump(summary, open(out,"w"), indent=2)

print("\n" + "="*60)
print("COMPLETE")
for r in all_results:
    print(f"  {r['name']:20s} {r['group']:12s} mmgbsa={r.get('mmgbsa_mean')} kcal/mol")
if summary["delta_allo_vs_as"]:
    print(f"  Delta allo vs active-site: {summary['delta_allo_vs_as']:.4f} kcal/mol")
print(f"\nSaved: {out}")
print(">>> cat results/md/holo_mmgbsa_results.json  — paste to Claude <<<")