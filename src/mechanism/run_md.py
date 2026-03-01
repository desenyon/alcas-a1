"""
ALCAS - Apo PETase MD Simulation
3 replicate NPT simulations of WT PETase (5XJH) using OpenMM on CUDA.

Protocol:
  - Force field: AMBER14-SB + TIP3P-FB water
  - Box: cubic, 10A padding, 0.15M NaCl, neutralized
  - Minimization round 1: 20000 steps (tolerance=10 kJ/mol/nm)
  - Minimization round 2: 5000 steps after velocity assignment
  - Gradual heating: 10K -> 300K in 30 stages over 100ps at 1fs timestep
  - NVT equilibration: 500ps at 1fs timestep (restrained heavy atoms, k=200)
  - NPT equilibration: 500ps at 1fs timestep (no restraints)
  - NPT production: 10ns at 2fs timestep
  - Save: every 10ps (1000 frames per replicate, 3000 total)
  - Temperature: 300K (Langevin middle integrator)
  - Pressure: 1 bar (Monte Carlo barostat)
"""

import openmm as mm
import openmm.app as app
import openmm.unit as unit
from pathlib import Path
import sys
import time
import json
import copy

# ============================================================
# PARAMETERS
# ============================================================
PDB_PATH         = Path('data/petase/5XJH_fixed.pdb')
OUT_DIR          = Path('data/petase/md')
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEMPERATURE      = 300  * unit.kelvin
PRESSURE         = 1.0  * unit.bar
TIMESTEP_EQUIL   = 1.0  * unit.femtoseconds   # conservative during equilibration
TIMESTEP_PROD    = 2.0  * unit.femtoseconds   # standard during production
FRICTION         = 1.0  / unit.picosecond
CUTOFF           = 1.0  * unit.nanometer
PADDING          = 1.0  * unit.nanometer
IONIC_STRENGTH   = 0.15 * unit.molar

MINIM1_STEPS     = 20_000   # first minimization
MINIM1_TOL       = 10.0     # kJ/mol/nm
MINIM2_STEPS     = 5_000    # second minimization after velocity assignment
MINIM2_TOL       = 50.0     # kJ/mol/nm (looser, just remove hot spots)

HEAT_STEPS       = 100_000  # 100ps total heating at 1fs
HEAT_STAGES      = 30       # 30 temperature increments
NVT_STEPS        = 500_000  # 500ps NVT restrained at 1fs
NPT_EQUIL_STEPS  = 500_000  # 500ps NPT unrestrained at 1fs
PROD_STEPS       = 5_000_000 # 10ns production at 2fs
SAVE_INTERVAL    = 5_000    # save every 10ps
STDOUT_INTERVAL  = 50_000   # print every 100ps

N_REPLICATES     = 3
SEEDS            = [42, 7, 13]
RESTRAINT_K      = 200.0    # kJ/mol/nm^2, soft restraints

CUDA_PROPS = {
    'CudaDeviceIndex': '0',
    'CudaPrecision':   'mixed',
}


# ============================================================
# BUILD BASE SYSTEM (once, reused across replicates)
# ============================================================
def build_base_system():
    print("Loading fixed PDB...")
    pdb = app.PDBFile(str(PDB_PATH))
    ff  = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

    print("Solvating (10A padding, 0.15M NaCl)...")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addSolvent(
        ff,
        padding       = PADDING,
        ionicStrength = IONIC_STRENGTH,
        neutralize    = True,
    )

    n_atoms = modeller.topology.getNumAtoms()
    n_res   = modeller.topology.getNumResidues()
    print(f"System: {n_atoms} atoms | {n_res} residues")

    print("Building AMBER14 force field system...")
    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod     = app.PME,
        nonbondedCutoff     = CUTOFF,
        constraints         = app.HBonds,
        rigidWater          = True,
        ewaldErrorTolerance = 0.0005,
    )

    return system, modeller


# ============================================================
# HELPERS
# ============================================================
def make_restrained_system(base_system, modeller):
    """Deep copy + add harmonic restraints on protein heavy atoms."""
    sys_r = copy.deepcopy(base_system)

    restraint = mm.CustomExternalForce(
        'k*((x-x0)^2+(y-y0)^2+(z-z0)^2)'
    )
    restraint.addGlobalParameter(
        'k', RESTRAINT_K * unit.kilojoules_per_mole / unit.nanometer**2
    )
    restraint.addPerParticleParameter('x0')
    restraint.addPerParticleParameter('y0')
    restraint.addPerParticleParameter('z0')

    positions   = modeller.positions
    skip_names  = {'HOH', 'WAT', 'SOL', 'NA', 'CL', 'SOD', 'CLA', 'K', 'MG', 'CA'}
    n_restrained = 0

    for atom in modeller.topology.atoms():
        if atom.residue.name in skip_names:
            continue
        if atom.element is not None and atom.element.symbol == 'H':
            continue
        pos = positions[atom.index]
        restraint.addParticle(atom.index, [pos.x, pos.y, pos.z])
        n_restrained += 1

    sys_r.addForce(restraint)
    print(f"    Restrained {n_restrained} heavy atoms (k={RESTRAINT_K} kJ/mol/nm²)")
    return sys_r


def make_npt_system(base_system):
    """Deep copy + barostat for NPT."""
    sys_npt = copy.deepcopy(base_system)
    sys_npt.addForce(mm.MonteCarloBarostat(PRESSURE, TEMPERATURE, 25))
    return sys_npt


def make_sim(topology, system, integrator):
    """Create CUDA simulation."""
    platform = mm.Platform.getPlatformByName('CUDA')
    return app.Simulation(topology, system, integrator, platform, CUDA_PROPS)


# ============================================================
# RUN ONE REPLICATE
# ============================================================
def run_replicate(rep_idx, seed, base_system, modeller):
    print(f"\n{'='*60}")
    print(f"Replicate {rep_idx+1}/{N_REPLICATES} | Seed {seed}")
    print(f"{'='*60}")

    prefix = OUT_DIR / f'rep{rep_idx+1}'

    # --------------------------------------------------------
    # PHASE 1: Minimization + heating + NVT (restrained, 1fs)
    # --------------------------------------------------------
    print("Phase 1: NVT equilibration")
    sys_nvt = make_restrained_system(base_system, modeller)
    intg_nvt = mm.LangevinMiddleIntegrator(10 * unit.kelvin, FRICTION, TIMESTEP_EQUIL)
    intg_nvt.setRandomNumberSeed(seed)
    sim_nvt = make_sim(modeller.topology, sys_nvt, intg_nvt)
    sim_nvt.context.setPositions(modeller.positions)

    # Round 1 minimization
    print(f"  Minimization 1 (max {MINIM1_STEPS} steps, tol={MINIM1_TOL})...")
    sim_nvt.minimizeEnergy(
        maxIterations = MINIM1_STEPS,
        tolerance     = MINIM1_TOL * unit.kilojoules_per_mole / unit.nanometer,
    )
    e1 = sim_nvt.context.getState(getEnergy=True).getPotentialEnergy()
    print(f"  Energy after min1: {e1.value_in_unit(unit.kilocalories_per_mole):.1f} kcal/mol")

    # Assign velocities at 10K then minimize again to remove hot spots
    print("  Assigning velocities at 10K...")
    sim_nvt.context.setVelocitiesToTemperature(10 * unit.kelvin, seed)

    print(f"  Minimization 2 (max {MINIM2_STEPS} steps, tol={MINIM2_TOL})...")
    sim_nvt.minimizeEnergy(
        maxIterations = MINIM2_STEPS,
        tolerance     = MINIM2_TOL * unit.kilojoules_per_mole / unit.nanometer,
    )
    e2 = sim_nvt.context.getState(getEnergy=True).getPotentialEnergy()
    print(f"  Energy after min2: {e2.value_in_unit(unit.kilocalories_per_mole):.1f} kcal/mol")

    # Gradual heating 10K -> 300K
    steps_per_stage = HEAT_STEPS // HEAT_STAGES
    print(f"  Heating 10K -> 300K ({HEAT_STAGES} stages, {HEAT_STEPS * 1e-6 * 1000:.0f} ps)...")
    for stage in range(HEAT_STAGES):
        T = 10.0 + (290.0 * (stage + 1) / HEAT_STAGES)
        intg_nvt.setTemperature(T * unit.kelvin)
        sim_nvt.step(steps_per_stage)
        if (stage + 1) % 10 == 0:
            state = sim_nvt.context.getState(getEnergy=True)
            print(f"    Stage {stage+1}/{HEAT_STAGES}: T={T:.0f}K | "
                  f"E={state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole):.0f} kcal/mol")
    intg_nvt.setTemperature(TEMPERATURE)

    # NVT at 300K with restraints
    print(f"  NVT at 300K with restraints ({NVT_STEPS * 1e-6 * 1000:.0f} ps)...")
    sim_nvt.step(NVT_STEPS)

    state_nvt = sim_nvt.context.getState(
        getPositions=True, getVelocities=True, enforcePeriodicBox=True
    )
    pos_nvt = state_nvt.getPositions()
    vel_nvt = state_nvt.getVelocities()
    box_nvt = state_nvt.getPeriodicBoxVectors()
    del sim_nvt, intg_nvt, sys_nvt
    print("  Phase 1 complete.")

    # --------------------------------------------------------
    # PHASE 2: NPT equilibration (no restraints, 1fs)
    # --------------------------------------------------------
    print("Phase 2: NPT equilibration (no restraints)")
    sys_npt_eq  = make_npt_system(base_system)
    intg_npt_eq = mm.LangevinMiddleIntegrator(TEMPERATURE, FRICTION, TIMESTEP_EQUIL)
    intg_npt_eq.setRandomNumberSeed(seed + 1000)
    sim_npt_eq  = make_sim(modeller.topology, sys_npt_eq, intg_npt_eq)
    sim_npt_eq.context.setPositions(pos_nvt)
    sim_npt_eq.context.setVelocities(vel_nvt)
    sim_npt_eq.context.setPeriodicBoxVectors(*box_nvt)

    print(f"  NPT equilibration ({NPT_EQUIL_STEPS * 1e-6 * 1000:.0f} ps)...")
    sim_npt_eq.step(NPT_EQUIL_STEPS)

    state_npt = sim_npt_eq.context.getState(
        getPositions=True, getVelocities=True, enforcePeriodicBox=True
    )
    pos_npt = state_npt.getPositions()
    vel_npt = state_npt.getVelocities()
    box_npt = state_npt.getPeriodicBoxVectors()
    del sim_npt_eq, intg_npt_eq, sys_npt_eq
    print("  Phase 2 complete.")

    # --------------------------------------------------------
    # PHASE 3: NPT production (2fs)
    # --------------------------------------------------------
    print("Phase 3: NPT production")
    sys_prod  = make_npt_system(base_system)
    intg_prod = mm.LangevinMiddleIntegrator(TEMPERATURE, FRICTION, TIMESTEP_PROD)
    intg_prod.setRandomNumberSeed(seed + 2000)
    sim       = make_sim(modeller.topology, sys_prod, intg_prod)
    sim.context.setPositions(pos_npt)
    sim.context.setVelocities(vel_npt)
    sim.context.setPeriodicBoxVectors(*box_npt)

    # Save topology PDB for MDAnalysis
    topo_path = str(prefix) + '_topology.pdb'
    with open(topo_path, 'w') as f:
        app.PDBFile.writeFile(
            sim.topology,
            sim.context.getState(getPositions=True).getPositions(),
            f
        )

    dcd_path = str(prefix) + '_traj.dcd'
    log_path = str(prefix) + '_md.log'

    sim.reporters.append(app.DCDReporter(dcd_path, SAVE_INTERVAL))
    sim.reporters.append(app.StateDataReporter(
        log_path, SAVE_INTERVAL,
        step=True, time=True,
        potentialEnergy=True, kineticEnergy=True,
        temperature=True, density=True,
        progress=True, remainingTime=True,
        totalSteps=PROD_STEPS, separator='\t',
    ))
    sim.reporters.append(app.StateDataReporter(
        sys.stdout, STDOUT_INTERVAL,
        step=True, time=True, temperature=True,
        progress=True, remainingTime=True,
        totalSteps=PROD_STEPS, separator='\t',
    ))

    print(f"  Production 10.0 ns ({PROD_STEPS} steps at 2fs)...")
    t0 = time.time()
    sim.step(PROD_STEPS)
    elapsed = time.time() - t0

    final_state = sim.context.getState(
        getPositions=True, enforcePeriodicBox=True
    )
    with open(str(prefix) + '_final.pdb', 'w') as f:
        app.PDBFile.writeFile(sim.topology, final_state.getPositions(), f)

    n_frames = PROD_STEPS // SAVE_INTERVAL
    print(f"\n  Replicate {rep_idx+1} complete: {elapsed/60:.1f} min | {n_frames} frames")

    return {
        'dcd':         dcd_path,
        'topology':    topo_path,
        'log':         log_path,
        'elapsed_min': elapsed / 60,
        'n_frames':    n_frames,
        'seed':        seed,
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("ALCAS - Apo PETase MD Simulation")
    print(f"Production: {PROD_STEPS * 2e-6 / 1000:.0f} ns x {N_REPLICATES} replicates")
    print(f"Frames:     {PROD_STEPS // SAVE_INTERVAL} per replicate ({SAVE_INTERVAL * 2e-3:.0f} ps interval)")
    print(f"Protocol:   min1 -> vel+min2 -> heat(30 stages) -> NVT(1fs) -> NPT_eq(1fs) -> NPT_prod(2fs)")
    print("=" * 60)

    for i in range(mm.Platform.getNumPlatforms()):
        print(f"Platform: {mm.Platform.getPlatform(i).getName()}")

    # CUDA sanity check
    try:
        _s = mm.System(); _s.addParticle(1.0)
        _i = mm.VerletIntegrator(0.001)
        _c = mm.Context(_s, _i, mm.Platform.getPlatformByName('CUDA'), CUDA_PROPS)
        print("CUDA: OK")
        del _c, _i, _s
    except Exception as e:
        print(f"CUDA FAILED: {e}")
        sys.exit(1)

    base_system, modeller = build_base_system()

    results  = []
    total_t0 = time.time()
    for i, seed in enumerate(SEEDS):
        result = run_replicate(i, seed, base_system, modeller)
        results.append(result)

    total_elapsed = time.time() - total_t0

    manifest = {
        'pdb':          str(PDB_PATH),
        'n_replicates': N_REPLICATES,
        'prod_ns':      PROD_STEPS * 2e-6 / 1000,
        'save_ps':      SAVE_INTERVAL * 2e-3,
        'total_ns':     N_REPLICATES * PROD_STEPS * 2e-6 / 1000,
        'total_frames': N_REPLICATES * PROD_STEPS // SAVE_INTERVAL,
        'elapsed_min':  total_elapsed / 60,
        'replicates':   results,
    }
    with open(OUT_DIR / 'md_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"MD COMPLETE: {total_elapsed/60:.1f} min total")
    print("Next: python src/mechanism/coupling_analysis.py")
    print("=" * 60)