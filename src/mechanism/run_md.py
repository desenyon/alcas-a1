"""
ALCAS - Apo PETase MD Simulation
Key fix: restraint reference positions update after each minimization stage.
"""

import openmm as mm
import openmm.app as app
import openmm.unit as unit
from pathlib import Path
import sys, time, json, copy

PDB_PATH       = Path('data/petase/5XJH_fixed.pdb')
OUT_DIR        = Path('data/petase/md')
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEMPERATURE    = 300  * unit.kelvin
PRESSURE       = 1.0  * unit.bar
CUTOFF         = 1.0  * unit.nanometer
PADDING        = 1.0  * unit.nanometer
IONIC_STRENGTH = 0.15 * unit.molar

N_REPLICATES = 3
SEEDS        = [42, 7, 13]
CUDA_PROPS   = {'CudaDeviceIndex': '0', 'CudaPrecision': 'mixed'}
SKIP_NAMES   = {'HOH','WAT','SOL','NA','CL','SOD','CLA','K','MG','CA'}


def build_base_system():
    print("Loading PDB...")
    pdb = app.PDBFile(str(PDB_PATH))
    ff  = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    print("Solvating...")
    mod = app.Modeller(pdb.topology, pdb.positions)
    mod.addSolvent(ff, padding=PADDING, ionicStrength=IONIC_STRENGTH, neutralize=True)
    print(f"System: {mod.topology.getNumAtoms()} atoms")
    print("Creating force field...")
    system = ff.createSystem(
        mod.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=CUTOFF,
        constraints=app.HBonds,
        rigidWater=True,
        ewaldErrorTolerance=0.0005,
    )
    return system, mod


def make_restrained_system(base_system, topology, positions, k):
    """
    Add restraints using CURRENT positions as reference.
    This is critical - always use the latest positions, not original.
    """
    s = copy.deepcopy(base_system)
    f = mm.CustomExternalForce('k*((x-x0)^2+(y-y0)^2+(z-z0)^2)')
    f.addGlobalParameter('k', k * unit.kilojoules_per_mole / unit.nanometer**2)
    f.addPerParticleParameter('x0')
    f.addPerParticleParameter('y0')
    f.addPerParticleParameter('z0')
    for atom in topology.atoms():
        if atom.residue.name in SKIP_NAMES: continue
        if atom.element is not None and atom.element.symbol == 'H': continue
        p = positions[atom.index]
        # positions may be in nm (OpenMM internal) or with units
        try:
            f.addParticle(atom.index, [p.x, p.y, p.z])
        except AttributeError:
            f.addParticle(atom.index, [float(p[0]), float(p[1]), float(p[2])])
    s.addForce(f)
    return s


def make_npt_system(base_system):
    s = copy.deepcopy(base_system)
    s.addForce(mm.MonteCarloBarostat(PRESSURE, TEMPERATURE, 25))
    return s


def make_sim(top, system, integrator):
    return app.Simulation(top, system, integrator,
                          mm.Platform.getPlatformByName('CUDA'), CUDA_PROPS)


def get_energy(sim):
    return sim.context.getState(getEnergy=True)\
               .getPotentialEnergy()\
               .value_in_unit(unit.kilocalories_per_mole)


def run_replicate(rep_idx, seed, base_system, modeller):
    print(f"\n{'='*60}")
    print(f"Replicate {rep_idx+1}/{N_REPLICATES} | Seed {seed}")
    print(f"{'='*60}")
    prefix = OUT_DIR / f'rep{rep_idx+1}'
    top    = modeller.topology

    # ----------------------------------------------------------
    # STEP 1: Minimize with strong restraints from ORIGINAL positions
    # ----------------------------------------------------------
    print("\n[1] Minimize strong restraints (k=5000, ref=original)...")
    s1   = make_restrained_system(base_system, top, modeller.positions, 5000.0)
    i1   = mm.LangevinMiddleIntegrator(10*unit.kelvin, 10/unit.picosecond, 0.5*unit.femtoseconds)
    sim1 = make_sim(top, s1, i1)
    sim1.context.setPositions(modeller.positions)
    sim1.minimizeEnergy(maxIterations=10000,
                        tolerance=10*unit.kilojoules_per_mole/unit.nanometer)
    e1 = get_energy(sim1)
    print(f"   E={e1:.0f} kcal/mol")
    st1  = sim1.context.getState(getPositions=True, enforcePeriodicBox=True)
    pos1 = st1.getPositions()
    box1 = st1.getPeriodicBoxVectors()
    del sim1, i1, s1

    # ----------------------------------------------------------
    # STEP 2: Minimize with medium restraints from STEP 1 positions
    # ----------------------------------------------------------
    print("[2] Minimize medium restraints (k=500, ref=step1 positions)...")
    s2   = make_restrained_system(base_system, top, pos1, 500.0)
    i2   = mm.LangevinMiddleIntegrator(10*unit.kelvin, 10/unit.picosecond, 0.5*unit.femtoseconds)
    sim2 = make_sim(top, s2, i2)
    sim2.context.setPositions(pos1)
    sim2.context.setPeriodicBoxVectors(*box1)
    sim2.minimizeEnergy(maxIterations=10000,
                        tolerance=5*unit.kilojoules_per_mole/unit.nanometer)
    e2 = get_energy(sim2)
    print(f"   E={e2:.0f} kcal/mol")
    st2  = sim2.context.getState(getPositions=True, enforcePeriodicBox=True)
    pos2 = st2.getPositions()
    box2 = st2.getPeriodicBoxVectors()
    del sim2, i2, s2

    # ----------------------------------------------------------
    # STEP 3: Minimize with weak restraints from STEP 2 positions
    # ----------------------------------------------------------
    print("[3] Minimize weak restraints (k=50, ref=step2 positions)...")
    s3   = make_restrained_system(base_system, top, pos2, 50.0)
    i3   = mm.LangevinMiddleIntegrator(10*unit.kelvin, 10/unit.picosecond, 0.5*unit.femtoseconds)
    sim3 = make_sim(top, s3, i3)
    sim3.context.setPositions(pos2)
    sim3.context.setPeriodicBoxVectors(*box2)
    sim3.minimizeEnergy(maxIterations=10000,
                        tolerance=5*unit.kilojoules_per_mole/unit.nanometer)
    e3 = get_energy(sim3)
    print(f"   E={e3:.0f} kcal/mol")
    st3  = sim3.context.getState(getPositions=True, enforcePeriodicBox=True)
    pos3 = st3.getPositions()
    box3 = st3.getPeriodicBoxVectors()
    del sim3, i3, s3

    # Sanity check
    if e3 > -1_000_000:
        print(f"   WARNING: energy {e3:.0f} suspiciously high, check structure")

    # ----------------------------------------------------------
    # STEP 4: Short NVT at 10K to absorb velocity stress
    # (restrained k=50 from step3 positions, 0.5fs, fric=10, 10ps)
    # ----------------------------------------------------------
    print("[4] Short NVT at 10K (10ps, 0.5fs, fric=10, k=50)...")
    s4   = make_restrained_system(base_system, top, pos3, 50.0)
    i4   = mm.LangevinMiddleIntegrator(10*unit.kelvin, 10/unit.picosecond, 0.5*unit.femtoseconds)
    i4.setRandomNumberSeed(seed)
    sim4 = make_sim(top, s4, i4)
    sim4.context.setPositions(pos3)
    sim4.context.setPeriodicBoxVectors(*box3)
    sim4.context.setVelocitiesToTemperature(10*unit.kelvin, seed)
    sim4.step(20_000)   # 10ps
    print(f"   E={get_energy(sim4):.0f} kcal/mol  -> OK")
    st4  = sim4.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
    pos4, vel4, box4 = st4.getPositions(), st4.getVelocities(), st4.getPeriodicBoxVectors()
    del sim4, i4, s4

    # ----------------------------------------------------------
    # STEP 5: Heat 10K -> 300K
    # (restrained k=50, 0.5fs, fric=10, 50 stages x 10ps = 500ps)
    # ----------------------------------------------------------
    print("[5] Heating 10K -> 300K (500ps, 50 stages, 0.5fs, fric=10)...")
    s5   = make_restrained_system(base_system, top, pos4, 50.0)
    i5   = mm.LangevinMiddleIntegrator(10*unit.kelvin, 10/unit.picosecond, 0.5*unit.femtoseconds)
    i5.setRandomNumberSeed(seed + 100)
    sim5 = make_sim(top, s5, i5)
    sim5.context.setPositions(pos4)
    sim5.context.setVelocities(vel4)
    sim5.context.setPeriodicBoxVectors(*box4)
    n_stages = 50
    steps_per_stage = 20_000   # 10ps per stage
    for stage in range(n_stages):
        T = 10.0 + (290.0 * (stage + 1) / n_stages)
        i5.setTemperature(T * unit.kelvin)
        sim5.step(steps_per_stage)
        if (stage + 1) % 10 == 0:
            print(f"   Stage {stage+1}/{n_stages}: T={T:.0f}K | E={get_energy(sim5):.0f} kcal/mol")
    i5.setTemperature(TEMPERATURE)
    st5  = sim5.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
    pos5, vel5, box5 = st5.getPositions(), st5.getVelocities(), st5.getPeriodicBoxVectors()
    del sim5, i5, s5
    print("   Heating complete.")

    # ----------------------------------------------------------
    # STEP 6: NVT 300K restrained (500ps, 0.5fs, fric=5, k=50)
    # ----------------------------------------------------------
    print("[6] NVT 300K restrained (500ps, 0.5fs, fric=5)...")
    s6   = make_restrained_system(base_system, top, pos5, 50.0)
    i6   = mm.LangevinMiddleIntegrator(TEMPERATURE, 5/unit.picosecond, 0.5*unit.femtoseconds)
    i6.setRandomNumberSeed(seed + 200)
    sim6 = make_sim(top, s6, i6)
    sim6.context.setPositions(pos5)
    sim6.context.setVelocities(vel5)
    sim6.context.setPeriodicBoxVectors(*box5)
    sim6.step(1_000_000)   # 500ps at 0.5fs
    st6  = sim6.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
    pos6, vel6, box6 = st6.getPositions(), st6.getVelocities(), st6.getPeriodicBoxVectors()
    del sim6, i6, s6
    print("   NVT complete.")

    # ----------------------------------------------------------
    # STEP 7: NPT equilibration unrestrained (500ps, 1fs, fric=2)
    # ----------------------------------------------------------
    print("[7] NPT equilibration unrestrained (500ps, 1fs, fric=2)...")
    s7   = make_npt_system(base_system)
    i7   = mm.LangevinMiddleIntegrator(TEMPERATURE, 2/unit.picosecond, 1.0*unit.femtoseconds)
    i7.setRandomNumberSeed(seed + 300)
    sim7 = make_sim(top, s7, i7)
    sim7.context.setPositions(pos6)
    sim7.context.setVelocities(vel6)
    sim7.context.setPeriodicBoxVectors(*box6)
    sim7.step(500_000)   # 500ps at 1fs
    st7  = sim7.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
    pos7, vel7, box7 = st7.getPositions(), st7.getVelocities(), st7.getPeriodicBoxVectors()
    del sim7, i7, s7
    print("   NPT equil complete.")

    # ----------------------------------------------------------
    # STEP 8: NPT production (10ns, 2fs, fric=1)
    # ----------------------------------------------------------
    print("[8] NPT production (10ns, 2fs, fric=1)...")
    PROD_STEPS    = 5_000_000
    SAVE_INTERVAL = 5_000

    s8   = make_npt_system(base_system)
    i8   = mm.LangevinMiddleIntegrator(TEMPERATURE, 1/unit.picosecond, 2.0*unit.femtoseconds)
    i8.setRandomNumberSeed(seed + 400)
    sim  = make_sim(top, s8, i8)
    sim.context.setPositions(pos7)
    sim.context.setVelocities(vel7)
    sim.context.setPeriodicBoxVectors(*box7)

    topo_path = str(prefix) + '_topology.pdb'
    with open(topo_path, 'w') as f:
        app.PDBFile.writeFile(sim.topology,
            sim.context.getState(getPositions=True).getPositions(), f)

    dcd_path = str(prefix) + '_traj.dcd'
    log_path = str(prefix) + '_md.log'

    sim.reporters.append(app.DCDReporter(dcd_path, SAVE_INTERVAL))
    sim.reporters.append(app.StateDataReporter(
        log_path, SAVE_INTERVAL,
        step=True, time=True, potentialEnergy=True, kineticEnergy=True,
        temperature=True, density=True, progress=True, remainingTime=True,
        totalSteps=PROD_STEPS, separator='\t',
    ))
    sim.reporters.append(app.StateDataReporter(
        sys.stdout, 50_000,
        step=True, time=True, temperature=True,
        progress=True, remainingTime=True,
        totalSteps=PROD_STEPS, separator='\t',
    ))

    t0 = time.time()
    sim.step(PROD_STEPS)
    elapsed = time.time() - t0

    final = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
    with open(str(prefix) + '_final.pdb', 'w') as f:
        app.PDBFile.writeFile(sim.topology, final.getPositions(), f)

    n_frames = PROD_STEPS // SAVE_INTERVAL
    print(f"\n   Replicate {rep_idx+1} done: {elapsed/60:.1f} min | {n_frames} frames")
    return {'dcd': dcd_path, 'topology': topo_path, 'log': log_path,
            'elapsed_min': elapsed/60, 'n_frames': n_frames, 'seed': seed}


if __name__ == '__main__':
    print("="*60)
    print("ALCAS - Apo PETase MD")
    print("Protocol: min(k=5000)->min(k=500,ref=updated)->min(k=50,ref=updated)")
    print("       -> NVT_10K -> heat(50 stages) -> NVT_300K -> NPT_eq -> NPT_prod")
    print("="*60)

    for i in range(mm.Platform.getNumPlatforms()):
        print(f"Platform: {mm.Platform.getPlatform(i).getName()}")

    try:
        _s = mm.System(); _s.addParticle(1.0)
        _i = mm.VerletIntegrator(0.001)
        _c = mm.Context(_s, _i, mm.Platform.getPlatformByName('CUDA'), CUDA_PROPS)
        print("CUDA: OK")
        del _c, _i, _s
    except Exception as e:
        print(f"CUDA FAILED: {e}"); sys.exit(1)

    base_system, modeller = build_base_system()

    results  = []
    t0_total = time.time()
    for i, seed in enumerate(SEEDS):
        results.append(run_replicate(i, seed, base_system, modeller))

    manifest = {
        'pdb': str(PDB_PATH), 'n_replicates': N_REPLICATES,
        'prod_ns': 10.0, 'total_ns': 30.0,
        'total_frames': N_REPLICATES * 5_000_000 // 5_000,
        'elapsed_min': (time.time()-t0_total)/60,
        'replicates': results,
    }
    with open(OUT_DIR / 'md_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"MD COMPLETE: {(time.time()-t0_total)/60:.1f} min total")
    print("Next: python src/mechanism/coupling_analysis.py")
    print("="*60)
