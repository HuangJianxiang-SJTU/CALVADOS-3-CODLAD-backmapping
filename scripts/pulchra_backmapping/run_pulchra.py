"""PULCHRA classical backmapping on the same dual inputs as CODLAD (Reviewer 1).

Two conditions, mirroring fig2/fig3 exactly:
  - PED  Cα  -> PULCHRA  (upper bound, paired GT; matches fig2 sampling: up to 300 conformers)
  - CALVADOS Cα -> PULCHRA  (CG-driven; matches fig3 sampling: 500 frames, every 8th of 4000)

PULCHRA is invoked with -f (preserve initial coordinates) so the input Cα framework
is preserved exactly, matching CODLAD's Cα-preservation property and making the
Cα-determined metrics (Rg, Dmax, Cα fidelity) a fair comparison.

Output cache layout is identical to fig3 (per-frame .npz + _top.pkl), so the
existing metric recomputation functions in fig3_step100_recompute_metrics.py
drop straight in.

Usage:
  python scripts/pulchra_backmapping/run_pulchra.py --condition ped  --systems PED00074
  python scripts/pulchra_backmapping/run_pulchra.py --condition cg   --systems PED00074
  python scripts/pulchra_backmapping/run_pulchra.py --condition both            # all 23 systems
"""
import os, sys, argparse, time, io, pickle, warnings
warnings.filterwarnings("ignore")

PROJECT = '/MDdata/data04/jxhuang/cg_cascade'
os.chdir(PROJECT)
sys.path.insert(0, os.path.join(PROJECT, 'scripts'))
sys.path.insert(0, os.path.join(PROJECT, 'src/cascade_codlad/eval_ensemble'))

import numpy as np
import mdtraj as md

PULCHRA_BIN = os.path.join(PROJECT, 'src/pulchra/src_euplotes/pulchra')
PED_DIR = os.path.join(PROJECT, 'data/processed/cascade')
CG_DIR = os.path.join(PROJECT, 'dataset/cg_simulations')
SYSTEMS_CSV = os.path.join(PROJECT, 'dataset/clean_v5.csv')
MAX_CONF = 300          # match fig2
N_FRAMES = 500          # match fig3
FRAME_STRIDE = 8        # every 8th of 4000

NONSTANDARD_RES = {'RCY', 'ZN', 'FE', 'CU', 'MG', 'MN', 'CA', 'CL', 'NA', 'K', 'HOH', 'SO4'}

# -- System list ----------------------------------------------------------------

def load_systems():
    systems = []
    with open(SYSTEMS_CSV) as f:
        next(f)
        for line in f:
            line = line.strip()
            if not line: continue
            systems.append(line.split(',')[0])
    return systems

# -- PED discovery (mirrors fig2_step100_worker.discover_peds) -------------------

def discover_peds():
    sources = {
        'ped_ensembles': os.path.join(PED_DIR, 'ped_ensembles'),
        'ped_staged':    os.path.join(PED_DIR, 'ped_staged'),
        'ped_raw':       os.path.join(PED_DIR, 'ped_raw'),
    }
    peds = {}
    priority = {'ped_ensembles': 1, 'ped_staged': 2, 'ped_raw': 3}
    import glob, re
    for src_name, src_dir in sources.items():
        if not os.path.isdir(src_dir): continue
        if src_name == 'ped_ensembles':
            for ped_id in os.listdir(src_dir):
                ped_path = os.path.join(src_dir, ped_id)
                if not os.path.isdir(ped_path): continue
                pdbs = sorted([f for f in os.listdir(ped_path) if f.endswith('.pdb')])
                if pdbs:
                    pdb_path = os.path.join(ped_path, pdbs[0])
                    if ped_id not in peds or priority[src_name] < peds[ped_id][2]:
                        peds[ped_id] = (src_name, pdb_path, priority[src_name])
        else:
            for pdb_file in glob.glob(os.path.join(src_dir, '*.pdb')):
                m = re.search(r'PED\d+', os.path.basename(pdb_file))
                if m:
                    ped_id = m.group()
                    if ped_id not in peds or priority[src_name] < peds[ped_id][2]:
                        peds[ped_id] = (src_name, pdb_file, priority[src_name])
    return peds

# -- PED PDB parsing (mirrors fig2 quick_pdb_check + read_sampled_models) -------

def count_models(pdb_path):
    n = 0
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("MODEL"): n += 1
    return n if n > 0 else 1

def quick_pdb_check(pdb_path):
    n_models = count_models(pdb_path)
    chains = set(); residues = set(); has_nonstandard = False
    seen_endmdl = False
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("MODEL"):
                if seen_endmdl: break
                continue
            elif line.startswith("ENDMDL"):
                seen_endmdl = True; continue
            elif line.startswith(("ATOM", "HETATM")):
                try:
                    resname = line[17:20].strip()
                    atomname = line[12:16].strip()
                    if resname in NONSTANDARD_RES: has_nonstandard = True
                    if atomname[0] != 'H':
                        chains.add(line[21])
                        residues.add((line[21], line[22:26].strip()))
                except: continue
    return n_models, len(chains), has_nonstandard, len(residues)

def read_model_ca(pdb_path, model_idx):
    """Return list of (resname, resseq, x, y, z) for CA atoms of a 0-based MODEL."""
    cur = None; model_count = -1; records = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("MODEL"):
                model_count += 1; cur = model_count
            elif line.startswith("ENDMDL"):
                if cur == model_idx: return records
                cur = None
            elif cur == model_idx and line.startswith(("ATOM", "HETATM")):
                if line[12:16].strip() == "CA":
                    try:
                        resname = line[17:20].strip()
                        resseq = int(line[22:26])
                        x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                        records.append((resname, resseq, x, y, z))
                    except: continue
    return records

# -- Cα-trace PDB writer (mdtraj-formatted for correct columns) -----------------

def write_ca_trace(resnames, ca_xyz_A, path):
    """ca_xyz_A: (N,3) Angstroms. Writes a single-model Cα-only PDB via mdtraj."""
    top = md.Topology()
    chain = top.add_chain()
    for rn in resnames:
        res = top.add_residue(rn, chain)
        top.add_atom('CA', md.element.carbon, res)
    traj = md.Trajectory(np.asarray(ca_xyz_A, dtype=np.float32)[None, ...] / 10.0, top)
    traj.save(path)

# -- PULCHRA invocation ---------------------------------------------------------

def run_pulchra(ca_trace_pdb):
    """Run PULCHRA -f on a Cα-trace PDB. Returns path to .rebuilt.pdb or None."""
    stem = ca_trace_pdb[:-4]
    out_pdb = stem + '.rebuilt.pdb'
    if os.path.exists(out_pdb): os.remove(out_pdb)
    cmd = f'"{PULCHRA_BIN}" -f "{ca_trace_pdb}" > /dev/null 2>&1'
    rc = os.system(cmd)
    if rc != 0 or not os.path.exists(out_pdb):
        return None
    return out_pdb

def load_rebuilt_as_xyz_top(rebuilt_pdb):
    """Load PULCHRA output, return (xyz_A, mdtraj_topology)."""
    t = md.load(rebuilt_pdb)
    return t.xyz[0] * 10.0, t.topology

# -- Cache I/O (fig3-compatible layout) -----------------------------------------

def write_cache(cache_dir, ped_id, frame_idx, xyz_recon, top, ca_input_A):
    key = f'{ped_id}_{frame_idx}'
    np.savez_compressed(os.path.join(cache_dir, f'{key}.npz'),
                        xyz_recon=xyz_recon, ca_input=ca_input_A)
    with open(os.path.join(cache_dir, f'{key}_top.pkl'), 'wb') as f:
        pickle.dump(top, f)

def cache_exists(cache_dir, ped_id, frame_idx):
    key = f'{ped_id}_{frame_idx}'
    return (os.path.exists(os.path.join(cache_dir, f'{key}.npz')) and
            os.path.exists(os.path.join(cache_dir, f'{key}_top.pkl')))

# -- PED condition --------------------------------------------------------------

def run_ped_condition(ped_id, cache_dir, tmp_dir, max_conf=MAX_CONF):
    peds = discover_peds()
    if ped_id not in peds:
        print(f"  {ped_id}: no PED file found, skip"); return 0
    pdb_path = peds[ped_id][1]
    try:
        n_models, n_chains, has_nonstandard, n_res = quick_pdb_check(pdb_path)
    except Exception as e:
        print(f"  {ped_id}: parse fail ({type(e).__name__})"); return 0
    if n_chains > 1 or has_nonstandard or n_res == 0:
        print(f"  {ped_id}: skip (chains={n_chains}, nonstd={has_nonstandard}, res={n_res})"); return 0

    if n_models > max_conf:
        model_indices = list(np.linspace(0, n_models - 1, max_conf, dtype=int))
    else:
        model_indices = list(range(n_models))

    n_done = 0; n_fail = 0; n_cached = 0
    t0 = time.time()
    for mi in model_indices:
        if cache_exists(cache_dir, ped_id, mi):
            n_cached += 1; continue
        recs = read_model_ca(pdb_path, mi)
        if len(recs) == 0:
            n_fail += 1; continue
        resnames = [r[0] for r in recs]
        ca_xyz = np.array([[r[2], r[3], r[4]] for r in recs], dtype=np.float32)
        trace_pdb = os.path.join(tmp_dir, f'{ped_id}_{mi}_ca.pdb')
        try:
            write_ca_trace(resnames, ca_xyz, trace_pdb)
            rebuilt = run_pulchra(trace_pdb)
            if rebuilt is None:
                print(f"  {ped_id} model {mi}: PULCHRA failed"); n_fail += 1; continue
            xyz_recon, top = load_rebuilt_as_xyz_top(rebuilt)
            write_cache(cache_dir, ped_id, mi, xyz_recon, top, ca_xyz)
            n_done += 1
        except Exception as e:
            print(f"  {ped_id} model {mi}: {type(e).__name__}: {str(e)[:120]}"); n_fail += 1
        finally:
            for ext in ('', '.rebuilt.pdb'):
                p = trace_pdb[:-4] + ext if ext else trace_pdb
                if os.path.exists(p): os.remove(p)
    dt = time.time() - t0
    print(f"  {ped_id}: PED {n_models} models -> sampled {len(model_indices)} "
          f"({n_done} new, {n_cached} cached, {n_fail} fail) in {dt:.1f}s")
    return n_done

# -- CALVADOS (CG) condition ----------------------------------------------------

def run_cg_condition(ped_id, cache_dir, tmp_dir, n_frames=N_FRAMES, stride=FRAME_STRIDE):
    dcd = os.path.join(CG_DIR, ped_id, f'{ped_id}.dcd')
    top_pdb = os.path.join(CG_DIR, ped_id, f'{ped_id}_first.pdb')
    if not os.path.exists(dcd):
        print(f"  {ped_id}: no CG trajectory, skip"); return 0
    traj = md.load(dcd, top=top_pdb)
    rnames = [r.name for r in traj.topology.residues]
    n_total = traj.n_frames
    frame_indices = list(range(0, min(n_total, 4000), stride))[:n_frames]

    n_done = 0; n_fail = 0; n_cached = 0
    t0 = time.time()
    ca_A_all = traj.xyz[frame_indices] * 10.0  # nm -> Å
    for fi, frame_idx in enumerate(frame_indices):
        if cache_exists(cache_dir, ped_id, frame_idx):
            n_cached += 1; continue
        ca_xyz = ca_A_all[fi].astype(np.float32)
        trace_pdb = os.path.join(tmp_dir, f'{ped_id}_{frame_idx}_ca.pdb')
        try:
            write_ca_trace(rnames, ca_xyz, trace_pdb)
            rebuilt = run_pulchra(trace_pdb)
            if rebuilt is None:
                print(f"  {ped_id} frame {frame_idx}: PULCHRA failed"); n_fail += 1; continue
            xyz_recon, top = load_rebuilt_as_xyz_top(rebuilt)
            write_cache(cache_dir, ped_id, frame_idx, xyz_recon, top, ca_xyz)
            n_done += 1
        except Exception as e:
            print(f"  {ped_id} frame {frame_idx}: {type(e).__name__}: {str(e)[:120]}"); n_fail += 1
        finally:
            for ext in ('', '.rebuilt.pdb'):
                p = trace_pdb[:-4] + ext if ext else trace_pdb
                if os.path.exists(p): os.remove(p)
    dt = time.time() - t0
    print(f"  {ped_id}: CG {n_total} frames -> sampled {len(frame_indices)} "
          f"({n_done} new, {n_cached} cached, {n_fail} fail) in {dt:.1f}s")
    return n_done

# -- Main -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--condition', choices=['ped', 'cg', 'both'], default='both')
    ap.add_argument('--systems', default=None, help='comma-separated PED IDs; default all 23')
    ap.add_argument('--max-conf', type=int, default=MAX_CONF)
    args = ap.parse_args()

    all_systems = load_systems()
    systems = args.systems.split(',') if args.systems else all_systems

    conds = []
    if args.condition in ('ped', 'both'): conds.append('ped')
    if args.condition in ('cg', 'both'):  conds.append('cg')

    for cond in conds:
        cache_dir = os.path.join(PROJECT, 'logs', f'pulchra_{cond}')
        tmp_dir = os.path.join(PROJECT, 'data', f'pulchra_{cond}_tmp')
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(tmp_dir, exist_ok=True)
        print(f"\n=== PULCHRA condition: {cond.upper()} ({len(systems)} systems) ===")
        for pi, ped_id in enumerate(systems):
            print(f"[{pi+1}/{len(systems)}] {ped_id}")
            if cond == 'ped':
                run_ped_condition(ped_id, cache_dir, tmp_dir, max_conf=args.max_conf)
            else:
                run_cg_condition(ped_id, cache_dir, tmp_dir)

if __name__ == '__main__':
    main()
