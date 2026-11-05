#!/usr/bin/env python3
"""c201 (R-234 item 2): the block-size study readout per arm (out.log PHASE_TIMERS, hst, the last prim_3d dump, mem.log).
  window: cycles 100 -> 200 (PHASE_TIMERS are cumulative; the first 100 include the start-up): wall/cycle, its split
  (stagen = the stage loop; mg = the CFC multigrid solves inside it; hydro_etc = the rest of the stage loop; regrid; io),
  zone-cycles/s per GPU (active cells x cycles / wall / 96), simulated time per wall-hour; nmb (min/max over the window's
  PHASE_TIMERS rows) and ceil(nmb/96)/(nmb/96) (the load-balance bound, argued: equal-cost blocks);
  per level of the last 3D dump: blocks and refined volume; hst at the end: rho_max/rho_c, the dense mass, K via the dump.
    c201_blocks_ana.py ARMDIR ..."""
import glob, re, sys, types, math
import numpy as np
sys.modules.setdefault('h5py', types.ModuleType('h5py'))
sys.path.insert(0, '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_tde_worktrees/riemann_frame/vis/python')
import bin_convert as bc   # noqa: E402
GAM, RC, K0 = 5.0/3.0, 3.24e-4, 1.2346626297e-2


def pt(line):
    return {k: float(v) for k, v in re.findall(r'(\w+)=([-0-9.e+]+)', line)}


for d in sys.argv[1:]:
    log = open(f'{d}/out.log').read().splitlines()
    rows = [pt(l) for l in log if l.startswith('PHASE_TIMERS ')]
    fin = [pt(l) for l in log if l.startswith('PHASE_TIMERS_FINAL')]
    by = {int(r['cycle']): r for r in rows}
    nb = re.search(r'nx1 *= *(\d+)', open(f'{d}/../{d.rstrip("/").split("/")[-1]}.par').read().split('<meshblock>')[1]).group(1)
    print(f'== {d} (meshblock {nb}^3)')
    if 100 in by and (200 in by or fin):
        a = by[100]; b = by.get(200, fin[-1] if fin else None)
        nc = b['cycle'] - a['cycle']; w = (b['wall'] - a['wall'])/nc
        sp = {k: (b[k] - a[k])/nc for k in ('stagen', 'mg', 'hydro_etc', 'regrid', 'io') if k in a and k in b}
        nmbs = [r['nmb'] for c, r in by.items() if 100 <= c <= 200] + ([b['nmb']] if 'nmb' in b else [])
        n = int(nb)
        cells = np.mean(nmbs)*n**3
        dt_sim = (b['time'] - a['time'])
        print(f'  cycles {int(a["cycle"])}-{int(b["cycle"])}: wall/cycle {w:.3f} s | ' + ' '.join(f'{k} {v:.3f}' for k, v in sp.items())
              + f' | mg share of stage loop {sp["mg"]/sp["stagen"]:.2f} | zone-cycles/s/GPU {cells/w/96:.3e} | M per wall-h {dt_sim/(b["wall"]-a["wall"])*3600:.2f}'
              + f' | dt {dt_sim/nc:.4f} | nmb {min(nmbs):.0f}-{max(nmbs):.0f} (lb bound {math.ceil(max(nmbs)/96)/(max(nmbs)/96):.2f}, blocks/rank {max(nmbs)/96:.1f})')
    else:
        print('  no 100->200 window:', sorted(by)[-3:] if by else 'no PHASE_TIMERS', fin[-1] if fin else '')
    fs = sorted(glob.glob(f'{d}/bin/*.prim_3d.*.bin'), key=lambda f: bc.read_binary(f)['time'])
    if fs:
        P = bc.read_binary(fs[-1]); g = np.asarray(P['mb_geometry'], float); w_ = g[:, 1] - g[:, 0]
        lv = {}
        for x in np.unique(np.round(w_, 9)):
            s = np.isclose(w_, x); lv[x] = (s.sum(), (w_[s]**3).sum())
        n = P['nx1_out_mb']
        cat = lambda k: np.concatenate([np.asarray(P['mb_data'][k][q]).ravel() for q in range(len(g))]).astype(float)
        rho, p = cat('dens'), cat('press'); m = rho*np.repeat((w_/n)**3, n**3); K = p/rho**GAM/K0 - 1
        dn = [rho >= f*RC for f in (0.01, 0.1)]
        print(f'  last 3D dump t={P["time"]:.2f}: blocks {len(g)}; per level (block width: count, volume): '
              + ', '.join(f'{x:g}: {c} {v:.3g}' for x, (c, v) in sorted(lv.items(), reverse=True)))
        print(f'    dense: M(>=0.01) {m[dn[0]].sum():.6e} K01 {(m[dn[0]]*K[dn[0]]).sum()/m[dn[0]].sum():+.4e} '
              f'K1 {(m[dn[1]]*K[dn[1]]).sum()/m[dn[1]].sum():+.4e} rho_max/rho_c {rho.max()/RC:.6f}')
    try:
        mem = [float(x) for x in re.findall(r'(\d+)\s*$', open(f'{d}/mem.log').read(), re.M)]
        print(f'  GPU memory used samples (xpu-smi, node 1 tile 0): {mem[:3]} ... max {max(mem) if mem else "n/a"}')
    except Exception as e:
        print('  mem.log:', e)
