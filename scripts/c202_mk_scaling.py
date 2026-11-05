#!/usr/bin/env python3
"""c202 (R-236 item 1): the strong-scaling decks. From c201's B16.par / F16.par (the P18 production deck, 16^3 blocks,
finest dx 1/16 / 1/32), only max_nmb_per_rank (sized ~1.5x the t = 0 blocks per rank: 2220 / 4124) and
phase_timers_every = 25 (the window 50 -> end) change, and the restart output is dropped (quota; timing only).  Arms <B16|F16>_n<nodes>, 12 ranks per node.
    c202_mk_scaling.py C201DIR OUTDIR"""
import re, sys, math
src, out = sys.argv[1], sys.argv[2]
def sub(s, block, key, val, add=False):
    pat = re.compile(rf'(<{block}>[^<]*?^\s*{key}\s*=\s*)(\S+)', re.M | re.S)
    s, c = pat.subn(lambda m: m.group(1) + str(val), s, count=1)
    if c == 0 and add:
        s, c = re.subn(rf'(<{block}>\n)', rf'\g<1>{key} = {val}\n', s, count=1)
    assert c == 1, (block, key)
    return s
NMB0 = {'B16': 2220, 'F16': 4124}
for arm in ('B16', 'F16'):
    s0 = open(f'{src}/{arm}.par').read()
    for nodes in (1, 2, 4, 8):
        mx = int(math.ceil(1.5*NMB0[arm]/(12*nodes)/10.0))*10
        s = sub(s0, 'mesh_refinement', 'max_nmb_per_rank', mx)
        s = sub(s, 'time', 'phase_timers_every', 25, add=True)
        s, c = re.subn(r'<output4>\nfile_type   = rst[^<]*', '', s); assert c == 1   # no restarts (2 x 11 GB per arm; timing only)
        s = f'# c202 strong-scaling arm {arm}_n{nodes}: {12*nodes} ranks, max_nmb_per_rank {mx}, phase_timers_every 25\n' + s
        open(f'{out}/{arm}_n{nodes}.par', 'w').write(s)
        print(arm, nodes, mx)
