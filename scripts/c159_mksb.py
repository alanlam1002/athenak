#!/usr/bin/env python3
"""c159 (R-188 item 3, Step B): dyn_grmhd decks on the FIXED analytic Kerr-Schild metric (a = 0, M = 1) from Step A's
SA_L{2,3,4}.par (c158_decks: the mesh, Michel parameters k_adi/r_crit, time, outputs; ppmx + HLLE + FOFC, RK3, cfl 0.4, as B.2
and Step A). The GR hydro module's <coord>/<hydro> blocks are replaced by dyn_grmhd's: <adm> alone (no <cfc>/<z4c>:
ADM::SetADMVariablesToKerrSchild each stage), <dyn_grmhd> metric_terms = kerr_schild (the face metric = the closed form at the
face, full gamma_ij; sources difference the face values), lapse excision at excise_lapse = 0.67 (r_KS < 1.63 M).
Arms: V (valencia), T (killing_t1), A (killing_ta), the latter two with killing_weights = kerr_schild; KCLOSE on for T/A.
    c159_mksb.py OUTDIR [tau_budget_every]   -> OUTDIR/SB{V,T,A}_L{2,3,4}.par"""
import os
import re
import sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'c158_decks')
out = sys.argv[1]
tbe = int(sys.argv[2]) if len(sys.argv) > 2 else 0
os.makedirs(out, exist_ok=True)

COORD = """<coord>
general_rel = true
m = 1.0
a = 0.0
excise = true
excision_scheme = lapse     # c159: alpha_KS <= 0.67 (r_KS <= 1.63 M; the horizon is r = 2 M)
excise_lapse = 0.67
excision_mask_refresh = true   # c159: the masks from the first stage (as the TDE decks)
dexcise = 1.0e-24
texcise = 1.0e-22
"""
MHD = """<mhd>
eos = ideal
dyn_eos = ideal
dyn_error = reset_floor
reconstruct = ppmx
rsolver = hlle
fofc = true
gamma = 1.6666666666666667
dfloor = 1.0e-24
tfloor = 1.0e-22
dthreshold = 1.02
dyn_scratch = 1
enforce_maximum = false
"""
KILL = """energy_variable = {ev}
killing_weights = kerr_schild
mass_closure = true
mass_closure_full = true
mass_closure_mstar = 1.0
mass_closure_tol = 1.0e-10
"""
TB = """tau_budget_every = {n}
tau_budget_bins = shells
tau_budget_tmin = 100.0     # C-148's window, t = 100 -> 120
"""


def blocks(txt):
    """[(name, body)] in file order"""
    parts = re.split(r'(?m)^<([A-Za-z0-9_]+)>\s*$', txt)
    return [(parts[q], parts[q + 1]) for q in range(1, len(parts), 2)]


for lv in (2, 3, 4):
    base = open(f'{SRC}/SA_L{lv}.par').read()
    keep = [(b, body) for b, body in blocks(base) if b not in ('coord', 'hydro')]
    for arm, ev in (('V', 'valencia'), ('T', 'killing_t1'), ('A', 'killing_ta')):
        s = (f'# c159 deck (R-188 item 3, Step B): {SRC}/SA_L{lv}.par with dyn_grmhd on the fixed KS metric '
             f'(c159_mksb.py), arm {arm} = {ev}\n')
        for b, body in keep:
            if b == 'job':
                body = re.sub(r'(?m)^basename\s*=.*$', 'basename = ks_michel', body)
            body = re.sub(r'(?m)^(variable\s*=\s*)hydro_w', r'\1mhd_w', body)   # dyn_grmhd's primitives
            s += f'<{b}>' + body
        s += COORD + MHD
        if ev != 'valencia':
            s += KILL.format(ev=ev)
            if tbe > 0:
                s += TB.format(n=tbe)
        elif tbe > 0:
            s += TB.format(n=tbe)
        s += '<adm>\n<dyn_grmhd>\nmetric_terms = kerr_schild\n'
        open(f'{out}/SB{arm}_L{lv}.par', 'w').write(s)
        print(f'{out}/SB{arm}_L{lv}.par')
