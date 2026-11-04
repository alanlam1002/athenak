#!/usr/bin/env python3
"""c190 (R-225 item 1) CPU 1D decks (built-in pgens; serial strict builds athena_dent_bi / athena_mrg_bi).
  flat/  the flat DE unit tests (R-103 set: col_M1.2/1.5/2/3, hom_c0.05/0.1/0.2/0.4, sod; dual_energy_tests_P2 with the
         production keys, covariant) and C-107's moving shocks (c108 base: M 2/1.2 x vb 0/0.1/0.2/0.3, nx 512):
         *_off (DE off), *_leg (production DE, legacy), *_cnd (conditioning). Legacy on both binaries = default bitwise.
  cold/  the cold boosted colliding flows (C-103's construction: two Mach-M shocks in the collision frame, boosted by vb),
         rho = 1, P0 in {3.2e-6, 1.6e-5} (upstream q_ad ~ 1e-4 / ~5e-4 at vb = 0.3), M in {1.5, 3, 6, 20},
         vb in {0, 0.3}, nx 512, arms off / leg / cnd; plus the ghost/owner pair: cnd M3 vb0.3 P3.2e-6 with
         meshblock nx1 = 512 (one block) vs 64 (eight blocks)."""
import math, os, re
P2 = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/dual_energy_tests_P2'
C108 = '/lus/flare/projects/CompactBinaryMerger/tlam/athenak_run/cfc/c108_demech'
PROD = ("dual_energy_flux = hll\nscalar_flux_hll = true\ndual_energy_recon_rhoy = true\nfofc_scalar_consistent = true\n"
        "cold_recovery = true\nfofc_cold_recovery = true\ndual_energy_diag = true\ndual_energy_div = covariant\n")
COND = "dual_energy_mode = conditioning\ndual_energy_q_eta1 = 3.0e-4\ndual_energy_q_eta2 = 3.0e-3\ndual_energy_q_delta = 0.1\n"
GAM = 5.0/3.0
n = 0
def put(d, nm, s):
    global n
    os.makedirs(d, exist_ok=True); open(f'{d}/{nm}.athinput', 'w').write(s); n += 1
def arms(d, base, s_on):
    assert re.search(r'^dual_energy = true$', s_on, re.M), base
    so = s_on.replace('dual_energy = true\n', 'dual_energy = false\n', 1)
    so = re.sub(r'^dual_energy_diag\s*=\s*true', 'dual_energy_diag = false', so, flags=re.M)
    so = re.sub(r'\n<output\d+>\s*\nfile_type\s*=\s*\S+\s*\nvariable\s*=\s*mhd_de_diag[^<]*', '\n', so)
    assert 'mhd_de_diag' not in so, base
    put(d, base + '_off', so)
    put(d, base + '_leg', s_on)
    put(d, base + '_cnd', s_on.replace('dual_energy = true\n', 'dual_energy = true\n' + COND, 1))
# flat unit tests
for t in [f'col_M{m}' for m in ('1.2', '1.5', '2', '3')] + [f'hom_c{c}' for c in ('0.05', '0.1', '0.2', '0.4')] + ['sod']:
    s = open(f'{P2}/{t}_p3.athinput').read().replace('dual_energy_shock_persist = 3\n', '')
    assert 'dual_energy_div' not in s
    s = s.replace('dual_energy = true\n', 'dual_energy = true\n' + PROD, 1)
    dt = re.search(r'<output6>.*?dt\s*=\s*(\S+)', s, re.S).group(1)
    s = s.rstrip('\n') + f'\n\n<output7>\nfile_type = tab\nvariable = mhd_de_diag\ndata_format = %24.17e\ndt = {dt}\n'
    arms('flat', t, s)
for M in ('2', '1.2'):
    for vb in ('0.0', '0.1', '0.2', '0.3'):
        s = open(f'{C108}/col_M{M}_vb{vb}_n512_base.athinput').read()
        arms('flat', f'c107_M{M}_vb{vb}', s)
# cold boosted colliding flows
tpl = open(f'{C108}/col_M2_vb0.0_n512_base.athinput').read()
TLIM = 1.35
for P0 in (3.2e-6, 1.6e-5):
    h = 1 + GAM/(GAM - 1)*P0
    cs = math.sqrt(GAM*P0/h)
    for M in (1.5, 3.0, 6.0, 20.0):
        d = 2*cs/(GAM + 1)*(M - 1/M)
        for vb in (0.0, 0.3):
            vl, vr = (vb + d)/(1 + vb*d), (vb - d)/(1 - vb*d)
            s = tpl
            # c190 v2: run to the warm test's c_s t (0.1 x 1.35), i.e. t = 0.135/c_s, on a domain widened by vb t
            # (the collision starts at x = 20 and moves with vb), dx = 1/256 as the warm nx 512 on [19, 21]
            tl = 0.135/cs; x1 = 21.0 + vb*tl; nxx = int(round(256*(x1 - 19.0)))
            for k, v in (('pl', P0), ('pr', P0), ('ul', vl), ('ur', vr), ('xshock', 20.0), ('x1max', x1), ('tlim', tl)):
                s, c = re.subn(rf'^{k}\s*=\s*\S+', f'{k} = {v!r}', s, count=1, flags=re.M); assert c == 1, k
            s = re.sub(r'(<mesh>.*?)nx1\s*=\s*\S+', lambda m: m.group(1) + f'nx1 = {nxx}', s, count=1, flags=re.S)
            s = re.sub(r'(<meshblock>.*?)nx1\s*=\s*\S+', lambda m: m.group(1) + f'nx1 = {nxx}', s, count=1, flags=re.S)
            s = re.sub(r'(<output\d+>\s*\nfile_type\s*=\s*(?:tab|hst)[^<]*?)dt\s*=\s*\S+', lambda m: m.group(1) + f'dt = {tl/10!r}', s, flags=re.S)
            # the tracer's kmax must cover K = P0 (k_1g: Y = (K/kmax)^(1/gamma)); the template's 1.0 does
            arms('cold', f'cold_P{P0:.1e}_M{M:g}_vb{vb:.1f}', s)
s = open('cold/cold_P3.2e-06_M3_vb0.3_cnd.athinput').read()
s2 = re.sub(r'(<meshblock>.*?)nx1\s*=\s*\S+', lambda m: m.group(1) + 'nx1 = 64', s, count=1, flags=re.S)  # v1 ghost pair kept below
assert s2 != s
put('cold', 'ghost_mb64_cold_P3.2e-06_M3_vb0.3_cnd', s2)
put('cold', 'ghost_mb512_cold_P3.2e-06_M3_vb0.3_cnd', s)
print(n, 'inputs')
# building shock (WHAM's case): a cold, boosted, finite-amplitude sound wave (linear_wave, wave_flag 0, periodic x in [0, 1])
# that steepens into a shock at t_b ~ 1/((gamma+1)/2 amp c_s k) ~ 100 (amp 0.5, P0 3.2e-6) and keeps strengthening
lw = open('../lw_test.athinput').read()
for P0 in (3.2e-6, 1.6e-5):
    for nx in (128, 256):
        s = lw
        for k, v in (('amp', '0.5'), ('pgas', repr(P0))):
            s, c = re.subn(rf'^{k}\s*=\s*\S+', f'{k} = {v}', s, count=1, flags=re.M); assert c == 1, k
        s = re.sub(r'(<mesh>.*?)nx1\s*=\s*\S+', lambda m: m.group(1) + f'nx1 = {nx}', s, count=1, flags=re.S)
        s = re.sub(r'(<meshblock>.*?)nx1\s*=\s*\S+', lambda m: m.group(1) + f'nx1 = {nx}', s, count=1, flags=re.S)
        s = re.sub(r'^tlim\s*=.*$', 'tlim = 400.0', s, flags=re.M)
        s = re.sub(r'(<output\d+>\s*\nfile_type\s*=\s*(?:tab|hst)[^<]*?)dt\s*=\s*\S+', lambda m: m.group(1) + 'dt = 10.0', s, flags=re.S)
        arms('build', f'lw_P{P0:.1e}_n{nx}', s)
print(n, 'inputs incl. build/')
