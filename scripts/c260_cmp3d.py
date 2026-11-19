#!/usr/bin/env python3
"""c260: B0 vs B1 from c260_rst.py outputs (R-257 2(c) metrics 1, 2, 3, 5) + a profile figure.
    c260_cmp3d.py CMP3D_DIR"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = sys.argv[1]
L = lambda n: dict(np.load(f'{D}/{n}.npz'))
T9 = L('T900')
print(f't=900 (shared start): f_bound {T9["fb"]:.6f} q10/50/90 {T9["q10"]:+.4f} {T9["q50"]:+.4f} {T9["q90"]:+.4f} '
      f'K/K0 zone {T9["Kzone"]:.3f}')
rel = lambda a, b: (b - a) / abs(a)
P = lambda ok: 'PASS' if ok else 'FAIL'
fig, ax = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
for j, c in enumerate(('T1000', 'T1100', 'END')):
    a, b = L(f'B0_{c}'), L(f'B1_{c}')
    print(f'\n## t = {float(a["t"]):.1f}  (nmb B0 {int(a["nmb"])}, B1 {int(b["nmb"])})')
    r = rel(a['fb'], b['fb'])
    print(f'1. f_bound B0 {a["fb"]:.6f} B1 {b["fb"]:.6f} rel {r:+.3%} {P(abs(r) < 0.002)} (0.2%)')
    for q in ('q10', 'q50', 'q90'):
        r = rel(a[q], b[q])
        print(f'   {q} E/dE B0 {a[q]:+.5f} B1 {b[q]:+.5f} rel {r:+.3%} {P(abs(r) < 0.005)} (0.5%)')
    for R in (50, 100, 150):
        k = f'fwhm{R}'
        if k in a and k in b:
            r1, r2 = rel(a[k], b[k]), rel(a[f'peak{R}'], b[f'peak{R}'])
            print(f'2. R={R}: phi B0 {a[f"phi{R}"]:.0f} B1 {b[f"phi{R}"]:.0f}; FWHM_z B0 {a[k]:.4f} B1 {b[k]:.4f} rel {r1:+.2%} '
                  f'{P(abs(r1) < 0.05)}; peak rho B0 {a[f"peak{R}"]:.4e} B1 {b[f"peak{R}"]:.4e} rel {r2:+.2%} {P(abs(r2) < 0.05)} (5%)')
        else:
            print(f'2. R={R}: bound stream missing in {"B0" if k not in a else ""} {"B1" if k not in b else ""}')
    r = rel(a['Kzone'], b['Kzone'])
    print(f'3. K/K0 bound stream r<120: B0 {a["Kzone"]:.4f} B1 {b["Kzone"]:.4f} rel {r:+.2%} {P(abs(r) < 0.10)} (10%); '
          f'M B0 {a["Mzone"]:.4e} B1 {b["Mzone"]:.4e}')
    rc = 0.5 * (a['rb'][1:] + a['rb'][:-1])
    for X, lab in ((a, 'B0'), (b, 'B1')):
        m = X['pM'] > 0
        K = np.where(m, X['pK'] / np.where(m, X['pM'], 1), np.nan)
        lv = np.where(m, X['pL'] / np.where(m, X['pM'], 1), np.nan)
        ax[0, j].semilogy(rc, np.where(m, X['pR'], np.nan), label=lab)
        ax[1, j].semilogy(rc, K, label=lab)
        ax[2, j].plot(rc, lv, label=lab)
    # 5: step test across r = 120: ratio B1/B0 of K in the 4 bins on each side
    m0, m1 = a['pM'] > 0, b['pM'] > 0
    K0_, K1_ = a['pK'] / np.where(m0, a['pM'], 1), b['pK'] / np.where(m1, b['pM'], 1)
    ins, out = (rc > 110) & (rc < 120) & m0 & m1, (rc > 120) & (rc < 130) & m0 & m1
    if ins.any() and out.any():
        print(f'5. K B1/B0 in 110-120: {np.mean(K1_[ins] / K0_[ins]):.3f}, in 120-130: {np.mean(K1_[out] / K0_[out]):.3f}; '
              f'peak rho B1/B0 in: {np.mean(b["pR"][ins] / a["pR"][ins]):.3f}, out: {np.mean(b["pR"][out] / a["pR"][out]):.3f}')
    ax[0, j].set_title(f't = {float(a["t"]):.0f}')
    for i in range(3):
        ax[i, j].axvline(120, color='k', ls=':', lw=1)
ax[0, 0].set_ylabel('peak rho (bound stream)'); ax[1, 0].set_ylabel('<K/K0>_m'); ax[2, 0].set_ylabel('<level>_m')
for j in range(3):
    ax[2, j].set_xlabel('r_BH [M]')
ax[0, 0].legend()
fig.suptitle('c260 bind-policy pilot: bound stream (E<0, 1e-6<=rho/rho_c<0.01) radial profiles; R_zone = 120 dotted')
fig.tight_layout()
fig.savefig(f'{D}/c260_profiles.png', dpi=110)
print(f'\nfigure {D}/c260_profiles.png')
