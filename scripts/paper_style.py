"""Shared style for the CFC paper's figures (R-240 item 1; FIGURES_PLAN.md): one colour + one line style per arm family
across all figures, t in M, t_p marked, single (3.4 in) / double (7 in) column widths, vector PDF.
Palette: the dataviz reference categorical palette (8 slots, fixed order, validated in that order); each figure uses a
contiguous run of slots (fig 1: 1-4; fig 5: 5-8), and every family also has its own line style (secondary encoding)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TP = 135.9          # B.1 pericentre (geodesic), M
W1, W2 = 3.4, 7.0   # column widths, in
OUT = '/home/tlam/athenak_tde_project/code/results/paper_figs'
INK, INK2, GRID = '#1f1f1e', '#5f5e58', '#d9d8d2'
FAM = {  # family: (label, colour, line style)
    'valencia':    ('Valencia ($K^{\\rm X}$)',      '#2a78d6', '-'),
    'keff':        ('$K_{\\rm eff}$',               '#eb6834', (0, (5, 1.5))),
    'kcfc':        ('kcfc',                         '#1baf7a', (0, (1, 1.2))),
    'kcfc_bg':     ('kcfc_bg',                    '#eda100', (0, (6, 1.5, 1, 1.5))),
    'energy_only': ('energy only',                  '#e87ba4', '-'),
    'prod_de':     ('production DE',                '#008300', (0, (5, 1.5))),
    'dent':        ('D-ent',                        '#6250d6', (0, (1, 1.2))),
    'dint':        ('D-int',                        '#e34948', (0, (6, 1.5, 1, 1.5))),
}


def setup():
    plt.rcParams.update({
        'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5, 'axes.axisbelow': True,
        'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.3,
        'legend.frameon': False, 'pdf.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})


def line(ax, fam, x, y, label=None, **kw):
    lab, c, ls = FAM[fam]
    kw.setdefault('ls', ls)
    return ax.plot(x, y, color=c, label=lab if label is None else label, **kw)


def mark_tp(ax, tp=TP, label=True):
    ax.axvline(tp, color=INK2, lw=0.7, ls=':', zorder=0)
    if label:
        ax.text(tp, 1.0, ' $t_p$', transform=ax.get_xaxis_transform(), va='top', ha='left', fontsize=7, color=INK2)


def save(fig, name):
    import os
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f'{OUT}/{name}.pdf')
    fig.savefig(f'{OUT}/{name}.png', dpi=200)
