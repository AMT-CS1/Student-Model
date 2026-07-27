"""
Visualisasi Trajektori Knowledge Concept (KC) -- gaya paper HELP-DKT.

Menampilkan untuk tiap KC (baris):
  - pita latar per attempt: HIJAU = KC dipakai & benar, MERAH = KC dipakai & salah,
    ABU = KC tidak terlibat di soal (dari Q-matrix).
  - garis trajektori mastery (ability) yang diwarnai per level:
    hijau = high, kuning = medium, oranye = low.
Baris paling bawah: trajektori attempt (checkmark = benar, x = salah),
dikelompokkan per soal (C-1 .. C-6).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

CSV_PATH = 'inference_dummy_result.CSV'
PMATRIX_PATH = '../ModelInput/P-matrix-out.CSV'   # KC yang diterapkan siswa per attempt

CONCEPT_NAMES = {
    'CO': 'Constants', 'VA': 'Variables', 'OP': 'Operators', 'ST': 'Strings',
    'EX': 'Expressions', 'LI': 'Lists', 'TU': 'Tuples', 'DI': 'Dictionaries',
    'CD': 'Conditionals', 'IO': 'Input / Output',
}
ROW_ORDER = ['CO', 'VA', 'OP', 'ST', 'EX', 'LI', 'TU', 'DI', 'CD', 'IO']
CSV_ORDER = ['ST', 'VA', 'OP', 'EX', 'IO', 'LI', 'CO', 'TU', 'DI', 'CD']

PROBLEM_QMATRIX = {
    '362': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    '371': [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    '406': [0, 1, 1, 1, 1, 0, 1, 0, 0, 0],
    '417': [0, 1, 0, 1, 1, 0, 0, 1, 0, 0],
    '449': [1, 1, 1, 1, 0, 0, 1, 0, 0, 1],
    '472': [1, 1, 0, 1, 1, 0, 1, 0, 1, 0],
}

C_HIGH, C_MED, C_LOW = '#2e9e3f', '#f2c200', '#e8892b'
B_CORRECT, B_WRONG, B_NA = '#7ec87e', '#f08a8a', '#d9d9d9'
GRID_C = '#9a9a9a'


def level_color(v):
    if v >= 0.60:
        return C_HIGH
    if v >= 0.35:
        return C_MED
    return C_LOW



def _smooth(y, sigma):
    """Gaussian smoothing 1-D tanpa scipy."""
    if sigma <= 0:
        return y
    rad = max(1, int(sigma * 3))
    k = np.exp(-(np.arange(-rad, rad + 1) ** 2) / (2 * sigma ** 2))
    k /= k.sum()
    yp = np.pad(y, rad, mode='edge')
    return np.convolve(yp, k, mode='valid')


def load_pmatrix(path=PMATRIX_PATH):
    """filename attempt -> vektor P (10 KC, urutan CSV_ORDER)."""
    import csv
    P = {}
    with open(path, encoding='utf-8-sig') as fh:
        for row in csv.reader(fh):
            if row and row[0]:
                P[row[0]] = [int(x) for x in row[1:11]]
    return P


def load(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path, encoding='utf-8-sig', skiprows=1)
    df = df.dropna(subset=['student']).reset_index(drop=True)
    for c in ['student', 'step', 'actual_next', 'pred_label']:
        df[c] = df[c].astype(int)
    return df


def build_attempts(df):
    files = df['attempt_file'].tolist() + [df['next_file'].iloc[-1]]
    attempts = []
    for f in files:
        pid = f.split('_')[1]
        correct = f.split('_')[0].startswith('c')
        attempts.append((pid, correct, f))
    return attempts


def plot_trajectory(df, save=None, shift=1, smooth=1.1, dens=12, pmatrix=None):
    attempts = build_attempts(df)
    if pmatrix is None:
        pmatrix = load_pmatrix()
    n = len(attempts)
    qidx = {k: i for i, k in enumerate(CSV_ORDER)}
    ability = df[['ability_' + k for k in CSV_ORDER]].to_numpy()
    x_ab = df['step'].to_numpy() + shift   # geser garis maju 'shift' step

    prob_order, seg = [], []
    prev, start = None, 0
    for i, (pid, _, _f) in enumerate(attempts):
        if pid != prev:
            if prev is not None:
                seg.append((prev, start, i))
            prob_order.append(pid); prev, start = pid, i
    seg.append((prev, start, n))

    row_h, gap = 1.0, 0.28
    n_rows = len(ROW_ORDER)
    fig, ax = plt.subplots(figsize=(17.5, 8))

    for r, kc in enumerate(ROW_ORDER):
        y0 = (n_rows - 1 - r) * (row_h + gap)
        yc = y0 + row_h / 2
        ci = qidx[kc]
        for a, (pid, correct, fname) in enumerate(attempts):
            q = PROBLEM_QMATRIX[pid][ci]                 # KC dituntut soal?
            p = pmatrix.get(fname, [0]*10)[ci]           # KC diterapkan siswa?
            if q == 0:
                color = B_NA                             # tidak terlibat
            elif p == 1:
                color = B_CORRECT                        # Q=1 & P=1 -> hijau
            else:
                color = B_WRONG                          # Q=1 & P=0 -> merah
            ax.add_patch(plt.Rectangle((a - 0.5, y0), 1.0, row_h,
                                       facecolor=color, edgecolor='white',
                                       linewidth=0.3, zorder=1))
        av = _smooth(ability[:, ci], smooth)            # haluskan nilai ability
        if dens and dens > 1:                            # rapatkan titik -> kurva mulus
            xd = np.linspace(x_ab[0], x_ab[-1], (len(x_ab) - 1) * dens + 1)
            ad = np.interp(xd, x_ab, av)
        else:
            xd, ad = x_ab, av
        yv = y0 + 0.12 + ad * (row_h - 0.24)
        pts = np.array([xd, yv]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        cols = [level_color(v) for v in ad[:-1]]
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=2.6,
                                         zorder=3, capstyle='round'))
        ax.text(-1.2, yc, CONCEPT_NAMES[kc], ha='right', va='center', fontsize=11)

    top = n_rows * (row_h + gap) - gap
    for k, (pid, s, e) in enumerate(seg, 1):
        if s > 0:
            ax.axvline(s - 0.5, color=GRID_C, ls=':', lw=1, zorder=4)
        ax.text((s + e) / 2 - 0.5, -1.35, f'C-{k}', ha='center', va='top',
                fontsize=11, color='#333')
    ax.axvline(n - 0.5, color=GRID_C, ls=':', lw=1, zorder=4)

    y_mark = -0.75
    for a, (pid, correct, fname) in enumerate(attempts):
        if correct:
            ax.text(a, y_mark, u'✓', color=C_HIGH, ha='center', va='center', fontsize=12)
        else:
            ax.text(a, y_mark, u'✗', color='#d33', ha='center', va='center', fontsize=11)
    ax.text(-1.2, y_mark, 'Attempt', ha='right', va='center', fontsize=10, style='italic')

    ax.set_xlim(-6, n + 0.5)
    ax.set_ylim(-1.7, top + 0.2)
    ax.axis('off')

    leg1 = [Line2D([0], [0], color=C_HIGH, lw=3, label='high level'),
            Line2D([0], [0], color=C_MED, lw=3, label='medium level'),
            Line2D([0], [0], color=C_LOW, lw=3, label='low level')]
    leg2 = [Patch(facecolor=B_WRONG, label='applied incorrectly'),
            Patch(facecolor=B_CORRECT, label='applied correctly'),
            Patch(facecolor=B_NA, label='not involved')]
    leg3 = [Line2D([0], [0], marker=r'$\checkmark$', color='w', markerfacecolor=C_HIGH,
                   markeredgecolor=C_HIGH, markersize=11, label='correct'),
            Line2D([0], [0], marker='x', color='w', markeredgecolor='#d33',
                   markersize=9, label='incorrect')]
    l1 = ax.legend(handles=leg1, title='Student Abilities on\nConcept Level',
                   loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True, fontsize=9)
    l2 = ax.legend(handles=leg2, title='Concept Trajectory',
                   loc='upper left', bbox_to_anchor=(1.01, 0.62), frameon=True, fontsize=9)
    l3 = ax.legend(handles=leg3, title='Attempt Trajectory',
                   loc='upper left', bbox_to_anchor=(1.01, 0.34), frameon=True, fontsize=9)
    ax.add_artist(l1); ax.add_artist(l2)
    for lg in (l1, l2, l3):
        lg.get_title().set_fontsize(9); lg.get_title().set_fontweight('bold')

    fig.suptitle(u'Trajektori Knowledge Concept — Mahasiswa 46724 (57 attempt)',
                 x=0.40, y=0.99, fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 0.78, 0.96])
    if save:
        fig.savefig(save, dpi=160, bbox_inches='tight')
    return fig


if __name__ == '__main__':
    df = load()
    plot_trajectory(df, save='kc_trajectory.png')
    print('saved kc_trajectory.png')
