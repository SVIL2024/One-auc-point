# -*- coding: utf-8 -*-
"""Figure 1 for the CAFE manuscript: census, exchange rate, reordering.

Design width = 5.42 in = 390 pt = the elsarticle preprint \textwidth, so the
figure is placed 1:1 and the 7-8 pt labels stay 7-8 pt in the PDF.
All numbers are the ones reported in the paper (Table 1, Corollary 1,
Table 2 / Section 6.6), so the figure cannot drift from the text.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.6,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "axes.titlesize": 7.8,
    "axes.labelsize": 7.2,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

OUT = sys.argv[1] if len(sys.argv) > 1 else r"E:\paper\drafts\2026\CAFE\fig1.pdf"

C_UC, C_XD, C_GREY, C_ACC = "#1f4e79", "#b35c00", "#666666", "#a01020"

ucf = (0.6317, 0.00388, 0.3645)
xd = (0.6985, 0.00071, 0.3008)

wB_ucf, wB_xd = 0.003880, 0.000710
sigma_theta = 0.0048

# (theta, B|z) of the eleven UCF-Crime checkpoints, as in Table 2 / Sec. 6.6
CKPT = [
    ("base 234", 0.8800, 0.6729),
    ("m6 789", 0.8778, 0.7230),
    ("base 789", 0.8772, 0.7169),
    ("cons0", 0.8759, 0.6744),
    ("attr", 0.8724, 0.6545),
    ("core14", 0.8717, 0.6728),
    ("m6 555", 0.8717, 0.6604),
    ("base 555", 0.8705, 0.6688),
    ("core8g1", 0.8684, 0.6621),
    ("res 234", 0.8682, 0.7093),
    ("m6-shuf", 0.8616, 0.6786),
]

fig = plt.figure(figsize=(5.42, 2.40))
gs = fig.add_gridspec(1, 3, width_ratios=[1.06, 1.0, 1.22], wspace=0.62)

# --------------------------------------------------------------- (a) census
ax = fig.add_subplot(gs[0, 0])
for k, (name, w, col) in enumerate([("UCF", ucf, C_UC), ("XD", xd, C_XD)]):
    y = 1 - k
    left = 0.0
    for frac, lab, alpha in ((w[0], r"$\mathcal{A}$", 1.0),
                             (w[2], r"$\mathcal{C}$", 0.45)):
        ax.barh(y, frac, left=left, height=0.50, color=col, alpha=alpha,
                edgecolor="white", linewidth=0.5)
        ax.text(left + frac / 2, y, lab, ha="center", va="center",
                fontsize=7.4, color="white")
        left += frac
    ax.text(-0.03, y, name, ha="right", va="center", fontsize=7.0)
ax.set_xlim(0, 1.0)
ax.set_ylim(-0.75, 1.90)
ax.set_yticks([])
ax.set_xlabel("share of scored pairs")
ax.set_title("(a) the blocks are unequal", loc="left")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.annotate(r"$\mathcal{B}$: 0.388% (UCF)" + "\n" + r"0.071% (XD)",
            xy=(0.65, 1.27), xytext=(0.0, 1.86), fontsize=6.4, color=C_ACC,
            ha="left", va="top", annotation_clip=False, linespacing=1.2,
            arrowprops=dict(arrowstyle="-|>", lw=0.55, color=C_ACC,
                            shrinkA=0, shrinkB=1))

# -------------------------------------------------------- (b) exchange rate
ax = fig.add_subplot(gs[0, 1])
db = np.linspace(0, 0.25, 250)
ax.plot(db, wB_ucf * db, color=C_UC, lw=1.1)
ax.plot(db, wB_xd * db, color=C_XD, lw=1.1)
ax.axhline(sigma_theta, color=C_GREY, ls=(0, (4, 2)), lw=0.8)
ax.text(0.006, sigma_theta + 0.0004, r"one seed $\sigma(\theta)$",
        ha="left", fontsize=6.6, color=C_GREY)
ax.set_xlim(0, 0.25)
ax.set_ylim(0, 0.0096)
ax.set_xlabel(r"gain in localization, $\Delta B$")
ax.set_ylabel(r"gain in reported AUC, $\Delta\theta$")
ax.set_title("(b) the exchange rate", loc="left")
ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
ax.text(0.05, 0.26, r"UCF: $\omega_B=0.0039$", color=C_UC, fontsize=6.5,
        transform=ax.transAxes, va="center")
ax.text(0.05, 0.15, r"XD: $\omega_B=0.0007$", color=C_XD, fontsize=6.5,
        transform=ax.transAxes, va="center")
ax.text(0.05, 0.95, r"$+0.01$ AUC needs" + "\n" + r"$\Delta B=2.6$ (UCF)",
        color=C_ACC, fontsize=6.5, transform=ax.transAxes, va="top", ha="left",
        linespacing=1.15)

# --------------------------------------------------------------- (c) ranks
ax = fig.add_subplot(gs[0, 2])
order_t = sorted(range(len(CKPT)), key=lambda i: -CKPT[i][1])
order_b = sorted(range(len(CKPT)), key=lambda i: -CKPT[i][2])
rank_t = {i: r for r, i in enumerate(order_t)}
rank_b = {i: r for r, i in enumerate(order_b)}
n_moved = 0
for i, (name, th, bz) in enumerate(CKPT):
    y0, y1 = rank_t[i], rank_b[i]
    moved = abs(y0 - y1) >= 3
    n_moved += moved
    ax.plot([0, 1], [y0, y1], color=C_ACC if moved else "#c2c2c2",
            lw=1.1 if moved else 0.6, zorder=3 if moved else 2)
    ax.plot([0], [y0], "o", ms=2.0, color=C_GREY, mew=0)
    ax.plot([1], [y1], "o", ms=2.0, color=C_GREY, mew=0)
    if moved:
        ax.text(-0.06, y0, name, ha="right", va="center", fontsize=6.4, color=C_ACC)
        ax.text(1.06, y1, name, ha="left", va="center", fontsize=6.4, color=C_ACC)
ax.set_xlim(-0.80, 1.80)
ax.set_ylim(len(CKPT) + 0.80, -0.75)
ax.set_xticks([0, 1])
ax.set_xticklabels([r"$\theta$", r"$B|z$"], fontsize=7.2)
ax.set_yticks([])
for s in ("left", "right", "top"):
    ax.spines[s].set_visible(False)
ax.set_title("(c) re-ranking by $B|z$", loc="left")
ax.text(0.5, len(CKPT) + 0.45, "%d of 11 move by $\\geq$ 3 places" % n_moved,
        ha="center", fontsize=6.6, color=C_ACC)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT)
fig.savefig(OUT.replace(".pdf", ".png"), dpi=240)
print("wrote", OUT, "moved:", n_moved)
