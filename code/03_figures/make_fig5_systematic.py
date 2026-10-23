# -*- coding: utf-8 -*-
"""Fig.5 (main text): systematic lattice thinning -> deterministic underestimation.
No top title, 300 dpi. Data from experiment_key_stats.json."""
import os, json
import numpy as np
from matplotlib import pyplot as plt

BASE = r"D:\python\try\2026\No_2\SCI_RS_Paper"
stats = json.load(open(os.path.join(BASE, "02_图表", "experiment_key_stats.json"), encoding="utf-8"))
bench = stats["benchmark_I"]
sysd = stats["systematic"]
steps = ["2", "3", "5"]
labels = ["2\u00d72 (25%)", "3\u00d73 (11.1%)", "5\u00d75 (4%)"]
I = [sysd[s]["I"] for s in steps]
bias = [sysd[s]["bias_pct"] for s in steps]

fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=300)
ax.axhline(bench, color="red", ls="--", lw=1.6,
            label="Wall-to-wall benchmark (0.6612)")
ax.plot(range(3), I, "-s", color="#b2182b", lw=2.2, ms=11, mfc="#b2182b")
for i, (v, b) in enumerate(zip(I, bias)):
    ax.annotate(f"\u2212{abs(b):.1f}%", (i, v), textcoords="offset points",
                xytext=(12, -16), fontsize=12, color="#b2182b", weight="bold")
ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=11)
ax.set_xlabel("Systematic lattice thinning (support spacing)", fontsize=11.5)
ax.set_ylabel("Moran's I", fontsize=11.5)
ax.set_ylim(0.34, 0.71)
ax.tick_params(labelsize=10.5)
ax.grid(ls=":", alpha=0.6)
ax.legend(loc="lower left", fontsize=10.5, framealpha=0.9)
fig.savefig(os.path.join(BASE, "02_图表", "Fig5_systematic_thinning.png"),
            bbox_inches="tight", dpi=300)
plt.close(fig)
print("Fig5_systematic_thinning.png done")
