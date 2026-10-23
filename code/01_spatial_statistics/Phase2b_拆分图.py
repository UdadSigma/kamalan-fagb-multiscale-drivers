# -*- coding: utf-8 -*-
"""Fig.4 拆分为两张单独图片，300 dpi：
Fig4a_random_subsampling.png / Fig4b_precision_loss.png
数据与样式沿用 Phase2。"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "02_图表"))
I0 = 0.661154

d = pd.read_csv(os.path.join(HERE, "subsampling_moran_results.csv"))
rates = sorted(d["rate_pct"].unique())
data_by_rate = [d.loc[d["rate_pct"] == r, "moran_I"].values for r in rates]
sds = [v.std() for v in data_by_rate]
maxabs = [max(v.max() - I0, I0 - v.min()) / I0 * 100 for v in data_by_rate]

# ---------- Fig.4a ----------
fig, ax1 = plt.subplots(figsize=(5.6, 4.4), dpi=300)
bp = ax1.boxplot(data_by_rate, positions=np.log10(rates), widths=0.45,
                 patch_artist=True, showfliers=True,
                 flierprops=dict(marker="o", ms=2.5, color="#999999", alpha=0.6),
                 medianprops=dict(color="#C00000", lw=1.4))
for p in bp["boxes"]:
    p.set(facecolor="#BDD7EE", edgecolor="#2E75B6", lw=1)
ax1.axhline(I0, color="red", ls="--", lw=1.2,
            label=f"Wall-to-wall benchmark ($I$ = {I0:.4f})")
ax1.set_xticks(np.log10(rates))
ax1.set_xticklabels([str(r) for r in rates])
ax1.set_xlabel("Sampling fraction (%)", fontsize=10)
ax1.set_ylabel("Moran's $I$ estimates (50 replicates)", fontsize=10)
ax1.set_title("(a) Random subsampling: unbiased but increasingly imprecise", fontsize=10.5)
ax1.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
ax1.grid(axis="y", ls=":", alpha=0.5)
ax1.tick_params(labelsize=9)
fig.tight_layout()
p1 = os.path.join(OUT, "Fig4a_random_subsampling.png")
fig.savefig(p1, dpi=300, bbox_inches="tight"); plt.close(fig)
print("saved:", p1)

# ---------- Fig.4b ----------
fig, ax2 = plt.subplots(figsize=(5.6, 4.4), dpi=300)
ax2.plot(np.log10(rates), sds, "o-", color="#2E75B6", lw=1.6, label="SD of $I$ estimates")
ax2.plot(np.log10(rates), [I0 * 0.124 / r for r in rates], ls=":", color="#2E75B6", alpha=0.7,
         label=r"$\propto 1/f$ reference")
ax2.set_xticks(np.log10(rates)); ax2.set_xticklabels([str(r) for r in rates])
ax2.set_xlabel("Sampling fraction (%)", fontsize=10)
ax2.set_ylabel("SD of Moran's $I$ estimates", fontsize=10)
ax2.set_yscale("log")
ax2b = ax2.twinx()
ax2b.bar(np.log10(rates), maxabs, width=0.35, color="#F4B183", alpha=0.75,
         label="Max |bias| of single replicates (%)")
ax2b.set_ylabel("Max |relative bias| of single replicates (%)", color="#C55A11", fontsize=10)
ax2b.tick_params(axis="y", labelcolor="#C55A11", labelsize=9)
ax2.set_title("(b) Precision loss grows as $\\sim 1/f$", fontsize=10.5)
h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax2b.get_legend_handles_labels()
ax2.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=8.5, framealpha=0.95)
ax2.grid(axis="y", ls=":", alpha=0.5)
ax2.tick_params(labelsize=9)
fig.tight_layout()
p2 = os.path.join(OUT, "Fig4b_precision_loss.png")
fig.savefig(p2, dpi=300, bbox_inches="tight"); plt.close(fig)
print("saved:", p2)
