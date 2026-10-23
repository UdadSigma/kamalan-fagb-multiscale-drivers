# -*- coding: utf-8 -*-
"""Phase 2：汇总统计 + 主文图 Fig.4（双面板）+ 补充图 Fig.S1（系统抽样）
输出至 ../02_图表/
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "02_图表"))
os.makedirs(OUT, exist_ok=True)
I0 = 0.661154

d = pd.read_csv(os.path.join(HERE, "subsampling_moran_results.csv"))
d["bias_pct"] = (d["moran_I"] - I0) / I0 * 100
s = pd.read_csv(os.path.join(HERE, "systematic_sampling_results.csv"))
s["bias_pct"] = (s["moran_I"] - I0) / I0 * 100

# ---------- 汇总表 ----------
def ci95(x):
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))

g = d.groupby("rate_pct")["moran_I"]
summ_r = pd.DataFrame({
    "n": g.size(),
    "I_mean": g.mean().round(6),
    "I_sd": g.std().round(6),
    "I_min": g.min().round(6),
    "I_max": g.max().round(6),
    "mean_bias_pct": ((g.mean() - I0) / I0 * 100).round(3),
    "max_abs_bias_pct": pd.concat([ (g.max()-I0)/I0*100, (I0-g.min())/I0*100 ]).groupby(level=0).max().round(3),
    "pass_rate_p001": d.groupby("rate_pct")["pass_p001"].mean().round(3),
})
summ_r.to_csv(os.path.join(OUT, "Table_subsampling_summary_random.csv"), encoding="utf-8-sig")

gs = s.groupby("step").agg(n=("moran_I", "size"), n_valid=("n_valid", "mean"),
                           I_mean=("moran_I", "mean"), bias_pct=("bias_pct", "mean"))
gs.to_csv(os.path.join(OUT, "Table_subsampling_summary_systematic.csv"), encoding="utf-8-sig")
print(summ_r.to_string()); print(gs.round(4).to_string())

stats = {
    "benchmark_I": I0,
    "random": {int(k): {"mean": float(g.mean()[k]), "sd": float(g.std()[k]),
                        "bias_pct": float((g.mean()[k]-I0)/I0*100),
                        "min": float(g.min()[k]), "max": float(g.max()[k]),
                        "pass_rate": float(d.groupby("rate_pct")["pass_p001"].mean()[k])}
               for k in g.size().index},
    "systematic": {int(k): {"I": float(v.I_mean), "bias_pct": float(v.bias_pct), "n": int(v.n_valid)}
                   for k, v in gs.iterrows()},
}
with open(os.path.join(OUT, "experiment_key_stats.json"), "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

# ---------- Fig.4 主文图 ----------
rates = sorted(d["rate_pct"].unique())
data_by_rate = [d.loc[d["rate_pct"] == r, "moran_I"].values for r in rates]
means = [v.mean() for v in data_by_rate]
sds = [v.std() for v in data_by_rate]
maxabs = [max(v.max() - I0, I0 - v.min()) / I0 * 100 for v in data_by_rate]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)

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
ax1.set_xlabel("Sampling fraction (%)")
ax1.set_ylabel("Moran's $I$ estimates (50 replicates)")
ax1.set_title("(a) Random subsampling: unbiased but increasingly imprecise", fontsize=10)
ax1.legend(loc="lower right", fontsize=8, framealpha=0.9)
ax1.grid(axis="y", ls=":", alpha=0.5)

ax2.plot(np.log10(rates), sds, "o-", color="#2E75B6", lw=1.6, label="SD of $I$ estimates")
ax2.plot(np.log10(rates), [I0 * 0.124 / r for r in rates], ls=":", color="#2E75B6", alpha=0.7,
         label=r"$\propto 1/f$ reference")
ax2.set_xticks(np.log10(rates)); ax2.set_xticklabels([str(r) for r in rates])
ax2.set_xlabel("Sampling fraction (%)")
ax2.set_ylabel("SD of Moran's $I$ estimates")
ax2.set_yscale("log")
ax2b = ax2.twinx()
ax2b.bar(np.log10(rates), maxabs, width=0.35, color="#F4B183", alpha=0.75,
         label="Max |bias| of single replicates (%)")
ax2b.set_ylabel("Max |relative bias| of single replicates (%)", color="#C55A11")
ax2b.tick_params(axis="y", labelcolor="#C55A11")
ax2.set_title("(b) Precision loss grows as $\\sim 1/f$", fontsize=10)
h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax2b.get_legend_handles_labels()
ax2.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=8, framealpha=0.95)
ax2.grid(axis="y", ls=":", alpha=0.5)

fig.tight_layout()
p1 = os.path.join(OUT, "Fig4_subsampling_sensitivity.png")
fig.savefig(p1, bbox_inches="tight"); plt.close(fig)
print("saved:", p1)

# ---------- Fig.S1 系统抽样（补充材料）----------
fig, ax = plt.subplots(figsize=(5.2, 3.8), dpi=300)
steps = [2, 3, 5]
lbl = ["2×2 (25%)", "3×3 (11.1%)", "5×5 (4%)"]
Im = [stats["systematic"][k]["I"] for k in steps]
bias = [stats["systematic"][k]["bias_pct"] for k in steps]
ax.plot(range(3), Im, "s-", color="#C00000", lw=1.6)
for i, (v, b) in enumerate(zip(Im, bias)):
    ax.annotate(f"{b:.1f}%", (i, v), textcoords="offset points", xytext=(0, -14),
                ha="center", fontsize=9, color="#C00000")
ax.axhline(I0, color="red", ls="--", lw=1.2, label=f"Wall-to-wall benchmark ({I0:.4f})")
ax.set_xticks(range(3)); ax.set_xticklabels(lbl)
ax.set_xlabel("Systematic lattice thinning (support spacing)")
ax.set_ylabel("Moran's $I$")
ax.set_ylim(0.35, 0.72)
ax.set_title("Systematic sampling: systematic underestimation\n(support effect, deterministic)", fontsize=10)
ax.legend(fontsize=8, loc="lower left")
ax.grid(ls=":", alpha=0.5)
fig.tight_layout()
p2 = os.path.join(OUT, "FigS1_systematic_sampling.png")
fig.savefig(p2, bbox_inches="tight"); plt.close(fig)
print("saved:", p2)
