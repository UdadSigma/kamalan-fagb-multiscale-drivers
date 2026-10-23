# -*- coding: utf-8 -*-
"""Fig.2 Technical route flowchart (English)."""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = r"D:\python\try\2026\No_2\SCI_RS_Paper\02_图表\Fig2_technical_route.png"

fig, ax = plt.subplots(figsize=(9.2, 6.4), dpi=300)
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

C_DATA = "#dce9f7"; C_PROC = "#fdf3d8"; C_OUT = "#e2f0dc"; C_EDGE = "#555555"

def box(x, y, w, h, text, fc, fs=8.2, bold=False, ec=C_EDGE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.6",
                                fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.35)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=13, color="#333333", lw=1.3))

# Title
ax.text(50, 97, "Technical route of this study", ha="center", fontsize=12, fontweight="bold")

# Stage 1: Data
box(2, 78, 24, 15, "Data\nPublished 10 m FAGB product\n(SSEL; Sentinel-2 + field plots)", C_DATA, fs=8.0)
box(2, 58, 24, 15, "26 candidate driving factors\n(terrain / hydrothermal /\npermafrost / disturbance),\nfully traced to data products", C_DATA, fs=8.0)

# Stage 2: screening
box(33, 66, 26, 15, "Redundancy-controlled screening\nsignificance |r| > 0.6 | VIF < 5\n26 \u2192 12 factors", C_PROC, fs=8.0)
arrow(26, 84, 32, 77); arrow(26, 65, 32, 72)

# Stage 3: analyses (three parallel branches)
box(66, 80, 30, 13, "(a) Full-pixel spatial autocorrelation\nGlobal Moran's I + LISA on\n21,795,474 pixels (no sampling)", C_PROC, fs=8.0)
box(66, 62, 30, 13, "(b) Subsampling sensitivity experiment\nrandom vs. systematic sampling;\nbias\u2013variance of Moran's I", C_PROC, fs=8.0)
box(66, 44, 30, 13, "(c) GeoDetector\nsingle-factor q + full 66-pair\ninteraction matrix (12 factors)", C_PROC, fs=8.0)
box(66, 26, 30, 13, "(d) MGWR\nscale-dependent non-stationary\ndriving mechanisms (n = 3000)", C_PROC, fs=8.0)
for y in (86.5, 68.5, 50.5):
    arrow(59, 73, 65, y)
arrow(59, 73, 65, 32.5)

# Outputs
box(32, 30, 27, 13, "Unbiased spatial-structure\nbenchmark + support-effect\nevidence for no-sampling\nstatistics", C_OUT, fs=8.0)
box(32, 8, 27, 13, "Weak-single / strong-interaction\ndriver structure; valley vs. slope\nlimiting regimes; scale-explicit\nlocal mechanisms", C_OUT, fs=8.0)
arrow(65, 32, 59.5, 36); arrow(65, 32, 59.5, 14)

ax.text(50, 1.5, "FAGB: forest aboveground biomass;  LISA: local indicators of spatial association;  "
        "VIF: variance inflation factor;  MGWR: multiscale geographically weighted regression",
        ha="center", fontsize=7.2, style="italic", color="#444444")

fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("saved", OUT)
