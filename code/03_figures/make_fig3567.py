# -*- coding: utf-8 -*-
"""Fig.3 FAGB map, Fig.5 LISA, Fig.6 interaction heatmap, Fig.7 MGWR coefficients.
All panels in WGS84 degrees, watershed boundary from AGB raster footprint."""
import os
import numpy as np
import pandas as pd
import rasterio
from matplotlib import pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Patch

BASE = r"D:\python\try\2026\No_2"
P0 = os.path.join(BASE, "P0_因子筛选")
OUTDIR = r"D:\python\try\2026\No_2\SCI_RS_Paper\02_图表"

# ---------- UTM 51N -> WGS84 ----------
def utm_to_ll(x, y, lon0=123.0):
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f); ep2 = e2 / (1 - e2); k0 = 0.9996
    x = np.asarray(x, float) - 500000.0
    y = np.asarray(y, float)
    M = y / k0
    mu = M / (a * (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256))
    e1 = (1 - np.sqrt(1 - e2)) / (1 + np.sqrt(1 - e2))
    fp = (mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * np.sin(2 * mu)
          + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * np.sin(4 * mu)
          + (151 * e1 ** 3 / 96) * np.sin(6 * mu)
          + (1097 * e1 ** 4 / 512) * np.sin(8 * mu))
    s, c, t = np.sin(fp), np.cos(fp), np.tan(fp)
    C1 = ep2 * c ** 2; T1 = t ** 2
    N1 = a / np.sqrt(1 - e2 * s ** 2)
    R1 = a * (1 - e2) / (1 - e2 * s ** 2) ** 1.5
    D = x / (N1 * k0)
    lat = fp - (N1 * t / R1) * (D ** 2 / 2 - (5 + 3 * T1 + 10 * C1 - 4 * C1 ** 2 - 9 * ep2) * D ** 4 / 24
             + (61 + 90 * T1 + 298 * C1 + 45 * T1 ** 2 - 252 * ep2 - 3 * C1 ** 2) * D ** 6 / 720)
    lon = np.radians(lon0) + (D - (1 + 2 * T1 + C1) * D ** 3 / 6
             + (5 - 2 * C1 + 28 * T1 - 3 * C1 ** 2 + 8 * ep2 + 24 * T1 ** 2) * D ** 5 / 120) / c
    return np.degrees(lon), np.degrees(lat)

def grid_extent(tr, shape, geographic=False):
    """extent in WGS84 degrees (left, right, bottom, top)"""
    cols = np.array([0, shape[1]])
    rows = np.array([0, shape[0]])
    xs = tr.c + (cols + 0.5) * tr.a
    ys = tr.f + (rows + 0.5) * tr.e
    if geographic:
        return (xs[0], xs[1], ys[1], ys[0])
    lons, _ = utm_to_ll(xs, np.full(2, tr.f))
    _, lats = utm_to_ll(np.full(2, tr.c), ys)
    return (lons[0], lons[1], lats[1], lats[0])

# ---------- watershed boundary (from AGB footprint) ----------
with rasterio.open(os.path.join(BASE, "NEW_AGB_Prediction_Map.tif")) as r:
    agb = r.read(1)
    agb_tr = r.transform
valid = np.isfinite(agb) & (agb > 0)
ds = 14
m = valid[::ds, ::ds].astype(float)
rows, cols = np.mgrid[0:m.shape[0], 0:m.shape[1]]
xu = agb_tr.c + (cols * ds + ds / 2) * agb_tr.a
yu = agb_tr.f + (rows * ds + ds / 2) * agb_tr.e
ftmp = plt.figure()
cs = plt.contour(xu, yu, m, levels=[0.5])
plt.close(ftmp)
paths = []
for p in cs.collections[0].get_paths():
    if len(p.vertices) > 500:
        lon, lat = utm_to_ll(p.vertices[:, 0], p.vertices[:, 1])
        paths.append(np.column_stack([lon, lat]))
allp = np.vstack(paths)
XL = (allp[:, 0].min() - 0.05, allp[:, 0].max() + 0.05)
YL = (allp[:, 1].min() - 0.05, allp[:, 1].max() + 0.05)

def base_map(ax, title, label_fs=10):
    for p in paths:
        ax.plot(p[:, 0], p[:, 1], color="k", lw=0.9)
    ax.set_xlim(*XL); ax.set_ylim(*YL)
    ax.set_xlabel("Longitude (°E)", fontsize=label_fs)
    ax.set_ylabel("Latitude (°N)", fontsize=label_fs)
    ax.set_title(title, fontsize=11)
    ax.set_aspect(1 / np.cos(np.radians(52.1)) * (XL[1] - XL[0]) / (YL[1] - YL[0]) * 1.0)
    ax.set_aspect("equal")

# ================= Fig.3 FAGB map =================
agb_d = agb[::5, ::5]
agb_d = np.where(np.isfinite(agb_d) & (agb_d > 0), agb_d, np.nan)
ext = grid_extent(agb_tr, agb.shape)
fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=300)
im = ax.imshow(agb_d, cmap="YlGn", extent=ext, origin="upper",
               vmin=0, vmax=240, interpolation="bilinear")
base_map(ax, "(a) FAGB distribution (10 m)")
cb = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.02)
cb.set_label("FAGB (Mg ha$^{-1}$)")
fig.savefig(os.path.join(OUTDIR, "Fig3_FAGB_map.png"), bbox_inches="tight")
plt.close(fig)
print("Fig.3 done")

# ================= Fig.5 LISA =================
with rasterio.open(os.path.join(BASE, "process", "mappingTAGB_LISA_Clustering.tif")) as r:
    lisa = r.read(1)
    lisa_tr = r.transform
lisa_d = lisa[::3, ::3]
ext5 = grid_extent(lisa_tr, lisa.shape)
colors = {0: "#f2f2f2", 1: "#d73027", 2: "#4575b4", 3: "#fdae61", 4: "#74add1"}
cmap = ListedColormap([colors[i] for i in range(5)])
fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=300)
ax.imshow(lisa_d, cmap=cmap, extent=ext5, origin="upper", vmin=-0.5, vmax=4.5,
          interpolation="nearest")
base_map(ax, "(a) LISA cluster map")
legend = [Patch(fc=colors[1], label="HH (hotspot)"),
          Patch(fc=colors[2], label="LL (coldspot)"),
          Patch(fc=colors[3], label="HL (outlier)"),
          Patch(fc=colors[4], label="LH (outlier)"),
          Patch(fc=colors[0], label="Not significant")]
ax.legend(handles=legend, loc="lower left", fontsize=8.5, framealpha=0.9,
          ncol=2, title="Cluster type", title_fontsize=9)
fig.savefig(os.path.join(OUTDIR, "Fig5_LISA.png"), bbox_inches="tight")
plt.close(fig)
print("Fig.5 done")

# ================= Fig.6 interaction heatmap =================
M = pd.read_csv(os.path.join(P0, "08_12因子交互矩阵.csv"), index_col=0)
abbr = {"Relief": "RD", "TPI": "TPI", "elevation": "DEM", "aspect": "ASP", "DDT": "DDT",
        "Prec": "PREC", "TWI": "TWI", "NDWI": "NDWI", "Humus": "HUM",
        "Permafrost": "PER", "Dist_Settlement": "DST", "Dist_Fire": "DFR"}
M = M.rename(index=abbr, columns=abbr)
V = M.values.copy()
mask = np.tril(np.ones_like(V, bool), k=-1)  # keep upper triangle incl. diagonal
Vu = np.where(mask, V, np.nan)
fig, ax = plt.subplots(figsize=(8.6, 7.4), dpi=300)
im = ax.imshow(Vu, cmap="YlOrRd", vmin=0, vmax=8)
ax.set_xticks(range(12)); ax.set_yticks(range(12))
ax.set_xticklabels(M.columns, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(M.index, fontsize=9)
for i in range(12):
    for j in range(i, 12):
        ax.text(j, i, f"{V[i, j]:.2f}", ha="center", va="center", fontsize=7.2,
                color="black" if V[i, j] < 6 else "white")
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("q value of interaction (%)")
ax.set_title("(a) Pairwise interaction effects on FAGB (GeoDetector)", fontsize=11.5)
fig.savefig(os.path.join(OUTDIR, "Fig6_Interaction_Heatmap.png"), bbox_inches="tight")
plt.close(fig)
print("Fig.6 done")

# ================= Fig.7 MGWR coefficients =================
panels = [("Intercept_P0_12因子.tif", "(a) Intercept"),
          ("TPI_P0_12因子.tif", "(b) TPI"),
          ("Prec_P0_12因子.tif", "(c) PREC"),
          ("TWI_P0_12因子.tif", "(d) TWI"),
          ("Dist_Fire_P0_12因子.tif", "(e) DFR"),
          ("Permafrost_P0_12因子.tif", "(f) PER")]
fig, axes = plt.subplots(3, 2, figsize=(11.6, 12.6), dpi=300)
for ax, (fn, title) in zip(axes.ravel(), panels):
    with rasterio.open(os.path.join(P0, "图5-5图5-6_系数空间分布", fn)) as r:
        c = r.read(1).astype(float)
        ctr = r.transform
        geo = r.crs.is_geographic
    c = np.where(np.isfinite(c), c, np.nan)
    vmax = np.nanpercentile(np.abs(c), 97)
    ext7 = grid_extent(ctr, c.shape, geographic=geo)
    im = ax.imshow(np.clip(c, -vmax, vmax), cmap="RdBu_r", extent=ext7, origin="upper",
                   vmin=-vmax, vmax=vmax, interpolation="bilinear")
    for p in paths:
        ax.plot(p[:, 0], p[:, 1], color="k", lw=0.5)
    ax.set_xlim(*XL); ax.set_ylim(*YL)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01)
    cb.ax.tick_params(labelsize=7.5)
    cb.set_label("Local coefficient", fontsize=8.5)
axes[2, 0].set_xticks([122.7, 122.9, 123.1, 123.3, 123.5])
axes[2, 0].set_xlabel("Longitude (°E)", fontsize=10)
axes[2, 1].set_xticks([122.7, 122.9, 123.1, 123.3, 123.5])
axes[2, 1].set_xlabel("Longitude (°E)", fontsize=10)
for ax in (axes[0, 0], axes[1, 0], axes[2, 0]):
    ax.set_yticks([51.9, 52.0, 52.1, 52.2, 52.3])
axes[1, 0].set_ylabel("Latitude (°N)", fontsize=10)
fig.suptitle("MGWR local regression coefficients", fontsize=13, y=0.995)
fig.savefig(os.path.join(OUTDIR, "Fig7_MGWR_Coefficients.png"), bbox_inches="tight")
plt.close(fig)
print("Fig.7 done")
