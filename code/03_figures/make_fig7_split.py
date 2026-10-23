# -*- coding: utf-8 -*-
"""Fig.7 MGWR coefficients -> 6 single-panel figures, 300 dpi.
Boundary extraction & UTM->WGS84 logic reused from make_fig3567.py."""
import os
import numpy as np
import rasterio
from matplotlib import pyplot as plt

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

# ---------- Fig.7 panels -> 6 single figures ----------
panels = [("Intercept_P0_12因子.tif", "(a) Intercept", "Fig7a_Intercept.png"),
          ("TPI_P0_12因子.tif", "(b) TPI", "Fig7b_TPI.png"),
          ("Prec_P0_12因子.tif", "(c) PREC", "Fig7c_PREC.png"),
          ("TWI_P0_12因子.tif", "(d) TWI", "Fig7d_TWI.png"),
          ("Dist_Fire_P0_12因子.tif", "(e) DFR", "Fig7e_DFR.png"),
          ("Permafrost_P0_12因子.tif", "(f) PER", "Fig7f_PER.png")]

for fn, title, out in panels:
    with rasterio.open(os.path.join(P0, "图5-5图5-6_系数空间分布", fn)) as r:
        c = r.read(1).astype(float)
        ctr = r.transform
        geo = r.crs.is_geographic
    c = np.where(np.isfinite(c), c, np.nan)
    vmax = np.nanpercentile(np.abs(c), 97)
    ext7 = grid_extent(ctr, c.shape, geographic=geo)

    fig, ax = plt.subplots(figsize=(7.0, 6.4), dpi=300)
    im = ax.imshow(np.clip(c, -vmax, vmax), cmap="RdBu_r", extent=ext7, origin="upper",
                   vmin=-vmax, vmax=vmax, interpolation="bilinear")
    for p in paths:
        ax.plot(p[:, 0], p[:, 1], color="k", lw=0.6)
    ax.set_xlim(*XL); ax.set_ylim(*YL)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude (°E)", fontsize=10.5)
    ax.set_ylabel("Latitude (°N)", fontsize=10.5)
    ax.tick_params(labelsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.02)
    cb.ax.tick_params(labelsize=8.5)
    cb.set_label("Local coefficient", fontsize=10)
    fig.savefig(os.path.join(OUTDIR, out), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(out, "done, vmax=%.4f" % vmax)

print("All Fig.7 panels split.")
