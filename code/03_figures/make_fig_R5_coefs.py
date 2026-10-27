# -*- coding: utf-8 -*-
"""R5 (8因子) MGWR 系数面图：9 张单独 PNG (300 dpi) + 1 张 3x3 组合图。
数据源: results/R5_MGWR局部系数_8因子.csv (3000 点, griddata 线性插值, 流域边界内)"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import rasterio
from scipy.interpolate import griddata

REPO = r"D:\python\try\2026\No_2\kamalan-fagb-multiscale-drivers"
BASE = r"D:\python\try\2026\No_2"
OUTDIR = os.path.join(BASE, "SCI_RS_Paper", "02_图表")
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Times New Roman', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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

with rasterio.open(os.path.join(BASE, "NEW_AGB_Prediction_Map.tif")) as r:
    agb = r.read(1); agb_tr = r.transform
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

coef = pd.read_csv(os.path.join(REPO, "results", "R5_MGWR局部系数_8因子.csv"))
lon_s, lat_s = utm_to_ll(coef['X'].values, coef['Y'].values)
pts = np.column_stack([lon_s, lat_s])
grid_lon = np.linspace(XL[0], XL[1], 400)
grid_lat = np.linspace(YL[0], YL[1], 400)
GL, GA = np.meshgrid(grid_lon, grid_lat)
centroid = paths[np.argmax([len(p) for p in paths])]
mpath_poly = mpath.Path(centroid)
inside = mpath_poly.contains_points(np.column_stack([GL.ravel(), GA.ravel()])).reshape(GL.shape)

panels = [('Intercept', '(a) Intercept'),
          ('TPI', '(b) TPI'), ('DDT', '(c) DDT'), ('PREC', '(d) PREC'),
          ('TWI', '(e) TWI'), ('NDWI', '(f) NDWI'), ('HUM', '(g) HUM'),
          ('PER', '(h) PER'), ('DST', '(i) DST')]
out_names = {'Intercept': 'a_Intercept', 'TPI': 'b_TPI', 'DDT': 'c_DDT', 'PREC': 'd_PREC',
             'TWI': 'e_TWI', 'NDWI': 'f_NDWI', 'HUM': 'g_HUM', 'PER': 'h_PER', 'DST': 'i_DST'}
tags = [chr(ord('a') + i) for i in range(9)]

vmax_map = {}
for col, ttl in panels:
    v = coef[col].values.astype(float)
    vmax_map[col] = np.nanpercentile(np.abs(v), 97)

# 单独图
for col, ttl in panels:
    v = coef[col].values.astype(float)
    gi = griddata(pts, v, (GL, GA), method='linear')
    gi = np.where(inside, gi, np.nan)
    vmax = vmax_map[col]
    fig, ax = plt.subplots(figsize=(7.0, 6.2), dpi=300)
    im = ax.imshow(np.clip(gi, -vmax, vmax), cmap='RdBu_r', extent=(XL[0], XL[1], YL[0], YL[1]),
                   origin='lower', vmin=-vmax, vmax=vmax, interpolation='bilinear')
    for p in paths:
        ax.plot(p[:, 0], p[:, 1], color='k', lw=0.6)
    ax.set_xlim(*XL); ax.set_ylim(*YL); ax.set_aspect('equal')
    ax.set_xlabel('Longitude (°E)', fontsize=10.5)
    ax.set_ylabel('Latitude (°N)', fontsize=10.5)
    ax.set_title(ttl, fontsize=12, fontweight='bold', loc='left')
    ax.tick_params(labelsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.02)
    cb.ax.tick_params(labelsize=8.5)
    cb.set_label('Local coefficient', fontsize=10)
    fn = os.path.join(OUTDIR, f'Fig8_R5_{out_names[col]}.png')
    fig.savefig(fn, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(fn, 'done, vmax=%.4f' % vmax)

# 组合图 3x3
fig, axes = plt.subplots(3, 3, figsize=(15.5, 13.8), dpi=300)
for ax, (col, ttl) in zip(axes.ravel(), panels):
    v = coef[col].values.astype(float)
    gi = griddata(pts, v, (GL, GA), method='linear')
    gi = np.where(inside, gi, np.nan)
    vmax = vmax_map[col]
    im = ax.imshow(np.clip(gi, -vmax, vmax), cmap='RdBu_r', extent=(XL[0], XL[1], YL[0], YL[1]),
                   origin='lower', vmin=-vmax, vmax=vmax, interpolation='bilinear')
    for p in paths:
        ax.plot(p[:, 0], p[:, 1], color='k', lw=0.5)
    ax.set_xlim(*XL); ax.set_ylim(*YL); ax.set_aspect('equal')
    ax.set_title(ttl, fontsize=11, fontweight='bold', loc='left')
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.015)
    cb.ax.tick_params(labelsize=7)
for ax in (axes[2, 0], axes[2, 1], axes[2, 2]):
    ax.set_xticks([122.7, 122.9, 123.1, 123.3, 123.5])
    ax.set_xlabel('Longitude (°E)', fontsize=10)
for ax in (axes[0, 0], axes[1, 0], axes[2, 0]):
    ax.set_yticks([51.9, 52.0, 52.1, 52.2, 52.3])
for ax in (axes[0, 0], axes[1, 0], axes[2, 0]):
    ax.set_ylabel('Latitude (°N)', fontsize=10)
fig.suptitle('MGWR local regression coefficients (8-factor clean core)', fontsize=13, y=0.995)
fn = os.path.join(OUTDIR, 'Fig8_R5_MGWR_Coefficients_combined.png')
fig.savefig(fn, bbox_inches='tight', dpi=300)
plt.close(fig)
print(fn, 'done')
print('ALL R5 FIGURES DONE')
