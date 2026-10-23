# -*- coding: utf-8 -*-
"""Fig.1 Study area map: DEM hillshade + watershed boundary (from FAGB raster
footprint) + China locator inset."""
import os, json, urllib.request
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.windows import from_bounds
from matplotlib import pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe

BASE = r"D:\python\try\2026\No_2"
DEM_DIR = os.path.join(BASE, "dem_cache")
AGB_TIF = os.path.join(BASE, "NEW_AGB_Prediction_Map.tif")
OUTDIR = r"D:\python\try\2026\No_2\SCI_RS_Paper\02_图表"
os.makedirs(OUTDIR, exist_ok=True)

# ---------- 1. watershed boundary from AGB raster footprint ----------
with rasterio.open(AGB_TIF) as r:
    agb = r.read(1)
    tr = r.transform
    crs = r.crs
valid = np.isfinite(agb) & (agb > 0)
# boundary via contour on downsampled mask
ds = 14
m = valid[::ds, ::ds].astype(float)
rows, cols = np.mgrid[0:m.shape[0], 0:m.shape[1]]
xu = tr.c + (cols * ds + ds / 2) * tr.a
yu = tr.f + (rows * ds + ds / 2) * tr.e
fig_tmp = plt.figure()
cs = plt.contour(xu, yu, m, levels=[0.5])
plt.close(fig_tmp)
paths = [p.vertices for p in cs.collections[0].get_paths() if len(p.vertices) > 50]
print("boundary rings:", len(paths), "max len:", max(len(p) for p in paths))

# ---------- 2. UTM 51N -> WGS84 ----------
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

polys_ll = []
for p in paths:
    lon, lat = utm_to_ll(p[:, 0], p[:, 1])
    polys_ll.append(np.column_stack([lon, lat]))
allpts = np.vstack(polys_ll)
lon_min, lon_max = allpts[:, 0].min(), allpts[:, 0].max()
lat_min, lat_max = allpts[:, 1].min(), allpts[:, 1].max()
print(f"bbox: lon {lon_min:.4f}~{lon_max:.4f}, lat {lat_min:.4f}~{lat_max:.4f}")

# ---------- 3. mosaic DEM ----------
tiles = [os.path.join(DEM_DIR, f"Copernicus_DSM_COG_10_N{n:02d}_00_E{e:03d}_00_DEM.tif")
         for e in range(int(np.floor(lon_min)), int(np.floor(lon_max)) + 1)
         for n in range(int(np.floor(lat_min)), int(np.floor(lat_max)) + 1)
         if os.path.exists(os.path.join(DEM_DIR, f"Copernicus_DSM_COG_10_N{n:02d}_00_E{e:03d}_00_DEM.tif"))]
print("tiles:", len(tiles))
mos, mos_tr = merge([rasterio.open(t) for t in tiles])
dem = mos[0]

def hillshade(dem, res_x, res_y, az=315, alt=45):
    gy, gx = np.gradient(dem.astype(float), abs(res_y), abs(res_x))
    slope = np.pi / 2 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(az), np.radians(alt)
    return np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)

step = max(1, int(dem.shape[0] / 800))
dem_d = dem[::step, ::step]
hs = hillshade(dem_d, mos_tr.a * step, mos_tr.e * step)

rows, cols = np.mgrid[0:dem_d.shape[0], 0:dem_d.shape[1]]
lons = mos_tr.c + (cols + 0.5) * mos_tr.a * step
lats = mos_tr.f + (rows + 0.5) * mos_tr.e * step
mask = np.zeros(dem_d.shape, bool)
pts = np.column_stack([lons.ravel(), lats.ravel()])
for p in polys_ll:
    mask |= MplPath(p).contains_points(pts).reshape(dem_d.shape)

dem_m = np.where(mask, dem_d, np.nan)
hs_m = np.where(mask, hs, np.nan)

# ---------- 4. plot ----------
fig = plt.figure(figsize=(7.4, 8.4), dpi=300)
ax = fig.add_axes([0.09, 0.06, 0.86, 0.79])
ax.set_facecolor("#dfe8f0")
vmin, vmax = np.nanpercentile(dem_d, 2), np.nanpercentile(dem_d, 98)
ax.imshow(np.clip(dem_m, vmin, vmax), cmap="terrain", extent=[lons.min(), lons.max(), lats.min(), lats.max()],
          origin="upper", vmin=vmin, vmax=vmax)
hsn = (hs_m - np.nanmin(hs_m)) / (np.nanmax(hs_m) - np.nanmin(hs_m))
rgba = np.dstack([plt.get_cmap("gray")(hsn)[:, :, :3], np.where(mask, 0.35, 0.0)])
ax.imshow(rgba, extent=[lons.min(), lons.max(), lats.min(), lats.max()], origin="upper")
for p in polys_ll:
    if len(p) > 500:
        ax.plot(p[:, 0], p[:, 1], color="k", lw=1.2)
pad = 0.05
ax_margin = 0.09
ax.set_xlim(lon_min - pad, lon_max + ax_margin)
ax.set_ylim(lat_min - pad, lat_max + pad)
ax.set_xlabel("Longitude (°E)", fontsize=11)
ax.set_ylabel("Latitude (°N)", fontsize=11)
ax.set_title("(a) Kamalan River watershed, northern Greater Khingan Mountains", fontsize=11.5)
sm = plt.cm.ScalarMappable(cmap="terrain", norm=plt.Normalize(vmin, vmax))
cb = fig.colorbar(sm, ax=ax, fraction=0.038, pad=0.02)
cb.set_label("Elevation (m)")
mid_lat = (lats.min() + lats.max()) / 2
bar_km = 20
bar_deg = bar_km / (111.32 * np.cos(np.radians(mid_lat)))
x0, y0 = lon_min + 0.01, lat_min + 0.01
ax.plot([x0, x0 + bar_deg], [y0, y0], color="k", lw=3, solid_capstyle="butt")
ax.text(x0 + bar_deg / 2, y0 + 0.028, f"{bar_km} km", ha="center", fontsize=9,
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
ax.annotate("N", xy=(0.955, 0.93), xycoords="axes fraction", ha="center", fontsize=13, fontweight="bold")
ax.annotate("", xy=(0.955, 0.905), xytext=(0.955, 0.81), xycoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color="k", lw=2))

# ---------- 5. locator inset (local TopoJSON, decoded manually) ----------
axins = fig.add_axes([0.585, 0.865, 0.365, 0.13])
try:
    topo = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "ne_110m_countries.geojson"), encoding="utf-8"))
    sc, tr0 = topo["transform"]["scale"], topo["transform"]["translate"]

    def arc_points(ai):
        pts, x, y = [], 0, 0
        for dx, dy in topo["arcs"][ai if ai >= 0 else ~ai]:
            x += dx; y += dy
            pts.append([x * sc[0] + tr0[0], y * sc[1] + tr0[1]])
        if ai < 0:
            pts = pts[::-1]
        return pts

    fc = {"China": "#e3d2b0", "Russia": "#ededed", "Mongolia": "#dcdcdc",
          "North Korea": "#e3d2b0", "South Korea": "#e3d2b0"}
    for g in topo["objects"]["countries"]["geometries"]:
        nm = g["properties"].get("name")
        if nm not in fc:
            continue
        geoms = g["geometries"] if g["type"] == "GeometryCollection" else [g]
        for gm in geoms:
            polys = gm["arcs"] if gm["type"] == "MultiPolygon" else [gm["arcs"]]
            for poly in polys:
                ring = []
                for a in poly[0]:
                    ring += arc_points(a)
                arr = np.array(ring)
                axins.fill(arr[:, 0], arr[:, 1], color=fc[nm], ec="0.45", lw=0.35, zorder=1)
    axins.set_xlim(105, 136); axins.set_ylim(30, 55)
    print("locator OK")
except Exception as e:
    print("locator failed:", e)
    axins.set_xlim(105, 136); axins.set_ylim(30, 55)
axins.add_patch(Rectangle((lon_min - 0.3, lat_min - 0.3), (lon_max - lon_min) + 0.6, (lat_max - lat_min) + 0.6,
                          fill=False, ec="red", lw=1.4, zorder=3))
axins.set_xticks([]); axins.set_yticks([])
axins.set_title("(b) Location in China", fontsize=9)
for s in axins.spines.values():
    s.set_linewidth(0.6)

out = os.path.join(OUTDIR, "Fig1_study_area.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print("saved", out)
