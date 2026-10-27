# -*- coding: utf-8 -*-
"""
P0-4c ASP 环向重编码(sin/cos) + 最终10因子 MGWR/OLS/GWR 重跑（v7）
背景：审查意见指出 aspect 以原始方位角(0-360°)作单一连续线性项不合理
      （0°与360°在数值上远离、圆周结构被破坏）。本脚本将 aspect 转为
      北向/东向分量（与 P0_1 相关诊断一致）：
        ASP_N = cos(radians(aspect))   # 坡向朝北分量, 北坡=+1
        ASP_E = sin(radians(aspect))   # 坡向朝东分量, 东坡=+1
      10因子 -> 11个预测列（ASP 拆为两列）, 建模点与 P0_4b 完全相同
      （同一50k文件 + 同一 linspace 等距索引, 保证可比）。
GeoDetector 侧不受影响：离散化数据中 aspect 为 9 个坡向扇区（无序类别,
      q 统计对层序无假设，圆周安全），无需重跑。
输出: R4 系列 CSV（results/ 目录）
"""
import os
import time
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, '..', '..', 'results')
os.makedirs(RESULTS, exist_ok=True)
torch.manual_seed(42)

FACTORS10 = ['TPI', 'elevation', 'ASP_N', 'ASP_E', 'DDT', 'Prec',
             'TWI', 'NDWI', 'Humus', 'Permafrost', 'Dist_Settlement']  # 11列（ASP拆两列）

df = pd.read_csv(os.path.join(BASE, 'Kamalan_Drivers_Continuous_50k_with_XY.csv'))

# ---------- ASP 环向重编码 ----------
asp = df['aspect'].values.astype(np.float64)
df['ASP_N'] = np.cos(np.radians(asp))
df['ASP_E'] = np.sin(np.radians(asp))
print(f"ASP_N: mean={df['ASP_N'].mean():.4f}, sd={df['ASP_N'].std():.4f}")
print(f"ASP_E: mean={df['ASP_E'].mean():.4f}, sd={df['ASP_E'].std():.4f}")
print(f"ASP_N vs ASP_E r={df['ASP_N'].corr(df['ASP_E']):.4f}")

# ---------- VIF 诊断（全5万样点, 11列） ----------
def vif_table(X, names):
    rows = []
    X = np.asarray(X, dtype=np.float64)
    for j, nm in enumerate(names):
        yj = X[:, j]
        Xo = np.column_stack([np.ones(len(yj)), np.delete(X, j, axis=1)])
        beta, *_ = np.linalg.lstsq(Xo, yj, rcond=None)
        yhat = Xo @ beta
        ss_res = np.sum((yj - yhat) ** 2)
        ss_tot = np.sum((yj - yj.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rows.append({'因子': nm, 'VIF': round(1.0 / max(1 - r2, 1e-8), 2)})
    return pd.DataFrame(rows)

vif_df = vif_table(df[FACTORS10].values, FACTORS10)
vif_df.to_csv(os.path.join(RESULTS, 'R4_VIF_11列.csv'), index=False, encoding='utf-8-sig')
print("\n===== VIF (11列, 阈值5) =====")
print(vif_df.to_string(index=False))
assert vif_df['VIF'].max() < 5, "VIF 超阈值，需检查"

# ---------- 全域等距抽取3000建模点（与P0_4b完全同一索引） ----------
n_fit = 3000
idx = np.linspace(0, len(df) - 1, n_fit).astype(int)
d3000 = df.iloc[idx].reset_index(drop=True)
d3000.to_csv(os.path.join(RESULTS, 'R4_MGWR建模3000点_ASPsincos.csv'), index=False, encoding='utf-8-sig')
print(f"\n建模点: {len(d3000)} (全样点等距抽取, 与P0_4b同一索引)")

coords = d3000[['X', 'Y']].values.astype(np.float32)
yv = d3000['TAGB'].values.astype(np.float32)
Xraw = d3000[FACTORS10].values.astype(np.float32)
mu, sd = Xraw.mean(0), Xraw.std(0)
Xs = (Xraw - mu) / sd
N, P = Xs.shape
K = P + 1
bw_grid = list(range(100, 2901, 100))
G = len(bw_grid)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"计算设备: {device}")

# ---------- 预计算（带磁盘缓存，可分段执行） ----------
cache = os.path.join(RESULTS, '_cache_W_ASPsincos.pt')
if os.path.exists(cache):
    print("加载预计算缓存...")
    W = torch.load(cache, map_location=device)
    Xt = torch.cat([torch.ones((N, 1), device=device),
                    torch.tensor(Xs, device=device)], dim=1)
    yt = torch.tensor(yv, device=device).reshape(-1, 1)
    D = torch.cdist(torch.tensor(coords, device=device),
                    torch.tensor(coords, device=device), p=2)
else:
    t0 = time.time()
    D = torch.cdist(torch.tensor(coords, device=device),
                    torch.tensor(coords, device=device), p=2)
    hn = torch.zeros(G, N, device=device)
    for gi, bw in enumerate(bw_grid):
        hn[gi] = D.topk(k=int(bw), largest=False, dim=1).values[:, -1]
    hn3 = hn[:, None, :]
    rr = (D[None] / hn3) ** 2
    W = torch.where(D[None] <= hn3, (1 - rr) ** 2, torch.zeros_like(rr))
    torch.save(W.cpu(), cache)
    print(f"预计算完成 {time.time()-t0:.1f}s (W已缓存)")
    Xt = torch.cat([torch.ones((N, 1), device=device),
                    torch.tensor(Xs, device=device)], dim=1)
    yt = torch.tensor(yv, device=device).reshape(-1, 1)

bw_t = torch.tensor(np.array(bw_grid, dtype=np.float32), device=device)
gcv_pen = (1.0 - (N / bw_t) / N) ** 2


# ==================== GWR (bw=400) ====================
def fit_gwr_joint(bw, ridge=1e-3):
    h = D.topk(k=int(bw), largest=False, dim=1).values[:, -1].unsqueeze(1)
    w = (1 - (D / h) ** 2) ** 2
    w = torch.where(D <= h, w, torch.zeros_like(w))
    Xe = Xt.unsqueeze(0); ye = yt.unsqueeze(0); Wd = w.unsqueeze(2)
    WX = Xe * Wd
    A = torch.matmul(WX.transpose(1, 2), Xe) + torch.eye(K, device=device).unsqueeze(0) * ridge
    try:
        Ainv = torch.linalg.inv(A)
    except Exception:
        Ainv = torch.linalg.pinv(A)
    b = torch.matmul(Ainv, torch.matmul(WX.transpose(1, 2), ye)).squeeze(2)
    y_hat = torch.sum(Xt * b, dim=1, keepdim=True)
    rss = ((yt - y_hat) ** 2).sum().item()
    return b, y_hat, rss


# ==================== MGWR: 反向拟合 + GCV ====================
def col_fit_allbw(x_k, y_res):
    Wxy = torch.einsum('gij,j->gi', W, x_k * y_res)
    Wxx = torch.einsum('gij,j->gi', W, x_k * x_k)
    beta = Wxy / (Wxx + 1e-6)
    f = x_k[None, :] * beta
    rss = ((y_res[None, :] - f) ** 2).sum(dim=1)
    return beta, f, rss


def fit_mgwr(max_iter=30, tol=1e-4, verbose=True):
    beta0, _, _ = fit_gwr_joint(400)
    betas = beta0.clone()
    f_hat = Xt * beta0
    bw_arr = np.full(K, 400, dtype=int)
    for it in range(max_iter):
        f_old = f_hat.clone()
        for k in range(K):
            y_res = (yt.squeeze() - (f_hat.sum(dim=1) - f_hat[:, k])).squeeze()
            b_all, f_all, rss_all = col_fit_allbw(Xt[:, k], y_res)
            gcv = (rss_all / N) / gcv_pen
            gi = int(torch.argmin(gcv).item())
            bw_arr[k] = bw_grid[gi]
            betas[:, k] = b_all[gi]
            f_hat[:, k] = f_all[gi]
        soc = float(((f_hat - f_old) ** 2).sum() / ((f_hat ** 2).sum() + 1e-8))
        yp = (Xt * betas).sum(dim=1, keepdim=True)
        r2 = 1 - ((yt - yp) ** 2).sum().item() / ((yt - yt.mean()) ** 2).sum().item()
        if verbose:
            print(f"  iter {it+1}: SOC={soc:.6f}, R2={r2:.4f}")
        if soc < tol:
            break
    y_pred = (Xt * betas).sum(dim=1, keepdim=True)
    rss = ((yt - y_pred) ** 2).sum().item()
    return bw_arr, betas, y_pred, rss


def model_stats(rss, edf):
    r2 = 1 - rss / np.sum((yv - yv.mean()) ** 2)
    rmse = np.sqrt(rss / N)
    adj_r2 = 1 - (1 - r2) * (N - 1) / max(N - edf - 1, 1)
    aicc = N * np.log(rss / N) + 2 * edf + 2 * edf * (edf + 1) / max(N - edf - 1, 1)
    return r2, adj_r2, rmse, aicc


def resid_morans_I(resid, k=8, n_perm=99, seed=42):
    tree = cKDTree(coords)
    _, nbrs = tree.query(coords, k=k + 1)
    nbrs = nbrs[:, 1:]
    Wn = np.zeros((N, N), dtype=np.float32)
    rows = np.repeat(np.arange(N), k)
    Wn[rows, nbrs.ravel()] = 1.0
    Wn = np.maximum(Wn, Wn.T)
    s0 = Wn.sum()
    z = resid - resid.mean()
    den = np.sum(z ** 2)
    I = (N / s0) * (z @ Wn @ z) / den
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        zp = rng.permutation(z)
        Ip = (N / s0) * (zp @ Wn @ zp) / den
        if abs(Ip) >= abs(I):
            cnt += 1
    return float(I), (cnt + 1) / (n_perm + 1)


# ---------- OLS ----------
Xo = np.column_stack([np.ones(N), Xraw])
beta_ols, *_ = np.linalg.lstsq(Xo, yv, rcond=None)
yp_ols = Xo @ beta_ols
rss_ols = float(np.sum((yv - yp_ols) ** 2))
r2o, ar2o, rmseo, aicco = model_stats(rss_ols, K)
I_ols, p_ols = resid_morans_I(yv - yp_ols)
print(f"\nOLS: R²={r2o:.4f}, RMSE={rmseo:.2f}, 残差I={I_ols:.4f}(p={p_ols:.3f})")

# ---------- GWR ----------
beta_g, yp_g, rss_g = fit_gwr_joint(400)
edf_g = N * K / 400.0
r2g, ar2g, rmseg, aiccg = model_stats(rss_g, edf_g)
I_g, p_g = resid_morans_I(yv - yp_g.squeeze().cpu().numpy())
print(f"GWR(bw=400): R²={r2g:.4f}, RMSE={rmseg:.2f}, 残差I={I_g:.4f}(p={p_g:.3f})")

# ---------- MGWR ----------
t0 = time.time()
bw_m, beta_m, yp_m, rss_m = fit_mgwr()
edf_m = float(np.sum(N / np.maximum(bw_m, 1)))
r2m, ar2m, rmsem, aicc_m = model_stats(rss_m, edf_m)
I_m, p_m = resid_morans_I(yv - yp_m.squeeze().cpu().numpy())
print(f"MGWR: R²={r2m:.4f}, RMSE={rmsem:.2f}, 残差I={I_m:.4f}(p={p_m:.3f}), 耗时{time.time()-t0:.0f}s")

# ---------- 系数换算回原量纲 ----------
def to_original_scale(betas_z):
    b = betas_z.cpu().numpy().copy()
    b_orig = b.copy()
    b_orig[:, 1:] = b[:, 1:] / sd[None, :]
    b_orig[:, 0] = b[:, 0] - np.sum(b[:, 1:] * (mu / sd)[None, :], axis=1)
    return b_orig

coef_orig = to_original_scale(beta_m)
g_orig = to_original_scale(beta_g)

# ---------- 保存 R4 系列 ----------
cmp_df = pd.DataFrame([
    {'模型': '全局OLS', 'RSS': round(rss_ols, 2), 'R2': round(r2o, 4), '调整R2': round(ar2o, 4),
     'RMSE(t/hm2)': round(rmseo, 2), 'AICc(近似)': round(aicco, 2),
     '有效自由度': round(K, 1), '残差Moran I': round(I_ols, 4), '残差I检验p值': round(p_ols, 4)},
    {'模型': '经典GWR(bw=400)', 'RSS': round(rss_g, 2), 'R2': round(r2g, 4), '调整R2': round(ar2g, 4),
     'RMSE(t/hm2)': round(rmseg, 2), 'AICc(近似)': round(aiccg, 2),
     '有效自由度': round(edf_g, 1), '残差Moran I': round(I_g, 4), '残差I检验p值': round(p_g, 4)},
    {'模型': '多尺度MGWR', 'RSS': round(rss_m, 2), 'R2': round(r2m, 4), '调整R2': round(ar2m, 4),
     'RMSE(t/hm2)': round(rmsem, 2), 'AICc(近似)': round(aicc_m, 2),
     '有效自由度': round(edf_m, 1), '残差Moran I': round(I_m, 4), '残差I检验p值': round(p_m, 4)},
])
cmp_df.to_csv(os.path.join(RESULTS, 'R4_OLS_GWR_MGWR对比表_ASPsincos.csv'), index=False, encoding='utf-8-sig')
print("\n===== 模型对比 =====")
print(cmp_df.to_string(index=False))

scale_label = []
for bwi, name in zip(bw_m, ['Intercept'] + FACTORS10):
    if bwi >= 2600:
        s = '全局平稳 (Global)'
    elif bwi >= 1500:
        s = '中尺度/区域性 (Regional)'
    else:
        s = '局域 (Local)'
    scale_label.append(s)
bw_df = pd.DataFrame({'因子': ['Intercept'] + FACTORS10, 'MGWR最佳带宽': bw_m, '作用尺度属性': scale_label})
bw_df.to_csv(os.path.join(RESULTS, 'R4_MGWR带宽表_ASPsincos.csv'), index=False, encoding='utf-8-sig')
print("\n===== MGWR 最优带宽 =====")
print(bw_df.to_string(index=False))

coef_df = pd.DataFrame(coef_orig, columns=['Intercept'] + FACTORS10)
coef_df.insert(0, 'Y', d3000['Y'])
coef_df.insert(0, 'X', d3000['X'])
coef_df.to_csv(os.path.join(RESULTS, 'R4_MGWR局部系数_ASPsincos.csv'), index=False, encoding='utf-8-sig')

gdf = pd.DataFrame(g_orig, columns=['Intercept'] + FACTORS10)
gdf.insert(0, 'Y', d3000['Y']); gdf.insert(0, 'X', d3000['X'])
gdf.to_csv(os.path.join(RESULTS, 'R4_GWR局部系数_ASPsincos.csv'), index=False, encoding='utf-8-sig')

pd.DataFrame([
    {'模型': 'OLS', '残差MoranI': I_ols, 'p值': p_ols, '权重方案': 'kNN(k=8)权重, 蒙特卡洛99次置换'},
    {'模型': 'GWR(bw=400)', '残差MoranI': I_g, 'p值': p_g, '权重方案': 'kNN(k=8)权重, 蒙特卡洛99次置换'},
    {'模型': 'MGWR', '残差MoranI': I_m, 'p值': p_m, '权重方案': 'kNN(k=8)权重, 蒙特卡洛99次置换'},
]).to_csv(os.path.join(RESULTS, 'R4_残差MoranI_ASPsincos.csv'), index=False, encoding='utf-8-sig')

# 系数统计（均值/正占比/范围）供正文引用
stat_rows = []
for j, nm in enumerate(['Intercept'] + FACTORS10):
    c = coef_orig[:, j]
    stat_rows.append({'项': nm, '均值': round(float(c.mean()), 4),
                      '最小': round(float(c.min()), 4), '最大': round(float(c.max()), 4),
                      '正占比(%)': round(float((c > 0).mean() * 100), 1)})
pd.DataFrame(stat_rows).to_csv(os.path.join(RESULTS, 'R4_MGWR局部系数统计_ASPsincos.csv'),
                               index=False, encoding='utf-8-sig')

print("\n✅ R4 系列已保存")
