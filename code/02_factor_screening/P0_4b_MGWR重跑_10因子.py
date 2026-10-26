# -*- coding: utf-8 -*-
"""
P0-4b 最终10因子 MGWR 重跑 + OLS/GWR/MGWR 对比（审计框架版）
方法说明(与论文5.4节描述对齐, 修正原代码与正文不符之处):
  * 因子数据: 10个审计后因子（剔除Relief/Dist_Fire）, 建模前标准化(z-score), 系数图已换算回原量纲
  * 建模点: 5万样点全域等距抽取3000点(修正原脚本head(3000)集中于北缘条带的问题)
  * OLS : 标准最小二乘
  * GWR : 联合多元局部加权回归 (X'WX+λI)^-1 X'Wy, 自适应双平方核, 固定邻域点数bw=400
  * MGWR: 反向拟合(Backfitting), 每列局部加权最小二乘 beta_i=Σwxy/Σwx²,
          逐列带宽网格寻优(100~2900, 步长100), 准则=GCV(广义交叉验证, edf≈N/bw),
          以联合GWR(400)解初始化 —— 与论文正文所述"GCV准则+反向拟合"一致
输出: R3 系列 CSV
"""
import os
import time
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

BASE = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(42)

FACTORS10 = ['TPI', 'elevation', 'aspect', 'DDT', 'Prec',
             'TWI', 'NDWI', 'Humus', 'Permafrost', 'Dist_Settlement']

df = pd.read_csv(os.path.join(BASE, 'Kamalan_Drivers_Continuous_50k_with_XY.csv'))

# ---------- 全域等距抽取3000建模点 ----------
n_fit = 3000
idx = np.linspace(0, len(df) - 1, n_fit).astype(int)
d3000 = df.iloc[idx].reset_index(drop=True)
d3000.to_csv(os.path.join(BASE, 'R3_MGWR10因子建模3000点.csv'), index=False, encoding='utf-8-sig')
print(f"建模点: {len(d3000)} (全样点等距抽取)")

coords = d3000[['X', 'Y']].values.astype(np.float32)
yv = d3000['TAGB'].values.astype(np.float32)
Xraw = d3000[FACTORS10].values.astype(np.float32)
mu, sd = Xraw.mean(0), Xraw.std(0)
Xs = (Xraw - mu) / sd                      # 标准化
N, P = Xs.shape
K = P + 1
bw_grid = list(range(100, 2901, 100))
G = len(bw_grid)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"计算设备: {device}")

# ---------- 预计算: 距离矩阵 + 29个带宽的双平方核 ----------
t0 = time.time()
D = torch.cdist(torch.tensor(coords, device=device),
                torch.tensor(coords, device=device), p=2)
hn = torch.zeros(G, N, device=device)
for gi, bw in enumerate(bw_grid):
    hn[gi] = D.topk(k=int(bw), largest=False, dim=1).values[:, -1]
hn3 = hn[:, None, :]
rr = (D[None] / hn3) ** 2
W = torch.where(D[None] <= hn3, (1 - rr) ** 2, torch.zeros_like(rr))   # (G,N,N)
ones = torch.ones((N, 1), device=device)
Xt = torch.cat([ones, torch.tensor(Xs, device=device)], dim=1)         # (N,K)
yt = torch.tensor(yv, device=device).reshape(-1, 1)
bw_t = torch.tensor(np.array(bw_grid, dtype=np.float32), device=device)
gcv_pen = (1.0 - (N / bw_t) / N) ** 2                                   # (1-edf/N)^2
print(f"预计算完成 {time.time()-t0:.1f}s")


# ==================== GWR: 联合多元局部加权回归 (bw=400) ====================
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


# ==================== MGWR: 反向拟合 + GCV逐列寻优 ====================
def col_fit_allbw(x_k, y_res):
    """教科书局部加权最小二乘: beta_i = Σwxy/Σwx², 批量29个带宽"""
    Wxy = torch.einsum('gij,j->gi', W, x_k * y_res)
    Wxx = torch.einsum('gij,j->gi', W, x_k * x_k)
    beta = Wxy / (Wxx + 1e-6)
    f = x_k[None, :] * beta
    rss = ((y_res[None, :] - f) ** 2).sum(dim=1)
    return beta, f, rss


def fit_mgwr(max_iter=30, tol=1e-4, verbose=True):
    beta0, _, _ = fit_gwr_joint(400)                 # 联合GWR解初始化
    betas = beta0.clone()
    f_hat = Xt * beta0
    bw_arr = np.full(K, 400, dtype=int)
    for it in range(max_iter):
        f_old = f_hat.clone()
        for k in range(K):
            y_res = (yt.squeeze() - (f_hat.sum(dim=1) - f_hat[:, k])).squeeze()
            b_all, f_all, rss_all = col_fit_allbw(Xt[:, k], y_res)
            gcv = (rss_all / N) / gcv_pen            # GCV = (RSS/N)/(1-edf/N)^2
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


# ==================== 统计与检验 ====================
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
Xo = np.column_stack([np.ones(N), Xv_raw := Xraw])
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
r2m, ar2m, rmsem, aiccm = model_stats(rss_m, edf_m)
I_m, p_m = resid_morans_I(yv - yp_m.squeeze().cpu().numpy())
print(f"MGWR: R²={r2m:.4f}, RMSE={rmsem:.2f}, 残差I={I_m:.4f}(p={p_m:.3f}), 耗时{time.time()-t0:.0f}s")

# ---------- 系数换算回原量纲: y = b0 + Σ bk·xk => b0_orig = b0z - Σ bkz·μ/σ; bk_orig = bkz/σ ----------
def to_original_scale(betas_z):
    b = betas_z.cpu().numpy().copy()
    b_orig = b.copy()
    b_orig[:, 1:] = b[:, 1:] / sd[None, :]
    b_orig[:, 0] = b[:, 0] - np.sum(b[:, 1:] * (mu / sd)[None, :], axis=1)
    return b_orig

coef_orig = to_original_scale(beta_m)
g_orig = to_original_scale(beta_g)

# ---------- 保存 ----------
cmp_df = pd.DataFrame([
    {'模型': '全局OLS', 'RSS': round(rss_ols, 2), 'R2': round(r2o, 4), '调整R2': round(ar2o, 4),
     'RMSE(t/hm2)': round(rmseo, 2), 'AICc(近似)': round(aicco, 2),
     '有效自由度': round(K, 1), '残差Moran I': round(I_ols, 4), '残差I检验p值': round(p_ols, 4)},
    {'模型': '经典GWR(bw=400)', 'RSS': round(rss_g, 2), 'R2': round(r2g, 4), '调整R2': round(ar2g, 4),
     'RMSE(t/hm2)': round(rmseg, 2), 'AICc(近似)': round(aiccg, 2),
     '有效自由度': round(edf_g, 1), '残差Moran I': round(I_g, 4), '残差I检验p值': round(p_g, 4)},
    {'模型': '多尺度MGWR', 'RSS': round(rss_m, 2), 'R2': round(r2m, 4), '调整R2': round(ar2m, 4),
     'RMSE(t/hm2)': round(rmsem, 2), 'AICc(近似)': round(aiccm, 2),
     '有效自由度': round(edf_m, 1), '残差Moran I': round(I_m, 4), '残差I检验p值': round(p_m, 4)},
])
cmp_df.to_csv(os.path.join(BASE, 'R3_OLS_GWR_MGWR对比表_10因子.csv'), index=False, encoding='utf-8-sig')
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
bw_df.to_csv(os.path.join(BASE, 'R3_MGWR带宽表_10因子.csv'), index=False, encoding='utf-8-sig')
print("\n===== MGWR 最优带宽 =====")
print(bw_df.to_string(index=False))

coef_df = pd.DataFrame(coef_orig, columns=['Intercept'] + FACTORS10)
coef_df.insert(0, 'Y', d3000['Y'])
coef_df.insert(0, 'X', d3000['X'])
coef_df.to_csv(os.path.join(BASE, 'R3_MGWR局部系数_10因子.csv'), index=False, encoding='utf-8-sig')

gdf = pd.DataFrame(g_orig, columns=['Intercept'] + FACTORS10)
gdf.insert(0, 'Y', d3000['Y']); gdf.insert(0, 'X', d3000['X'])
gdf.to_csv(os.path.join(BASE, 'R3_GWR局部系数_10因子.csv'), index=False, encoding='utf-8-sig')

pd.DataFrame([
    {'模型': 'OLS', '残差MoranI': I_ols, 'p值': p_ols, '权重方案': 'kNN(k=8)权重, 蒙特卡洛99次置换'},
    {'模型': 'GWR(bw=400)', '残差MoranI': I_g, 'p值': p_g, '权重方案': 'kNN(k=8)权重, 蒙特卡洛99次置换'},
    {'模型': 'MGWR', '残差MoranI': I_m, 'p值': p_m, '权重方案': 'kNN(k=8)权重, 蒙特卡洛99次置换'},
]).to_csv(os.path.join(BASE, 'R3_残差MoranI_10因子.csv'), index=False, encoding='utf-8-sig')

print("\n✅ R3 系列已保存")
