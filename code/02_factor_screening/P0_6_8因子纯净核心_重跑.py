# -*- coding: utf-8 -*-
"""
P0-6 八因子纯净核心重跑（R5 系列）
背景：审查决定——在审计框架（剔 Relief、Dist_Fire）之上，
      进一步剔除 SSEL 特征空间内的地形变量（elevation、aspect），
      彻底消除"用模型自身输入解释模型输出"的内生性风险。
      ASP 既已剔除，其环向编码问题（A06）随之消解（在 2.2.3 中声明）。
最终8因子: TPI, DDT, Prec, TWI, NDWI, Humus, Permafrost, Dist_Settlement
输出 (results/):
  R5_8因子单因子q与F检验.csv
  R5_8因子交互矩阵.csv
  R5_8因子交互明细_28组.csv
  R5_8因子交互热力图.png
  R5_Pearson相关矩阵_8因子.csv
  R5_VIF_8因子.csv
  R5_MGWR建模3000点_8因子.csv
  R5_OLS_GWR_MGWR对比表_8因子.csv
  R5_MGWR带宽表_8因子.csv
  R5_MGWR局部系数_8因子.csv
  R5_GWR局部系数_8因子.csv
  R5_MGWR局部系数统计_8因子.csv
  R5_残差MoranI_8因子.csv
"""
import os
import time
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from scipy import stats as sps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, '..', '..', 'results')
DATA = os.path.join(BASE, '..', '..', 'data')
os.makedirs(RESULTS, exist_ok=True)
torch.manual_seed(42)

FACTORS8 = ['TPI', 'DDT', 'Prec', 'TWI', 'NDWI', 'Humus', 'Permafrost', 'Dist_Settlement']
DISP = {'TPI': 'TPI', 'DDT': 'DDT', 'Prec': 'PREC', 'TWI': 'TWI', 'NDWI': 'NDWI',
        'Humus': 'HUM', 'Permafrost': 'PER', 'Dist_Settlement': 'DST'}

# ==================== Part A: GeoDetector (离散化50k) ====================
disc = pd.read_csv(os.path.join(DATA, 'Kamalan_Drivers_Discretized_50k_Fixed.csv'))
y = disc['TAGB'].values.astype(np.float64)
N = len(y)
SST = np.sum((y - y.mean()) ** 2)
print(f"[GeoDetector] 样点数: {N}")

Xc = {f: disc[f].values.astype(np.int64) for f in FACTORS8}

def q_value(x1, x2=None):
    if x2 is None:
        keys = x1
    else:
        keys = x1 * 100 + x2
    _, inv = np.unique(keys, return_inverse=True)
    G = inv.max() + 1
    sums = np.bincount(inv, weights=y)
    cnts = np.bincount(inv).astype(np.float64)
    means = sums / cnts
    yhat = means[inv]
    ssw = np.sum((y - yhat) ** 2)
    q = max(0.0, (1.0 - ssw / SST) * 100.0)
    return q, G

K = len(FACTORS8)
M = np.zeros((K, K))
for i in range(K):
    for j in range(i + 1):
        q, _ = q_value(Xc[FACTORS8[i]], None if i == j else Xc[FACTORS8[j]])
        M[i, j] = M[j, i] = q
mat = pd.DataFrame(M, index=[DISP[f] for f in FACTORS8], columns=[DISP[f] for f in FACTORS8])
mat.to_csv(os.path.join(RESULTS, 'R5_8因子交互矩阵.csv'), encoding='utf-8-sig')

# 单因子 q + F 检验 + p 值（A06: complete q and p results）
srows = []
for f in FACTORS8:
    q, L = q_value(Xc[f])
    qq = q / 100.0
    F = (N - L) / (L - 1) * qq / max(1 - qq, 1e-12)
    p = sps.f.sf(F, L - 1, N - L)
    srows.append({'因子': DISP[f], '分层数': int(L), 'q值(%)': round(q, 2),
                  'F统计量': round(F, 2), 'p值': p})
single = pd.DataFrame(srows).sort_values('q值(%)', ascending=False)
single.to_csv(os.path.join(RESULTS, 'R5_8因子单因子q与F检验.csv'), index=False, encoding='utf-8-sig')
print("\n===== 8因子单因子q（含F检验）=====")
print(single.to_string(index=False))

# 28组交互明细
rows = []
for i in range(K):
    for j in range(i + 1, K):
        f1, f2 = FACTORS8[i], FACTORS8[j]
        q1, q2, q12 = mat.iloc[i, i], mat.iloc[j, j], mat.iloc[i, j]
        mx, sm = max(q1, q2), q1 + q2
        if q12 > sm + 1e-9:
            t = '非线性增强'
        elif q12 > mx + 1e-9:
            t = '双因子增强'
        elif abs(q12 - mx) <= 1e-9:
            t = '独立'
        else:
            t = '减弱'
        rows.append({'因子1': DISP[f1], '因子2': DISP[f2], 'q(X1)(%)': round(q1, 2),
                     'q(X2)(%)': round(q2, 2), 'q(X1∩X2)(%)': round(q12, 2),
                     '较max增益(%)': round(q12 - mx, 2), '交互类型': t})
det = pd.DataFrame(rows).sort_values('q(X1∩X2)(%)', ascending=False)
det.to_csv(os.path.join(RESULTS, 'R5_8因子交互明细_28组.csv'), index=False, encoding='utf-8-sig')
print("\n===== 28组交互 Top10 =====")
print(det.head(10).to_string(index=False))
print("\n===== 交互类型统计 =====")
print(det['交互类型'].value_counts().to_string())

mask = np.triu(np.ones_like(M, dtype=bool), k=1)
fig, ax = plt.subplots(figsize=(9, 7.5), dpi=300)
sns.heatmap(mat, mask=mask, cmap='YlOrRd', annot=True, fmt='.2f',
            linewidths=0.5, square=True,
            cbar_kws={'shrink': 0.8, 'label': '交互探测解释力 q 值 (%)'},
            annot_kws={'size': 9, 'weight': 'bold'}, ax=ax)
ax.set_title('卡马兰河流域TAGB驱动因子交互探测矩阵（8因子纯净核心, q值%）',
             fontsize=13, fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, 'R5_8因子交互热力图.png'), dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ Part A (GeoDetector) 完成")

# ==================== Part B: 相关矩阵 + VIF ====================
cont = pd.read_csv(os.path.join(DATA, 'Kamalan_Drivers_Continuous_50k_with_XY.csv'))
cm = cont[FACTORS8].corr(method='pearson')
cm.index = [DISP[f] for f in FACTORS8]
cm.columns = [DISP[f] for f in FACTORS8]
cm.round(3).to_csv(os.path.join(RESULTS, 'R5_Pearson相关矩阵_8因子.csv'), encoding='utf-8-sig')
print("\n===== Pearson 相关矩阵 =====")
print(cm.round(3).to_string())

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

vif_df = vif_table(cont[FACTORS8].values, [DISP[f] for f in FACTORS8])
vif_df.to_csv(os.path.join(RESULTS, 'R5_VIF_8因子.csv'), index=False, encoding='utf-8-sig')
print("\n===== VIF (8因子, 阈值5) =====")
print(vif_df.to_string(index=False))
assert vif_df['VIF'].max() < 5, "VIF 超阈值，需检查"

# ==================== Part C: MGWR / OLS / GWR (3000点) ====================
n_fit = 3000
idx = np.linspace(0, len(cont) - 1, n_fit).astype(int)
d3000 = cont.iloc[idx].reset_index(drop=True)
d3000.to_csv(os.path.join(RESULTS, 'R5_MGWR建模3000点_8因子.csv'), index=False, encoding='utf-8-sig')
print(f"\n[MGWR] 建模点: {len(d3000)} (与P0_4b/P0_4c同一等距索引)")

coords = d3000[['X', 'Y']].values.astype(np.float32)
yv = d3000['TAGB'].values.astype(np.float32)
Xraw = d3000[FACTORS8].values.astype(np.float32)
mu, sd = Xraw.mean(0), Xraw.std(0)
Xs = (Xraw - mu) / sd
Nn, P = Xs.shape
Kk = P + 1
bw_grid = list(range(100, 2901, 100))
G = len(bw_grid)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"计算设备: {device}")

cache = os.path.join(RESULTS, '_cache_W_ASPsincos.pt')  # 坐标/带宽相同，直接复用P0-4c缓存
if os.path.exists(cache):
    print("复用 P0-4c 预计算权重缓存...")
    W = torch.load(cache, map_location=device)
    D = torch.cdist(torch.tensor(coords, device=device),
                    torch.tensor(coords, device=device), p=2)
else:
    t0 = time.time()
    D = torch.cdist(torch.tensor(coords, device=device),
                    torch.tensor(coords, device=device), p=2)
    hn = torch.zeros(G, Nn, device=device)
    for gi, bw in enumerate(bw_grid):
        hn[gi] = D.topk(k=int(bw), largest=False, dim=1).values[:, -1]
    hn3 = hn[:, None, :]
    rr = (D[None] / hn3) ** 2
    W = torch.where(D[None] <= hn3, (1 - rr) ** 2, torch.zeros_like(rr))
    torch.save(W.cpu(), os.path.join(RESULTS, '_cache_W_8因子.pt'))
    print(f"预计算完成 {time.time()-t0:.1f}s")
Xt = torch.cat([torch.ones((Nn, 1), device=device),
                torch.tensor(Xs, device=device)], dim=1)
yt = torch.tensor(yv, device=device).reshape(-1, 1)
bw_t = torch.tensor(np.array(bw_grid, dtype=np.float32), device=device)
gcv_pen = (1.0 - (Nn / bw_t) / Nn) ** 2

def fit_gwr_joint(bw, ridge=1e-3):
    h = D.topk(k=int(bw), largest=False, dim=1).values[:, -1].unsqueeze(1)
    w = (1 - (D / h) ** 2) ** 2
    w = torch.where(D <= h, w, torch.zeros_like(w))
    Xe = Xt.unsqueeze(0); ye = yt.unsqueeze(0); Wd = w.unsqueeze(2)
    WX = Xe * Wd
    A = torch.matmul(WX.transpose(1, 2), Xe) + torch.eye(Kk, device=device).unsqueeze(0) * ridge
    try:
        Ainv = torch.linalg.inv(A)
    except Exception:
        Ainv = torch.linalg.pinv(A)
    b = torch.matmul(Ainv, torch.matmul(WX.transpose(1, 2), ye)).squeeze(2)
    y_hat = torch.sum(Xt * b, dim=1, keepdim=True)
    rss = ((yt - y_hat) ** 2).sum().item()
    return b, y_hat, rss

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
    bw_arr = np.full(Kk, 400, dtype=int)
    for it in range(max_iter):
        f_old = f_hat.clone()
        for k in range(Kk):
            y_res = (yt.squeeze() - (f_hat.sum(dim=1) - f_hat[:, k])).squeeze()
            b_all, f_all, rss_all = col_fit_allbw(Xt[:, k], y_res)
            gcv = (rss_all / Nn) / gcv_pen
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
    rmse = np.sqrt(rss / Nn)
    adj_r2 = 1 - (1 - r2) * (Nn - 1) / max(Nn - edf - 1, 1)
    aicc = Nn * np.log(rss / Nn) + 2 * edf + 2 * edf * (edf + 1) / max(Nn - edf - 1, 1)
    return r2, adj_r2, rmse, aicc

def resid_morans_I(resid, k=8, n_perm=99, seed=42):
    tree = cKDTree(coords)
    _, nbrs = tree.query(coords, k=k + 1)
    nbrs = nbrs[:, 1:]
    Wn = np.zeros((Nn, Nn), dtype=np.float32)
    rows_i = np.repeat(np.arange(Nn), k)
    Wn[rows_i, nbrs.ravel()] = 1.0
    Wn = np.maximum(Wn, Wn.T)
    s0 = Wn.sum()
    z = resid - resid.mean()
    den = np.sum(z ** 2)
    I = (Nn / s0) * (z @ Wn @ z) / den
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        zp = rng.permutation(z)
        Ip = (Nn / s0) * (zp @ Wn @ zp) / den
        if abs(Ip) >= abs(I):
            cnt += 1
    return float(I), (cnt + 1) / (n_perm + 1)

Xo = np.column_stack([np.ones(Nn), Xraw])
beta_ols, *_ = np.linalg.lstsq(Xo, yv, rcond=None)
yp_ols = Xo @ beta_ols
rss_ols = float(np.sum((yv - yp_ols) ** 2))
r2o, ar2o, rmseo, aicco = model_stats(rss_ols, Kk)
I_ols, p_ols = resid_morans_I(yv - yp_ols)
print(f"\nOLS: R²={r2o:.4f}, RMSE={rmseo:.2f}, 残差I={I_ols:.4f}(p={p_ols:.3f})")

beta_g, yp_g, rss_g = fit_gwr_joint(400)
edf_g = Nn * Kk / 400.0
r2g, ar2g, rmseg, aiccg = model_stats(rss_g, edf_g)
I_g, p_g = resid_morans_I(yv - yp_g.squeeze().cpu().numpy())
print(f"GWR(bw=400): R²={r2g:.4f}, RMSE={rmseg:.2f}, 残差I={I_g:.4f}(p={p_g:.3f})")

t0 = time.time()
bw_m, beta_m, yp_m, rss_m = fit_mgwr()
edf_m = float(np.sum(Nn / np.maximum(bw_m, 1)))
r2m, ar2m, rmsem, aicc_m = model_stats(rss_m, edf_m)
I_m, p_m = resid_morans_I(yv - yp_m.squeeze().cpu().numpy())
print(f"MGWR: R²={r2m:.4f}, RMSE={rmsem:.2f}, 残差I={I_m:.4f}(p={p_m:.3f}), 耗时{time.time()-t0:.0f}s")

def to_original_scale(betas_z):
    b = betas_z.cpu().numpy().copy()
    b_orig = b.copy()
    b_orig[:, 1:] = b[:, 1:] / sd[None, :]
    b_orig[:, 0] = b[:, 0] - np.sum(b[:, 1:] * (mu / sd)[None, :], axis=1)
    return b_orig

names9 = ['Intercept'] + [DISP[f] for f in FACTORS8]
coef_orig = to_original_scale(beta_m)
g_orig = to_original_scale(beta_g)

cmp_df = pd.DataFrame([
    {'模型': '全局OLS', 'RSS': round(rss_ols, 2), 'R2': round(r2o, 4), '调整R2': round(ar2o, 4),
     'RMSE(t/hm2)': round(rmseo, 2), 'AICc(近似)': round(aicco, 2),
     '有效自由度': round(Kk, 1), '残差Moran I': round(I_ols, 4), '残差I检验p值': round(p_ols, 4)},
    {'模型': '经典GWR(bw=400)', 'RSS': round(rss_g, 2), 'R2': round(r2g, 4), '调整R2': round(ar2g, 4),
     'RMSE(t/hm2)': round(rmseg, 2), 'AICc(近似)': round(aiccg, 2),
     '有效自由度': round(edf_g, 1), '残差Moran I': round(I_g, 4), '残差I检验p值': round(p_g, 4)},
    {'模型': '多尺度MGWR', 'RSS': round(rss_m, 2), 'R2': round(r2m, 4), '调整R2': round(ar2m, 4),
     'RMSE(t/hm2)': round(rmse_m := rmsem, 2), 'AICc(近似)': round(aicc_m, 2),
     '有效自由度': round(edf_m, 1), '残差Moran I': round(I_m, 4), '残差I检验p值': round(p_m, 4)},
])
cmp_df.to_csv(os.path.join(RESULTS, 'R5_OLS_GWR_MGWR对比表_8因子.csv'), index=False, encoding='utf-8-sig')
print("\n===== 模型对比 =====")
print(cmp_df.to_string(index=False))

scale_label = []
for bwi in bw_m:
    if bwi >= 2600:
        scale_label.append('全局平稳 (Global)')
    elif bwi >= 1500:
        scale_label.append('中尺度/区域性 (Regional)')
    else:
        scale_label.append('局域 (Local)')
bw_df = pd.DataFrame({'因子': names9, 'MGWR最佳带宽': bw_m, '作用尺度属性': scale_label})
bw_df.to_csv(os.path.join(RESULTS, 'R5_MGWR带宽表_8因子.csv'), index=False, encoding='utf-8-sig')
print("\n===== MGWR 最优带宽 =====")
print(bw_df.to_string(index=False))

coef_df = pd.DataFrame(coef_orig, columns=names9)
coef_df.insert(0, 'Y', d3000['Y'])
coef_df.insert(0, 'X', d3000['X'])
coef_df.to_csv(os.path.join(RESULTS, 'R5_MGWR局部系数_8因子.csv'), index=False, encoding='utf-8-sig')

gdf = pd.DataFrame(g_orig, columns=names9)
gdf.insert(0, 'Y', d3000['Y']); gdf.insert(0, 'X', d3000['X'])
gdf.to_csv(os.path.join(RESULTS, 'R5_GWR局部系数_8因子.csv'), index=False, encoding='utf-8-sig')

pd.DataFrame([
    {'模型': 'OLS', '残差MoranI': I_ols, 'p值': p_ols, '权重方案': 'kNN(k=8)权重, 蒙特卡洛99次置换'},
    {'模型': 'GWR(bw=400)', '残差MoranI': I_g, 'p值': p_g, '权重方案': 'kNN(k=8)权重, 蒙特卡洛99次置换'},
    {'模型': 'MGWR', '残差MoranI': I_m, 'p值': p_m, '权重方案': 'kNN(k=8)权重, 蒙特卡洛99次置换'},
]).to_csv(os.path.join(RESULTS, 'R5_残差MoranI_8因子.csv'), index=False, encoding='utf-8-sig')

stat_rows = []
for j, nm in enumerate(names9):
    c = coef_orig[:, j]
    stat_rows.append({'项': nm, '均值': round(float(c.mean()), 4),
                      '最小': round(float(c.min()), 4), '最大': round(float(c.max()), 4),
                      '正占比(%)': round(float((c > 0).mean() * 100), 1)})
pd.DataFrame(stat_rows).to_csv(os.path.join(RESULTS, 'R5_MGWR局部系数统计_8因子.csv'),
                               index=False, encoding='utf-8-sig')

print("\n✅ R5 系列（8因子纯净核心）全部完成")
