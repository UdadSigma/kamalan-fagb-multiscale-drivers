# -*- coding: utf-8 -*-
"""
P0-1 驱动因子筛选：相关性矩阵 + 层次聚类 + VIF(<5) 迭代剔除
输入: Kamalan_Drivers_Continuous_50k_with_XY.csv (5万连续值样点)
输出:
  01_Pearson相关矩阵.csv / 01_Spearman相关矩阵.csv
  01_相关性聚类树状图.png / 01_相关性热图.png
  02_因子聚类分组结果.csv
  03_VIF迭代剔除日志.csv
  04_最终入选因子清单.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.stats import spearmanr

BASE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

DF_PATH = os.path.join(BASE, 'Kamalan_Drivers_Continuous_50k_with_XY.csv')

# 26个候选驱动因子(已由早期VIF剔除SPI/FDD)
DRIVERS = ['TPI', 'elevation', 'slope', 'aspect', 'Relief', 'Roughness',
           'TEM', 'GST', 'DDT', 'Prec', 'Snow', 'NDWI',
           'Depth', 'Humus', 'Sand', 'Moist', 'TWI', 'Flow',
           'PH', 'SOC', 'Bulk_density', 'Permafrost',
           'LULC', 'Dist_Settlement', 'Dist_Road', 'Dist_Fire']

# 表5-4 单因子q值(%)——用于聚类簇内代表因子择优
Q_VALUES = {'Relief': 4.13, 'slope': 3.68, 'DDT': 2.76, 'TPI': 2.16, 'aspect': 2.11,
            'GST': 1.87, 'NDWI': 1.62, 'Dist_Settlement': 1.49, 'TWI': 1.43,
            'elevation': 1.27, 'Prec': 1.23, 'Humus': 1.19, 'Permafrost': 0.89,
            'Dist_Fire': 0.81, 'LULC': 0.65, 'Roughness': 0.64, 'TEM': 0.56,
            'SOC': 0.32, 'Snow': 0.20, 'Sand': 0.19, 'Moist': 0.15, 'Depth': 0.14,
            'Dist_Road': 0.12, 'Bulk_density': 0.03, 'PH': 0.01}

CORR_CLUSTER_THRESHOLD = 0.7   # |r| > 0.7 视为同一机理簇
VIF_THRESHOLD = 5.0            # 本轮收紧的VIF阈值

# ---------- 读取与环向变量处理 ----------
df = pd.read_csv(DF_PATH)
print(f"样点数: {len(df)}")

# 坡向为环向变量(0~360°)，线性相关系数无意义，转换为北向性/东向性分量参与相关计算
df['aspect_northness'] = np.cos(np.deg2rad(df['aspect']))
df['aspect_eastness'] = np.sin(np.deg2rad(df['aspect']))

CORR_COLS = [c for c in DRIVERS if c != 'aspect'] + ['aspect_northness', 'aspect_eastness']
X = df[CORR_COLS].astype(np.float64)

# ---------- 1. 相关矩阵 ----------
pear = X.corr(method='pearson')
spear = X.corr(method='spearman')
pear.to_csv(os.path.join(BASE, '01_Pearson相关矩阵.csv'), encoding='utf-8-sig')
spear.to_csv(os.path.join(BASE, '01_Spearman相关矩阵.csv'), encoding='utf-8-sig')

# 热图
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(pear.values, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(pear.columns))); ax.set_xticklabels(pear.columns, rotation=90, fontsize=8)
ax.set_yticks(range(len(pear.index))); ax.set_yticklabels(pear.index, fontsize=8)
for i in range(len(pear.index)):
    for j in range(len(pear.columns)):
        v = pear.values[i, j]
        if abs(v) >= CORR_CLUSTER_THRESHOLD:
            ax.text(j, i, f"{v:.2f}", ha='center', va='center', fontsize=7,
                    color='white' if abs(v) > 0.85 else 'black', fontweight='bold')
plt.colorbar(im, shrink=0.8)
ax.set_title('26驱动因子Pearson相关矩阵(|r|≥0.7加粗标注)')
plt.tight_layout(); plt.savefig(os.path.join(BASE, '01_相关性热图.png'), dpi=300); plt.close()

# ---------- 2. 层次聚类(基于|r|) ----------
abs_r = pear.abs()
dist = 1.0 - abs_r.values          # 距离 = 1 - |r|
np.fill_diagonal(dist, 0.0)
Z = linkage(dist, method='average')
fig, ax = plt.subplots(figsize=(13, 6))
dendrogram(Z, labels=pear.columns.tolist(), ax=ax, color_threshold=1 - CORR_CLUSTER_THRESHOLD)
ax.axhline(1 - CORR_CLUSTER_THRESHOLD, ls='--', c='red',
           label=f'截断线 |r|={CORR_CLUSTER_THRESHOLD}')
ax.set_ylabel('距离 (1 - |r|)'); ax.legend()
ax.set_title('驱动因子相关性层次聚类树状图')
plt.tight_layout(); plt.savefig(os.path.join(BASE, '01_相关性聚类树状图.png'), dpi=300); plt.close()

labels = fcluster(Z, t=1 - CORR_CLUSTER_THRESHOLD, criterion='distance')
clusters = {}
for c, lab in zip(pear.columns, labels):
    clusters.setdefault(lab, []).append(c)

print("\n===== 因子聚类结果(簇内|r|>0.7) =====")
cluster_rows = []
for lab, members in sorted(clusters.items()):
    print(f"簇{lab}: {members}")
    for m in members:
        is_rep = False
        cluster_rows.append({'簇编号': lab, '因子': m,
                             '簇内代表': '待定', 'q值(%)': Q_VALUES.get(m, np.nan)})
rep_list = []

# ---------- 3. 簇内择优：保留q值最高的代表因子 ----------
for lab, members in sorted(clusters.items()):
    # aspect的两个分量合并视为'aspect'参与择优
    mapped = []
    for m in members:
        mapped.append('aspect' if m.startswith('aspect_') else m)
    mapped = list(dict.fromkeys(mapped))
    rep = max(mapped, key=lambda f: Q_VALUES.get(f, -1))
    rep_list.append(rep)
    for m in members:
        base_f = 'aspect' if m.startswith('aspect_') else m
        for row in cluster_rows:
            if row['簇编号'] == lab and row['因子'] == m:
                row['簇内代表'] = 'YES' if base_f == rep else 'NO'
    print(f"簇{lab} 代表因子: {rep} (q={Q_VALUES.get(rep)}%)")

# aspect分量若分属不同簇, 需要去重
rep_list = list(dict.fromkeys(rep_list))
print(f"\n聚类择优后剩余因子({len(rep_list)}个): {rep_list}")

# ---------- 4. VIF迭代剔除(阈值5.0) ----------
def calc_vif(Xdf):
    cols = list(Xdf.columns)
    vifs = {}
    for c in cols:
        others = [o for o in cols if o != c]
        Xo = Xdf[others].values
        yv = Xdf[c].values
        Xo1 = np.column_stack([np.ones(len(Xo)), Xo])
        beta, *_ = np.linalg.lstsq(Xo1, yv, rcond=None)
        pred = Xo1 @ beta
        ss_res = np.sum((yv - pred) ** 2)
        ss_tot = np.sum((yv - yv.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vifs[c] = 1.0 / max(1e-10, 1 - r2)
    return vifs

# 用原始(未标准化)数值；aspect用northness代替
vif_input = ['aspect_northness' if f == 'aspect' else f for f in rep_list]
vif_df = df[vif_input].astype(np.float64)
# 标准化, 消除量纲
vif_df = (vif_df - vif_df.mean()) / vif_df.std(ddof=0)

vif_log = []
remaining = list(vif_input)
rnd = 1
while True:
    vifs = calc_vif(vif_df[remaining])
    vmax = max(vifs.values())
    worst = max(vifs, key=vifs.get)
    for k, v in sorted(vifs.items(), key=lambda kv: -kv[1]):
        vif_log.append({'轮次': rnd, '因子': k, 'VIF': round(v, 3),
                        '是否剔除': 'YES' if (vmax > VIF_THRESHOLD and k == worst) else ''})
    if vmax <= VIF_THRESHOLD:
        print(f"\n✅ VIF收敛, 全部因子VIF≤{VIF_THRESHOLD}")
        break
    # 剔除VIF最高者; 若并列, 剔除q值更低者
    cands = [k for k, v in vifs.items() if v == vmax]
    def to_orig(k): return 'aspect' if k == 'aspect_northness' else k
    worst = min(cands, key=lambda k: Q_VALUES.get(to_orig(k), -1))
    print(f"轮{rnd}: 剔除 {worst} (VIF={vmax:.2f})")
    remaining.remove(worst)
    for row in vif_log:
        if row['轮次'] == rnd and row['因子'] == worst:
            row['是否剔除'] = 'YES'
    rnd += 1

final_factors = ['aspect' if f == 'aspect_northness' else f for f in remaining]
final_vifs = calc_vif(vif_df[remaining])
print(f"\n===== 最终入选因子({len(final_factors)}个) =====")
for f in final_factors:
    print(f"  {f:16s} VIF={final_vifs['aspect_northness' if f=='aspect' else f]:6.2f}  q={Q_VALUES.get(f)}%")

# ---------- 5. 保存 ----------
pd.DataFrame(cluster_rows).to_csv(os.path.join(BASE, '02_因子聚类分组结果.csv'),
                                  index=False, encoding='utf-8-sig')
pd.DataFrame(vif_log).to_csv(os.path.join(BASE, '03_VIF迭代剔除日志.csv'),
                             index=False, encoding='utf-8-sig')
final_df = pd.DataFrame({
    '入选因子': final_factors,
    'VIF': [round(final_vifs['aspect_northness' if f == 'aspect' else f], 2) for f in final_factors],
    '单因子q值(%)': [Q_VALUES.get(f) for f in final_factors]
}).sort_values('单因子q值(%)', ascending=False)
final_df.to_csv(os.path.join(BASE, '04_最终入选因子清单.csv'),
                index=False, encoding='utf-8-sig')
print("\n✅ 全部过程文件已保存至 P0_因子筛选/")
