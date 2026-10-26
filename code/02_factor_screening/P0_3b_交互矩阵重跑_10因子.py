# -*- coding: utf-8 -*-
"""
P0-3b 基于最终10因子的地理探测器重跑（审计框架：剔除Relief[坡度替身]与Dist_Fire[火点记录窗口过短]）
输入: Kamalan_Drivers_Discretized_50k_Fixed.csv (与论文表5-4同源的离散化样点表（10因子版）)
输出:
  R1_10因子单因子q.csv
  R1_10因子交互矩阵.csv           (10×10)
  R1_10因子交互明细_45组.csv      (含增强类型判定)
  R1_10因子交互热力图.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

FACTORS10 = ['TPI', 'elevation', 'aspect', 'DDT', 'Prec',
             'TWI', 'NDWI', 'Humus', 'Permafrost', 'Dist_Settlement']

df = pd.read_csv(os.path.join(BASE, 'Kamalan_Drivers_Discretized_50k_Fixed.csv'))
y = df['TAGB'].values.astype(np.float64)
N = len(y)
SST = np.sum((y - y.mean()) ** 2)
print(f"样点数: {N}, TAGB均值={y.mean():.2f}, SST={SST:.1f}")

Xc = {f: df[f].values.astype(np.int64) for f in FACTORS10}

def q_value(x1, x2=None):
    """与论文一致的q值: q = 1 - SSW/SST, 层内均值重构"""
    if x2 is None:
        keys = x1
    else:
        keys = x1 * 100 + x2        # 两因子组合编码 (层数<100, 无碰撞)
    _, inv = np.unique(keys, return_inverse=True)
    G = inv.max() + 1
    sums = np.bincount(inv, weights=y)
    cnts = np.bincount(inv).astype(np.float64)
    means = sums / cnts
    yhat = means[inv]
    ssw = np.sum((y - yhat) ** 2)
    return max(0.0, (1.0 - ssw / SST) * 100.0), G

# ---------- 12×12 矩阵 ----------
K = len(FACTORS10)
M = np.zeros((K, K))
for i in range(K):
    for j in range(i + 1):
        q, _ = q_value(Xc[FACTORS10[i]], None if i == j else Xc[FACTORS10[j]])
        M[i, j] = M[j, i] = q

mat = pd.DataFrame(M, index=FACTORS10, columns=FACTORS10)
mat.to_csv(os.path.join(BASE, 'R1_10因子交互矩阵.csv'), encoding='utf-8-sig')

# ---------- 单因子q复核(覆盖旧文件) ----------
single = pd.DataFrame({
    '因子': FACTORS10,
    '分层数': [int(df[f].nunique()) for f in FACTORS10],
    'q值(%)': [round(mat.loc[f, f], 2) for f in FACTORS10],
}).sort_values('q值(%)', ascending=False)
single.to_csv(os.path.join(BASE, 'R1_10因子单因子q.csv'), index=False, encoding='utf-8-sig')
print("\n===== 10因子单因子q (与表5-4同源) =====")
print(single.to_string(index=False))

# ---------- 66组交互明细 ----------
rows = []
for i in range(K):
    for j in range(i + 1, K):
        f1, f2 = FACTORS10[i], FACTORS10[j]
        q1, q2, q12 = mat.loc[f1, f1], mat.loc[f2, f2], mat.loc[f1, f2]
        mx, sm = max(q1, q2), q1 + q2
        if q12 > sm + 1e-9:
            t = '非线性增强'
        elif q12 > mx + 1e-9:
            t = '双因子增强'
        elif abs(q12 - mx) <= 1e-9:
            t = '独立'
        else:
            t = '减弱'
        rows.append({'因子1': f1, '因子2': f2, 'q(X1)(%)': round(q1, 2),
                     'q(X2)(%)': round(q2, 2), 'q(X1∩X2)(%)': round(q12, 2),
                     '较max增益(%)': round(q12 - mx, 2), '交互类型': t})
det = pd.DataFrame(rows).sort_values('q(X1∩X2)(%)', ascending=False)
det.to_csv(os.path.join(BASE, 'R1_10因子交互明细_45组.csv'), index=False, encoding='utf-8-sig')

type_counts = det['交互类型'].value_counts()
print("\n===== 45组交互类型统计 =====")
print(type_counts.to_string())
print("\n===== 交互解释力 Top12 =====")
print(det.head(12).to_string(index=False))

# ---------- 热力图 ----------
mask = np.triu(np.ones_like(M, dtype=bool), k=1)
fig, ax = plt.subplots(figsize=(11, 9), dpi=300)
sns.heatmap(mat, mask=mask, cmap='YlOrRd', annot=True, fmt='.2f',
            linewidths=0.5, square=True,
            cbar_kws={'shrink': 0.8, 'label': '交互探测解释力 q 值 (%)'},
            annot_kws={'size': 9, 'weight': 'bold'}, ax=ax)
ax.set_title('卡马兰河流域TAGB驱动因子交互探测矩阵（审计后10因子, q值%）',
             fontsize=14, fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'R1_10因子交互热力图.png'), dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ R1 系列已保存")
