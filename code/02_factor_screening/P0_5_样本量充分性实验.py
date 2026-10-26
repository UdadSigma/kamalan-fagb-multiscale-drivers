# -*- coding: utf-8 -*-
"""
P0-5 驱动诊断样本量充分性实验（审计框架新增，回应"应通过对比试验确定适宜样本量"意见）
设计:
  在50k离散化主样表上, 按 1%/5%/10%/20%/50%/100% 六个抽样水平(对应 n=500/2500/5000/10000/25000/50000),
  每个水平简单随机无放回抽样 50 次重复, 计算10个审计后因子的单因子 GeoDetector q 值,
  评估 q 的均值稳定性(相对全样基准的偏差)与离散度(重复间SD),
  给出"驱动诊断进入平台期的最低样本量"判定:
      平台期判定准则 —— 某样本量 n* 下, 全部因子的 |mean q(n*) - q(100%)| / q(100%) < 5%
  注: 本实验关注效应量(q)稳定性而非显著性; 在 n>=500 时所有因子 p<0.001 恒成立, 无信息量。
输出:
  R2_样本量充分性_q稳定性.csv   (因子×水平: mean/sd/min/max/cv/相对偏差%)
  R2_样本量充分性_判定摘要.txt
  R2_样本量充分性_q稳定性曲线.png  (q均值±SD带, 横轴log)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

FACTORS10 = ['TPI', 'elevation', 'aspect', 'DDT', 'Prec',
             'TWI', 'NDWI', 'Humus', 'Permafrost', 'Dist_Settlement']
N_FULL = 50000
RATES = [0.01, 0.05, 0.10, 0.20, 0.50, 1.00]
N_REPS = 50
TOL = 0.05          # 平台期相对偏差阈值 5%
SEED = 42

df = pd.read_csv(os.path.join(BASE, 'Kamalan_Drivers_Discretized_50k_Fixed.csv'))
assert len(df) == N_FULL, f"主样表应为{N_FULL}行, 实际{len(df)}"
y_all = df['TAGB'].values.astype(np.float64)
X_all = {f: df[f].values.astype(np.int64) for f in FACTORS10}

def q_value(y, x):
    """q = 1 - SSW/SST, 层内均值重构 (与P0-3b完全一致)"""
    _, inv = np.unique(x, return_inverse=True)
    sums = np.bincount(inv, weights=y)
    cnts = np.bincount(inv).astype(np.float64)
    means = sums / cnts
    yhat = means[inv]
    sst = np.sum((y - y.mean()) ** 2)
    ssw = np.sum((y - yhat) ** 2)
    return max(0.0, (1.0 - ssw / sst) * 100.0)

# ---------- 全样基准 ----------
q_full = {f: q_value(y_all, X_all[f]) for f in FACTORS10}
print("===== 全样(100%)基准q =====")
for f in FACTORS10:
    print(f"  {f:18s} {q_full[f]:.3f}")

# ---------- 分水平抽样 ----------
rng = np.random.default_rng(SEED)
records = []
for rate in RATES:
    n = int(round(N_FULL * rate))
    if n >= N_FULL:
        # 100%: 单次确定值
        for f in FACTORS10:
            records.append({'抽样率(%)': rate * 100, '样本量n': n, '重复': 1,
                            '因子': f, 'q(%)': round(q_full[f], 4)})
        continue
    for rep in range(1, N_REPS + 1):
        idx = rng.choice(N_FULL, size=n, replace=False)
        ys = y_all[idx]
        for f in FACTORS10:
            q = q_value(ys, X_all[f][idx])
            records.append({'抽样率(%)': rate * 100, '样本量n': n, '重复': rep,
                            '因子': f, 'q(%)': round(q, 4)})

raw = pd.DataFrame(records)
raw.to_csv(os.path.join(BASE, 'R2_样本量充分性_q值明细.csv'), index=False, encoding='utf-8-sig')

# ---------- 汇总 ----------
rows = []
for (rate, n, f), g in raw.groupby(['抽样率(%)', '样本量n', '因子']):
    qm = g['q(%)'].mean()
    qs = g['q(%)'].std(ddof=1) if len(g) > 1 else 0.0
    rows.append({
        '抽样率(%)': rate, '样本量n': n, '因子': f,
        'q均值(%)': round(qm, 4), 'q标准差(%)': round(qs, 4),
        'q最小(%)': round(g['q(%)'].min(), 4), 'q最大(%)': round(g['q(%)'].max(), 4),
        'CV(%)': round(qs / qm * 100, 2) if qm > 0 else np.nan,
        '相对全样偏差(%)': round((qm - q_full[f]) / q_full[f] * 100, 2),
    })
summ = pd.DataFrame(rows)
summ = summ.sort_values(['因子', '样本量n'])
summ.to_csv(os.path.join(BASE, 'R2_样本量充分性_q稳定性.csv'), index=False, encoding='utf-8-sig')

# ---------- 平台期判定 ----------
lines = []
lines.append("驱动诊断样本量充分性实验 —— 平台期判定摘要")
lines.append(f"判定准则: 某样本量下, 全部{len(FACTORS10)}个因子的 |mean q(n) - q(100%)|/q(100%) < {TOL*100:.0f}%")
lines.append("")
adequate = None
for rate in RATES:
    n = int(round(N_FULL * rate))
    sub = summ[summ['样本量n'] == n]
    worst = sub['相对全样偏差(%)'].abs().max()
    ok = worst < TOL * 100
    lines.append(f"抽样率 {rate*100:5.1f}%  (n={n:6d}):  最大相对偏差 = {worst:6.2f}%   {'✅ 通过' if ok else '❌ 未通过'}")
    if ok and adequate is None:
        adequate = n
lines.append("")
if adequate is not None:
    lines.append(f"结论: 满足全部因子相对偏差<{TOL*100:.0f}% 的最低样本量 n* = {adequate} "
                 f"(占全样 {adequate/N_FULL*100:.1f}%)")
else:
    lines.append("结论: 在所测水平内未达到平台期准则, 需提高样本量或放宽阈值。")
# SD 附注
sub20 = summ[summ['样本量n'] == 10000]
lines.append("")
lines.append("参考: n=10000 时各因子重复间SD (%):")
for _, r in sub20.iterrows():
    lines.append(f"  {r['因子']:18s} SD={r['q标准差(%)']:.4f}  CV={r['CV(%)']:.2f}")
summary_txt = "\n".join(lines)
with open(os.path.join(BASE, 'R2_样本量充分性_判定摘要.txt'), 'w', encoding='utf-8') as fp:
    fp.write(summary_txt)
print("\n" + summary_txt)

# ---------- 图: q均值±SD带 vs n (log x) ----------
fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
colors = plt.cm.tab10(np.linspace(0, 1, len(FACTORS10)))
for ci, f in enumerate(FACTORS10):
    sub = summ[summ['因子'] == f].sort_values('样本量n')
    ns = sub['样本量n'].values
    qm = sub['q均值(%)'].values
    qs = sub['q标准差(%)'].values
    ax.plot(ns, qm, marker='o', ms=4, lw=1.6, color=colors[ci], label=f)
    ax.fill_between(ns, qm - qs, qm + qs, color=colors[ci], alpha=0.15)
ax.set_xscale('log')
ax.axvline(adequate if adequate else N_FULL, ls='--', c='k', lw=1,
           label=f'平台期阈值 n*={adequate}' if adequate else '')
ax.set_xlabel('样本量 n (对数轴)', fontsize=12)
ax.set_ylabel('单因子解释力 q (%)', fontsize=12)
ax.set_title('卡马兰河流域TAGB驱动诊断的样本量充分性\n(10因子GeoDetector q值, 50次随机重复, 阴影=±1SD)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9, ncol=2, framealpha=0.9)
ax.grid(True, which='both', ls=':', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'R2_样本量充分性_q稳定性曲线.png'), dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ R2 系列已保存 (CSV明细/CSV汇总/TXT摘要/PNG曲线)")
