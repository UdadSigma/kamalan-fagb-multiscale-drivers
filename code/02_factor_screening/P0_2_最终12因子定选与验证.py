# -*- coding: utf-8 -*-
"""
P0-2 最终因子定选: 显著性过滤 → 机理分组择优 → VIF(<5)验证 → 单因子q值复核
输出:
  05_最终12因子与剔除理由.csv      (定选清单+每个剔除因子的理由)
  06_最终12因子VIF验证.csv
  07_最终12因子单因子q复核.csv      (基于离散化50k样点表重算, 与表5-4对照)
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
CONT = os.path.join(BASE, 'Kamalan_Drivers_Continuous_50k_with_XY.csv')
SAMP = os.path.join(BASE, 'Kamalan_Drivers_Sampling_50k.csv')

# ============ 机理分组择优结果 (每组保留q值最高或机理上不可替代的代表) ============
FINAL_12 = {
    # 地形地貌维度 (4)
    'Relief':        ('地形陡峭度代表', 'q=4.13%全因子第一; 与slope r=0.96、与Roughness r=0.49, 保留其一'),
    'TPI':           ('坡位/相对地形', 'q=2.16%; 刻画山脊-沟谷局地分异, 与elevation仅r=0.24不冗余'),
    'elevation':     ('垂直水热梯度', 'q=1.27%; 控制热量垂直分异与高海拔风蚀薄土冷点'),
    'aspect':        ('坡向辐射差异', 'q=2.11%; 阴阳坡水热分异的直接度量'),
    # 气候水文维度 (4)
    'DDT':           ('热量/冻融强度', 'q=2.76%气候类第一; 与GST r=0.86、与TEM r=0.70, 保留q最高的DDT'),
    'Prec':          ('水分输入', 'q=1.23%; 区域水分总供给'),
    'TWI':           ('水分汇聚(地形湿度)', 'q=1.43%; 河谷沼泽化机制的直接度量, 与Flow r=0.50保留q高者'),
    'NDWI':          ('地表水分现状(遥感)', 'q=1.62%; 实际水体的遥感观测, 与TWI(潜在湿度)机理互补'),
    # 土壤微生境维度 (2)
    'Humus':         ('土壤肥力代表', 'q=1.19%; 与SOC r=-0.14、与PH r=-0.09基本独立, 土壤养分簇q最高'),
    'Permafrost':    ('冻土环境(区域特有)', 'q=0.89%; 寒温带多年冻土区不可替代的隔水层控制因子'),
    # 干扰维度 (2)
    'Dist_Settlement': ('人为干扰可达性', 'q=1.49%干扰类第一; 与Dist_Road同为人类通达度量, 保留q高者'),
    'Dist_Fire':     ('历史林火干扰', 'q=0.81%; 火适应型生态系统的关键自然干扰烙印'),
}

DROPPED = {
    'slope':       ('地形陡峭度', '与Relief r=0.96, 簇内q较低(3.68%<4.13%), 剔除'),
    'Roughness':   ('地形粗糙度', '与Relief r=0.48, q仅0.64%, 陡峭度已由Relief表征'),
    'TEM':         ('年均温', '与DDT r=0.70、与Snow r=-0.70; VIF检验首轮剔除(VIF=7.8>5)'),
    'GST':         ('地表温度', '与DDT r=0.86强相关, 簇内保留q更高的DDT(2.76%>1.87%)'),
    'Snow':        ('积雪日数', '与TEM r=-0.70、与DDT r=-0.37, 热量簇由DDT统一表征, q仅0.20%'),
    'Flow':        ('汇流累积量', '与TWI r=0.50同源于DEM水文分析, q仅0.12%, 保留TWI'),
    'Moist':       ('土壤含水量', 'q=0.15%解释力极弱, 土壤水分由TWI+NDWI联合表征'),
    'SOC':         ('土壤有机碳', '与Humus同表证土壤肥力, q仅0.32%, 保留Humus'),
    'Bulk_density':('土壤容重', 'q=0.03%解释力极弱, 与Humus同属肥力簇'),
    'PH':          ('土壤酸碱度', 'p=0.274未通过显著性检验(唯一不显著因子), 剔除'),
    'Depth':       ('基岩深度', 'q=0.14%解释力极弱, 土壤物理阻隔机理由其他因子覆盖'),
    'Sand':        ('砂粒含量', 'q=0.19%解释力极弱, 为控制因子总量(12个)让位'),
    'LULC':        ('土地利用覆被', 'q=0.65%且与因变量森林掩膜同源, 作为采样本底使用, 不入模'),
    'Dist_Road':   ('距道路距离', 'q=0.12%解释力最弱, 人类通达性由Dist_Settlement表征'),
}

VIF_THRESHOLD = 5.0

# ---------- VIF验证 ----------
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

df = pd.read_csv(CONT, usecols=['aspect'] + list(FINAL_12.keys()))
Xv = df[list(FINAL_12.keys())].astype(np.float64)
Xv = (Xv - Xv.mean()) / Xv.std(ddof=0)
vifs = calc_vif(Xv)
print("===== 最终12因子 VIF 验证 =====")
ok = True
for f, v in sorted(vifs.items(), key=lambda kv: -kv[1]):
    flag = '✅' if v < VIF_THRESHOLD else '❌超标'
    if v >= VIF_THRESHOLD: ok = False
    print(f"  {f:16s} VIF={v:6.2f} {flag}")
assert ok, "存在VIF超标因子!"
pd.DataFrame({'因子': list(vifs.keys()), 'VIF': [round(v, 2) for v in vifs.values()]}) \
  .sort_values('VIF', ascending=False) \
  .to_csv(os.path.join(BASE, '06_最终12因子VIF验证.csv'), index=False, encoding='utf-8-sig')

# ---------- 单因子q值复核(离散化样点表) ----------
dcls = pd.read_csv(SAMP)
y = dcls['TAGB'].values.astype(np.float64)

def geodetector_q(x, y):
    N = len(y); sst = np.sum((y - y.mean()) ** 2)
    ssw = 0.0
    _, inv = np.unique(x, return_inverse=True)
    sums = np.bincount(inv, weights=y)
    sums2 = np.bincount(inv, weights=y ** 2)
    cnts = np.bincount(inv)
    means = sums / cnts
    ssw = np.sum(sums2 - cnts * means ** 2)   # 层内平方和
    return 1.0 - ssw / sst

rows = []
for f in FINAL_12:
    q = geodetector_q(dcls[f].values.astype(int), y)
    n_strata = dcls[f].nunique()
    rows.append({'因子': f, '分层数': n_strata, 'q值(%)': round(q * 100, 2)})
    print(f"  {f:16s} q={q*100:6.2f}%  ({n_strata}层)")

qdf = pd.DataFrame(rows).sort_values('q值(%)', ascending=False)
qdf.to_csv(os.path.join(BASE, '07_最终12因子单因子q复核.csv'), index=False, encoding='utf-8-sig')

# ---------- 定选清单保存 ----------
keep_rows = [{'因子': f, '处理': '保留', '机理角色': info[0], '理由': info[1]}
             for f, info in FINAL_12.items()]
drop_rows = [{'因子': f, '处理': '剔除', '机理角色': info[0], '理由': info[1]}
             for f, info in DROPPED.items()]
out = pd.DataFrame(keep_rows + drop_rows)
out.to_csv(os.path.join(BASE, '05_最终12因子与剔除理由.csv'), index=False, encoding='utf-8-sig')
print("\n✅ 05/06/07 已保存")
