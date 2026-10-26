# -*- coding: utf-8 -*-
"""由12因子脚本派生10因子重跑脚本（P0-3b / P0-4b），不覆盖原文件。"""
import os, re

BASE = os.path.dirname(os.path.abspath(__file__))

OLD_LIST = """FACTORS12 = ['Relief', 'TPI', 'elevation', 'aspect', 'DDT', 'Prec',
             'TWI', 'NDWI', 'Humus', 'Permafrost', 'Dist_Settlement', 'Dist_Fire']"""
NEW_LIST = """FACTORS10 = ['TPI', 'elevation', 'aspect', 'DDT', 'Prec',
             'TWI', 'NDWI', 'Humus', 'Permafrost', 'Dist_Settlement']"""

def patch(src_name, dst_name, replacements, list_replace=True):
    with open(os.path.join(BASE, src_name), encoding='utf-8') as f:
        s = f.read()
    if list_replace:
        assert OLD_LIST in s, f"list literal not found in {src_name}"
        s = s.replace(OLD_LIST, NEW_LIST)
    s = s.replace("FACTORS12", "FACTORS10")
    for a, b in replacements:
        assert a in s, f"pattern not found: {a}"
        s = s.replace(a, b)
    with open(os.path.join(BASE, dst_name), 'w', encoding='utf-8') as f:
        f.write(s)
    print("written:", dst_name)

# ---------- P0-3b: 交互矩阵 ----------
patch("P0_3_交互矩阵重跑_12因子.py", "P0_3b_交互矩阵重跑_10因子.py", [
    ("P0-3 基于最终12因子的地理探测器重跑", "P0-3b 基于最终10因子的地理探测器重跑（审计框架：剔除Relief[坡度替身]与Dist_Fire[火点记录窗口过短]）"),
    ("与论文表5-4完全一致的离散化样点表", "与论文表5-4同源的离散化样点表（10因子版）"),
    ("07_最终12因子单因子q复核.csv   (覆盖早期版本; 对角线, 与表5-4同源可比)", "R1_10因子单因子q.csv"),
    ("08_12因子交互矩阵.csv           (12×12)", "R1_10因子交互矩阵.csv           (10×10)"),
    ("09_12因子交互明细_66组.csv      (含增强类型判定)", "R1_10因子交互明细_45组.csv      (含增强类型判定)"),
    ("10_12因子交互热力图.png", "R1_10因子交互热力图.png"),
    ("'07_最终12因子单因子q复核.csv'", "'R1_10因子单因子q.csv'"),
    ("'08_12因子交互矩阵.csv'", "'R1_10因子交互矩阵.csv'"),
    ("'09_12因子交互明细_66组.csv'", "'R1_10因子交互明细_45组.csv'"),
    ("12因子单因子q (与表5-4同源)", "10因子单因子q (与表5-4同源)"),
    ("66组交互类型统计", "45组交互类型统计"),
    ("交互解释力 Top12", "交互解释力 Top12"),
    ("卡马兰河流域TAGB驱动因子交互探测矩阵（筛选后12因子, q值%）", "卡马兰河流域TAGB驱动因子交互探测矩阵（审计后10因子, q值%）"),
    ("✅ 07/08/09/10 已保存", "✅ R1 系列已保存"),
])

# ---------- P0-4b: MGWR ----------
patch("P0_4_MGWR重跑_12因子.py", "P0_4b_MGWR重跑_10因子.py", [
    ("P0-4 最终12因子 MGWR 重跑 + OLS/GWR/MGWR 对比 (v3 定稿)", "P0-4b 最终10因子 MGWR 重跑 + OLS/GWR/MGWR 对比（审计框架版）"),
    ("* 因子数据: 12个筛选后因子", "* 因子数据: 10个审计后因子（剔除Relief/Dist_Fire）"),
    ("输出: 11~15号 CSV", "输出: R3 系列 CSV"),
    ("'11_MGWR建模用3000点样本.csv'", "'R3_MGWR10因子建模3000点.csv'"),
    ("'12_OLS_GWR_MGWR模型对比表.csv'", "'R3_OLS_GWR_MGWR对比表_10因子.csv'"),
    ("'13_MGWR各因子最优带宽表.csv'", "'R3_MGWR带宽表_10因子.csv'"),
    ("'14_MGWR局部回归系数_3000点.csv'", "'R3_MGWR局部系数_10因子.csv'"),
    ("'14b_GWR局部回归系数_3000点.csv'", "'R3_GWR局部系数_10因子.csv'"),
    ("'15_残差莫兰指数检验.csv'", "'R3_残差MoranI_10因子.csv'"),
    ("✅ 11~15 号文件已保存", "✅ R3 系列已保存"),
])

print("ALL DONE")
