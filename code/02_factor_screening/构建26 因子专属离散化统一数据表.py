import os
import numpy as np
import pandas as pd
import torch

# ==========================================
# 1. 26 个驱动因子的 Rulebook (区间边界已防爆处理)
# ==========================================
# 注意：代码会自动为这些 bins 的首尾加上 -inf 和 +inf，不用担心超出范围
DISCRETIZATION_RULES = {
    'slope': [2.0, 5.0, 15.0, 25.0, 35.0, 55.0],
    'elevation': [600.0, 800.0, 900.0, 1000.0, 1200.0, 1400.0],
    'TPI': 'CUSTOM_TPI',
    'aspect': [0.0, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 360.0],
    'Relief': [8.49, 16.45, 24.83, 35.27, 77.13],
    'Roughness': [1.005, 1.015, 1.03, 1.06, 1.12, 1.25],
    'TEM': [-5.0, -4.2, -3.4, -2.7],
    'GST': [0.0, 1.2, 2.4, 3.5],
    'DDT': [2250.0, 2450.0, 2650.0, 2850.0],
    'Prec': [1.500, 1.590, 1.680, 1.770],
    'Snow': [105.0, 130.0, 155.0, 180.0],
    'NDWI': [-0.634, -0.494, -0.354, -0.214],
    'Depth': [13.500, 21.500, 29.500, 37.500],
    'Humus': [10.500, 14.500, 18.500, 22.500],
    'Sand': [38.0, 44.0, 50.0, 56.0],
    'Moist': [30.0, 35.0, 40.0, 46.0],
    'TWI': [4.0, 6.5, 9.0, 12.0],
    'Flow': [2.0, 10.0, 40.0, 100.0, 196.0],
    'PH': [57.0, 60.0, 63.0, 66.0, 75.0],  # 💡 若 CSV 中 pH 是 5.7~7.5，请改成 [5.7, 6.0, 6.3, 6.6, 7.5]
    'SOC': [6.0, 8.0, 11.0, 18.0, 70.0],
    'Bulk_density': [120.0, 127.0, 130.0, 134.0, 150.0],  # 💡 若 CSV 中是 1.2~1.5，请改成 [1.2, 1.27, 1.30, 1.34, 1.50]
    'Permafrost': None,
    'Dist_Settlement': [1000, 3000, 5000, 10000],
    'Dist_Road': [500, 1500, 3000, 6000],
    'Dist_Fire': [2000, 5000, 10000, 20000],
    'LULC': None,
}


def apply_discretization_and_check(df, rules, slope_thr=5.0):
  df_disc = df.copy()

  print("🔍 【数据体检】正在对 26 个因子进行离散化与有效性检查...\n")
  print(
      f"{'变量名':<16} | {'原始最小值':<10} | {'原始最大值':<10} |"
      f" {'有效类别数':<10} | {'状态评估':<12}"
  )
  print("-" * 70)

  # 1. 优先处理 TPI
  if 'TPI' in df_disc.columns and rules.get('TPI') == 'CUSTOM_TPI':
    tpi = df_disc['TPI']
    slope = df_disc['slope'] if 'slope' in df_disc.columns else df_disc['Slope']
    conditions = [
        tpi < -1.0,
        (tpi >= -1.0) & (tpi < -0.5),
        (tpi >= -0.5) & (tpi <= 0.5) & (slope <= slope_thr),
        (tpi >= -0.5) & (tpi <= 0.5) & (slope > slope_thr),
        (tpi > 0.5) & (tpi <= 1.0),
        tpi > 1.0,
    ]
    df_disc['TPI'] = np.select(conditions, [1, 2, 3, 4, 5, 6], default=3)
    n_unique = df_disc['TPI'].nunique()
    print(
        f"{'TPI':<16} | {tpi.min():<10.2f} | {tpi.max():<10.2f} |"
        f" {n_unique:<10} | {'✅ 正常' if n_unique > 1 else '❌ 塌陷为单类别'}"
    )

  # 2. 处理其他变量
  for col, bins in rules.items():
    if col == 'TPI' or col not in df_disc.columns:
      continue

    val_min = df_disc[col].min()
    val_max = df_disc[col].max()

    if isinstance(bins, list):
      # 🌟 防爆核心：自动补齐 -inf 与 +inf，确保无空值产生
      full_bins = (
          bins.copy()
          if col == 'aspect'
          else [-np.inf] + [b for b in bins if not np.isinf(b)] + [np.inf]
      )
      full_bins = sorted(list(set(full_bins)))

      df_disc[col] = (
          pd.cut(
              df_disc[col], bins=full_bins, labels=False, include_lowest=True
          )
          + 1
      )

      if col == 'aspect':
        df_disc['aspect'] = df_disc['aspect'].replace({9: 1})

    n_unique = df_disc[col].nunique()
    status = '✅ 正常' if n_unique > 1 else '❌ 异常(量级/单位不匹配)'
    print(
        f'{col:<16} | {val_min:<10.2f} | {val_max:<10.2f} | {n_unique:<10} |'
        f' {status}'
    )

  print('-' * 70)
  return df_disc


# ==========================================
# 2. 读取数据与执行检查
# ==========================================
raw_csv_path = (
    r"D:\BaiduNetdiskDownload\used_bianliang\Kamalan_Drivers_Sampling_50k.csv"
)
df_raw = pd.read_csv(raw_csv_path)

df_disc = apply_discretization_and_check(df_raw, DISCRETIZATION_RULES)

# 保存修正后的离散化 CSV
disc_csv_path = os.path.join(
    os.path.dirname(raw_csv_path), "Kamalan_Drivers_Discretized_50k_Fixed.csv"
)
df_disc.to_csv(disc_csv_path, index=False)

# ==========================================
# 3. GPU 交互矩阵计算
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
y_col = 'TAGB'
x_cols = [c for c in df_disc.columns if c != y_col]
num_factors = len(x_cols)

Y = torch.tensor(df_disc[y_col].values, dtype=torch.float32, device=device)
SST = torch.sum((Y - torch.mean(Y)) ** 2)
X_tensors = [
    torch.tensor(df_disc[col].values, dtype=torch.long, device=device)
    for col in x_cols
]


def calc_q_exact(x1_tensor, x2_tensor=None):
  if x2_tensor is None:
    _, inverse_indices = torch.unique(x1_tensor, return_inverse=True)
    num_groups = len(torch.unique(x1_tensor))
  else:
    stacked = torch.stack([x1_tensor, x2_tensor], dim=1)
    unique_rows, inverse_indices = torch.unique(
        stacked, dim=0, return_inverse=True
    )
    num_groups = len(unique_rows)

  group_sums = torch.zeros(num_groups, device=device).scatter_add_(
      0, inverse_indices, Y
  )
  group_counts = torch.bincount(inverse_indices).float()
  group_means = group_sums / group_counts
  y_hat = group_means[inverse_indices]

  ssw = torch.sum((Y - y_hat) ** 2)
  return max(0.0, float((1.0 - ssw / SST) * 100.0))


q_matrix = np.zeros((num_factors, num_factors))
print('\n🚀 正在重新计算 GPU 地理探测器交互矩阵...')
for i in range(num_factors):
  for j in range(i + 1):
    val = calc_q_exact(
        X_tensors[i], None if i == j else X_tensors[j]
    )
    q_matrix[i, j] = q_matrix[j, i] = val

df_matrix = pd.DataFrame(q_matrix, index=x_cols, columns=x_cols)

matrix_csv_path = os.path.join(
    os.path.dirname(raw_csv_path), "GeoDetector_Interaction_Matrix_Fixed.csv"
)
df_matrix.to_csv(matrix_csv_path)

print(
    f'\n🎉 重新计算完成！矩阵已保存至: {matrix_csv_path}\n'
    f"📌 slope 与自身的单因子 q 值: {df_matrix.loc['slope', 'slope']:.4f}%\n"
)