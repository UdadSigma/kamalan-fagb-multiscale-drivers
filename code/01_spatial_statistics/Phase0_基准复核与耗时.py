# -*- coding: utf-8 -*-
"""抽样敏感性实验 Phase 0：基准复核 + 单次耗时测定
基准：全像元 Global Moran's I 应复现 0.661154（与学位论文表5-2一致）
耗时：1次全像元(含100次置换) + 1次10%抽样(含100次置换)
"""
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
import time

TIF = r"D:\python\try\2026\No_2\NEW_AGB_Prediction_Map.tif"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
KERNEL = torch.tensor([[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]],
                      dtype=torch.float64, device=DEVICE).view(1, 1, 3, 3)

print(f"设备: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""))

with rasterio.open(TIF) as src:
    band = src.read(1).astype(np.float64)
    nodata = src.nodata
data = torch.from_numpy(band).to(DEVICE)
base_valid = (~torch.isnan(data)) & (~torch.isinf(data))
if nodata is not None:
    base_valid &= (data != nodata)
base_valid &= (data > 0.0)
N_full = int(base_valid.sum().item())
print(f"有效像元 N = {N_full:,}")


def moran_with_perms(sub_mask: torch.Tensor, n_perm: int = 100):
    """在给定布尔掩膜上计算 Moran's I + 蒙特卡洛置换（与原论文脚本同方法）"""
    t0 = time.time()
    m = sub_mask.to(torch.float64)
    N = m.sum()
    x = torch.where(sub_mask, data, torch.zeros((), dtype=torch.float64, device=DEVICE))
    mean_v = x.sum() / N
    z = torch.where(sub_mask, x - mean_v, torch.zeros((), dtype=torch.float64, device=DEVICE))
    zu, mu = z.unsqueeze(0).unsqueeze(0), m.unsqueeze(0).unsqueeze(0)
    nsum = F.conv2d(zu, KERNEL, padding=1).squeeze()
    ncnt = F.conv2d(mu, KERNEL, padding=1).squeeze()
    num = torch.sum(z * nsum).item()
    den = torch.sum(z ** 2).item()
    W = torch.sum(ncnt * m).item()
    I = (N.item() / W) * (num / den)
    E = -1.0 / (N.item() - 1)

    idx = torch.nonzero(sub_mask, as_tuple=True)
    zvals = z[idx]
    perm_I = np.empty(n_perm)
    for j in range(n_perm):
        sp = zvals[torch.randperm(zvals.numel(), device=DEVICE)]
        zp = torch.zeros_like(z)
        zp[idx] = sp
        pn = torch.sum(zp * F.conv2d(zp.unsqueeze(0).unsqueeze(0), KERNEL, padding=1).squeeze()).item()
        perm_I[j] = (N.item() / W) * (pn / den)
    sd = perm_I.std()
    p = (np.sum(np.abs(perm_I - E) >= abs(I - E)) + 1) / (n_perm + 1)
    return I, E, sd, p, time.time() - t0


# ---- 1. 全像元基准 ----
t_all = time.time()
I0, E0, sd0, p0, t_main = moran_with_perms(base_valid, 100)
print(f"\n[基准] I = {I0:.6f} (论文值 0.661154, 差 {I0-0.661154:+.6f})")
print(f"[基准] E = {E0:.10f}, SD = {sd0:.6f}, p = {p0:.6f}")
print(f"[基准] 单次含置换耗时 {time.time()-t_all:.2f} s (其中主计算 {t_main:.2f} s)")

# ---- 2. 10% 抽样单次耗时 ----
g = torch.Generator().manual_seed(20261021)
k = int(round(N_full * 0.10))
perm_idx = torch.randperm(N_full, generator=g)[:k].to(DEVICE)
vidx = torch.nonzero(base_valid, as_tuple=True)[0]
sub = torch.zeros(N_full, dtype=torch.bool, device=DEVICE)
sub[vidx[perm_idx]] = True
sub_mask = sub.view(base_valid.shape)
I1, E1, sd1, p1, t1 = moran_with_perms(sub_mask, 100)
print(f"\n[10%] I = {I1:.6f}, p = {p1:.6f}, 单次含置换耗时 {t1:.2f} s")

# ---- 3. 总时长估算 ----
est = 5 * 50 * t1 + 1 * 50 * (time.time() - t_all)
print(f"\n估算总时长: 5档×50次×{t1:.1f}s + 50次全像元 ≈ {est/60:.1f} 分钟")
