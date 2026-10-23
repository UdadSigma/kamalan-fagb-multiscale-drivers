# -*- coding: utf-8 -*-
"""抽样敏感性实验 主实验（含断点续跑）
设计：简单随机抽样（不放回），抽样率 1/5/10/20/50/100%，每档 n=50 次独立重复（固定种子）
每次重复：3×3 Queen 卷积仅在有效像元累计（权重自动按幸存像元归一化）+ 100 次全域随机置换
输出：subsampling_moran_results.csv（逐次记录）；systematic_sampling_results.csv（系统抽样对照，2x2/3x3/5x5）
调用方式：重复运行本脚本直至 300 次主计算全部完成（每次调用限时 TIME_BUDGET_S 秒后主动退出）
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
TIF = r"D:\python\try\2026\No_2\NEW_AGB_Prediction_Map.tif"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
KERNEL = torch.tensor([[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]],
                      dtype=torch.float64, device=DEVICE).view(1, 1, 3, 3)
BASE_SEED = 20261021
RATES = [1, 5, 10, 20, 50, 100]          # %
REPS = 50
N_PERM = 100
TIME_BUDGET_S = 260                       # 每次调用的时间预算
MAIN_CSV = os.path.join(HERE, "subsampling_moran_results.csv")
SYS_CSV = os.path.join(HERE, "systematic_sampling_results.csv")
GRID_SHAPE = None

print(f"设备: {DEVICE}", flush=True)
with rasterio.open(TIF) as src:
    band = src.read(1).astype(np.float64)
    nodata = src.nodata
data = torch.from_numpy(band).to(DEVICE)
GRID_SHAPE = data.shape
base_valid = (~torch.isnan(data)) & (~torch.isinf(data))
if nodata is not None:
    base_valid &= (data != nodata)
base_valid &= (data > 0.0)
N_full = int(base_valid.sum().item())
flat_valid_idx = torch.nonzero(base_valid.reshape(-1), as_tuple=True)[0].to(DEVICE)
print(f"有效像元 N = {N_full:,}", flush=True)


def moran_with_perms(sub_mask: torch.Tensor, n_perm: int = N_PERM):
    m = sub_mask.to(torch.float64)
    N = m.sum()
    x = torch.where(sub_mask, data, torch.zeros((), dtype=torch.float64, device=DEVICE))
    mean_v = x.sum() / N
    z = torch.where(sub_mask, x - mean_v, torch.zeros((), dtype=torch.float64, device=DEVICE))
    nsum = F.conv2d(z.unsqueeze(0).unsqueeze(0), KERNEL, padding=1).squeeze()
    ncnt = F.conv2d(m.unsqueeze(0).unsqueeze(0), KERNEL, padding=1).squeeze()
    num = torch.sum(z * nsum).item()
    den = torch.sum(z ** 2).item()
    W = torch.sum(ncnt * m).item()
    Nv = N.item()
    if W == 0 or den == 0:                       # 防御：无邻接或零方差
        return float("nan"), W, float("nan"), float("nan")
    I = (Nv / W) * (num / den)
    E = -1.0 / (Nv - 1)
    idx = torch.nonzero(sub_mask, as_tuple=True)
    zvals = z[idx]
    perm_I = np.empty(n_perm)
    for j in range(n_perm):
        sp = zvals[torch.randperm(zvals.numel(), device=DEVICE)]
        zp = torch.zeros_like(z)
        zp[idx] = sp
        pn = torch.sum(zp * F.conv2d(zp.unsqueeze(0).unsqueeze(0), KERNEL, padding=1).squeeze()).item()
        perm_I[j] = (Nv / W) * (pn / den)
    sd = float(perm_I.std())
    p = float((np.sum(np.abs(perm_I - E) >= abs(I - E)) + 1) / (n_perm + 1))
    return I, W, sd, p


def random_sub_mask(rate_pct: int, seed: int):
    k = int(round(N_full * rate_pct / 100.0))
    g = torch.Generator().manual_seed(seed)
    sel = flat_valid_idx[torch.randperm(N_full, generator=g)[:k].to(DEVICE)]
    m = torch.zeros(data.numel(), dtype=torch.bool, device=DEVICE)
    m[sel] = True
    return m.view(GRID_SHAPE)


def moran_on_lattice(step: int, seed: int):
    """系统抽样（step×step 取 1）：在抽稀后的规则子栅格上用其自身的 3×3 Queen 邻接计算。
    注意：若直接在原网格上置 NoData，保留像元间距 ≥ step，3×3 核内无邻接（W=0），
    因此须在子栅格上重建一阶邻接——这正是文献系统抽样的实际做法（支撑尺度变为 step×10 m）。"""
    torch.manual_seed(seed)
    sd_ = data[::step, ::step].contiguous()
    sv = base_valid[::step, ::step]
    m = sv.to(torch.float64)
    N = m.sum()
    x = torch.where(sv, sd_, torch.zeros((), dtype=torch.float64, device=DEVICE))
    mean_v = x.sum() / N
    z = torch.where(sv, x - mean_v, torch.zeros((), dtype=torch.float64, device=DEVICE))
    nsum = F.conv2d(z.unsqueeze(0).unsqueeze(0), KERNEL, padding=1).squeeze()
    ncnt = F.conv2d(m.unsqueeze(0).unsqueeze(0), KERNEL, padding=1).squeeze()
    num = torch.sum(z * nsum).item()
    den = torch.sum(z ** 2).item()
    W = torch.sum(ncnt * m).item()
    Nv = N.item()
    if W == 0 or den == 0:
        return Nv, W, float("nan"), float("nan"), float("nan")
    I = (Nv / W) * (num / den)
    E = -1.0 / (Nv - 1)
    idx = torch.nonzero(sv, as_tuple=True)
    zvals = z[idx]
    perm_I = np.empty(N_PERM)
    for j in range(N_PERM):
        sp = zvals[torch.randperm(zvals.numel(), device=DEVICE)]
        zp = torch.zeros_like(z)
        zp[idx] = sp
        pn = torch.sum(zp * F.conv2d(zp.unsqueeze(0).unsqueeze(0), KERNEL, padding=1).squeeze()).item()
        perm_I[j] = (Nv / W) * (pn / den)
    sd = float(perm_I.std())
    p = float((np.sum(np.abs(perm_I - E) >= abs(I - E)) + 1) / (N_PERM + 1))
    return Nv, W, I, sd, p


def run_block(csv_path, tasks, mask_fn, tag):
    """tasks: list of (label, seed, extra)；mask_fn(label, seed)->mask。断点续跑。"""
    done = set()
    if os.path.exists(csv_path):
        done = set(pd.read_csv(csv_path)["task_key"].tolist())
    rows, t0 = [], time.time()
    for label, seed, extra in tasks:
        key = f"{label}|{seed}"
        if key in done:
            continue
        if time.time() - t0 > TIME_BUDGET_S:
            break
        mask = mask_fn(label, seed)
        torch.manual_seed(seed)          # 置换检验同样可复现
        ts = time.time()
        I, W, sd, p = moran_with_perms(mask)
        n = int(mask.sum().item())
        rows.append({
            "task_key": key, "group": tag, "label": label, "repeat_seed": seed,
            "n_valid": n, "W": W, "moran_I": round(I, 6), "perm_sd": sd,
            "p_value": p, "pass_p001": int(p < 0.01),
            "elapsed_s": round(time.time() - ts, 2),
            **extra,
        })
        print(f"[{tag}] {label} seed={seed}: I={I:.6f} p={p:.4f} pass={p<0.01} ({time.time()-ts:.1f}s)", flush=True)
        if len(rows) % 10 == 0:                      # 每10条落盘一次
            pd.DataFrame(rows).to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)
            rows = []
    if rows:
        pd.DataFrame(rows).to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)
    remain = sum(1 for label, seed, _ in tasks if f"{label}|{seed}" not in
                 (set(pd.read_csv(csv_path)["task_key"].tolist()) if os.path.exists(csv_path) else set()))
    print(f"[{tag}] 本次完成，剩余 {remain} 次", flush=True)
    return remain


# ---- 主实验：随机抽样（已完成则自动跳过）----
tasks_main = [
    (f"r{rate}", BASE_SEED + rate * 10000 + rep, {"rate_pct": rate, "repeat": rep})
    for rate in RATES for rep in range(REPS)
]
remain = run_block(MAIN_CSV, tasks_main,
                   lambda label, seed: random_sub_mask(int(label[1:]), seed),
                   "random")

# ---- 补充：系统抽样对照（2x2/3x3/5x5 子栅格，各50次，断点续跑）----
if remain == 0:
    done = set()
    if os.path.exists(SYS_CSV):
        done = set(pd.read_csv(SYS_CSV)["task_key"].tolist())
    rows, t0 = [], time.time()
    for s in [2, 3, 5]:
        for rep in range(REPS):
            key = f"g{s}|{rep}"
            if key in done:
                continue
            if time.time() - t0 > TIME_BUDGET_S:
                break
            ts = time.time()
            n, W, I, sd, p = moran_on_lattice(s, BASE_SEED + s * 777 + rep)
            rows.append({
                "task_key": key, "group": "systematic", "label": f"g{s}",
                "repeat_seed": BASE_SEED + s * 777 + rep,
                "n_valid": n, "W": W, "moran_I": round(I, 6) if I == I else None,
                "perm_sd": sd, "p_value": p,
                "pass_p001": int(p < 0.01) if p == p else None,
                "elapsed_s": round(time.time() - ts, 2),
                "step": s, "repeat": rep,
            })
            print(f"[systematic] g{s} rep={rep}: I={I:.6f} p={p:.4f} ({time.time()-ts:.1f}s)", flush=True)
            if len(rows) % 10 == 0:
                pd.DataFrame(rows).to_csv(SYS_CSV, mode="a", header=not os.path.exists(SYS_CSV), index=False)
                rows = []
    if rows:
        pd.DataFrame(rows).to_csv(SYS_CSV, mode="a", header=not os.path.exists(SYS_CSV), index=False)
    n_done = len(pd.read_csv(SYS_CSV)) if os.path.exists(SYS_CSV) else 0
    print(f"[systematic] 累计 {n_done}/150", flush=True)
    if n_done >= 150:
        print("ALL_DONE", flush=True)
