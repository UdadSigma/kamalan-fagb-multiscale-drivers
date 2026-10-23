import numpy as np
import rasterio
import torch
import torch.nn.functional as F
import time

def calculate_lisa_gpu_4090(tif_path, output_tif_path, p_threshold=0.05, permutations=99):
    # 1. 强制检测 GPU 并行设备
    if not torch.cuda.is_available():
        print("❌ 未检测到 GPU，请先排查 CUDA 环境！")
        return
    
    device = torch.device('cuda:0')
    print(f"🔥 成功启用旗舰显卡: {torch.cuda.get_device_name(0)}")
    print(f"当前可用显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    start_time = time.time()
    
    # 2. 读取 TIF 栅格数据
    print(f"\n正在读取栅格数据: {tif_path} ...")
    with rasterio.open(tif_path) as src:
        meta = src.meta.copy()
        band1 = src.read(1).astype(np.float32)  # GPU 运行 float32 效率极高且 4090 精度足够
        nodata = src.nodata
        
    # 直接载入显存
    data_tensor = torch.from_numpy(band1).to(device)
    
    # 3. 极速掩膜过滤
    valid_mask = (~torch.isnan(data_tensor)) & (~torch.isinf(data_tensor))
    if nodata is not None:
        valid_mask = valid_mask & (data_tensor != nodata)
    valid_mask = valid_mask & (data_tensor > 0.0)  # 剔除生物量异常值
    
    mask = valid_mask.to(torch.float32)
    N = mask.sum().item()
    print(f"有效像元总数 (N): {N:,}")
    
    # 4. 数据的标准化与去均值（Z-Score）
    clean_data = torch.where(valid_mask, data_tensor, torch.tensor(0.0, device=device))
    mean_val = clean_data.sum() / N
    z = torch.where(valid_mask, clean_data - mean_val, torch.tensor(0.0, device=device))
    std_val = torch.sqrt(torch.sum(z ** 2) / N)
    z_std = torch.where(valid_mask, z / std_val, torch.tensor(0.0, device=device))
    
    # 5. 定义一阶 Queen 邻近权重卷积核
    kernel = torch.tensor([[1., 1., 1.],
                           [1., 0., 1.],
                           [1., 1., 1.]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    
    z_unsqueezed = z_std.unsqueeze(0).unsqueeze(0)
    mask_unsqueezed = mask.unsqueeze(0).unsqueeze(0)
    
    print("正在 GPU 上执行空间邻域卷积...")
    # 空间滞后（Spatial Lag）计算
    neighbor_sum = F.conv2d(z_unsqueezed, kernel, padding=1).squeeze()
    valid_neighbor_counts = F.conv2d(mask_unsqueezed, kernel, padding=1).squeeze()
    
    # 防止孤立像元除零
    valid_neighbor_counts = torch.where(valid_neighbor_counts == 0, torch.tensor(1.0, device=device), valid_neighbor_counts)
    spatial_lag = neighbor_sum / valid_neighbor_counts
    
    # 计算局部莫兰指数 Local Moran's I
    local_moran = z_std * spatial_lag
    
    # 6. GPU 级并行的蒙特卡洛随机置换检验（Permutation Test）
    print(f"正在 4090 上全速运行置换检验 ({permutations} 次迭代)...")
    valid_indices = torch.nonzero(valid_mask, as_tuple=True)
    valid_z_values = z_std[valid_indices]
    
    larger_count = torch.zeros_like(local_moran, dtype=torch.float32, device=device)
    num_valid = len(valid_z_values)
    
    for _ in range(permutations):
        # 4090 极其擅长这种大规模一维矩阵的随机排列
        shuffled = valid_z_values[torch.randperm(num_valid, device=device)]
        z_perm = torch.zeros_like(z_std)
        z_perm[valid_indices] = shuffled
        
        # 快速卷积置换后的空间滞后
        z_perm_unsqueezed = z_perm.unsqueeze(0).unsqueeze(0)
        perm_neighbor_sum = F.conv2d(z_perm_unsqueezed, kernel, padding=1).squeeze()
        perm_spatial_lag = perm_neighbor_sum / valid_neighbor_counts
        perm_local_moran = z_perm * perm_spatial_lag
        
        # 累加满足极端条件的频次
        larger_count += (torch.abs(perm_local_moran) >= torch.abs(local_moran)).to(torch.float32)
        
    p_values = (larger_count + 1.0) / (permutations + 1.0)
    
    # 7. 划分 LISA 空间集聚类型
    print("正在进行局部冷热点空间分类...")
    lisa_class = torch.zeros_like(local_moran, dtype=torch.uint8, device=device)
    
    # 显著性条件
    significant = (p_values <= p_threshold) & valid_mask.bool()
    
    # 分类逻辑
    hh_cond = significant & (z_std > 0) & (spatial_lag > 0)
    ll_cond = significant & (z_std < 0) & (spatial_lag < 0)
    hl_cond = significant & (z_std > 0) & (spatial_lag < 0)
    lh_cond = significant & (z_std < 0) & (spatial_lag > 0)
    
    lisa_class[hh_cond] = 1  # High-High
    lisa_class[ll_cond] = 2  # Low-Low
    lisa_class[hl_cond] = 3  # High-Low
    lisa_class[lh_cond] = 4  # Low-High
    
    # 8. 回写 CPU 并输出 TIF 文件
    print(f"正在将结果写回硬盘: {output_tif_path} ...")
    lisa_class_np = lisa_class.cpu().numpy()
    
    meta.update({
        'dtype': 'uint8',
        'count': 1,
        'nodata': 0
    })
    
    with rasterio.open(output_tif_path, 'w', **meta) as dst:
        dst.write(lisa_class_np, 1)
        # 经典红蓝 LISA 图色彩映射表（ArcGIS/QGIS中可直接加载）
        dst.write_colormap(1, {
            0: (200, 200, 200, 255),  # 灰色: 非显著
            1: (255, 0, 0, 255),      # 红色: High-High (热点)
            2: (0, 0, 255, 255),      # 蓝色: Low-Low (冷点)
            3: (255, 192, 203, 255),  # 粉色: High-Low (异常)
            4: (173, 216, 230, 255)   # 浅蓝: Low-High (异常)
        })
        
    # 主动释放显存
    del data_tensor, z_std, spatial_lag, local_moran, larger_count, lisa_class
    torch.cuda.empty_cache()
    
    end_time = time.time()
    print(f"🎉 LISA计算顺利完成！4090 耗时: {end_time - start_time:.2f} 秒 (包含IO和保存时间)")
    
    # 打印最终统计
    print("\n" + "="*50)
    print("             LISA 聚类像元统计结果")
    print("="*50)
    print(f"0 - 非显著区 (Not Significant): {np.sum(lisa_class_np == 0):,} 个像元")
    print(f"1 - 高-高集聚 (High-High)     : {np.sum(lisa_class_np == 1):,} 个像元")
    print(f"2 - 低-低集聚 (Low-Low)       : {np.sum(lisa_class_np == 2):,} 个像元")
    print(f"3 - 高-低异常 (High-Low)       : {np.sum(lisa_class_np == 3):,} 个像元")
    print(f"4 - 低-高异常 (Low-High)       : {np.sum(lisa_class_np == 4):,} 个像元")
    print("="*50)

if __name__ == "__main__":
    input_tif = r"D:\python\try\2025\NW_TRY\Iterative_AGB_Model_Dynamic_V1119_1_ref+VIs_20\NEW_AGB_Prediction_Map.tif"
    output_tif = r"D:\python\try\2026\No_2\process\mappingTAGB_LISA_Clustering.tif"
    
    # 99次置换在 4090 上非常轻松，你可以直接挑战 100 次以内的极限速度
    calculate_lisa_gpu_4090(input_tif, output_tif, p_threshold=0.05, permutations=99)  