#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
性能优化验证脚本
用于测试GPU缓存和批量规划的效果
"""

import torch
import numpy as np
import time
import sys

def test_gpu_availability():
    """测试GPU是否可用"""
    print("=" * 60)
    print("GPU可用性测试")
    print("=" * 60)
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA可用: {cuda_available}")
    
    if cuda_available:
        print(f"GPU设备数: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  设备 {i}: {torch.cuda.get_device_name(i)}")
        print(f"当前设备: {torch.cuda.current_device()}")
    else:
        print("警告: 未检测到CUDA支持的GPU")
        print("优化效果将受限，建议安装CUDA版PyTorch")
    
    return cuda_available

def test_gpu_cache_performance():
    """测试GPU缓存性能"""
    print("\n" + "=" * 60)
    print("GPU缓存性能测试")
    print("=" * 60)
    
    # 模拟地形法向量数据（100x100）
    H, W = 100, 100
    normal_x = np.random.randn(H, W).astype(np.float32)
    normal_y = np.random.randn(H, W).astype(np.float32)
    normal_z = np.abs(np.random.randn(H, W).astype(np.float32))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试1: 无缓存（重复转换）
    print("\n测试1: 无缓存（重复CPU->GPU传输）")
    times_no_cache = []
    for i in range(10):
        start = time.time()
        tx = torch.tensor(normal_x, dtype=torch.float32, device=device)
        ty = torch.tensor(normal_y, dtype=torch.float32, device=device)
        tz = torch.tensor(normal_z, dtype=torch.float32, device=device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - start
        times_no_cache.append(elapsed)
        print(f"  迭代 {i+1}: {elapsed*1000:.2f}ms")
    
    avg_no_cache = np.mean(times_no_cache) * 1000
    print(f"平均时间（无缓存）: {avg_no_cache:.2f}ms")
    
    # 测试2: 有缓存（只转换一次）
    print("\n测试2: 有缓存（仅首次传输）")
    cached_x = torch.tensor(normal_x, dtype=torch.float32, device=device)
    cached_y = torch.tensor(normal_y, dtype=torch.float32, device=device)
    cached_z = torch.tensor(normal_z, dtype=torch.float32, device=device)
    
    times_with_cache = []
    for i in range(10):
        start = time.time()
        # 直接使用缓存的张量（无数据传输）
        tx = cached_x
        ty = cached_y
        tz = cached_z
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - start
        times_with_cache.append(elapsed)
        print(f"  迭代 {i+1}: {elapsed*1000:.2f}ms")
    
    avg_with_cache = np.mean(times_with_cache) * 1000
    print(f"平均时间（有缓存）: {avg_with_cache:.2f}ms")
    
    # 性能提升
    if avg_no_cache > 0:
        speedup = (avg_no_cache - avg_with_cache) / avg_no_cache * 100
        print(f"\n性能提升: {speedup:.1f}%")
        print(f"加速比: {avg_no_cache/avg_with_cache:.2f}x")
    
    return avg_no_cache, avg_with_cache

def test_tensor_operations():
    """测试张量操作性能"""
    print("\n" + "=" * 60)
    print("张量操作性能测试")
    print("=" * 60)
    
    H, W, Y = 100, 100, 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    data = torch.randn(H, W, Y, device=device)
    
    # 测试向量化操作
    print("\n测试向量化操作...")
    start = time.time()
    for _ in range(100):
        result = torch.where(data > 0, torch.ones_like(data), torch.zeros_like(data))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"100次向量化操作: {elapsed*1000:.2f}ms")
    print(f"单次操作: {elapsed*10:.2f}ms")
    
    return elapsed

def estimate_optimization_benefit(map_size_list=[40, 80, 100]):
    """估算不同地图尺寸的优化收益"""
    print("\n" + "=" * 60)
    print("优化收益估算")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n设备: {device}")
    print(f"{'地图尺寸':<15} {'无缓存(ms)':<15} {'有缓存(ms)':<15} {'提升':<10}")
    print("-" * 60)
    
    for size in map_size_list:
        # 模拟数据
        normal_x = np.random.randn(size, size).astype(np.float32)
        
        # 无缓存
        times_no_cache = []
        for _ in range(5):
            start = time.time()
            tx = torch.tensor(normal_x, dtype=torch.float32, device=device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times_no_cache.append((time.time() - start) * 1000)
        avg_no_cache = np.mean(times_no_cache)
        
        # 有缓存
        cached_x = torch.tensor(normal_x, dtype=torch.float32, device=device)
        times_with_cache = []
        for _ in range(5):
            start = time.time()
            tx = cached_x
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times_with_cache.append((time.time() - start) * 1000)
        avg_with_cache = np.mean(times_with_cache)
        
        speedup = (avg_no_cache - avg_with_cache) / avg_no_cache * 100 if avg_no_cache > 0 else 0
        print(f"{size}x{size:<10} {avg_no_cache:<15.2f} {avg_with_cache:<15.2f} {speedup:<10.1f}%")

def main():
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "          性能优化验证工具          ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "═" * 58 + "╝")
    
    # 测试1: GPU可用性
    gpu_available = test_gpu_availability()
    
    if not gpu_available:
        print("\n警告: 未检测到GPU，性能优化效果有限")
        print("建议安装CUDA版PyTorch以获得最佳性能")
        response = input("\n是否继续CPU测试? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # 测试2: GPU缓存性能
    avg_no_cache, avg_with_cache = test_gpu_cache_performance()
    
    # 测试3: 张量操作性能
    test_tensor_operations()
    
    # 测试4: 不同地图尺寸的优化收益
    estimate_optimization_benefit()
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    if avg_no_cache > 0 and avg_with_cache > 0:
        speedup = avg_no_cache / avg_with_cache
        time_saved_per_map = (avg_no_cache - avg_with_cache) / 1000  # 秒
        
        print(f"\nGPU缓存优化:")
        print(f"  - 加速比: {speedup:.2f}x")
        print(f"  - 每张地图节省时间: {time_saved_per_map:.3f}秒")
        print(f"  - 1000张地图节省时间: {time_saved_per_map*1000/60:.1f}分钟")
        
        if gpu_available:
            print(f"\n设备信息:")
            print(f"  - GPU: {torch.cuda.get_device_name(0)}")
            print(f"  - 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        print(f"\n建议:")
        if speedup > 2:
            print("  ✓ GPU缓存优化效果显著，强烈建议启用")
        elif speedup > 1.5:
            print("  ✓ GPU缓存优化有效，建议启用")
        else:
            print("  - GPU缓存优化效果一般，可根据实际情况选择")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

if __name__ == '__main__':
    main()
