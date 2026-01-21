#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
数据集格式转换器测试脚本
用于测试转换功能和验证结果
"""

import os
import pickle
import numpy as np
import tempfile
import shutil
from pathlib import Path

def create_test_old_format_dataset(dataset_dir):
    """创建测试用的旧格式数据集"""
    dataset_dir = Path(dataset_dir)
    
    # 创建目录结构
    train_dir = dataset_dir / 'train'
    val_dir = dataset_dir / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建测试环境
    for split_dir, split_name in [(train_dir, 'train'), (val_dir, 'val')]:
        for env_id in range(2):  # 创建2个测试环境
            env_name = f"env{env_id:06d}"
            env_dir = split_dir / env_name
            env_dir.mkdir(exist_ok=True)
            
            # 创建旧格式地图文件
            tensor = np.random.rand(50, 50, 4).astype(np.float32)
            old_map_data = {
                'grid_map': tensor,
                'bounds': (-5.0, 5.0, -5.0, 5.0),
                'resolution': 0.2,
                'center': (0.0, 0.0),
                'size': (50, 50),
                'terrain_info': {'seed': 12345},
                'heightmap': np.random.rand(250, 250)
            }
            
            map_file = env_dir / 'map.p'
            with open(map_file, 'wb') as f:
                pickle.dump(old_map_data, f)
            
            # 创建旧格式路径文件
            for path_id in range(3):  # 每个环境3条路径
                path_data = np.random.rand(20, 3).astype(np.float32)  # 20个路径点，每个点3维(x,y,yaw)
                old_path_data = {
                    'path': path_data,
                    'env_id': env_id,
                    'path_id': path_id,
                    'phase': split_name
                }
                
                path_file = env_dir / f'path_{path_id}.p'
                with open(path_file, 'wb') as f:
                    pickle.dump(old_path_data, f)
            
            # 创建可视化文件（空文件用于测试）
            (env_dir / 'terrain_2d.png').touch()
            (env_dir / 'terrain_3d.png').touch()
    
    print(f"Created test old format dataset at: {dataset_dir}")
    return dataset_dir

def test_conversion():
    """测试转换功能"""
    print("="*60)
    print("Testing Dataset Format Converter")
    print("="*60)
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 创建测试数据
        old_dataset_dir = temp_path / 'old_dataset'
        new_dataset_dir = temp_path / 'new_dataset'
        
        print("\n1. Creating test old format dataset...")
        create_test_old_format_dataset(old_dataset_dir)
        
        # 导入转换器
        print("\n2. Importing converter...")
        import sys
        sys.path.append(str(Path(__file__).parent))
        from dataset_format_converter import DatasetFormatConverter
        
        # 创建转换器实例
        print("\n3. Creating converter instance...")
        converter = DatasetFormatConverter(
            old_dataset_dir=old_dataset_dir,
            new_dataset_dir=new_dataset_dir,
            backup=True
        )
        
        # 执行转换
        print("\n4. Converting dataset...")
        converter.convert_dataset()
        
        # 验证转换结果
        print("\n5. Verifying conversion results...")
        
        # 检查新格式文件
        for split in ['train', 'val']:
            for env_id in range(2):
                env_name = f"env{env_id:06d}"
                
                # 检查地图文件
                new_map_file = new_dataset_dir / split / env_name / 'map.p'
                if new_map_file.exists():
                    with open(new_map_file, 'rb') as f:
                        new_data = pickle.load(f)
                    
                    # 验证新格式字段
                    required_keys = ['tensor', 'bounds', 'resolution', 'map_name', 'channels', 'shape']
                    missing_keys = [key for key in required_keys if key not in new_data]
                    if missing_keys:
                        print(f"  ERROR: Missing keys in {split}/{env_name}/map.p: {missing_keys}")
                        return False
                    
                    if new_data['map_name'] != env_name:
                        print(f"  ERROR: Wrong map_name in {split}/{env_name}/map.p: {new_data['map_name']}")
                        return False
                    
                    if new_data['tensor'].shape != (50, 50, 4):
                        print(f"  ERROR: Wrong tensor shape in {split}/{env_name}/map.p: {new_data['tensor'].shape}")
                        return False
                    
                    print(f"  ✓ {split}/{env_name}/map.p verified")
                
                # 检查路径文件
                for path_id in range(3):
                    new_path_file = new_dataset_dir / split / env_name / f'path_{path_id}.p'
                    if new_path_file.exists():
                        with open(new_path_file, 'rb') as f:
                            new_path_data = pickle.load(f)
                        
                        # 验证新格式字段
                        if 'path' not in new_path_data or 'map_name' not in new_path_data:
                            print(f"  ERROR: Missing keys in {split}/{env_name}/path_{path_id}.p")
                            return False
                        
                        if new_path_data['map_name'] != env_name:
                            print(f"  ERROR: Wrong map_name in path file")
                            return False
                        
                        if len(new_path_data) != 2:  # 应该只有path和map_name两个字段
                            print(f"  ERROR: Too many keys in new path format: {list(new_path_data.keys())}")
                            return False
                        
                        print(f"  ✓ {split}/{env_name}/path_{path_id}.p verified")
        
        # 检查备份
        backup_dir = new_dataset_dir / 'backup'
        if not backup_dir.exists():
            print("  ERROR: Backup directory not created")
            return False
        
        print("  ✓ Backup directory verified")
        
        # 使用内置验证功能
        print("\n6. Running built-in verification...")
        result = converter.verify_conversion('env000000', 'train')
        if not result:
            print("  ERROR: Built-in verification failed")
            return False
        
        print("  ✓ Built-in verification passed")
        
        print("\n" + "="*60)
        print("All tests PASSED! ✓")
        print("Dataset format converter is working correctly.")
        print("="*60)
        
        return True

def create_sample_dataset_for_real_test():
    """创建一个真实的示例数据集用于实际测试"""
    sample_dir = Path.home() / 'sample_old_dataset'
    
    if sample_dir.exists():
        print(f"Sample dataset already exists at: {sample_dir}")
        return sample_dir
    
    print(f"Creating sample old format dataset at: {sample_dir}")
    create_test_old_format_dataset(sample_dir)
    
    print("\nYou can now test the converter with:")
    print(f"python3 dataset_format_converter.py {sample_dir} ~/sample_new_dataset")
    
    return sample_dir

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test dataset format converter')
    parser.add_argument('--create-sample', action='store_true', 
                       help='Create a sample old format dataset for testing')
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_dataset_for_real_test()
    else:
        test_conversion()
