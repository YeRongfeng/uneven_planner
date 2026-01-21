#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
数据集格式转换器
将旧格式的数据集转换为新格式，避免重新生成数据

旧格式 map.p:
{
    'grid_map': tensor,
    'bounds': bounds,
    'resolution': resolution,
    'center': center,
    'size': size,
    'terrain_info': terrain_info,
    'heightmap': heightmap
}

新格式 map.p:
{
    'tensor': tensor,
    'bounds': bounds,
    'resolution': resolution,
    'map_name': env_name,
    'channels': ['elevation', 'normal_x', 'normal_y', 'normal_z'],
    'shape': tensor.shape
}

旧格式 path_*.p:
{
    'path': trajectory_path,
    'env_id': env_id,
    'path_id': path_id,
    'phase': phase
}

新格式 path_*.p:
{
    'path': trajectory_path,
    'map_name': env_name
}
"""

import os
import pickle
import shutil
import argparse
from pathlib import Path


class DatasetFormatConverter:
    """数据集格式转换器"""
    
    def __init__(self, old_dataset_dir, new_dataset_dir, backup=True):
        """
        初始化转换器
        
        Args:
            old_dataset_dir: 旧格式数据集目录
            new_dataset_dir: 新格式数据集输出目录
            backup: 是否备份原始数据
        """
        self.old_dataset_dir = Path(old_dataset_dir)
        self.new_dataset_dir = Path(new_dataset_dir)
        self.backup = backup
        
        # 验证输入目录
        if not self.old_dataset_dir.exists():
            raise ValueError(f"Old dataset directory does not exist: {old_dataset_dir}")
        
        # 创建输出目录
        self.new_dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建备份目录（如果需要）
        if self.backup:
            self.backup_dir = self.new_dataset_dir / "backup"
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            
        print(f"Dataset Format Converter initialized:")
        print(f"  Old dataset: {self.old_dataset_dir}")
        print(f"  New dataset: {self.new_dataset_dir}")
        print(f"  Backup enabled: {self.backup}")
    
    def convert_map_file(self, old_map_file, new_map_file, env_name):
        """
        转换单个地图文件
        
        Args:
            old_map_file: 旧格式地图文件路径
            new_map_file: 新格式地图文件路径
            env_name: 环境名称
        """
        try:
            # 加载旧格式数据
            with open(old_map_file, 'rb') as f:
                old_data = pickle.load(f)
            
            # 检查旧格式数据的键
            if 'grid_map' in old_data:
                tensor = old_data['grid_map']
            elif 'tensor' in old_data:
                # 已经是新格式，直接复制
                print(f"  Map file already in new format: {old_map_file.name}")
                shutil.copy2(old_map_file, new_map_file)
                return True
            else:
                print(f"  Warning: No 'grid_map' or 'tensor' key found in {old_map_file.name}")
                return False
            
            # 创建新格式数据
            new_data = {
                'tensor': tensor,
                'bounds': old_data.get('bounds', (-5.0, 5.0, -5.0, 5.0)),
                'resolution': old_data.get('resolution', 0.2),
                'map_name': env_name,
                'channels': ['elevation', 'normal_x', 'normal_y', 'normal_z'],
                'shape': tensor.shape
            }
            
            # 保存新格式数据
            with open(new_map_file, 'wb') as f:
                pickle.dump(new_data, f)
            
            print(f"  Converted map file: {old_map_file.name} -> {new_map_file.name}")
            print(f"    Tensor shape: {tensor.shape}")
            print(f"    Resolution: {new_data['resolution']}")
            print(f"    Bounds: {new_data['bounds']}")
            
            return True
            
        except Exception as e:
            print(f"  Error converting map file {old_map_file.name}: {e}")
            return False
    
    def convert_path_file(self, old_path_file, new_path_file, env_name):
        """
        转换单个路径文件
        
        Args:
            old_path_file: 旧格式路径文件路径
            new_path_file: 新格式路径文件路径
            env_name: 环境名称
        """
        try:
            # 加载旧格式数据
            with open(old_path_file, 'rb') as f:
                old_data = pickle.load(f)
            
            # 检查是否已经是新格式
            if 'map_name' in old_data and len(old_data) == 2:
                # 已经是新格式，直接复制
                print(f"    Path file already in new format: {old_path_file.name}")
                shutil.copy2(old_path_file, new_path_file)
                return True
            
            # 创建新格式数据
            new_data = {
                'path': old_data['path'],
                'map_name': env_name
            }
            
            # 保存新格式数据
            with open(new_path_file, 'wb') as f:
                pickle.dump(new_data, f)
            
            print(f"    Converted path file: {old_path_file.name} -> {new_path_file.name}")
            print(f"      Path points: {len(old_data['path'])}")
            
            return True
            
        except Exception as e:
            print(f"    Error converting path file {old_path_file.name}: {e}")
            return False
    
    def convert_environment(self, old_env_dir, new_env_dir):
        """
        转换单个环境目录
        
        Args:
            old_env_dir: 旧格式环境目录
            new_env_dir: 新格式环境目录
        """
        env_name = old_env_dir.name
        print(f"Converting environment: {env_name}")
        
        # 创建新环境目录
        new_env_dir.mkdir(parents=True, exist_ok=True)
        
        # 备份原始数据（如果需要）
        if self.backup:
            backup_env_dir = self.backup_dir / env_name
            if not backup_env_dir.exists():
                shutil.copytree(old_env_dir, backup_env_dir)
                print(f"  Backed up to: {backup_env_dir}")
        
        converted_files = 0
        total_files = 0
        
        # 转换地图文件
        old_map_file = old_env_dir / 'map.p'
        if old_map_file.exists():
            new_map_file = new_env_dir / 'map.p'
            total_files += 1
            if self.convert_map_file(old_map_file, new_map_file, env_name):
                converted_files += 1
        else:
            print(f"  Warning: No map.p found in {env_name}")
        
        # 转换路径文件
        path_files = list(old_env_dir.glob('path_*.p'))
        total_files += len(path_files)
        
        print(f"  Converting {len(path_files)} path files...")
        for old_path_file in path_files:
            new_path_file = new_env_dir / old_path_file.name
            if self.convert_path_file(old_path_file, new_path_file, env_name):
                converted_files += 1
        
        # 复制可视化文件
        visualization_files = ['terrain_2d.png', 'terrain_3d.png']
        for vis_file in visualization_files:
            old_vis_file = old_env_dir / vis_file
            if old_vis_file.exists():
                new_vis_file = new_env_dir / vis_file
                shutil.copy2(old_vis_file, new_vis_file)
                print(f"  Copied visualization: {vis_file}")
                converted_files += 1
                total_files += 1
        
        print(f"  Environment {env_name}: {converted_files}/{total_files} files converted")
        return converted_files, total_files
    
    def convert_dataset_split(self, split_name):
        """
        转换数据集的一个分割（train或val）
        
        Args:
            split_name: 分割名称（'train' 或 'val'）
        """
        print(f"\n=== Converting {split_name} split ===")
        
        old_split_dir = self.old_dataset_dir / split_name
        new_split_dir = self.new_dataset_dir / split_name
        
        if not old_split_dir.exists():
            print(f"Split directory does not exist: {old_split_dir}")
            return 0, 0
        
        # 创建新分割目录
        new_split_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取所有环境目录
        env_dirs = [d for d in old_split_dir.iterdir() if d.is_dir() and d.name.startswith('env')]
        env_dirs.sort()  # 确保按顺序处理
        
        print(f"Found {len(env_dirs)} environments in {split_name} split")
        
        total_converted = 0
        total_files = 0
        
        for old_env_dir in env_dirs:
            new_env_dir = new_split_dir / old_env_dir.name
            converted, files = self.convert_environment(old_env_dir, new_env_dir)
            total_converted += converted
            total_files += files
        
        print(f"{split_name} split conversion completed: {total_converted}/{total_files} files")
        return total_converted, total_files
    
    def convert_dataset(self):
        """转换整个数据集"""
        print("\n" + "="*60)
        print("Starting dataset format conversion...")
        print("="*60)
        
        total_converted = 0
        total_files = 0
        
        # 转换训练集
        train_converted, train_files = self.convert_dataset_split('train')
        total_converted += train_converted
        total_files += train_files
        
        # 转换验证集
        val_converted, val_files = self.convert_dataset_split('val')
        total_converted += val_converted
        total_files += val_files
        
        # 转换测试集（如果存在）
        if (self.old_dataset_dir / 'test').exists():
            test_converted, test_files = self.convert_dataset_split('test')
            total_converted += test_converted
            total_files += test_files
        
        # 复制其他文件（如README等）
        other_files = [f for f in self.old_dataset_dir.iterdir() 
                      if f.is_file() and f.name not in ['train', 'val', 'test']]
        for other_file in other_files:
            new_other_file = self.new_dataset_dir / other_file.name
            shutil.copy2(other_file, new_other_file)
            print(f"Copied additional file: {other_file.name}")
            total_converted += 1
            total_files += 1
        
        print("\n" + "="*60)
        print("Dataset format conversion completed!")
        print(f"Total files processed: {total_files}")
        print(f"Total files converted: {total_converted}")
        print(f"Success rate: {total_converted/max(total_files,1)*100:.1f}%")
        print(f"New dataset location: {self.new_dataset_dir}")
        if self.backup:
            print(f"Backup location: {self.backup_dir}")
        print("="*60)
    
    def verify_conversion(self, env_name, split='train'):
        """
        验证转换结果
        
        Args:
            env_name: 环境名称
            split: 数据集分割
        """
        print(f"\n=== Verifying conversion for {env_name} ({split}) ===")
        
        # 检查旧格式文件
        old_env_dir = self.old_dataset_dir / split / env_name
        old_map_file = old_env_dir / 'map.p'
        
        # 检查新格式文件
        new_env_dir = self.new_dataset_dir / split / env_name
        new_map_file = new_env_dir / 'map.p'
        
        if not old_map_file.exists():
            print(f"Old map file not found: {old_map_file}")
            return False
        
        if not new_map_file.exists():
            print(f"New map file not found: {new_map_file}")
            return False
        
        try:
            # 加载并比较数据
            with open(old_map_file, 'rb') as f:
                old_data = pickle.load(f)
            
            with open(new_map_file, 'rb') as f:
                new_data = pickle.load(f)
            
            # 检查tensor数据
            if 'grid_map' in old_data:
                old_tensor = old_data['grid_map']
            else:
                old_tensor = old_data['tensor']
            
            new_tensor = new_data['tensor']
            
            # 比较tensor形状和部分数据
            print(f"Old tensor shape: {old_tensor.shape}")
            print(f"New tensor shape: {new_tensor.shape}")
            
            if old_tensor.shape != new_tensor.shape:
                print("ERROR: Tensor shapes don't match!")
                return False
            
            # 检查数据是否相同（检查几个点）
            import numpy as np
            if not np.array_equal(old_tensor, new_tensor):
                print("ERROR: Tensor data doesn't match!")
                return False
            
            # 检查新格式必需字段
            required_keys = ['tensor', 'bounds', 'resolution', 'map_name', 'channels', 'shape']
            for key in required_keys:
                if key not in new_data:
                    print(f"ERROR: Missing required key in new format: {key}")
                    return False
            
            print(f"Map file verification PASSED")
            
            # 检查路径文件
            old_path_files = list(old_env_dir.glob('path_*.p'))
            new_path_files = list(new_env_dir.glob('path_*.p'))
            
            if len(old_path_files) != len(new_path_files):
                print(f"ERROR: Path file count mismatch: {len(old_path_files)} vs {len(new_path_files)}")
                return False
            
            # 检查第一个路径文件
            if old_path_files and new_path_files:
                with open(old_path_files[0], 'rb') as f:
                    old_path_data = pickle.load(f)
                
                with open(new_path_files[0], 'rb') as f:
                    new_path_data = pickle.load(f)
                
                if not np.array_equal(old_path_data['path'], new_path_data['path']):
                    print("ERROR: Path data doesn't match!")
                    return False
                
                if new_path_data['map_name'] != env_name:
                    print(f"ERROR: Map name mismatch: {new_path_data['map_name']} vs {env_name}")
                    return False
            
            print(f"Path files verification PASSED")
            print(f"Verification for {env_name} completed successfully!")
            return True
            
        except Exception as e:
            print(f"ERROR during verification: {e}")
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Convert dataset from old format to new format')
    parser.add_argument('old_dataset', help='Path to old format dataset directory')
    parser.add_argument('new_dataset', help='Path to new format dataset output directory')
    parser.add_argument('--no-backup', action='store_true', help='Disable backup of original data')
    parser.add_argument('--verify', help='Verify conversion for specific environment (e.g., env000001)')
    parser.add_argument('--verify-split', default='train', help='Dataset split for verification (default: train)')
    
    args = parser.parse_args()
    
    # 创建转换器
    converter = DatasetFormatConverter(
        old_dataset_dir=args.old_dataset,
        new_dataset_dir=args.new_dataset,
        backup=not args.no_backup
    )
    
    if args.verify:
        # 仅验证指定环境
        converter.verify_conversion(args.verify, args.verify_split)
    else:
        # 执行完整转换
        converter.convert_dataset()
        
        # 验证第一个环境作为样本
        old_train_dir = Path(args.old_dataset) / 'train'
        if old_train_dir.exists():
            env_dirs = [d for d in old_train_dir.iterdir() if d.is_dir() and d.name.startswith('env')]
            if env_dirs:
                first_env = sorted(env_dirs)[0].name
                print(f"\nPerforming sample verification on {first_env}...")
                converter.verify_conversion(first_env, 'train')


if __name__ == '__main__':
    main()
