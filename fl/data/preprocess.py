#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据预处理模块

将原始数据集预处理为 memmap 格式，供 OfflineCIFAR10/OfflineCIFAR100 类使用。

Usage:
    python -m fl.data.preprocess --dataset cifar100 --image_size 224
    python -m fl.data.preprocess --dataset cifar10 --image_size 224 128
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import datasets, transforms

from ..utils.paths import DATA_DIR, PREPROCESSED_DATA_DIR


# =============================================================================
# 支持的数据集配置
# =============================================================================
DATASET_CONFIG = {
    'cifar10': {
        'class': datasets.CIFAR10,
        'train_samples': 50000,
        'test_samples': 10000,
        'num_classes': 10,
    },
    'cifar100': {
        'class': datasets.CIFAR100,
        'train_samples': 50000,
        'test_samples': 10000,
        'num_classes': 100,
    },
}


def preprocess_dataset(dataset_name: str, image_size: int, force: bool = False) -> bool:
    """
    预处理单个数据集
    
    Args:
        dataset_name: 数据集名称 (cifar10, cifar100)
        image_size: 目标图像尺寸
        force: 是否强制覆盖已存在的文件
    
    Returns:
        bool: 是否成功
    """
    if dataset_name not in DATASET_CONFIG:
        print(f"[Error] Unsupported dataset: {dataset_name}")
        print(f"[Error] Supported: {list(DATASET_CONFIG.keys())}")
        return False
    
    config = DATASET_CONFIG[dataset_name]
    data_root = os.path.join(DATA_DIR, dataset_name)
    output_dir = os.path.join(PREPROCESSED_DATA_DIR, f'{dataset_name}_{image_size}x{image_size}')
    
    print(f"\n{'='*60}")
    print(f"Preprocessing {dataset_name.upper()} @ {image_size}x{image_size}")
    print(f"{'='*60}")
    print(f"  Data root: {data_root}")
    print(f"  Output dir: {output_dir}")
    
    # 检查是否已存在
    train_images_path = os.path.join(output_dir, 'train_images.npy')
    if os.path.exists(train_images_path) and not force:
        print(f"\n[Skip] Already exists: {train_images_path}")
        print(f"[Skip] Use --force to overwrite")
        return True
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 下载数据集
    print(f"\n[1/3] Downloading {dataset_name.upper()}...")
    DatasetClass = config['class']
    train_dataset = DatasetClass(data_root, train=True, download=True)
    test_dataset = DatasetClass(data_root, train=False, download=True)
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    # 预处理 transform
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    # 处理训练集和测试集
    for split, dataset, num_samples in [
        ('train', train_dataset, config['train_samples']),
        ('test', test_dataset, config['test_samples']),
    ]:
        print(f"\n[2/3] Processing {split} set...")
        
        images_path = os.path.join(output_dir, f'{split}_images.npy')
        labels_path = os.path.join(output_dir, f'{split}_labels.npy')
        
        # 创建 memmap 文件
        images = np.memmap(
            images_path, dtype='float32', mode='w+',
            shape=(num_samples, 3, image_size, image_size)
        )
        labels = np.zeros(num_samples, dtype=np.int64)
        
        for idx in tqdm(range(num_samples), desc=f"  {split}"):
            img, label = dataset[idx]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            images[idx] = transform(img).numpy()
            labels[idx] = label
        
        images.flush()
        np.save(labels_path, labels)
        
        size_mb = os.path.getsize(images_path) / (1024 * 1024)
        print(f"  Saved: {images_path} ({size_mb:.1f} MB)")
    
    # 验证
    print(f"\n[3/3] Verifying...")
    try:
        images = np.memmap(
            os.path.join(output_dir, 'train_images.npy'),
            dtype='float32', mode='r',
            shape=(config['train_samples'], 3, image_size, image_size)
        )
        labels = np.load(os.path.join(output_dir, 'train_labels.npy'))
        print(f"  Image shape: {images.shape}")
        print(f"  Label range: [{labels.min()}, {labels.max()}]")
        print(f"  ✓ Verification passed")
    except Exception as e:
        print(f"  ✗ Verification failed: {e}")
        return False
    
    print(f"\n{'='*60}")
    print(f"✓ {dataset_name.upper()} @ {image_size}x{image_size} completed!")
    print(f"{'='*60}")
    return True


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='Preprocess datasets for offline loading',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m fl.data.preprocess --dataset cifar100 --image_size 224
  python -m fl.data.preprocess --dataset cifar10 --image_size 224 128
  python -m fl.data.preprocess --dataset cifar100 --image_size 224 --force
        """
    )
    parser.add_argument('--dataset', type=str, required=True,
                        choices=list(DATASET_CONFIG.keys()),
                        help='Dataset name')
    parser.add_argument('--image_size', type=int, nargs='+', required=True,
                        help='Target image size(s)')
    parser.add_argument('--force', action='store_true',
                        help='Force overwrite existing files')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Data Preprocessing Tool")
    print("="*60)
    print(f"  Dataset: {args.dataset}")
    print(f"  Image sizes: {args.image_size}")
    print(f"  Force: {args.force}")
    print(f"  Output: {PREPROCESSED_DATA_DIR}")
    
    success = True
    for size in args.image_size:
        if not preprocess_dataset(args.dataset, size, args.force):
            success = False
    
    if success:
        print(f"\n✓ All preprocessing completed!")
    else:
        print(f"\n✗ Some preprocessing failed!")
        exit(1)


if __name__ == '__main__':
    main()
