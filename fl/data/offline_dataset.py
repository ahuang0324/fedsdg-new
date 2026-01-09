#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Offline preprocessed dataset classes.

Features:
  1. Zero CPU overhead for Resize (done offline)
  2. Multi-process shared memory (mmap_mode='r')
  3. Fast data loading, improved GPU utilization
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from ..utils.paths import DATA_DIR, PREPROCESSED_DATA_DIR


class OfflineCIFAR10(Dataset):
    """
    Offline preprocessed CIFAR-10 dataset.
    
    Uses numpy memmap to load preprocessed data, avoiding real-time Resize.
    Supports multi-process shared memory, greatly reducing CPU load.
    
    Args:
        root: Root directory for preprocessed data
        train: True for training set, False for test set
        image_size: Image size (must match preprocessing)
        transform: Additional transforms (optional, usually only Normalize)
        use_imagenet_norm: Whether to use ImageNet normalization (for pretrained models)
    """
    
    def __init__(self, root=None, train=True, image_size=224,
                 transform=None, use_imagenet_norm=True):
        if root is None:
            root = PREPROCESSED_DATA_DIR
        self.root = root
        self.train = train
        self.image_size = image_size
        self.use_imagenet_norm = use_imagenet_norm
        
        # Build data path
        data_dir = os.path.join(root, f'cifar10_{image_size}x{image_size}')
        split = 'train' if train else 'test'
        
        self.images_path = os.path.join(data_dir, f'{split}_images.npy')
        self.labels_path = os.path.join(data_dir, f'{split}_labels.npy')
        
        # Check files exist
        if not os.path.exists(self.images_path):
            raise FileNotFoundError(
                f"Preprocessed image file not found: {self.images_path}\n"
                f"Please run: python scripts/preprocess/preprocess_cifar10.py --image_size {image_size}"
            )
        if not os.path.exists(self.labels_path):
            raise FileNotFoundError(
                f"Preprocessed label file not found: {self.labels_path}\n"
                f"Please run: python scripts/preprocess/preprocess_cifar10.py --image_size {image_size}"
            )
        
        # Load data with memory mapping
        print(f"[OfflineCIFAR10] Loading preprocessed data (memory-mapped mode)...")
        print(f"  - Image file: {self.images_path}")
        print(f"  - Label file: {self.labels_path}")
        
        num_samples = 50000 if train else 10000
        
        self.images = np.memmap(
            self.images_path,
            dtype='float32',
            mode='r',
            shape=(num_samples, 3, image_size, image_size)
        )
        
        self.labels = np.load(self.labels_path, allow_pickle=True)
        
        print(f"  - Samples: {len(self.labels)}")
        print(f"  - Image shape: {self.images.shape}")
        print(f"  - Memory mapping: enabled (shared memory)")
        
        # Setup transform
        if transform is not None:
            self.transform = transform
        else:
            if use_imagenet_norm:
                normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
                print(f"  - Normalization: ImageNet (pretrained model)")
            else:
                normalize = transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
                print(f"  - Normalization: Standard (from scratch)")
            
            self.transform = normalize
        
        # Compatibility attributes
        self.targets = self.labels.tolist()
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck']
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = int(self.labels[idx])
        
        image = torch.from_numpy(image.copy())
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label
    
    def __repr__(self):
        split = "Train" if self.train else "Test"
        return (f"OfflineCIFAR10(split={split}, size={len(self)}, "
                f"image_size={self.image_size}x{self.image_size}, "
                f"imagenet_norm={self.use_imagenet_norm})")


class OfflineCIFAR100(Dataset):
    """
    Offline preprocessed CIFAR-100 dataset.
    
    Same features as OfflineCIFAR10 but for CIFAR-100 (100 classes).
    """
    
    def __init__(self, root=None, train=True, image_size=224,
                 transform=None, use_imagenet_norm=True):
        if root is None:
            root = PREPROCESSED_DATA_DIR
        self.root = root
        self.train = train
        self.image_size = image_size
        self.use_imagenet_norm = use_imagenet_norm
        
        data_dir = os.path.join(root, f'cifar100_{image_size}x{image_size}')
        split = 'train' if train else 'test'
        
        self.images_path = os.path.join(data_dir, f'{split}_images.npy')
        self.labels_path = os.path.join(data_dir, f'{split}_labels.npy')
        
        if not os.path.exists(self.images_path):
            raise FileNotFoundError(
                f"Preprocessed image file not found: {self.images_path}\n"
                f"Please run: python scripts/preprocess/preprocess_cifar100.py"
            )
        if not os.path.exists(self.labels_path):
            raise FileNotFoundError(
                f"Preprocessed label file not found: {self.labels_path}\n"
                f"Please run: python scripts/preprocess/preprocess_cifar100.py"
            )
        
        print(f"[OfflineCIFAR100] Loading preprocessed data (memory-mapped mode)...")
        print(f"  - Image file: {self.images_path}")
        print(f"  - Label file: {self.labels_path}")
        
        num_samples = 50000 if train else 10000
        
        self.images = np.memmap(
            self.images_path,
            dtype='float32',
            mode='r',
            shape=(num_samples, 3, image_size, image_size)
        )
        
        self.labels = np.load(self.labels_path, allow_pickle=True)
        
        print(f"  - Samples: {len(self.labels)}")
        print(f"  - Image shape: {self.images.shape}")
        print(f"  - Memory mapping: enabled (shared memory)")
        
        if transform is not None:
            self.transform = transform
        else:
            if use_imagenet_norm:
                normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
                print(f"  - Normalization: ImageNet (pretrained model)")
            else:
                normalize = transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
                print(f"  - Normalization: Standard (from scratch)")
            
            self.transform = normalize
        
        self.targets = self.labels.tolist()
        self.num_classes = 100
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = int(self.labels[idx])
        
        image = torch.from_numpy(image.copy())
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label
    
    def __repr__(self):
        split = "Train" if self.train else "Test"
        return (f"OfflineCIFAR100(split={split}, size={len(self)}, "
                f"image_size={self.image_size}x{self.image_size}, "
                f"imagenet_norm={self.use_imagenet_norm})")


def get_offline_cifar10(root=None, image_size=224, use_imagenet_norm=True):
    """Convenience function to get offline CIFAR-10 train and test datasets."""
    if root is None:
        root = PREPROCESSED_DATA_DIR
    train_dataset = OfflineCIFAR10(root=root, train=True, image_size=image_size, use_imagenet_norm=use_imagenet_norm)
    test_dataset = OfflineCIFAR10(root=root, train=False, image_size=image_size, use_imagenet_norm=use_imagenet_norm)
    return train_dataset, test_dataset


def get_offline_cifar100(root=None, image_size=224, use_imagenet_norm=True):
    """Convenience function to get offline CIFAR-100 train and test datasets."""
    if root is None:
        root = PREPROCESSED_DATA_DIR
    train_dataset = OfflineCIFAR100(root=root, train=True, image_size=image_size, use_imagenet_norm=use_imagenet_norm)
    test_dataset = OfflineCIFAR100(root=root, train=False, image_size=image_size, use_imagenet_norm=use_imagenet_norm)
    return train_dataset, test_dataset

