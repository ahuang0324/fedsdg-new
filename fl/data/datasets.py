#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dataset loading utilities for federated learning.

Supports:
- MNIST, Fashion-MNIST
- CIFAR-10, CIFAR-100
- Offline preprocessed datasets
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from ..utils.paths import DATA_DIR, PREPROCESSED_DATA_DIR, VISUALIZATIONS_DIR, ensure_dir
from .sampling import dirichlet_partition_train_test
from .offline_dataset import OfflineCIFAR10, OfflineCIFAR100


class DatasetSplit(Dataset):
    """A Dataset class that wraps around another dataset with specific indices."""

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def get_dataset(args):
    """
    Returns train and test datasets and user groups for federated learning.
    
    The user groups are dictionaries where keys are client indices and 
    values are the corresponding data indices for each client.
    
    Args:
        args: Command line arguments with dataset configuration
    
    Returns:
        train_dataset: Training dataset
        test_dataset: Test dataset
        user_groups: {client_id: train_indices} for training
        user_groups_test: {client_id: test_indices} for evaluation
    """

    if args.dataset == 'cifar':
        # Check if using offline preprocessed data
        if hasattr(args, 'use_offline_data') and args.use_offline_data:
            print("\n" + "="*70)
            print("[Offline Data] Using offline preprocessed data mode".center(70))
            print("="*70)
            
            # Use path from args or default
            offline_root = getattr(args, 'offline_data_root', PREPROCESSED_DATA_DIR)
            # Convert relative path to absolute if needed
            if not os.path.isabs(offline_root):
                offline_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), offline_root.lstrip('./').lstrip('../'))
            
            print(f"[Offline Data] Data path: {offline_root}")
            print(f"[Offline Data] Image size: {args.image_size}x{args.image_size}")
            
            use_imagenet_norm = (hasattr(args, 'model_variant') and 
                                args.model_variant == 'pretrained')
            
            if use_imagenet_norm:
                print(f"[Offline Data] Normalization: ImageNet (pretrained model)")
            else:
                print(f"[Offline Data] Normalization: Standard (from scratch)")
            
            print("="*70 + "\n")
            
            train_dataset = OfflineCIFAR10(
                root=offline_root,
                train=True,
                image_size=args.image_size,
                use_imagenet_norm=use_imagenet_norm
            )
            
            test_dataset = OfflineCIFAR10(
                root=offline_root,
                train=False,
                image_size=args.image_size,
                use_imagenet_norm=use_imagenet_norm
            )
            
            print(f"[Offline Data] Training set: {len(train_dataset)} samples")
            print(f"[Offline Data] Test set: {len(test_dataset)} samples")
            print(f"[Offline Data] Advantages: Zero Resize overhead + Shared memory + Low CPU load\n")
        else:
            # Traditional online data loading
            data_dir = os.path.join(DATA_DIR, 'cifar')
            
            if hasattr(args, 'model_variant') and args.model_variant == 'pretrained':
                imagenet_mean = (0.485, 0.456, 0.406)
                imagenet_std = (0.229, 0.224, 0.225)
                
                transform_list = [transforms.ToTensor()]
                
                if hasattr(args, 'image_size') and args.image_size != 32:
                    transform_list.insert(0, transforms.Resize(args.image_size))
                    print(f"[Data] CIFAR-10 will be resized to {args.image_size}x{args.image_size}")
                
                transform_list.append(transforms.Normalize(imagenet_mean, imagenet_std))
                apply_transform = transforms.Compose(transform_list)
                print(f"[Data] Using ImageNet normalization (pretrained model mode)")
            else:
                apply_transform = transforms.Compose(
                    [transforms.ToTensor(),
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                print(f"[Data] Using standard normalization (from scratch mode)")

            train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                           transform=apply_transform)
            test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                          transform=apply_transform)

    elif args.dataset == 'cifar100':
        if hasattr(args, 'use_offline_data') and args.use_offline_data:
            print("\n" + "="*70)
            print("[Offline Data] Using offline preprocessed data mode (CIFAR-100)".center(70))
            print("="*70)
            
            offline_root = getattr(args, 'offline_data_root', PREPROCESSED_DATA_DIR)
            if not os.path.isabs(offline_root):
                offline_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), offline_root.lstrip('./').lstrip('../'))
            
            print(f"[Offline Data] Data path: {offline_root}")
            print(f"[Offline Data] Image size: {args.image_size}x{args.image_size}")
            
            use_imagenet_norm = (hasattr(args, 'model_variant') and 
                                args.model_variant == 'pretrained')
            
            if use_imagenet_norm:
                print(f"[Offline Data] Normalization: ImageNet (pretrained model)")
            else:
                print(f"[Offline Data] Normalization: Standard (from scratch)")
            
            print("="*70 + "\n")
            
            train_dataset = OfflineCIFAR100(
                root=offline_root,
                train=True,
                image_size=args.image_size,
                use_imagenet_norm=use_imagenet_norm
            )
            
            test_dataset = OfflineCIFAR100(
                root=offline_root,
                train=False,
                image_size=args.image_size,
                use_imagenet_norm=use_imagenet_norm
            )
            
            print(f"[Offline Data] Training set: {len(train_dataset)} samples")
            print(f"[Offline Data] Test set: {len(test_dataset)} samples")
            print(f"[Offline Data] Classes: 100")
            print(f"[Offline Data] Advantages: Zero Resize overhead + Shared memory + Low CPU load\n")
        else:
            data_dir = os.path.join(DATA_DIR, 'cifar100')
            
            if hasattr(args, 'model_variant') and args.model_variant == 'pretrained':
                imagenet_mean = (0.485, 0.456, 0.406)
                imagenet_std = (0.229, 0.224, 0.225)
                
                transform_list = [transforms.ToTensor()]
                
                if hasattr(args, 'image_size') and args.image_size != 32:
                    transform_list.insert(0, transforms.Resize(args.image_size))
                    print(f"[Data] CIFAR-100 will be resized to {args.image_size}x{args.image_size}")
                
                transform_list.append(transforms.Normalize(imagenet_mean, imagenet_std))
                apply_transform = transforms.Compose(transform_list)
                print(f"[Data] Using ImageNet normalization (pretrained model mode)")
            else:
                apply_transform = transforms.Compose(
                    [transforms.ToTensor(),
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                print(f"[Data] Using standard normalization (from scratch mode)")

            train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                           transform=apply_transform)
            test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                          transform=apply_transform)

    elif args.dataset in ('mnist', 'fmnist'):
        if args.dataset == 'mnist':
            data_dir = os.path.join(DATA_DIR, 'mnist')
        else:
            data_dir = os.path.join(DATA_DIR, 'fmnist')

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

    else:
        raise ValueError('Error: unrecognized dataset')

    if args.unequal:
        raise NotImplementedError()

    alpha = float(args.dirichlet_alpha)
    
    # Dual evaluation: partition both train and test sets
    user_groups_train, user_groups_test = dirichlet_partition_train_test(
        train_dataset, test_dataset, args.num_users, alpha=alpha, min_size=20
    )
    print(f"Partitioning complete (Train + Test): Alpha: {alpha}")
    print(f"  - Train samples per client: min={min(len(v) for v in user_groups_train.values())}, "
          f"max={max(len(v) for v in user_groups_train.values())}, "
          f"avg={np.mean([len(v) for v in user_groups_train.values()]):.1f}")
    print(f"  - Test samples per client: min={min(len(v) for v in user_groups_test.values())}, "
          f"max={max(len(v) for v in user_groups_test.values())}, "
          f"avg={np.mean([len(v) for v in user_groups_test.values()]):.1f}")

    # Generate client-class distribution heatmap
    try:
        targets = train_dataset.targets
        if hasattr(targets, 'cpu'):
            targets = targets.cpu()
        targets = np.asarray(targets)
    except Exception:
        targets = None

    if targets is not None:
        num_classes = int(np.max(targets)) + 1
        dist = np.zeros((args.num_users, num_classes), dtype=np.int64)
        for client_id, idxs in user_groups_train.items():
            idxs = np.asarray(idxs, dtype=np.int64)
            dist[client_id] = np.bincount(targets[idxs], minlength=num_classes)

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            viz_dir = ensure_dir(VISUALIZATIONS_DIR)
            out_path = os.path.join(
                viz_dir,
                f"client_class_heatmap_{args.dataset}_{args.num_users}_alpha{args.dirichlet_alpha}.png",
            )

            plt.figure(figsize=(max(8, num_classes * 0.8), max(6, args.num_users * 0.12)))
            plt.imshow(dist, aspect='auto', interpolation='nearest')
            plt.colorbar(label='num_samples')
            plt.xlabel('class_id')
            plt.ylabel('client_id')
            plt.title(f"Client-Class Distribution Heatmap (alpha={args.dirichlet_alpha})")
            plt.tight_layout()
            plt.savefig(out_path, dpi=200)
            plt.close()

            print(f"Heatmap saved to: {out_path}")
        except ImportError:
            print('matplotlib not installed; skip client-class heatmap plotting.')

    return train_dataset, test_dataset, user_groups_train, user_groups_test

