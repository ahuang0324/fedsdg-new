#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data sampling and partitioning utilities for federated learning.

Supports:
- Dirichlet distribution-based Non-IID partitioning
- Train/Test simultaneous partitioning for dual evaluation
"""

import numpy as np


def _get_targets_numpy(dataset):
    """Extract labels/targets from a dataset as numpy array."""
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
        if hasattr(targets, 'cpu'):
            targets = targets.cpu()
        return np.asarray(targets)
    if hasattr(dataset, 'train_labels'):
        targets = dataset.train_labels
        if hasattr(targets, 'cpu'):
            targets = targets.cpu()
        return np.asarray(targets)
    if hasattr(dataset, 'labels'):
        return np.asarray(dataset.labels)
    raise AttributeError('Cannot infer labels/targets from dataset')


def dirichlet_partition(dataset, num_users, alpha, min_size=20, max_retries=100):
    """
    Partition dataset using Dirichlet distribution for Non-IID data.
    
    Args:
        dataset: PyTorch dataset with targets attribute
        num_users: Number of clients
        alpha: Dirichlet concentration parameter (smaller = more heterogeneous)
        min_size: Minimum samples per client
        max_retries: Maximum retry attempts
    
    Returns:
        dict_users: {client_id: np.array(indices)}
    """
    targets = _get_targets_numpy(dataset)
    num_classes = int(np.max(targets)) + 1

    for _ in range(max_retries):
        dict_users = {i: [] for i in range(num_users)}

        for c in range(num_classes):
            idx_c = np.where(targets == c)[0]
            np.random.shuffle(idx_c)

            proportions = np.random.dirichlet([alpha] * num_users)
            counts = np.random.multinomial(len(idx_c), proportions)

            start = 0
            for user_id, cnt in enumerate(counts):
                if cnt == 0:
                    continue
                dict_users[user_id].extend(idx_c[start:start + cnt].tolist())
                start += cnt

        sizes = [len(v) for v in dict_users.values()]
        if min(sizes) >= min_size:
            dict_users = {k: np.asarray(v, dtype=np.int64) for k, v in dict_users.items()}

            all_idxs = np.concatenate(list(dict_users.values())) if num_users > 0 else np.asarray([], dtype=np.int64)
            if all_idxs.size != len(dataset):
                raise ValueError(
                    f"Dirichlet partition produced wrong total size: {all_idxs.size} vs {len(dataset)}"
                )
            if np.any(all_idxs < 0) or np.any(all_idxs >= len(dataset)):
                raise ValueError("Dirichlet partition produced out-of-range indices")
            if np.unique(all_idxs).size != all_idxs.size:
                raise ValueError("Dirichlet partition produced duplicate indices across users")

            return dict_users

    # Fallback: uniform partition
    dict_users = {i: [] for i in range(num_users)}
    all_idxs = np.random.permutation(len(dataset)).tolist()
    for i, idx in enumerate(all_idxs):
        dict_users[i % num_users].append(idx)

    return {k: np.asarray(v, dtype=np.int64) for k, v in dict_users.items()}


def dirichlet_partition_train_test(train_dataset, test_dataset, num_users, alpha, 
                                    train_ratio=0.8, min_size=20, max_retries=100):
    """
    Partition both train and test datasets with matching label distributions.
    
    This is the core function for dual evaluation mechanism:
    1. Partition training set using Dirichlet distribution
    2. Partition test set using the same class proportions
    3. Ensure each client's test set matches its training set distribution
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        num_users: Number of clients
        alpha: Dirichlet concentration parameter
        train_ratio: Unused, kept for API compatibility
        min_size: Minimum samples per client in training set
        max_retries: Maximum retry attempts
    
    Returns:
        dict_users_train: {client_id: np.array(train_indices)}
        dict_users_test: {client_id: np.array(test_indices)}
    """
    train_targets = _get_targets_numpy(train_dataset)
    test_targets = _get_targets_numpy(test_dataset)
    num_classes = int(np.max(train_targets)) + 1
    
    for _ in range(max_retries):
        dict_users_train = {i: [] for i in range(num_users)}
        dict_users_test = {i: [] for i in range(num_users)}
        
        # Record proportions for each client-class pair
        client_class_proportions = np.zeros((num_users, num_classes))
        
        # Step 1: Partition training set
        for c in range(num_classes):
            idx_c = np.where(train_targets == c)[0]
            np.random.shuffle(idx_c)
            
            proportions = np.random.dirichlet([alpha] * num_users)
            client_class_proportions[:, c] = proportions
            
            counts = np.random.multinomial(len(idx_c), proportions)
            
            start = 0
            for user_id, cnt in enumerate(counts):
                if cnt == 0:
                    continue
                dict_users_train[user_id].extend(idx_c[start:start + cnt].tolist())
                start += cnt
        
        # Check training set partition
        train_sizes = [len(v) for v in dict_users_train.values()]
        if min(train_sizes) < min_size:
            continue
        
        # Step 2: Partition test set with same proportions
        for c in range(num_classes):
            idx_c = np.where(test_targets == c)[0]
            np.random.shuffle(idx_c)
            
            proportions = client_class_proportions[:, c]
            proportions = proportions / (proportions.sum() + 1e-10)
            
            counts = np.random.multinomial(len(idx_c), proportions)
            
            start = 0
            for user_id, cnt in enumerate(counts):
                if cnt == 0:
                    continue
                dict_users_test[user_id].extend(idx_c[start:start + cnt].tolist())
                start += cnt
        
        # Validation
        dict_users_train = {k: np.asarray(v, dtype=np.int64) for k, v in dict_users_train.items()}
        dict_users_test = {k: np.asarray(v, dtype=np.int64) for k, v in dict_users_test.items()}
        
        all_train_idxs = np.concatenate(list(dict_users_train.values())) if num_users > 0 else np.asarray([], dtype=np.int64)
        if all_train_idxs.size != len(train_dataset):
            raise ValueError(f"Train partition size mismatch: {all_train_idxs.size} vs {len(train_dataset)}")
        if np.unique(all_train_idxs).size != all_train_idxs.size:
            raise ValueError("Train partition has duplicate indices")
        
        all_test_idxs = np.concatenate(list(dict_users_test.values())) if num_users > 0 else np.asarray([], dtype=np.int64)
        if all_test_idxs.size != len(test_dataset):
            raise ValueError(f"Test partition size mismatch: {all_test_idxs.size} vs {len(test_dataset)}")
        if np.unique(all_test_idxs).size != all_test_idxs.size:
            raise ValueError("Test partition has duplicate indices")
        
        return dict_users_train, dict_users_test
    
    # Fallback: uniform partition
    print("[Warning] Dirichlet partition failed after max retries, falling back to uniform partition")
    
    dict_users_train = {i: [] for i in range(num_users)}
    dict_users_test = {i: [] for i in range(num_users)}
    
    all_train_idxs = np.random.permutation(len(train_dataset)).tolist()
    for i, idx in enumerate(all_train_idxs):
        dict_users_train[i % num_users].append(idx)
    
    all_test_idxs = np.random.permutation(len(test_dataset)).tolist()
    for i, idx in enumerate(all_test_idxs):
        dict_users_test[i % num_users].append(idx)
    
    dict_users_train = {k: np.asarray(v, dtype=np.int64) for k, v in dict_users_train.items()}
    dict_users_test = {k: np.asarray(v, dtype=np.int64) for k, v in dict_users_test.items()}
    
    return dict_users_train, dict_users_test


