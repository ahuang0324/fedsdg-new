#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Command line argument parser for federated learning experiments.

Supports loading configuration from YAML files:
    python main.py --config configs/fedsdg.yaml
    python main.py --config configs/fedsdg.yaml --dataset cifar --epochs 50
    
Priority: Command line args > Config file > Default values
"""

import argparse
import os
import sys
from typing import Dict, Any, Optional

# Try to import yaml, provide helpful message if not available
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dictionary containing configuration
    """
    if not YAML_AVAILABLE:
        raise ImportError(
            "PyYAML is required for config file support. "
            "Install with: pip install pyyaml"
        )
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def flatten_config(config: Dict[str, Any], dataset: str = None) -> Dict[str, Any]:
    """
    Flatten nested config structure to argument format.
    
    Args:
        config: Nested config dictionary
        dataset: Dataset name to use for dataset-specific settings
        
    Returns:
        Flat dictionary suitable for argparse
    """
    flat = {}
    
    # Algorithm
    if 'algorithm' in config:
        flat['alg'] = config['algorithm']
    
    # Dataset (can be overridden by command line)
    if dataset:
        flat['dataset'] = dataset
    elif 'dataset' in config:
        flat['dataset'] = config['dataset']
    
    # Get dataset-specific settings
    dataset_name = flat.get('dataset', 'cifar100')
    if 'datasets' in config and dataset_name in config['datasets']:
        ds_config = config['datasets'][dataset_name]
        for key, value in ds_config.items():
            flat[key] = value
    
    # LoRA settings
    if 'lora' in config:
        lora = config['lora']
        if 'r' in lora:
            flat['lora_r'] = lora['r']
        if 'alpha' in lora:
            flat['lora_alpha'] = lora['alpha']
        if 'train_mlp_head' in lora:
            flat['lora_train_mlp_head'] = 1 if lora['train_mlp_head'] else 0
    
    # FedSDG settings
    if 'fedsdg' in config:
        sdg = config['fedsdg']
        if 'server_agg_method' in sdg:
            flat['server_agg_method'] = sdg['server_agg_method']
        if 'lambda1' in sdg:
            flat['lambda1'] = sdg['lambda1']
        if 'lambda2' in sdg:
            flat['lambda2'] = sdg['lambda2']
        if 'gate_penalty_type' in sdg:
            flat['gate_penalty_type'] = sdg['gate_penalty_type']
        if 'lr_gate' in sdg:
            flat['lr_gate'] = sdg['lr_gate']
        if 'grad_clip' in sdg:
            flat['grad_clip'] = sdg['grad_clip']
    
    # Federated settings
    if 'federated' in config:
        fed = config['federated']
        if 'num_users' in fed:
            flat['num_users'] = fed['num_users']
        if 'frac' in fed:
            flat['frac'] = fed['frac']
        if 'dirichlet_alpha' in fed:
            flat['dirichlet_alpha'] = fed['dirichlet_alpha']
    
    # Training settings
    if 'training' in config:
        train = config['training']
        if 'epochs' in train:
            flat['epochs'] = train['epochs']
        if 'local_ep' in train:
            flat['local_ep'] = train['local_ep']
        if 'local_bs' in train:
            flat['local_bs'] = train['local_bs']
        if 'lr' in train:
            flat['lr'] = train['lr']
        if 'optimizer' in train:
            flat['optimizer'] = train['optimizer']
        if 'momentum' in train:
            flat['momentum'] = train['momentum']
    
    # Evaluation settings
    if 'evaluation' in config:
        evl = config['evaluation']
        if 'test_frac' in evl:
            flat['test_frac'] = evl['test_frac']
    
    # System settings
    if 'system' in config:
        sys_cfg = config['system']
        if 'gpu' in sys_cfg:
            flat['gpu'] = sys_cfg['gpu']
        if 'seed' in sys_cfg:
            flat['seed'] = sys_cfg['seed']
        if 'verbose' in sys_cfg:
            flat['verbose'] = sys_cfg['verbose']
    
    # Checkpoint settings
    if 'checkpoint' in config:
        ckpt = config['checkpoint']
        if 'enable' in ckpt:
            flat['enable_checkpoint'] = 1 if ckpt['enable'] else 0
        if 'save_frequency' in ckpt:
            flat['save_frequency'] = ckpt['save_frequency']
        if 'save_client_weights' in ckpt:
            flat['save_client_weights'] = 1 if ckpt['save_client_weights'] else 0
    
    return flat


def args_parser():
    """
    Parse command line arguments with optional config file support.
    
    Returns:
        Parsed arguments namespace
    """
    # First pass: check for --config argument
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config', type=str, default=None,
                            help='Path to YAML config file (e.g., configs/fedsdg.yaml)')
    pre_args, remaining = pre_parser.parse_known_args()
    
    # Load config file if specified
    config_defaults = {}
    if pre_args.config:
        try:
            config = load_config(pre_args.config)
            # Check if dataset is specified in remaining args
            temp_parser = argparse.ArgumentParser(add_help=False)
            temp_parser.add_argument('--dataset', type=str, default=None)
            temp_args, _ = temp_parser.parse_known_args(remaining)
            
            config_defaults = flatten_config(config, dataset=temp_args.dataset)
            print(f"[Config] Loaded configuration from: {pre_args.config}")
        except Exception as e:
            print(f"[Config] Warning: Failed to load config file: {e}")
    
    # Main parser
    parser = argparse.ArgumentParser(
        description='Federated Learning Experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config file argument
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file')

    # ==========================================================================
    # Federated Learning Arguments
    # ==========================================================================
    parser.add_argument('--epochs', type=int, 
                        default=config_defaults.get('epochs', 10),
                        help="Number of global communication rounds")
    parser.add_argument('--num_users', type=int, 
                        default=config_defaults.get('num_users', 100),
                        help="Total number of clients")
    parser.add_argument('--frac', type=float, 
                        default=config_defaults.get('frac', 0.1),
                        help='Fraction of clients per round')
    parser.add_argument('--local_ep', type=int, 
                        default=config_defaults.get('local_ep', 10),
                        help="Number of local epochs per round")
    parser.add_argument('--local_bs', type=int, 
                        default=config_defaults.get('local_bs', 10),
                        help="Local batch size")
    parser.add_argument('--lr', type=float, 
                        default=config_defaults.get('lr', 0.01),
                        help='Learning rate')
    parser.add_argument('--momentum', type=float, 
                        default=config_defaults.get('momentum', 0.5),
                        help='SGD momentum')
    
    # ==========================================================================
    # Algorithm Selection
    # ==========================================================================
    parser.add_argument('--alg', type=str, 
                        default=config_defaults.get('alg', 'fedavg'),
                        choices=['fedavg', 'fedlora', 'fedsdg'],
                        help='Federated learning algorithm')
    
    # ==========================================================================
    # LoRA Arguments (FedLoRA/FedSDG)
    # ==========================================================================
    parser.add_argument('--lora_r', type=int, 
                        default=config_defaults.get('lora_r', 8),
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, 
                        default=config_defaults.get('lora_alpha', 16),
                        help='LoRA scaling factor')
    parser.add_argument('--lora_train_mlp_head', type=int, 
                        default=config_defaults.get('lora_train_mlp_head', 1),
                        help='Train classification head (1=True, 0=False)')
    
    # ==========================================================================
    # FedSDG Arguments
    # ==========================================================================
    parser.add_argument('--server_agg_method', type=str, 
                        default=config_defaults.get('server_agg_method', 'fedavg'),
                        choices=['fedavg', 'alignment'],
                        help='FedSDG server aggregation method')
    parser.add_argument('--lambda1', type=float, 
                        default=config_defaults.get('lambda1', 1e-3),
                        help='FedSDG: Gate sparsity penalty (L1)')
    parser.add_argument('--lambda2', type=float, 
                        default=config_defaults.get('lambda2', 1e-4),
                        help='FedSDG: Private parameter regularization (L2)')
    parser.add_argument('--gate_penalty_type', type=str, 
                        default=config_defaults.get('gate_penalty_type', 'bilateral'),
                        choices=['unilateral', 'bilateral'],
                        help='FedSDG: Gate penalty type')
    parser.add_argument('--lr_gate', type=float, 
                        default=config_defaults.get('lr_gate', 1e-2),
                        help='FedSDG: Gate parameter learning rate')
    parser.add_argument('--grad_clip', type=float, 
                        default=config_defaults.get('grad_clip', 1.0),
                        help='FedSDG: Gradient clipping norm (0=disabled)')
    
    # ==========================================================================
    # Evaluation Arguments
    # ==========================================================================
    parser.add_argument('--test_frac', type=float, 
                        default=config_defaults.get('test_frac', 0.2),
                        help='Fraction of clients for local evaluation')
    
    # ==========================================================================
    # Checkpoint Arguments
    # ==========================================================================
    parser.add_argument('--enable_checkpoint', type=int, 
                        default=config_defaults.get('enable_checkpoint', 1),
                        help='Enable checkpoint saving (1=True, 0=False)')
    parser.add_argument('--save_frequency', type=int, 
                        default=config_defaults.get('save_frequency', 5),
                        help='Checkpoint save frequency')
    parser.add_argument('--save_client_weights', type=int, 
                        default=config_defaults.get('save_client_weights', 1),
                        help='Save client weights (1=True, 0=False)')
    parser.add_argument('--max_checkpoints', type=int, default=-1,
                        help='Max checkpoints to keep (-1=unlimited)')

    # ==========================================================================
    # Model Arguments
    # ==========================================================================
    parser.add_argument('--model', type=str, 
                        default=config_defaults.get('model', 'mlp'),
                        choices=['mlp', 'cnn', 'vit'],
                        help='Model architecture')
    parser.add_argument('--model_variant', type=str, 
                        default=config_defaults.get('model_variant', 'scratch'),
                        choices=['scratch', 'pretrained'],
                        help='Model variant')
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Path to local pretrained weights')
    parser.add_argument('--image_size', type=int, 
                        default=config_defaults.get('image_size', 32),
                        help='Input image size')
    
    # ==========================================================================
    # Offline Data Arguments
    # ==========================================================================
    use_offline_default = config_defaults.get('use_offline', False)
    parser.add_argument('--use_offline_data', action='store_true',
                        default=use_offline_default,
                        help='Use offline preprocessed data')
    parser.add_argument('--offline_data_root', type=str, 
                        default='./datasets/preprocessed',
                        help='Offline data directory')
    
    # ==========================================================================
    # Other Model Arguments
    # ==========================================================================
    parser.add_argument('--kernel_num', type=int, default=9)
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5')
    parser.add_argument('--num_channels', type=int, 
                        default=config_defaults.get('num_channels', 1))
    parser.add_argument('--norm', type=str, default='batch_norm')
    parser.add_argument('--num_filters', type=int, default=32)
    parser.add_argument('--max_pool', type=str, default='True')

    # ==========================================================================
    # Dataset & System Arguments
    # ==========================================================================
    parser.add_argument('--dataset', type=str, 
                        default=config_defaults.get('dataset', 'mnist'),
                        choices=['mnist', 'fmnist', 'cifar', 'cifar10', 'cifar100'],
                        help="Dataset name")
    parser.add_argument('--num_classes', type=int, 
                        default=config_defaults.get('num_classes', 10),
                        help="Number of classes")
    parser.add_argument('--gpu', type=int, 
                        default=config_defaults.get('gpu', -1),
                        help="GPU ID (-1 for CPU)")
    parser.add_argument('--optimizer', type=str, 
                        default=config_defaults.get('optimizer', 'sgd'),
                        help="Optimizer type")
    parser.add_argument('--dirichlet_alpha', type=float, 
                        default=config_defaults.get('dirichlet_alpha', 0.5),
                        help='Dirichlet alpha for Non-IID')
    parser.add_argument('--unequal', type=int, default=0,
                        help='Unequal data splits')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='Early stopping rounds')
    parser.add_argument('--verbose', type=int, 
                        default=config_defaults.get('verbose', 1),
                        help='Verbosity level')
    parser.add_argument('--seed', type=int, 
                        default=config_defaults.get('seed', 1),
                        help='Random seed')
    
    args = parser.parse_args()
    
    # ==========================================================================
    # Validation
    # ==========================================================================
    if args.alg in ('fedlora', 'fedsdg') and args.model != 'vit':
        raise ValueError(
            f"{args.alg.upper()} requires ViT model. "
            f"Use --model vit or switch to --alg fedavg"
        )
    
    if args.model_variant == 'pretrained' and args.model != 'vit':
        raise ValueError("Pretrained variant only supports ViT")
    
    if args.model_variant == 'pretrained' and args.image_size == 32:
        print("[Warning] Pretrained ViT with image_size=32. Consider --image_size 224")
    
    # ==========================================================================
    # Auto-configure num_classes
    # ==========================================================================
    if args.dataset == 'cifar100':
        args.num_classes = 100
    elif args.dataset in ('cifar', 'cifar10'):
        args.num_classes = 10
        if args.dataset == 'cifar10':
            args.dataset = 'cifar'
    elif args.dataset in ('mnist', 'fmnist'):
        args.num_classes = 10
    
    # Print config summary if using config file
    if pre_args.config:
        print(f"[Config] Algorithm: {args.alg}")
        print(f"[Config] Dataset: {args.dataset} ({args.num_classes} classes)")
        print(f"[Config] Model: {args.model} ({args.model_variant})")
        if args.alg in ('fedlora', 'fedsdg'):
            print(f"[Config] LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
        if args.alg == 'fedsdg':
            print(f"[Config] FedSDG: λ1={args.lambda1}, λ2={args.lambda2}, "
                  f"agg={args.server_agg_method}")
    
    return args
