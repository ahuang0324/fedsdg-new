#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
路径管理工具

提供项目中所有目录路径的统一管理，避免使用 '../' 相对路径。

目录结构：
    project_root/
    ├── datasets/               # 数据目录
    │   └── preprocessed/       # 预处理数据
    ├── configs/                # 配置文件
    ├── logs/                   # TensorBoard 日志
    │   └── {algorithm}/        # 按算法分类
    │       └── {experiment}/   # 每个实验一个子目录
    └── outputs/                # 所有输出文件
        ├── checkpoints/        # 训练检查点
        ├── models/             # 最终模型
        ├── results/            # 实验结果 (.pkl)
        ├── summaries/          # 实验摘要 (.txt)
        └── visualizations/     # 可视化图表
"""

import os

# =============================================================================
# 项目根目录
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# 数据目录
# =============================================================================
DATA_DIR = os.path.join(PROJECT_ROOT, 'datasets')
PREPROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'preprocessed')

# =============================================================================
# 配置目录
# =============================================================================
CONFIGS_DIR = os.path.join(PROJECT_ROOT, 'configs')

# =============================================================================
# 日志目录 (TensorBoard)
# 结构: logs/{algorithm}/{experiment_name}/
# =============================================================================
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')

# =============================================================================
# 输出目录 (统一的输出位置)
# =============================================================================
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')
CHECKPOINTS_DIR = os.path.join(OUTPUTS_DIR, 'checkpoints')
MODELS_DIR = os.path.join(OUTPUTS_DIR, 'models')
RESULTS_DIR = os.path.join(OUTPUTS_DIR, 'results')
SUMMARIES_DIR = os.path.join(OUTPUTS_DIR, 'summaries')
VISUALIZATIONS_DIR = os.path.join(OUTPUTS_DIR, 'visualizations')


# =============================================================================
# 工具函数
# =============================================================================

def ensure_dir(path: str) -> str:
    """确保目录存在，不存在则创建"""
    os.makedirs(path, exist_ok=True)
    return path


def get_data_dir(dataset_name: str) -> str:
    """获取指定数据集的目录路径"""
    return os.path.join(DATA_DIR, dataset_name)


def get_log_dir(algorithm: str, experiment_name: str) -> str:
    """
    获取日志目录路径
    
    结构: logs/{algorithm}/{experiment_name}/
    示例: logs/fedlora/cifar100_vit_E50_r8_alpha0.5/
    
    Args:
        algorithm: 算法名称 (fedavg, fedlora, fedsdg)
        experiment_name: 实验名称
    
    Returns:
        日志目录的完整路径
    """
    log_path = os.path.join(LOGS_DIR, algorithm, experiment_name)
    return ensure_dir(log_path)


def get_checkpoint_dir(algorithm: str, experiment_name: str) -> str:
    """
    获取检查点目录路径
    
    结构: outputs/checkpoints/{algorithm}/{experiment_name}/
    """
    checkpoint_path = os.path.join(CHECKPOINTS_DIR, algorithm, experiment_name)
    return ensure_dir(checkpoint_path)


def get_model_path(algorithm: str, experiment_name: str, suffix: str = 'final') -> str:
    """
    获取模型保存路径
    
    结构: outputs/models/{algorithm}/{experiment_name}_{suffix}.pth
    """
    ensure_dir(os.path.join(MODELS_DIR, algorithm))
    return os.path.join(MODELS_DIR, algorithm, f'{experiment_name}_{suffix}.pth')


def get_result_path(algorithm: str, experiment_name: str) -> str:
    """
    获取结果文件路径
    
    结构: outputs/results/{algorithm}/{experiment_name}.pkl
    """
    ensure_dir(os.path.join(RESULTS_DIR, algorithm))
    return os.path.join(RESULTS_DIR, algorithm, f'{experiment_name}.pkl')


def get_summary_path(algorithm: str, experiment_name: str) -> str:
    """
    获取摘要文件路径
    
    结构: outputs/summaries/{algorithm}/{experiment_name}.txt
    """
    ensure_dir(os.path.join(SUMMARIES_DIR, algorithm))
    return os.path.join(SUMMARIES_DIR, algorithm, f'{experiment_name}.txt')


def get_visualization_path(algorithm: str, experiment_name: str, filename: str) -> str:
    """
    获取可视化文件路径
    
    结构: outputs/visualizations/{algorithm}/{experiment_name}/{filename}
    """
    viz_dir = os.path.join(VISUALIZATIONS_DIR, algorithm, experiment_name)
    ensure_dir(viz_dir)
    return os.path.join(viz_dir, filename)


def generate_experiment_name(args) -> str:
    """
    根据参数生成规范化的实验名称
    
    格式: {dataset}_{model}_{variant}_E{epochs}_C{frac}_alpha{dirichlet_alpha}
    LoRA: 额外添加 _r{lora_r}_la{lora_alpha}
    FedSDG: 额外添加 _l1{lambda1}_l2{lambda2}
    
    示例:
        - cifar100_vit_pretrained_E50_C0.1_alpha0.5
        - cifar100_vit_pretrained_E50_C0.1_alpha0.5_r8_la16
        - cifar100_vit_pretrained_E50_C0.1_alpha0.5_r8_la16_l10.01_l20.001
    """
    # 基础部分
    model_variant = getattr(args, 'model_variant', 'scratch')
    parts = [
        args.dataset,
        args.model,
        model_variant,
        f'E{args.epochs}',
        f'C{args.frac}',
        f'alpha{args.dirichlet_alpha}',
    ]
    
    # LoRA 参数
    if args.alg in ('fedlora', 'fedsdg'):
        parts.extend([
            f'r{args.lora_r}',
            f'la{args.lora_alpha}',
        ])
    
    # FedSDG 参数
    if args.alg == 'fedsdg':
        parts.extend([
            f'l1_{args.lambda1}',
            f'l2_{args.lambda2}',
        ])
    
    return '_'.join(parts)
