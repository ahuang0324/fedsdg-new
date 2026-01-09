#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
联邦学习实验主入口

支持的算法：
- FedAvg: 联邦平均 (传输完整模型)
- FedLoRA: 联邦低秩适应 (仅传输 LoRA 参数)
- FedSDG: 联邦结构解耦门控 (双路架构: 全局 + 私有分支)

使用方法：
    python main.py --alg fedavg --model cnn --dataset cifar --epochs 100
    python main.py --alg fedlora --model vit --model_variant pretrained --dataset cifar100 --epochs 50
    python main.py --alg fedsdg --model vit --model_variant pretrained --dataset cifar100 --epochs 50
    python main.py --config configs/fedsdg.yaml
"""

import os
import sys
import copy
import time
import pickle
import math
import random
from typing import Dict, List, Any, Optional, Union, Tuple

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from options import args_parser
from fl.utils.paths import (
    PROJECT_ROOT, 
    get_log_dir, get_result_path, get_summary_path, get_model_path,
    generate_experiment_name, ensure_dir
)
from fl.utils import exp_details, get_communication_stats, print_communication_profile
from fl.utils import test_inference, evaluate_local_personalization
from fl.utils import CheckpointManager, create_checkpoint_manager
from fl.utils import visualize_all_gates
from fl.data import get_dataset
from fl.models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, ViT, get_pretrained_vit
from fl.models import inject_lora, inject_lora_timm
from fl.algorithms import average_weights, average_weights_lora
from fl.clients import LocalUpdate


# =============================================================================
# 辅助函数
# =============================================================================

def set_random_seed(seed: int) -> None:
    """
    设置全局随机种子以确保实验可复现
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 确保 CUDA 卷积操作的确定性（可能影响性能）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def validate_args(args) -> None:
    """
    验证命令行参数的有效性
    
    Args:
        args: 解析后的参数
        
    Raises:
        ValueError: 参数无效时抛出异常
    """
    # 基本参数验证
    if args.epochs <= 0:
        raise ValueError(f"训练轮次 (epochs) 必须为正整数，当前值: {args.epochs}")
    
    if args.num_users <= 0:
        raise ValueError(f"客户端数量 (num_users) 必须为正整数，当前值: {args.num_users}")
    
    if not (0 < args.frac <= 1):
        raise ValueError(f"客户端参与率 (frac) 必须在 (0, 1] 范围内，当前值: {args.frac}")
    
    if args.local_ep <= 0:
        raise ValueError(f"本地训练轮次 (local_ep) 必须为正整数，当前值: {args.local_ep}")
    
    if args.local_bs <= 0:
        raise ValueError(f"本地批次大小 (local_bs) 必须为正整数，当前值: {args.local_bs}")
    
    if args.lr <= 0:
        raise ValueError(f"学习率 (lr) 必须为正数，当前值: {args.lr}")
    
    # 数据集验证
    valid_datasets = ['mnist', 'fmnist', 'cifar', 'cifar10', 'cifar100']
    if args.dataset not in valid_datasets:
        raise ValueError(f"不支持的数据集: {args.dataset}，有效选项: {valid_datasets}")
    
    # 模型验证
    valid_models = ['mlp', 'cnn', 'vit']
    if args.model not in valid_models:
        raise ValueError(f"不支持的模型: {args.model}，有效选项: {valid_models}")
    
    # 算法验证
    valid_algs = ['fedavg', 'fedlora', 'fedsdg']
    if args.alg not in valid_algs:
        raise ValueError(f"不支持的算法: {args.alg}，有效选项: {valid_algs}")
    
    # FedLoRA/FedSDG 特定验证
    if args.alg in ('fedlora', 'fedsdg'):
        if args.model != 'vit':
            raise ValueError(
                f"{args.alg.upper()} 目前仅支持 ViT 模型，但当前模型为 '{args.model}'。"
                f"请使用 --model vit 或切换到 --alg fedavg"
            )
        
        if args.lora_r <= 0:
            raise ValueError(f"LoRA 秩 (lora_r) 必须为正整数，当前值: {args.lora_r}")
        
        if args.lora_alpha <= 0:
            raise ValueError(f"LoRA Alpha (lora_alpha) 必须为正整数，当前值: {args.lora_alpha}")
    
    # FedSDG 特定验证
    if args.alg == 'fedsdg':
        if args.lambda1 < 0:
            raise ValueError(f"lambda1 必须为非负数，当前值: {args.lambda1}")
        
        if args.lambda2 < 0:
            raise ValueError(f"lambda2 必须为非负数，当前值: {args.lambda2}")
        
        valid_agg_methods = ['fedavg', 'alignment']
        if args.server_agg_method not in valid_agg_methods:
            raise ValueError(
                f"不支持的聚合方法: {args.server_agg_method}，有效选项: {valid_agg_methods}"
            )
    
    # 预训练模型验证
    if args.model_variant == 'pretrained':
        if args.model != 'vit':
            raise ValueError("预训练模型 (pretrained) 目前仅支持 ViT 模型")
        
        if args.image_size < 32:
            raise ValueError(f"图像尺寸 (image_size) 必须 >= 32，当前值: {args.image_size}")
        
        if args.image_size == 32:
            print("[警告] 预训练 ViT 使用 image_size=32 可能效果不佳，建议使用 --image_size 224")
    
    # Dirichlet alpha 验证
    if args.dirichlet_alpha <= 0:
        raise ValueError(f"Dirichlet alpha 必须为正数，当前值: {args.dirichlet_alpha}")
    
    # GPU 验证
    if args.gpu is not None and args.gpu >= 0:
        if not torch.cuda.is_available():
            print(f"[警告] 指定了 GPU {args.gpu}，但 CUDA 不可用，将使用 CPU")
        elif args.gpu >= torch.cuda.device_count():
            raise ValueError(
                f"指定的 GPU {args.gpu} 不存在，可用 GPU 数量: {torch.cuda.device_count()}"
            )


def validate_dataset(train_dataset, test_dataset, user_groups, user_groups_test) -> None:
    """
    验证数据集和用户分组的有效性
    
    Args:
        train_dataset: 训练数据集
        test_dataset: 测试数据集
        user_groups: 训练数据用户分组
        user_groups_test: 测试数据用户分组
        
    Raises:
        ValueError: 数据无效时抛出异常
    """
    if len(train_dataset) == 0:
        raise ValueError("训练数据集为空")
    
    if len(test_dataset) == 0:
        raise ValueError("测试数据集为空")
    
    if len(user_groups) == 0:
        raise ValueError("用户分组为空")
    
    # 验证每个用户至少有一些数据
    empty_users = [uid for uid, idxs in user_groups.items() if len(idxs) == 0]
    if empty_users:
        print(f"[警告] 以下用户没有训练数据: {empty_users[:10]}...")
    
    # 验证数据集维度
    sample_data, sample_label = train_dataset[0]
    print(f"\n[数据集信息]")
    print(f"  训练集大小: {len(train_dataset)}")
    print(f"  测试集大小: {len(test_dataset)}")
    print(f"  样本形状: {sample_data.shape}")
    print(f"  用户数量: {len(user_groups)}")
    
    # 统计每个用户的数据量
    user_data_counts = [len(idxs) for idxs in user_groups.values()]
    print(f"  每用户样本数: 均值={np.mean(user_data_counts):.1f}, "
          f"最小={min(user_data_counts)}, 最大={max(user_data_counts)}")


def build_model(args, train_dataset, device: str) -> nn.Module:
    """
    根据参数构建模型
    
    Args:
        args: 命令行参数
        train_dataset: 训练数据集（用于获取输入维度）
        device: 计算设备
        
    Returns:
        构建好的模型
        
    Raises:
        ValueError: 模型类型无效时抛出异常
    """
    global_model: Optional[nn.Module] = None
    
    if args.model == 'cnn':
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset in ('cifar', 'cifar10', 'cifar100'):
            global_model = CNNCifar(args=args)
        else:
            raise ValueError(f"CNN 不支持数据集: {args.dataset}")
    
    elif args.model == 'vit':
        if hasattr(args, 'model_variant') and args.model_variant == 'pretrained':
            global_model = get_pretrained_vit(
                num_classes=args.num_classes,
                image_size=args.image_size if hasattr(args, 'image_size') else 224,
                pretrained_path=args.pretrained_path if hasattr(args, 'pretrained_path') else None
            )
        else:
            # 从零训练的 ViT
            try:
                img_size = train_dataset[0][0].shape[-1]
                channels = train_dataset[0][0].shape[0]
            except (IndexError, AttributeError) as e:
                raise ValueError(f"无法从数据集获取图像尺寸: {e}")
            
            global_model = ViT(
                image_size=img_size,
                patch_size=4,
                num_classes=args.num_classes,
                dim=128,
                depth=6,
                heads=8,
                mlp_dim=256,
                channels=channels,
            )
    
    elif args.model == 'mlp':
        try:
            img_size = train_dataset[0][0].shape
            len_in = 1
            for x in img_size:
                len_in *= x
        except (IndexError, AttributeError) as e:
            raise ValueError(f"无法从数据集获取输入维度: {e}")
        
        global_model = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)
    
    else:
        raise ValueError(f"不支持的模型类型: {args.model}")
    
    if global_model is None:
        raise ValueError("模型构建失败")
    
    # 移动到设备并设置为训练模式
    global_model.to(device)
    global_model.train()
    
    return global_model


def inject_lora_to_model(model: nn.Module, args) -> nn.Module:
    """
    为模型注入 LoRA 层
    
    Args:
        model: 原始模型
        args: 命令行参数
        
    Returns:
        注入 LoRA 后的模型
    """
    is_fedsdg = (args.alg == 'fedsdg')
    alg_name = "FedSDG" if is_fedsdg else "FedLoRA"
    
    if hasattr(args, 'model_variant') and args.model_variant == 'pretrained':
        # 预训练模型：使用 timm 专用的 LoRA 注入函数
        model = inject_lora_timm(
            model, 
            r=args.lora_r, 
            lora_alpha=args.lora_alpha,
            train_head=bool(args.lora_train_mlp_head),
            is_fedsdg=is_fedsdg
        )
    else:
        # 从零训练模型：使用手写 ViT 的 LoRA 注入函数
        print("\n" + "="*60)
        print(f"[{alg_name}] 注入 {alg_name} 到手写 ViT 模型...")
        print("="*60)
        model = inject_lora(
            model, 
            r=args.lora_r, 
            lora_alpha=args.lora_alpha,
            train_mlp_head=bool(args.lora_train_mlp_head),
            is_fedsdg=is_fedsdg
        )
        print("="*60 + "\n")
    
    return model


def generate_summary_report(
    args,
    train_accuracy: List[float],
    train_loss: List[float],
    test_acc: float,
    test_loss: float,
    final_local_acc: float,
    final_local_loss: float,
    comm_stats: Dict[str, Any],
    cumulative_comm_volume_mb: float,
    total_time: float,
    efficiency_score: float,
    efficiency_score_per_gb: float,
    best_efficiency_epoch: int,
    best_efficiency_score: float,
) -> str:
    """
    生成详细的实验总结报告
    
    Args:
        args: 命令行参数
        train_accuracy: 训练准确率列表
        train_loss: 训练损失列表
        test_acc: 全局测试准确率
        test_loss: 全局测试损失
        final_local_acc: 本地个性化准确率
        final_local_loss: 本地个性化损失
        comm_stats: 通信量统计
        cumulative_comm_volume_mb: 累计通信量 (MB)
        total_time: 总训练时间 (秒)
        efficiency_score: 效率得分 (准确率/MB)
        efficiency_score_per_gb: 效率得分 (准确率/GB)
        best_efficiency_epoch: 最佳效率轮次
        best_efficiency_score: 最佳效率得分
        
    Returns:
        格式化的报告文本
    """
    # 计算完整模型大小
    full_model_size_mb = comm_stats['total_params'] * 4 / (1024 * 1024)
    savings_ratio_percent = (1 - comm_stats['comm_size_mb'] / full_model_size_mb) * 100
    
    # 相比 FedAvg 的节省量
    fedavg_estimated_comm_mb = full_model_size_mb * 2 * args.epochs
    saved_comm_mb = fedavg_estimated_comm_mb - cumulative_comm_volume_mb
    saved_comm_gb = saved_comm_mb / 1024
    savings_multiplier = fedavg_estimated_comm_mb / cumulative_comm_volume_mb if cumulative_comm_volume_mb > 0 else 1
    
    # 构建报告
    summary_text = f"""
# 联邦学习实验总结报告

## 基本配置
- **算法**: {args.alg.upper()}
- **模型**: {args.model.upper()} ({args.model_variant if hasattr(args, 'model_variant') else 'scratch'})
- **数据集**: {args.dataset.upper()} ({args.num_classes} 类)
- **训练轮次**: {args.epochs}
- **客户端数量**: {args.num_users}
- **参与率**: {args.frac * 100:.1f}%
- **本地训练轮次**: {args.local_ep}
- **本地批次大小**: {args.local_bs}
- **学习率**: {args.lr}
- **优化器**: {args.optimizer}
- **Dirichlet Alpha**: {args.dirichlet_alpha}

## 性能指标
### 全局模型性能
- **最终训练准确率**: {train_accuracy[-1] * 100:.2f}%
- **全局测试准确率**: {test_acc * 100:.2f}%
- **最终训练损失**: {train_loss[-1]:.4f}
- **全局测试损失**: {test_loss:.4f}

### 本地个性化性能（双重评估）
- **本地平均测试准确率**: {final_local_acc * 100:.2f}%
- **本地平均测试损失**: {final_local_loss:.4f}
- **准确率差异 (Local - Global)**: {(final_local_acc - test_acc) * 100:+.2f}%

### 训练时间
- **总训练时间**: {total_time / 60:.2f} 分钟 ({total_time:.2f} 秒)
- **平均每轮时间**: {total_time / args.epochs:.2f} 秒

## 通信效率分析
### 模型参数统计
- **总参数量**: {comm_stats['total_params']:,} ({full_model_size_mb:.2f} MB)
- **可训练参数**: {comm_stats['trainable_params']:,}
- **每轮通信参数**: {comm_stats['comm_params']:,} ({comm_stats['comm_size_mb']:.2f} MB)
- **压缩率**: {comm_stats['compression_ratio']:.2f}%

### 通信量统计
- **单轮通信量（双向）**: {comm_stats['comm_size_mb'] * 2:.2f} MB
- **总通信量**: {cumulative_comm_volume_mb:.2f} MB ({cumulative_comm_volume_mb / 1024:.2f} GB)
- **通信节省率**: {savings_ratio_percent:.2f}%

### 相比 FedAvg 的优势 (假设 FedAvg 传输完整模型)
- **FedAvg 预估通信量**: {fedavg_estimated_comm_mb:.2f} MB ({fedavg_estimated_comm_mb / 1024:.2f} GB)
- **节省的通信量**: {saved_comm_mb:.2f} MB ({saved_comm_gb:.2f} GB)
- **节省倍数**: {savings_multiplier:.2f}x
- **通信效率提升**: {(savings_multiplier - 1) * 100:.1f}%

### 效率评分
- **准确率/MB**: {efficiency_score:.6f}
- **准确率/GB**: {efficiency_score_per_gb:.4f}
- **最佳效率轮次**: 第 {best_efficiency_epoch + 1} 轮
- **最佳效率得分**: {best_efficiency_score:.6f}
"""
    
    # 添加 LoRA 配置信息
    if args.alg in ('fedlora', 'fedsdg'):
        summary_text += f"""
## LoRA 配置 (FedLoRA/FedSDG)
- **LoRA 秩 (r)**: {args.lora_r}
- **LoRA Alpha**: {args.lora_alpha}
- **训练分类头**: {'是' if args.lora_train_mlp_head else '否'}
"""
    
    # 添加 FedSDG 特定信息
    if args.alg == 'fedsdg':
        agg_method_desc = {
            'fedavg': 'FedAvg 均匀加权聚合',
            'alignment': '基于对齐度加权的 FedSDG 聚合算法'
        }
        summary_text += f"""
## FedSDG 配置
- **双路架构**: 全局分支 + 私有分支
- **门控惩罚类型**: {args.gate_penalty_type}
- **Lambda1 (门控惩罚)**: {args.lambda1}
- **Lambda2 (私有 L2)**: {args.lambda2}
- **门控学习率**: {args.lr_gate}
- **服务端聚合算法**: {args.server_agg_method} ({agg_method_desc.get(args.server_agg_method, 'unknown')})
"""
    
    # 添加结论
    summary_text += f"""
## 结论
"""
    
    if args.alg == 'fedlora':
        summary_text += f"""
本次实验使用 **FedLoRA** 算法，成功将通信开销降低至原来的 **{100 - savings_ratio_percent:.2f}%**。
相比传统 FedAvg，节省了 **{saved_comm_gb:.2f} GB** 的通信流量，相当于减少了 **{savings_multiplier:.2f}** 倍的通信成本。
同时保持了 **{test_acc * 100:.2f}%** 的测试准确率，展现了参数高效联邦学习（PEFT）的强大优势。

**投入产出比**: 每传输 1 MB 数据，获得 {efficiency_score:.6f} 的准确率提升，
即每 GB 流量可换取 {efficiency_score_per_gb:.4f} 的准确率收益。
"""
    elif args.alg == 'fedsdg':
        summary_text += f"""
本次实验使用 **FedSDG** 算法，通过双路架构（全局分支 + 私有分支）对抗 Non-IID 数据分布。
通信开销与 FedLoRA 保持一致，降低至原来的 **{100 - savings_ratio_percent:.2f}%**。
相比传统 FedAvg，节省了 **{saved_comm_gb:.2f} GB** 的通信流量，相当于减少了 **{savings_multiplier:.2f}** 倍的通信成本。

**FedSDG 特点**:
- 私有参数（lora_A_private, lora_B_private, lambda_k）仅在客户端本地更新
- 全局参数（lora_A, lora_B）参与服务器聚合
- 通过门控机制自动学习全局/私有分支的最优权重
- 最终测试准确率: **{test_acc * 100:.2f}%**
- 本地个性化准确率: **{final_local_acc * 100:.2f}%**

**投入产出比**: 每传输 1 MB 数据，获得 {efficiency_score:.6f} 的准确率提升，
即每 GB 流量可换取 {efficiency_score_per_gb:.4f} 的准确率收益。
"""
    else:
        summary_text += f"""
本次实验使用 **FedAvg** 算法，传输完整模型参数进行联邦学习。
总通信量为 **{cumulative_comm_volume_mb / 1024:.2f} GB**，最终测试准确率达到 **{test_acc * 100:.2f}%**。

**投入产出比**: 每传输 1 MB 数据，获得 {efficiency_score:.6f} 的准确率提升，
即每 GB 流量可换取 {efficiency_score_per_gb:.4f} 的准确率收益。
"""
    
    summary_text += f"""
---
*报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    return summary_text


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数"""
    start_time = time.time()
    
    # ==================== 参数解析与验证 ====================
    args = args_parser()
    
    try:
        validate_args(args)
    except ValueError as e:
        print(f"\n[错误] 参数验证失败: {e}")
        sys.exit(1)
    
    # 设置随机种子
    if hasattr(args, 'seed') and args.seed > 0:
        set_random_seed(args.seed)
        print(f"\n[随机种子] 设置为 {args.seed}")
    
    # 显示实验配置
    exp_details(args)
    
    # ==================== 路径设置 ====================
    path_project = PROJECT_ROOT
    experiment_name = generate_experiment_name(args)
    
    # 设置日志目录: logs/{algorithm}/{experiment_name}/
    log_dir = get_log_dir(args.alg, experiment_name)
    logger = SummaryWriter(log_dir)
    print(f"\n[输出] TensorBoard 日志: {log_dir}")
    
    # ==================== 设备设置 ====================
    use_cuda = (args.gpu is not None) and (int(args.gpu) >= 0) and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(int(args.gpu))
        print(f"[设备] 使用 GPU {args.gpu}: {torch.cuda.get_device_name(int(args.gpu))}")
    else:
        print("[设备] 使用 CPU")
    device = 'cuda' if use_cuda else 'cpu'
    
    # ==================== 加载数据集 ====================
    print("\n[数据] 加载数据集...")
    try:
        train_dataset, test_dataset, user_groups, user_groups_test = get_dataset(args)
        validate_dataset(train_dataset, test_dataset, user_groups, user_groups_test)
    except Exception as e:
        print(f"\n[错误] 数据集加载失败: {e}")
        logger.close()
        sys.exit(1)
    
    # ==================== 构建模型 ====================
    print("\n[模型] 构建模型...")
    try:
        global_model = build_model(args, train_dataset, device)
    except Exception as e:
        print(f"\n[错误] 模型构建失败: {e}")
        logger.close()
        sys.exit(1)
    
    print(global_model)
    
    # ==================== 注入 LoRA (FedLoRA/FedSDG) ====================
    if args.alg in ('fedlora', 'fedsdg'):
        try:
            global_model = inject_lora_to_model(global_model, args)
        except Exception as e:
            print(f"\n[错误] LoRA 注入失败: {e}")
            logger.close()
            sys.exit(1)
    
    # 复制初始权重
    global_weights = global_model.state_dict()
    
    # ==================== 通信量统计 ====================
    comm_stats = get_communication_stats(global_model, args.alg)
    print_communication_profile(comm_stats, args)
    
    # 记录基础通信量信息到 TensorBoard
    logger.add_scalar('info/total_params', comm_stats['total_params'], 0)
    logger.add_scalar('info/trainable_params', comm_stats['trainable_params'], 0)
    logger.add_scalar('info/comm_params_per_round', comm_stats['comm_params'], 0)
    logger.add_scalar('info/comm_size_per_round_MB', comm_stats['comm_size_mb'], 0)
    logger.add_scalar('info/compression_ratio_percent', comm_stats['compression_ratio'], 0)
    
    # 计算总通信量（双向：上传 + 下载）
    comm_per_round_2way_mb = comm_stats['comm_size_mb'] * 2
    total_comm_volume_mb = comm_per_round_2way_mb * args.epochs
    logger.add_scalar('info/estimated_total_volume_MB', total_comm_volume_mb, 0)
    logger.add_scalar('info/estimated_total_volume_GB', total_comm_volume_mb / 1024, 0)
    
    # 效率指标
    full_model_size_mb = comm_stats['total_params'] * 4 / (1024 * 1024)
    savings_ratio_percent = (1 - comm_stats['comm_size_mb'] / full_model_size_mb) * 100
    logger.add_scalar('Efficiency/communication_savings_percent', savings_ratio_percent, 0)
    
    log10_comm_size = math.log10(comm_stats['comm_size_mb']) if comm_stats['comm_size_mb'] > 0 else 0
    logger.add_scalar('Efficiency/log10_comm_size_MB', log10_comm_size, 0)
    
    print(f"\n{'='*70}")
    print("[效率指标]")
    print(f"{'='*70}")
    print(f"  完整模型大小: {full_model_size_mb:.2f} MB")
    print(f"  实际通信大小: {comm_stats['comm_size_mb']:.2f} MB")
    print(f"  通信节省率: {savings_ratio_percent:.2f}%")
    print(f"{'='*70}\n")
    
    # ==================== FedSDG 私有状态管理 ====================
    local_private_states: Optional[Dict[int, Dict[str, torch.Tensor]]] = {} if args.alg == 'fedsdg' else None
    
    if args.alg == 'fedsdg':
        print("\n" + "="*70)
        print("[FedSDG] 客户端私有状态管理已初始化")
        print("="*70)
        print(f"  每个客户端维护独立的私有参数（lora_A_private, lora_B_private, lambda_k）")
        print(f"  私有参数不参与服务器聚合")
        print("="*70 + "\n")
    
    # ==================== 检查点管理器 ====================
    checkpoint_manager: Optional[CheckpointManager] = None
    if args.enable_checkpoint:
        checkpoint_manager = create_checkpoint_manager(args, path_project, experiment_name)
    
    # ==================== 训练循环 ====================
    train_loss: List[float] = []
    train_accuracy: List[float] = []
    val_acc_list: List[float] = []
    net_list: List[Any] = []
    cv_loss: List[float] = []
    cv_acc: List[float] = []
    print_every = 2
    val_loss_pre, counter = 0, 0
    
    cumulative_comm_volume_mb = 0.0
    best_efficiency_score = 0.0
    best_efficiency_epoch = 0
    efficiency_score = 0.0
    efficiency_score_per_gb = 0.0
    
    print("\n" + "="*70)
    print("[训练开始]")
    print("="*70)
    
    for epoch in tqdm(range(args.epochs), desc="全局训练轮次"):
        start_epoch_time = time.time()
        local_weights: List[Dict[str, torch.Tensor]] = []
        local_losses: List[float] = []
        print(f'\n | 全局训练轮次 : {epoch+1} |\n')
        
        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        for idx in idxs_users:
            # FedSDG: 加载客户端私有状态
            if args.alg == 'fedsdg' and local_private_states is not None:
                local_model_copy = copy.deepcopy(global_model)
                
                if idx in local_private_states:
                    current_state = local_model_copy.state_dict()
                    for param_name, param_value in local_private_states[idx].items():
                        if param_name in current_state:
                            current_state[param_name] = param_value.clone()
                    local_model_copy.load_state_dict(current_state)
            else:
                local_model_copy = copy.deepcopy(global_model)
            
            local_model = LocalUpdate(
                args=args, 
                dataset=train_dataset,
                idxs=user_groups[idx], 
                logger=logger
            )
            w, loss = local_model.update_weights(model=local_model_copy, global_round=epoch)
            
            # FedSDG: 保存客户端私有状态
            if args.alg == 'fedsdg' and local_private_states is not None:
                private_state: Dict[str, torch.Tensor] = {}
                for name, param in local_model_copy.named_parameters():
                    if '_private' in name or 'lambda_k' in name:
                        private_state[name] = param.data.clone().cpu()
                local_private_states[idx] = private_state
            
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            
            # FedSDG 调试信息
            if args.alg == 'fedsdg' and epoch % 5 == 0 and idx == idxs_users[0]:
                print(f"\n[FedSDG Debug - 轮次 {epoch+1}, 客户端 {idx}]")
                lambda_k_values = []
                for name, param in local_model_copy.named_parameters():
                    if 'lambda_k_logit' in name:
                        lambda_k = torch.sigmoid(param).item()
                        lambda_k_values.append(lambda_k)
                if lambda_k_values:
                    print(f"  Lambda_k 均值: {np.mean(lambda_k_values):.4f} "
                          f"(范围: {min(lambda_k_values):.4f} - {max(lambda_k_values):.4f})")
                print(f"[FedSDG Debug 结束]\n")
        
        # ==================== 聚合全局权重 ====================
        aggregation_info: Optional[Dict[str, Any]] = None
        previous_global_state = copy.deepcopy(global_model.state_dict()) if checkpoint_manager else None
        
        if args.alg in ('fedlora', 'fedsdg'):
            agg_method = args.server_agg_method if args.alg == 'fedsdg' else 'fedavg'
            global_weights, aggregation_info = average_weights_lora(
                local_weights, 
                global_model.state_dict(), 
                agg_method=agg_method,
                return_aggregation_info=True
            )
            
            # FedSDG 聚合调试信息
            if args.alg == 'fedsdg' and epoch % 5 == 0:
                print(f"\n[FedSDG 聚合 Debug - 轮次 {epoch+1}]")
                aggregated_keys = [k for k in global_weights.keys() 
                                   if ('lora_' in k or 'head' in k) 
                                   and '_private' not in k 
                                   and 'lambda_k' not in k]
                print(f"  聚合的参数键数量: {len(aggregated_keys)}")
                print(f"[FedSDG 聚合 Debug 结束]\n")
        else:
            global_weights = average_weights(local_weights)
        
        global_model.load_state_dict(global_weights)
        
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        
        logger.add_scalar('global/train_loss_avg', loss_avg, epoch)
        logger.add_scalar('client/train_loss_mean', np.mean(local_losses), epoch)
        logger.add_scalar('client/train_loss_var', np.var(local_losses), epoch)
        logger.add_scalar('lr', args.lr, epoch)
        
        # 评估训练准确率（每 5 轮）
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            list_acc: List[float] = []
            list_loss: List[float] = []
            global_model.eval()
            for idx in idxs_users:
                local_model = LocalUpdate(
                    args=args, 
                    dataset=train_dataset,
                    idxs=user_groups[idx], 
                    logger=logger
                )
                acc, loss = local_model.inference(model=global_model, loader='train')
                list_acc.append(acc)
                list_loss.append(loss)
            train_accuracy.append(sum(list_acc)/len(list_acc))
            
            logger.add_scalar('global/train_acc_avg', train_accuracy[-1], epoch)
            logger.add_scalar('global/train_loss_eval', sum(list_loss)/len(list_loss), epoch)
        else:
            if train_accuracy:
                train_accuracy.append(train_accuracy[-1])
            else:
                train_accuracy.append(0.0)
        
        cumulative_comm_volume_mb += comm_per_round_2way_mb
        logger.add_scalar('info/cumulative_comm_volume_MB', cumulative_comm_volume_mb, epoch)
        logger.add_scalar('info/cumulative_comm_volume_GB', cumulative_comm_volume_mb / 1024, epoch)
        
        # 评估测试准确率（每 5 轮）
        round_test_acc: Optional[float] = None
        round_test_loss: Optional[float] = None
        round_local_acc: Optional[float] = None
        round_local_loss: Optional[float] = None
        
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            # 全局测试
            test_acc, test_loss = test_inference(args, global_model, test_dataset)
            logger.add_scalar('global/test_acc', test_acc, epoch)
            logger.add_scalar('global/test_loss', test_loss, epoch)
            round_test_acc, round_test_loss = test_acc, test_loss
            
            # 本地个性化评估
            num_test_clients = max(1, int(args.test_frac * args.num_users))
            test_client_idxs = np.random.choice(range(args.num_users), num_test_clients, replace=False)
            local_avg_acc, local_avg_loss, _ = evaluate_local_personalization(
                args=args,
                global_model=global_model,
                test_dataset=test_dataset,
                user_groups_test=user_groups_test,
                local_private_states=local_private_states,
                sample_clients=test_client_idxs
            )
            logger.add_scalar('local/test_acc_avg', local_avg_acc, epoch)
            logger.add_scalar('local/test_loss_avg', local_avg_loss, epoch)
            round_local_acc, round_local_loss = local_avg_acc, local_avg_loss
            
            acc_gap = local_avg_acc - test_acc
            logger.add_scalar('local/acc_gap_vs_global', acc_gap, epoch)
            
            # 效率指标
            logger.add_scalar('Efficiency/communication_savings_percent', savings_ratio_percent, epoch)
            
            log10_cumulative_comm = math.log10(cumulative_comm_volume_mb) if cumulative_comm_volume_mb > 0 else 0
            logger.add_scalar('Efficiency/log10_cumulative_comm_MB', log10_cumulative_comm, epoch)
            
            comm_for_efficiency = cumulative_comm_volume_mb if cumulative_comm_volume_mb > 0 else comm_per_round_2way_mb
            efficiency_score = test_acc / comm_for_efficiency if comm_for_efficiency > 0 else 0
            logger.add_scalar('Efficiency/accuracy_per_MB', efficiency_score, epoch)
            
            if efficiency_score > best_efficiency_score:
                best_efficiency_score = efficiency_score
                best_efficiency_epoch = epoch
            
            efficiency_score_per_gb = test_acc / (cumulative_comm_volume_mb / 1024) if cumulative_comm_volume_mb > 0 else 0
            logger.add_scalar('Efficiency/accuracy_per_GB', efficiency_score_per_gb, epoch)
        
        logger.add_scalar('time/round', time.time() - start_epoch_time, epoch)
        
        # 保存检查点
        if checkpoint_manager is not None:
            if previous_global_state is not None:
                previous_global_state = {k: v.cpu() for k, v in previous_global_state.items()}
            
            checkpoint_manager.save_round_checkpoint(
                round_idx=epoch,
                global_model=global_model,
                local_weights=local_weights,
                local_losses=local_losses,
                selected_clients=list(idxs_users),
                aggregation_info=aggregation_info,
                local_private_states=local_private_states,
                train_loss=loss_avg,
                train_acc=train_accuracy[-1] if train_accuracy else None,
                test_acc=round_test_acc,
                test_loss=round_test_loss,
                local_test_acc=round_local_acc,
                local_test_loss=round_local_loss,
                comm_volume_mb=cumulative_comm_volume_mb,
                previous_global_state=previous_global_state,
            )
        
        if (epoch+1) % print_every == 0:
            print(f' \n训练统计（{epoch+1} 轮后）:')
            print(f'  平均训练损失 : {np.mean(np.array(train_loss)):.4f}')
            print(f'  训练准确率: {100*train_accuracy[-1]:.2f}%')
    
    # ==================== 最终评估 ====================
    print("\n" + "="*70)
    print("[最终评估]")
    print("="*70)
    
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    
    final_local_acc, final_local_loss, final_client_results = evaluate_local_personalization(
        args=args,
        global_model=global_model,
        test_dataset=test_dataset,
        user_groups_test=user_groups_test,
        local_private_states=local_private_states,
        sample_clients=None
    )
    
    print(f'\n {args.epochs} 轮全局训练后的结果:')
    print(f"|---- 平均训练准确率: {100*train_accuracy[-1]:.2f}%")
    print(f"|---- 全局测试准确率: {100*test_acc:.2f}%")
    print(f"|---- 本地个性化准确率 (平均): {100*final_local_acc:.2f}%")
    print(f"|---- 准确率差异 (Local - Global): {100*(final_local_acc - test_acc):+.2f}%")
    
    # ==================== 保存结果 ====================
    result_path = get_result_path(args.alg, experiment_name)
    
    results = {
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'comm_stats': comm_stats,
        'total_comm_volume_mb': cumulative_comm_volume_mb,
        'global_test_acc': test_acc,
        'global_test_loss': test_loss,
        'local_avg_acc': final_local_acc,
        'local_avg_loss': final_local_loss,
        'client_results': final_client_results,
        'args': vars(args)
    }
    
    with open(result_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n[输出] 结果已保存: {result_path}")
    
    total_time = time.time() - start_time
    print(f'\n 总运行时间: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)')
    
    # ==================== 生成总结报告 ====================
    summary_text = generate_summary_report(
        args=args,
        train_accuracy=train_accuracy,
        train_loss=train_loss,
        test_acc=test_acc,
        test_loss=test_loss,
        final_local_acc=final_local_acc,
        final_local_loss=final_local_loss,
        comm_stats=comm_stats,
        cumulative_comm_volume_mb=cumulative_comm_volume_mb,
        total_time=total_time,
        efficiency_score=efficiency_score,
        efficiency_score_per_gb=efficiency_score_per_gb,
        best_efficiency_epoch=best_efficiency_epoch,
        best_efficiency_score=best_efficiency_score,
    )
    
    logger.add_text('Experiment_Summary', summary_text, 0)
    
    summary_path = get_summary_path(args.alg, experiment_name)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"[输出] 摘要已保存: {summary_path}")
    
    # 保存最终模型
    model_path = get_model_path(args.alg, experiment_name, suffix='final')
    torch.save(global_model.state_dict(), model_path)
    print(f"[输出] 最终模型已保存: {model_path}")
    
    # 保存最终检查点
    if checkpoint_manager is not None:
        final_results = {
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'comm_stats': comm_stats,
            'total_comm_volume_mb': cumulative_comm_volume_mb,
            'global_test_acc': test_acc,
            'global_test_loss': test_loss,
            'local_avg_acc': final_local_acc,
            'local_avg_loss': final_local_loss,
            'client_results': final_client_results,
        }
        
        checkpoint_manager.save_final(
            global_model=global_model,
            local_private_states=local_private_states,
            final_results=final_results,
            args=args,
        )
    
    # ==================== FedSDG 门控可视化 ====================
    if args.alg == 'fedsdg':
        try:
            visualize_all_gates(
                model=global_model,
                local_private_states=local_private_states,
                algorithm=args.alg,
                experiment_name=experiment_name,
                prefix=f'{args.dataset}_{args.alg}_E{args.epochs}'
            )
        except Exception as e:
            print(f"[警告] 门控可视化失败: {e}")
    
    logger.close()
    
    print("\n" + "="*70)
    print("[训练完成]")
    print("="*70)
    print(f"实验名称: {experiment_name}")
    print(f"输出位置:")
    print(f"  - TensorBoard 日志: {log_dir}")
    print(f"  - 实验总结: {summary_path}")
    print(f"  - 最终模型: {model_path}")
    print(f"  - 结果文件: {result_path}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()