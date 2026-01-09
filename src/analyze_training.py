#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
联邦学习训练历史分析与可视化工具

功能：
1. 加载训练检查点
2. 分析参数演化
3. 可视化训练曲线
4. 分析聚合权重和对齐度
5. 分析客户端差异
6. FedSDG 门控参数分析

使用方法：
    python analyze_training.py --experiment_dir /path/to/experiment
    python analyze_training.py --experiment_dir /path/to/experiment --visualize
    python analyze_training.py --experiment_dir /path/to/experiment --compare /path/to/another/experiment
"""

import os
import sys
import argparse
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any

# 添加 src 目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from checkpoint_manager import CheckpointManager, TrainingAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(description='联邦学习训练历史分析工具')
    
    parser.add_argument('--experiment_dir', type=str, required=True,
                        help='实验目录路径（包含 training_history.pkl 等文件）')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录（默认为实验目录下的 analysis/）')
    parser.add_argument('--visualize', action='store_true',
                        help='生成可视化图表')
    parser.add_argument('--export_json', action='store_true',
                        help='导出 JSON 格式数据')
    parser.add_argument('--compare', type=str, default=None,
                        help='比较的另一个实验目录')
    parser.add_argument('--checkpoint_round', type=int, default=None,
                        help='分析特定轮次的检查点')
    
    return parser.parse_args()


def load_experiment(experiment_dir: str) -> Dict:
    """
    加载实验数据
    
    Args:
        experiment_dir: 实验目录
        
    Returns:
        实验数据字典
    """
    data = {}
    
    # 加载训练历史
    history_path = os.path.join(experiment_dir, 'training_history.pkl')
    if os.path.exists(history_path):
        with open(history_path, 'rb') as f:
            data['training_history'] = pickle.load(f)
    
    # 加载参数演化
    evolution_path = os.path.join(experiment_dir, 'param_evolution.pkl')
    if os.path.exists(evolution_path):
        with open(evolution_path, 'rb') as f:
            data['param_evolution'] = pickle.load(f)
    
    # 加载实验摘要
    summary_path = os.path.join(experiment_dir, 'experiment_summary.pkl')
    if os.path.exists(summary_path):
        with open(summary_path, 'rb') as f:
            data['summary'] = pickle.load(f)
    
    # 加载私有状态（FedSDG）
    private_path = os.path.join(experiment_dir, 'final_private_states.pkl')
    if os.path.exists(private_path):
        with open(private_path, 'rb') as f:
            data['private_states'] = pickle.load(f)
    
    # 加载 JSON 摘要
    json_path = os.path.join(experiment_dir, 'summary.json')
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data['json_summary'] = json.load(f)
    
    return data


def print_summary(data: Dict):
    """打印实验摘要"""
    print("\n" + "="*70)
    print("实验摘要".center(70))
    print("="*70)
    
    if 'json_summary' in data:
        summary = data['json_summary']
        print(f"  实验名称: {summary.get('experiment_name', 'N/A')}")
        print(f"  算法类型: {summary.get('alg', 'N/A')}")
        print(f"  总轮次: {summary.get('total_rounds', 'N/A')}")
        print(f"  最终训练准确率: {summary.get('final_train_acc', 'N/A')}")
        print(f"  最终测试准确率: {summary.get('final_test_acc', 'N/A')}")
        print(f"  最终本地准确率: {summary.get('final_local_acc', 'N/A')}")
        print(f"  检查点数量: {summary.get('checkpoint_count', 'N/A')}")
    
    if 'training_history' in data:
        history = data['training_history']
        
        # 性能统计
        if history.get('test_acc'):
            test_accs = [x for x in history['test_acc'] if x is not None]
            if test_accs:
                print(f"\n  测试准确率统计:")
                print(f"    最大值: {max(test_accs):.4f}")
                print(f"    最小值: {min(test_accs):.4f}")
                print(f"    平均值: {np.mean(test_accs):.4f}")
        
        # 门控参数统计（FedSDG）
        if history.get('gate_values_mean'):
            gate_means = [x for x in history['gate_values_mean'] if x is not None]
            if gate_means:
                print(f"\n  门控参数统计 (FedSDG):")
                print(f"    最终均值: {gate_means[-1]:.4f}")
                print(f"    范围: [{min(gate_means):.4f}, {max(gate_means):.4f}]")
                print(f"    变化趋势: {gate_means[-1] - gate_means[0]:+.4f}")
        
        # 聚合权重统计
        if history.get('aggregation_weights'):
            weights = history['aggregation_weights']
            if weights:
                # 计算权重的方差演化
                weight_vars = [np.var(w) if w else 0 for w in weights]
                print(f"\n  聚合权重统计:")
                print(f"    权重方差范围: [{min(weight_vars):.6f}, {max(weight_vars):.6f}]")
    
    if 'param_evolution' in data:
        evolution = data['param_evolution']
        if evolution.get('update_magnitudes'):
            mags = evolution['update_magnitudes']
            print(f"\n  参数更新统计:")
            print(f"    平均更新幅度: {np.mean(mags):.6f}")
            print(f"    更新幅度范围: [{min(mags):.6f}, {max(mags):.6f}]")
    
    print("="*70 + "\n")


def visualize_training_curves(data: Dict, output_dir: str, experiment_name: str = "experiment"):
    """
    绘制训练曲线
    """
    history = data.get('training_history', {})
    if not history:
        print("  ⚠ 无训练历史数据，跳过训练曲线绘制")
        return
    
    rounds = history.get('rounds', [])
    
    # 1. 准确率曲线
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 训练/测试准确率
    ax1 = axes[0, 0]
    if history.get('train_acc'):
        train_accs = [x if x is not None else np.nan for x in history['train_acc']]
        ax1.plot(rounds, train_accs, 'b-', label='Train Acc', linewidth=2)
    if history.get('test_acc'):
        test_accs = [x if x is not None else np.nan for x in history['test_acc']]
        ax1.plot(rounds, test_accs, 'r-', label='Test Acc', linewidth=2)
    if history.get('local_test_acc'):
        local_accs = [x if x is not None else np.nan for x in history['local_test_acc']]
        ax1.plot(rounds, local_accs, 'g-', label='Local Acc', linewidth=2)
    
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 训练损失
    ax2 = axes[0, 1]
    if history.get('train_loss'):
        train_losses = [x if x is not None else np.nan for x in history['train_loss']]
        ax2.plot(rounds, train_losses, 'b-', label='Train Loss', linewidth=2)
    if history.get('test_loss'):
        test_losses = [x if x is not None else np.nan for x in history['test_loss']]
        ax2.plot(rounds, test_losses, 'r-', label='Test Loss', linewidth=2)
    
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # LoRA 参数范数
    ax3 = axes[1, 0]
    if history.get('lora_param_norms'):
        norms = history['lora_param_norms']
        ax3.plot(rounds[:len(norms)], norms, 'purple', linewidth=2)
        ax3.set_xlabel('Round')
        ax3.set_ylabel('LoRA Param Norm')
        ax3.set_title('LoRA Parameter Evolution')
        ax3.grid(True, alpha=0.3)
    
    # 通信量
    ax4 = axes[1, 1]
    if history.get('comm_volume_mb'):
        comm = history['comm_volume_mb']
        ax4.plot(rounds[:len(comm)], comm, 'orange', linewidth=2)
        ax4.set_xlabel('Round')
        ax4.set_ylabel('Cumulative Comm (MB)')
        ax4.set_title('Communication Volume')
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Training Curves - {experiment_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 训练曲线已保存: {save_path}")


def visualize_param_evolution(data: Dict, output_dir: str, experiment_name: str = "experiment"):
    """
    绘制参数演化图
    """
    evolution = data.get('param_evolution', {})
    if not evolution or not evolution.get('update_magnitudes'):
        print("  ⚠ 无参数演化数据，跳过参数演化图绘制")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    rounds = evolution.get('rounds', [])
    magnitudes = evolution.get('update_magnitudes', [])
    directions = evolution.get('update_directions', [])
    
    # 更新幅度
    ax1 = axes[0]
    ax1.bar(rounds, magnitudes, color='steelblue', alpha=0.7)
    ax1.axhline(y=np.mean(magnitudes), color='red', linestyle='--', label=f'Mean: {np.mean(magnitudes):.4f}')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Update Magnitude')
    ax1.set_title('Parameter Update Magnitude per Round')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 更新方向一致性
    ax2 = axes[1]
    if directions:
        ax2.plot(rounds[1:len(directions)+1], directions, 'g-o', linewidth=2, markersize=4)
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Direction Similarity')
        ax2.set_title('Update Direction Consistency (Cosine Similarity)')
        ax2.set_ylim(-1.1, 1.1)
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Parameter Evolution - {experiment_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'param_evolution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 参数演化图已保存: {save_path}")


def visualize_aggregation_weights(data: Dict, output_dir: str, experiment_name: str = "experiment"):
    """
    绘制聚合权重分析图
    """
    history = data.get('training_history', {})
    weights_history = history.get('aggregation_weights', [])
    scores_history = history.get('alignment_scores', [])
    
    if not weights_history:
        print("  ⚠ 无聚合权重数据，跳过聚合权重图绘制")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    rounds = history.get('rounds', range(len(weights_history)))
    
    # 1. 权重方差演化
    ax1 = axes[0, 0]
    weight_vars = [np.var(w) if w else 0 for w in weights_history]
    ax1.plot(rounds[:len(weight_vars)], weight_vars, 'b-', linewidth=2)
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Weight Variance')
    ax1.set_title('Aggregation Weight Variance Evolution')
    ax1.grid(True, alpha=0.3)
    
    # 2. 权重分布箱线图（选取几个关键轮次）
    ax2 = axes[0, 1]
    if len(weights_history) > 5:
        sample_indices = np.linspace(0, len(weights_history)-1, min(10, len(weights_history)), dtype=int)
        sample_weights = [weights_history[i] for i in sample_indices if weights_history[i]]
        sample_labels = [f'R{rounds[i]}' for i in sample_indices if weights_history[i]]
        
        if sample_weights:
            bp = ax2.boxplot(sample_weights, labels=sample_labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            ax2.axhline(y=1/len(weights_history[0]) if weights_history[0] else 0, 
                       color='red', linestyle='--', label='Uniform')
            ax2.set_xlabel('Round')
            ax2.set_ylabel('Weight')
            ax2.set_title('Aggregation Weight Distribution')
            ax2.legend()
    
    # 3. 对齐度分数演化（如果有）
    ax3 = axes[1, 0]
    if scores_history:
        score_means = [np.mean(s) if s else 0 for s in scores_history]
        score_stds = [np.std(s) if s else 0 for s in scores_history]
        
        ax3.errorbar(rounds[:len(score_means)], score_means, yerr=score_stds, 
                    fmt='o-', color='green', capsize=3, label='Mean ± Std')
        ax3.set_xlabel('Round')
        ax3.set_ylabel('Alignment Score')
        ax3.set_title('Alignment Score Evolution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. 权重热力图（最后一轮）
    ax4 = axes[1, 1]
    if weights_history and len(weights_history[-1]) > 0:
        # 取最后几轮的权重
        last_n = min(20, len(weights_history))
        heatmap_data = np.array([weights_history[-i] if weights_history[-i] else [0] 
                                for i in range(last_n, 0, -1) if weights_history[-i]])
        
        if heatmap_data.size > 0:
            im = ax4.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
            ax4.set_xlabel('Client Index')
            ax4.set_ylabel('Round (recent)')
            ax4.set_title(f'Aggregation Weights Heatmap (Last {last_n} Rounds)')
            plt.colorbar(im, ax=ax4, label='Weight')
    
    plt.suptitle(f'Aggregation Analysis - {experiment_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'aggregation_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 聚合权重分析图已保存: {save_path}")


def visualize_gate_evolution(data: Dict, output_dir: str, experiment_name: str = "experiment"):
    """
    绘制门控参数演化图（FedSDG 专用）
    """
    history = data.get('training_history', {})
    
    gate_means = history.get('gate_values_mean', [])
    gate_stds = history.get('gate_values_std', [])
    gate_mins = history.get('gate_values_min', [])
    gate_maxs = history.get('gate_values_max', [])
    
    if not gate_means:
        print("  ⚠ 无门控参数数据（非 FedSDG 实验），跳过门控参数图绘制")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    rounds = history.get('rounds', range(len(gate_means)))[:len(gate_means)]
    
    # 1. 门控值演化曲线
    ax1 = axes[0]
    ax1.plot(rounds, gate_means, 'b-', linewidth=2, label='Mean')
    ax1.fill_between(rounds, 
                     [m - s for m, s in zip(gate_means, gate_stds)],
                     [m + s for m, s in zip(gate_means, gate_stds)],
                     alpha=0.3, color='blue', label='±1 Std')
    ax1.plot(rounds, gate_mins, 'g--', linewidth=1, alpha=0.7, label='Min')
    ax1.plot(rounds, gate_maxs, 'r--', linewidth=1, alpha=0.7, label='Max')
    ax1.axhline(y=0.5, color='gray', linestyle=':', linewidth=2, label='Neutral (0.5)')
    
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Gate Value (m_k)')
    ax1.set_title('Gate Parameter Evolution')
    ax1.set_ylim(0, 1)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 2. 门控分化程度
    ax2 = axes[1]
    # 计算分化指标：距离 0.5 的平均距离
    differentiation = [abs(m - 0.5) for m in gate_means]
    ax2.bar(rounds, differentiation, color='purple', alpha=0.7)
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Differentiation (|m_k - 0.5|)')
    ax2.set_title('Gate Differentiation Level')
    ax2.set_ylim(0, 0.5)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'FedSDG Gate Analysis - {experiment_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'gate_evolution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 门控参数演化图已保存: {save_path}")


def visualize_comparison(data1: Dict, data2: Dict, output_dir: str, 
                         name1: str = "Experiment 1", name2: str = "Experiment 2"):
    """
    比较两个实验的结果
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 测试准确率比较
    ax1 = axes[0]
    
    history1 = data1.get('training_history', {})
    history2 = data2.get('training_history', {})
    
    if history1.get('test_acc'):
        rounds1 = history1.get('rounds', range(len(history1['test_acc'])))
        test_accs1 = [x if x is not None else np.nan for x in history1['test_acc']]
        ax1.plot(rounds1, test_accs1, 'b-', linewidth=2, label=name1)
    
    if history2.get('test_acc'):
        rounds2 = history2.get('rounds', range(len(history2['test_acc'])))
        test_accs2 = [x if x is not None else np.nan for x in history2['test_acc']]
        ax1.plot(rounds2, test_accs2, 'r-', linewidth=2, label=name2)
    
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Test Accuracy Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 通信效率比较
    ax2 = axes[1]
    
    if history1.get('comm_volume_mb') and history1.get('test_acc'):
        comm1 = history1['comm_volume_mb']
        acc1 = [x if x is not None else 0 for x in history1['test_acc']]
        ax2.plot(comm1[:len(acc1)], acc1, 'b-', linewidth=2, label=name1)
    
    if history2.get('comm_volume_mb') and history2.get('test_acc'):
        comm2 = history2['comm_volume_mb']
        acc2 = [x if x is not None else 0 for x in history2['test_acc']]
        ax2.plot(comm2[:len(acc2)], acc2, 'r-', linewidth=2, label=name2)
    
    ax2.set_xlabel('Communication Volume (MB)')
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title('Communication Efficiency Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Experiment Comparison: {name1} vs {name2}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 比较图已保存: {save_path}")


def analyze_checkpoint(checkpoint_path: str, output_dir: str):
    """
    分析特定检查点
    """
    checkpoint = CheckpointManager.load_checkpoint(checkpoint_path)
    
    print(f"\n检查点分析: {checkpoint_path}")
    print("="*70)
    print(f"  轮次: {checkpoint.get('round_idx', 'N/A')}")
    print(f"  算法: {checkpoint.get('alg', 'N/A')}")
    print(f"  训练准确率: {checkpoint.get('train_acc', 'N/A')}")
    print(f"  测试准确率: {checkpoint.get('test_acc', 'N/A')}")
    print(f"  本地准确率: {checkpoint.get('local_test_acc', 'N/A')}")
    
    if checkpoint.get('local_weights'):
        print(f"\n  客户端本地权重:")
        print(f"    客户端数量: {len(checkpoint['local_weights'])}")
        
        # 计算客户端间差异
        if len(checkpoint['local_weights']) > 1:
            weights = checkpoint['local_weights']
            diffs = []
            for i in range(len(weights)):
                for j in range(i+1, len(weights)):
                    diff = sum(
                        (weights[i][k].float() - weights[j][k].float()).norm().item()**2
                        for k in weights[i].keys()
                    )
                    diffs.append(np.sqrt(diff))
            print(f"    客户端间平均差异: {np.mean(diffs):.6f}")
            print(f"    客户端间差异范围: [{min(diffs):.6f}, {max(diffs):.6f}]")
    
    if checkpoint.get('aggregation_info'):
        agg = checkpoint['aggregation_info']
        print(f"\n  聚合信息:")
        print(f"    聚合方法: {agg.get('agg_method', 'N/A')}")
        if agg.get('weights'):
            weights = agg['weights']
            print(f"    权重范围: [{min(weights):.6f}, {max(weights):.6f}]")
            print(f"    权重方差: {np.var(weights):.6f}")
    
    if checkpoint.get('private_states'):
        print(f"\n  FedSDG 私有状态:")
        print(f"    客户端数量: {len(checkpoint['private_states'])}")
        
        # 提取门控值
        gate_values = []
        for client_id, state in checkpoint['private_states'].items():
            for name, param in state.items():
                if 'lambda_k_logit' in name:
                    import torch
                    m_k = torch.sigmoid(param).item()
                    gate_values.append(m_k)
        
        if gate_values:
            print(f"    门控值范围: [{min(gate_values):.4f}, {max(gate_values):.4f}]")
            print(f"    门控值均值: {np.mean(gate_values):.4f}")
    
    print("="*70 + "\n")


def main():
    args = parse_args()
    
    # 检查实验目录
    if not os.path.exists(args.experiment_dir):
        print(f"错误: 实验目录不存在: {args.experiment_dir}")
        sys.exit(1)
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = os.path.join(args.experiment_dir, 'analysis')
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("联邦学习训练历史分析工具".center(70))
    print("="*70)
    print(f"  实验目录: {args.experiment_dir}")
    print(f"  输出目录: {args.output_dir}")
    print("="*70 + "\n")
    
    # 加载实验数据
    data = load_experiment(args.experiment_dir)
    
    if not data:
        print("错误: 无法加载实验数据")
        sys.exit(1)
    
    # 获取实验名称
    experiment_name = data.get('json_summary', {}).get('experiment_name', 'experiment')
    
    # 打印摘要
    print_summary(data)
    
    # 分析特定检查点
    if args.checkpoint_round is not None:
        checkpoint_path = os.path.join(
            args.experiment_dir, 'checkpoints', f'round_{args.checkpoint_round:04d}.pkl'
        )
        if os.path.exists(checkpoint_path):
            analyze_checkpoint(checkpoint_path, args.output_dir)
        else:
            print(f"  ⚠ 检查点不存在: {checkpoint_path}")
    
    # 生成可视化
    if args.visualize:
        print("\n生成可视化图表...")
        visualize_training_curves(data, args.output_dir, experiment_name)
        visualize_param_evolution(data, args.output_dir, experiment_name)
        visualize_aggregation_weights(data, args.output_dir, experiment_name)
        visualize_gate_evolution(data, args.output_dir, experiment_name)
    
    # 比较实验
    if args.compare:
        if os.path.exists(args.compare):
            print(f"\n比较实验: {args.compare}")
            data2 = load_experiment(args.compare)
            if data2:
                name2 = data2.get('json_summary', {}).get('experiment_name', 'experiment2')
                visualize_comparison(data, data2, args.output_dir, experiment_name, name2)
        else:
            print(f"  ⚠ 比较实验目录不存在: {args.compare}")
    
    # 导出 JSON
    if args.export_json:
        analyzer = TrainingAnalyzer(args.experiment_dir)
        json_path = os.path.join(args.output_dir, 'visualization_data.json')
        analyzer.export_for_visualization(json_path)
    
    print("\n分析完成！")
    print(f"结果保存在: {args.output_dir}\n")


if __name__ == '__main__':
    main()

