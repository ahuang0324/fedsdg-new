#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FedSDG 门控系数可视化模块

提供多种可视化方法来分析门控参数 (lambda_k) 的分布和分化情况：
1. 热力图 (Heatmap): 展示各层门控值
2. 分布直方图 (Histogram): 展示门控值的整体分布
3. 箱线图 (Boxplot): 展示门控值的统计特征
4. 极坐标图 (Polar): 展示各层门控值的雷达图
5. 分化指标图 (Metrics): 展示分化程度的量化指标
6. 客户端对比图 (Client Comparison): 对比不同客户端的门控分布
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适合服务器环境
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Optional, Tuple, Any

from .paths import get_visualization_path, ensure_dir


def extract_gate_values(model: Any) -> Dict[str, float]:
    """
    从模型中提取所有门控参数值
    
    Args:
        model: 包含 FedSDG LoRA 层的模型
        
    Returns:
        Dict[str, float]: {层名称: 门控值 m_k}
    """
    gate_values = {}
    for name, param in model.named_parameters():
        if 'lambda_k_logit' in name:
            # m_k = sigmoid(lambda_k_logit)
            m_k = torch.sigmoid(param).item()
            # 提取层名称（去掉 .lambda_k_logit 后缀）
            layer_name = name.replace('.lambda_k_logit', '')
            gate_values[layer_name] = m_k
    return gate_values


def extract_gate_values_from_private_states(
    local_private_states: Dict[int, Dict[str, torch.Tensor]]
) -> Dict[int, Dict[str, float]]:
    """
    从客户端私有状态中提取所有门控参数值
    
    Args:
        local_private_states: {client_id: {param_name: tensor}}
        
    Returns:
        Dict[int, Dict[str, float]]: {client_id: {层名称: 门控值}}
    """
    all_client_gates: Dict[int, Dict[str, float]] = {}
    for client_id, private_state in local_private_states.items():
        gate_values: Dict[str, float] = {}
        for name, param in private_state.items():
            if 'lambda_k_logit' in name:
                m_k = torch.sigmoid(param).item()
                layer_name = name.replace('.lambda_k_logit', '')
                gate_values[layer_name] = m_k
        if gate_values:
            all_client_gates[client_id] = gate_values
    return all_client_gates


def compute_aggregated_gate_values(
    all_client_gates: Dict[int, Dict[str, float]]
) -> Dict[str, float]:
    """
    计算所有客户端门控值的聚合统计（平均值）
    
    Args:
        all_client_gates: {client_id: {层名称: 门控值}}
        
    Returns:
        Dict[str, float]: {层名称: 平均门控值}
    """
    if not all_client_gates:
        return {}
    
    # 获取所有层名称
    first_client = list(all_client_gates.values())[0]
    layer_names = list(first_client.keys())
    
    # 计算每层的平均门控值
    aggregated: Dict[str, float] = {}
    for layer in layer_names:
        values = [all_client_gates[cid].get(layer, 0.5) for cid in all_client_gates]
        aggregated[layer] = float(np.mean(values))
    
    return aggregated


def visualize_gate_heatmap(
    gate_values: Dict[str, float], 
    save_path: str, 
    title: str = "Gate Values Heatmap"
) -> None:
    """
    热力图可视化：展示各层门控值
    
    Args:
        gate_values: {层名称: 门控值}
        save_path: 保存路径
        title: 图表标题
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 准备数据
    layers = list(gate_values.keys())
    values = list(gate_values.values())
    
    # 简化层名称（只保留关键部分）
    short_names = []
    for name in layers:
        parts = name.split('.')
        if 'blocks' in name:
            block_idx = parts[parts.index('blocks') + 1] if 'blocks' in parts else '?'
            layer_type = 'attn' if 'attn' in name else 'mlp' if 'mlp' in name else 'other'
            sublayer = parts[-1] if len(parts) > 0 else ''
            short_names.append(f"B{block_idx}.{layer_type}.{sublayer}")
        else:
            short_names.append(name[-20:])  # 取最后20个字符
    
    # 创建热力图数据（1行 x N列）
    data = np.array(values).reshape(1, -1)
    
    # 绘制热力图
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # 设置坐标轴
    ax.set_yticks([])
    ax.set_xticks(range(len(short_names)))
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
    
    # 在每个格子中显示数值
    for i, v in enumerate(values):
        color = 'white' if v < 0.3 or v > 0.7 else 'black'
        ax.text(i, 0, f'{v:.2f}', ha='center', va='center', color=color, fontsize=8)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.2)
    cbar.set_label('Gate Value (m_k): 0=Global, 1=Private', fontsize=10)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 热力图已保存: {save_path}")


def visualize_gate_histogram(
    gate_values: Dict[str, float], 
    save_path: str, 
    title: str = "Gate Values Distribution"
) -> None:
    """
    直方图可视化：展示门控值的整体分布
    
    Args:
        gate_values: {层名称: 门控值}
        save_path: 保存路径
        title: 图表标题
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    values = list(gate_values.values())
    
    # 左图：直方图
    ax1 = axes[0]
    n, bins, patches = ax1.hist(values, bins=20, range=(0, 1), edgecolor='black', alpha=0.7)
    
    # 根据值着色
    for i, patch in enumerate(patches):
        bin_center = (bins[i] + bins[i+1]) / 2
        if bin_center < 0.3:
            patch.set_facecolor('green')  # 偏向全局
        elif bin_center > 0.7:
            patch.set_facecolor('red')    # 偏向私有
        else:
            patch.set_facecolor('yellow') # 中间值
    
    ax1.axvline(x=0.5, color='blue', linestyle='--', linewidth=2, label='Neutral (0.5)')
    ax1.axvline(x=np.mean(values), color='orange', linestyle='-', linewidth=2, 
                label=f'Mean ({np.mean(values):.3f})')
    
    ax1.set_xlabel('Gate Value (m_k)', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('Distribution of Gate Values', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.set_xlim(0, 1)
    
    # 添加分化区域标注
    ax1.axvspan(0, 0.3, alpha=0.1, color='green', label='Global-biased')
    ax1.axvspan(0.7, 1, alpha=0.1, color='red', label='Private-biased')
    
    # 右图：核密度估计 (KDE)
    ax2 = axes[1]
    try:
        from scipy import stats
        kde = stats.gaussian_kde(values)
        x_range = np.linspace(0, 1, 200)
        ax2.fill_between(x_range, kde(x_range), alpha=0.5, color='steelblue')
        ax2.plot(x_range, kde(x_range), color='navy', linewidth=2)
    except Exception:
        # 如果 KDE 失败，使用简单的直方图
        ax2.hist(values, bins=20, range=(0, 1), density=True, alpha=0.7, color='steelblue')
    
    ax2.axvline(x=0.5, color='blue', linestyle='--', linewidth=2)
    ax2.axvline(x=np.mean(values), color='orange', linestyle='-', linewidth=2)
    
    ax2.set_xlabel('Gate Value (m_k)', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Kernel Density Estimation', fontsize=12)
    ax2.set_xlim(0, 1)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 分布直方图已保存: {save_path}")


def visualize_gate_boxplot(
    gate_values: Dict[str, float], 
    save_path: str, 
    title: str = "Gate Values Statistics"
) -> None:
    """
    箱线图可视化：展示门控值的统计特征
    
    Args:
        gate_values: {层名称: 门控值}
        save_path: 保存路径
        title: 图表标题
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    values = list(gate_values.values())
    
    # 左图：整体箱线图
    ax1 = axes[0]
    bp = ax1.boxplot(values, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['medians'][0].set_color('red')
    bp['medians'][0].set_linewidth(2)
    
    # 添加散点
    ax1.scatter([1] * len(values), values, alpha=0.5, color='navy', s=30, zorder=3)
    
    ax1.axhline(y=0.5, color='green', linestyle='--', linewidth=2, label='Neutral (0.5)')
    ax1.set_ylabel('Gate Value (m_k)', fontsize=11)
    ax1.set_title('Overall Distribution', fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.set_xticklabels(['All Layers'])
    ax1.legend()
    
    # 右图：按层类型分组的箱线图
    ax2 = axes[1]
    
    # 按层类型分组
    attn_values = [v for k, v in gate_values.items() if 'attn' in k]
    mlp_values = [v for k, v in gate_values.items() if 'mlp' in k]
    other_values = [v for k, v in gate_values.items() if 'attn' not in k and 'mlp' not in k]
    
    data_groups = []
    labels = []
    if attn_values:
        data_groups.append(attn_values)
        labels.append(f'Attention\n(n={len(attn_values)})')
    if mlp_values:
        data_groups.append(mlp_values)
        labels.append(f'MLP\n(n={len(mlp_values)})')
    if other_values:
        data_groups.append(other_values)
        labels.append(f'Other\n(n={len(other_values)})')
    
    if data_groups:
        bp2 = ax2.boxplot(data_groups, vert=True, patch_artist=True)
        colors = ['lightcoral', 'lightgreen', 'lightyellow']
        for patch, color in zip(bp2['boxes'], colors[:len(data_groups)]):
            patch.set_facecolor(color)
        
        ax2.axhline(y=0.5, color='green', linestyle='--', linewidth=2)
        ax2.set_xticklabels(labels)
        ax2.set_ylabel('Gate Value (m_k)', fontsize=11)
        ax2.set_title('By Layer Type', fontsize=12)
        ax2.set_ylim(0, 1)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 箱线图已保存: {save_path}")


def visualize_gate_polar(
    gate_values: Dict[str, float], 
    save_path: str, 
    title: str = "Gate Values Radar"
) -> None:
    """
    极坐标雷达图可视化：展示各层门控值
    
    Args:
        gate_values: {层名称: 门控值}
        save_path: 保存路径
        title: 图表标题
    """
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    layers = list(gate_values.keys())
    values = list(gate_values.values())
    
    # 简化层名称
    short_names = []
    for name in layers:
        parts = name.split('.')
        if 'blocks' in name:
            block_idx = parts[parts.index('blocks') + 1] if 'blocks' in parts else '?'
            layer_type = 'A' if 'attn' in name else 'M' if 'mlp' in name else 'O'
            short_names.append(f"B{block_idx}{layer_type}")
        else:
            short_names.append(name[-10:])
    
    # 计算角度
    num_vars = len(values)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # 闭合图形
    values_closed = values + [values[0]]
    angles_closed = angles + [angles[0]]
    
    # 绘制雷达图
    ax.plot(angles_closed, values_closed, 'o-', linewidth=2, color='steelblue', markersize=6)
    ax.fill(angles_closed, values_closed, alpha=0.25, color='steelblue')
    
    # 添加 0.5 参考线
    ref_values = [0.5] * (num_vars + 1)
    ax.plot(angles_closed, ref_values, '--', linewidth=1.5, color='red', alpha=0.7, label='Neutral (0.5)')
    
    # 设置标签
    ax.set_xticks(angles)
    ax.set_xticklabels(short_names, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=8)
    
    ax.set_title(title, fontsize=14, fontweight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 雷达图已保存: {save_path}")


def visualize_gate_metrics(
    gate_values: Dict[str, float], 
    save_path: str, 
    title: str = "Gate Differentiation Metrics"
) -> None:
    """
    分化指标可视化：展示分化程度的量化指标
    
    Args:
        gate_values: {层名称: 门控值}
        save_path: 保存路径
        title: 图表标题
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    values = np.array(list(gate_values.values()))
    
    # 计算各种分化指标
    mean_val = np.mean(values)
    std_val = np.std(values)
    range_val = np.max(values) - np.min(values)
    
    # 分化度量：距离 0.5 的平均距离
    differentiation = np.mean(np.abs(values - 0.5))
    
    # 极化度量：接近 0 或 1 的比例
    polarized_ratio = np.mean((values < 0.2) | (values > 0.8))
    
    # 双边惩罚值：min(m_k, 1-m_k) 的平均值
    bilateral_penalty = np.mean(np.minimum(values, 1 - values))
    
    # 1. 指标仪表盘
    ax1 = axes[0, 0]
    metrics = {
        'Mean': mean_val,
        'Std Dev': std_val,
        'Range': range_val,
        'Differentiation': differentiation,
        'Polarized Ratio': polarized_ratio,
        'Bilateral Penalty': bilateral_penalty
    }
    
    y_pos = np.arange(len(metrics))
    colors = ['steelblue', 'orange', 'green', 'red', 'purple', 'brown']
    bars = ax1.barh(y_pos, list(metrics.values()), color=colors, alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(list(metrics.keys()))
    ax1.set_xlim(0, 1)
    ax1.set_xlabel('Value')
    ax1.set_title('Differentiation Metrics', fontsize=12)
    
    # 在条形图上显示数值
    for bar, val in zip(bars, metrics.values()):
        ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=10)
    
    # 2. 分化程度饼图
    ax2 = axes[0, 1]
    global_biased = int(np.sum(values < 0.3))
    neutral = int(np.sum((values >= 0.3) & (values <= 0.7)))
    private_biased = int(np.sum(values > 0.7))
    
    sizes = [global_biased, neutral, private_biased]
    labels_pie = [f'Global-biased\n(<0.3): {global_biased}', 
                  f'Neutral\n(0.3-0.7): {neutral}', 
                  f'Private-biased\n(>0.7): {private_biased}']
    colors_pie = ['green', 'yellow', 'red']
    explode = (0.05, 0, 0.05)
    
    if sum(sizes) > 0:
        ax2.pie(sizes, explode=explode, labels=labels_pie, colors=colors_pie, autopct='%1.1f%%',
                shadow=True, startangle=90)
    ax2.set_title('Gate Value Categories', fontsize=12)
    
    # 3. 累积分布函数 (CDF)
    ax3 = axes[1, 0]
    sorted_values = np.sort(values)
    cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    ax3.plot(sorted_values, cdf, 'b-', linewidth=2, label='Empirical CDF')
    ax3.axvline(x=0.5, color='red', linestyle='--', label='Neutral (0.5)')
    ax3.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax3.fill_between(sorted_values, 0, cdf, alpha=0.3)
    ax3.set_xlabel('Gate Value (m_k)')
    ax3.set_ylabel('Cumulative Probability')
    ax3.set_title('Cumulative Distribution Function', fontsize=12)
    ax3.legend()
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    
    # 4. 分化趋势（按层顺序）
    ax4 = axes[1, 1]
    layer_indices = range(len(values))
    ax4.bar(layer_indices, values, color='steelblue', alpha=0.7, edgecolor='navy')
    ax4.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Neutral')
    ax4.axhline(y=mean_val, color='orange', linestyle='-', linewidth=2, label=f'Mean ({mean_val:.3f})')
    ax4.fill_between(layer_indices, 0.3, 0.7, alpha=0.1, color='yellow', label='Neutral Zone')
    ax4.set_xlabel('Layer Index')
    ax4.set_ylabel('Gate Value (m_k)')
    ax4.set_title('Gate Values by Layer Order', fontsize=12)
    ax4.legend(loc='upper right')
    ax4.set_ylim(0, 1)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 分化指标图已保存: {save_path}")


def visualize_client_comparison(
    all_client_gates: Dict[int, Dict[str, float]], 
    save_path: str, 
    max_clients: int = 20, 
    title: str = "Client Gate Comparison"
) -> None:
    """
    客户端对比可视化：对比不同客户端的门控分布
    
    Args:
        all_client_gates: {client_id: {层名称: 门控值}}
        save_path: 保存路径
        max_clients: 最多显示的客户端数量
        title: 图表标题
    """
    if not all_client_gates:
        print("  ⚠ 没有客户端门控数据，跳过客户端对比图")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 限制客户端数量
    client_ids = list(all_client_gates.keys())[:max_clients]
    
    # 1. 客户端门控均值对比
    ax1 = axes[0, 0]
    client_means = [np.mean(list(all_client_gates[cid].values())) for cid in client_ids]
    colors = ['green' if m < 0.4 else 'red' if m > 0.6 else 'yellow' for m in client_means]
    ax1.bar(range(len(client_ids)), client_means, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=0.5, color='blue', linestyle='--', linewidth=2, label='Neutral')
    ax1.set_xlabel('Client ID')
    ax1.set_ylabel('Mean Gate Value')
    ax1.set_title('Mean Gate Value per Client', fontsize=12)
    ax1.set_xticks(range(len(client_ids)))
    ax1.set_xticklabels(client_ids, rotation=45)
    ax1.set_ylim(0, 1)
    ax1.legend()
    
    # 2. 客户端门控标准差对比
    ax2 = axes[0, 1]
    client_stds = [np.std(list(all_client_gates[cid].values())) for cid in client_ids]
    ax2.bar(range(len(client_ids)), client_stds, color='steelblue', alpha=0.7, edgecolor='navy')
    ax2.set_xlabel('Client ID')
    ax2.set_ylabel('Std Dev of Gate Values')
    ax2.set_title('Gate Value Variance per Client', fontsize=12)
    ax2.set_xticks(range(len(client_ids)))
    ax2.set_xticklabels(client_ids, rotation=45)
    
    # 3. 热力图：客户端 x 层
    ax3 = axes[1, 0]
    
    # 获取所有层名称（取第一个客户端的层）
    first_client = list(all_client_gates.values())[0]
    layer_names = list(first_client.keys())
    
    # 构建热力图数据
    heatmap_data = []
    for cid in client_ids:
        client_values = [all_client_gates[cid].get(layer, 0.5) for layer in layer_names]
        heatmap_data.append(client_values)
    
    heatmap_data = np.array(heatmap_data)
    
    im = ax3.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax3.set_xlabel('Layer Index')
    ax3.set_ylabel('Client ID')
    ax3.set_yticks(range(len(client_ids)))
    ax3.set_yticklabels(client_ids)
    ax3.set_title('Gate Values Heatmap (Client x Layer)', fontsize=12)
    plt.colorbar(im, ax=ax3, label='Gate Value')
    
    # 4. 箱线图：各客户端门控分布
    ax4 = axes[1, 1]
    client_data = [list(all_client_gates[cid].values()) for cid in client_ids]
    bp = ax4.boxplot(client_data, vert=True, patch_artist=True)
    
    for i, patch in enumerate(bp['boxes']):
        mean_val = client_means[i]
        if mean_val < 0.4:
            patch.set_facecolor('lightgreen')
        elif mean_val > 0.6:
            patch.set_facecolor('lightcoral')
        else:
            patch.set_facecolor('lightyellow')
    
    ax4.axhline(y=0.5, color='blue', linestyle='--', linewidth=2)
    ax4.set_xlabel('Client ID')
    ax4.set_ylabel('Gate Value')
    ax4.set_title('Gate Value Distribution per Client', fontsize=12)
    ax4.set_xticklabels(client_ids, rotation=45)
    ax4.set_ylim(0, 1)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 客户端对比图已保存: {save_path}")


def visualize_all_gates(
    model: Any = None, 
    local_private_states: Optional[Dict[int, Dict[str, torch.Tensor]]] = None, 
    algorithm: str = 'fedsdg',
    experiment_name: str = 'experiment',
    prefix: str = 'fedsdg'
) -> Tuple[Optional[Dict[str, float]], Optional[Dict[int, Dict[str, float]]]]:
    """
    生成所有门控可视化图表
    
    重要说明：
    - FedSDG 中门控参数 (lambda_k_logit) 是私有参数，不参与联邦聚合
    - 全局模型的门控值始终为初始值 0.5，没有分析意义
    - 本函数优先使用客户端私有状态中的门控值进行可视化
    
    Args:
        model: 包含 FedSDG LoRA 层的模型
        local_private_states: 客户端私有状态（必需，用于获取真实的门控分化结果）
        algorithm: 算法名称
        experiment_name: 实验名称
        prefix: 文件名前缀
        
    Returns:
        Tuple[gate_values, all_client_gates]: 门控值和客户端门控值
    """
    print("\n" + "="*70)
    print("[FedSDG] 生成门控系数可视化图表")
    print("="*70)
    
    # 优先从客户端私有状态提取门控值（这才是有意义的分化结果）
    all_client_gates: Dict[int, Dict[str, float]] = {}
    if local_private_states:
        all_client_gates = extract_gate_values_from_private_states(local_private_states)
    
    if all_client_gates:
        # 使用客户端门控值的聚合统计作为主要可视化数据
        gate_values = compute_aggregated_gate_values(all_client_gates)
        print(f"  ✓ 从 {len(all_client_gates)} 个客户端私有状态提取门控值")
        print(f"  ✓ 发现 {len(gate_values)} 个门控参数（每层）")
        print(f"  ✓ 聚合门控值范围: {min(gate_values.values()):.4f} - {max(gate_values.values()):.4f}")
        print(f"  ✓ 聚合门控值均值: {np.mean(list(gate_values.values())):.4f}")
        use_client_data = True
    else:
        # 回退到全局模型（但会警告用户这些值没有意义）
        if model is None:
            print("  ⚠ 未提供客户端私有状态或模型，跳过可视化")
            return None, None
        gate_values = extract_gate_values(model)
        if not gate_values:
            print("  ⚠ 未找到门控参数，跳过可视化")
            return None, None
        print(f"  ⚠ 警告：未提供客户端私有状态，使用全局模型门控值")
        print(f"  ⚠ 全局模型门控值为初始值 0.5，不反映真实的分化情况")
        print(f"  发现 {len(gate_values)} 个门控参数")
        use_client_data = False
    
    print()
    
    # 根据数据来源设置标题后缀
    title_suffix = " (Client Aggregated)" if use_client_data else " (Global Model - Initial Values)"
    
    # 1. 热力图
    visualize_gate_heatmap(
        gate_values, 
        get_visualization_path(algorithm, experiment_name, f'{prefix}_gate_heatmap.png'),
        title="FedSDG Gate Values Heatmap" + title_suffix
    )
    
    # 2. 分布直方图
    visualize_gate_histogram(
        gate_values,
        get_visualization_path(algorithm, experiment_name, f'{prefix}_gate_histogram.png'),
        title="FedSDG Gate Values Distribution" + title_suffix
    )
    
    # 3. 箱线图
    visualize_gate_boxplot(
        gate_values,
        get_visualization_path(algorithm, experiment_name, f'{prefix}_gate_boxplot.png'),
        title="FedSDG Gate Values Statistics" + title_suffix
    )
    
    # 4. 雷达图
    visualize_gate_polar(
        gate_values,
        get_visualization_path(algorithm, experiment_name, f'{prefix}_gate_radar.png'),
        title="FedSDG Gate Values Radar" + title_suffix
    )
    
    # 5. 分化指标图
    visualize_gate_metrics(
        gate_values,
        get_visualization_path(algorithm, experiment_name, f'{prefix}_gate_metrics.png'),
        title="FedSDG Gate Differentiation Metrics" + title_suffix
    )
    
    # 6. 客户端对比图（如果有私有状态）
    if all_client_gates:
        visualize_client_comparison(
            all_client_gates,
            get_visualization_path(algorithm, experiment_name, f'{prefix}_client_comparison.png'),
            title="FedSDG Client Gate Comparison"
        )
    
    print()
    viz_dir = get_visualization_path(algorithm, experiment_name, '')
    print(f"  所有可视化图表已保存到: {viz_dir}")
    print("="*70 + "\n")
    
    return gate_values, all_client_gates if all_client_gates else None


# =============================================================================
# 命令行入口 - 用于手动可视化已保存的实验
# =============================================================================

def find_fedsdg_experiments() -> List[Tuple[str, str, float]]:
    """
    查找所有 FedSDG 实验的 checkpoint 目录
    
    Returns:
        list: [(实验名称, checkpoint路径, 修改时间), ...]
    """
    from .paths import CHECKPOINTS_DIR
    from datetime import datetime
    
    fedsdg_dir = os.path.join(CHECKPOINTS_DIR, 'fedsdg')
    experiments = []
    
    if not os.path.exists(fedsdg_dir):
        return experiments
    
    for exp_name in os.listdir(fedsdg_dir):
        exp_path = os.path.join(fedsdg_dir, exp_name)
        if os.path.isdir(exp_path):
            private_states_path = os.path.join(exp_path, 'final_private_states.pkl')
            if os.path.exists(private_states_path):
                mtime = os.path.getmtime(exp_path)
                experiments.append((exp_name, exp_path, mtime))
    
    experiments.sort(key=lambda x: x[2], reverse=True)
    return experiments


def load_private_states_from_checkpoint(checkpoint_dir: str) -> Optional[Dict[int, Dict[str, torch.Tensor]]]:
    """
    从 checkpoint 目录加载私有状态
    
    Args:
        checkpoint_dir: checkpoint 目录路径
        
    Returns:
        私有状态字典，或 None
    """
    import pickle
    
    private_states_path = os.path.join(checkpoint_dir, 'final_private_states.pkl')
    
    if not os.path.exists(private_states_path):
        print(f"⚠ 未找到私有状态文件: {private_states_path}")
        return None
    
    print(f"加载私有状态: {private_states_path}")
    
    with open(private_states_path, 'rb') as f:
        return pickle.load(f)


def main():
    """命令行入口 - 手动可视化 FedSDG 门控系数"""
    import argparse
    from .paths import CHECKPOINTS_DIR
    from datetime import datetime
    
    parser = argparse.ArgumentParser(
        description='FedSDG 门控系数可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m fl.utils.visualization --list
  python -m fl.utils.visualization --latest
  python -m fl.utils.visualization --experiment cifar100_vit_pretrained
  python -m fl.utils.visualization --checkpoint outputs/checkpoints/fedsdg/your_exp
        """
    )
    
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                        help='Checkpoint 目录路径')
    parser.add_argument('--experiment', '-e', type=str, default=None,
                        help='实验名称（支持模糊匹配）')
    parser.add_argument('--prefix', '-p', type=str, default='analysis',
                        help='输出文件前缀（默认: analysis）')
    parser.add_argument('--list', '-l', action='store_true',
                        help='列出所有可用的 FedSDG 实验')
    parser.add_argument('--latest', action='store_true',
                        help='使用最新的实验')
    
    args = parser.parse_args()
    
    # 列出实验
    if args.list:
        experiments = find_fedsdg_experiments()
        if not experiments:
            print(f"\n⚠ 未找到任何 FedSDG 实验检查点")
            print(f"  检查点目录: {os.path.join(CHECKPOINTS_DIR, 'fedsdg')}")
            return 0
        
        print("\n" + "="*80)
        print("可用的 FedSDG 实验")
        print("="*80)
        for i, (exp_name, exp_path, mtime) in enumerate(experiments, 1):
            time_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n  [{i}] {exp_name}")
            print(f"      路径: {exp_path}")
            print(f"      时间: {time_str}")
        print("\n" + "="*80)
        print(f"共 {len(experiments)} 个实验")
        print("="*80 + "\n")
        return 0
    
    # 解析 checkpoint 目录
    checkpoint_dir = None
    
    if args.checkpoint:
        if not os.path.exists(args.checkpoint):
            print(f"✗ 指定目录不存在: {args.checkpoint}")
            return 1
        checkpoint_dir = args.checkpoint
    elif args.experiment:
        fedsdg_dir = os.path.join(CHECKPOINTS_DIR, 'fedsdg')
        candidate = os.path.join(fedsdg_dir, args.experiment)
        if os.path.exists(candidate):
            checkpoint_dir = candidate
        elif os.path.exists(fedsdg_dir):
            for exp_name in os.listdir(fedsdg_dir):
                if args.experiment in exp_name:
                    checkpoint_dir = os.path.join(fedsdg_dir, exp_name)
                    print(f"模糊匹配到实验: {exp_name}")
                    break
        if checkpoint_dir is None:
            print(f"✗ 未找到匹配的实验: {args.experiment}")
            return 1
    elif args.latest:
        experiments = find_fedsdg_experiments()
        if experiments:
            _, checkpoint_dir, _ = experiments[0]
            print(f"使用最新实验: {os.path.basename(checkpoint_dir)}")
        else:
            print("✗ 未找到任何 FedSDG 实验")
            return 1
    else:
        parser.print_help()
        print("\n⚠ 请指定 --checkpoint, --experiment, --latest 或 --list")
        return 0
    
    # 加载私有状态并可视化
    private_states = load_private_states_from_checkpoint(checkpoint_dir)
    if private_states is None:
        return 1
    
    experiment_name = os.path.basename(checkpoint_dir)
    
    # 调用可视化函数
    gate_values, _ = visualize_all_gates(
        model=None,
        local_private_states=private_states,
        algorithm='fedsdg',
        experiment_name=experiment_name,
        prefix=args.prefix
    )
    
    return 0 if gate_values else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
