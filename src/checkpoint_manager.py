#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
联邦学习训练检查点管理器

提供全面的训练历史保存功能，支持：
1. 每轮全局模型参数保存
2. LoRA 参数演化追踪
3. 客户端本地参数保存
4. 聚合权重和对齐度分数保存
5. FedSDG 私有参数演化保存
6. 门控系数历史追踪

设计原则：
- 可扩展性：易于添加新的联邦学习算法
- 灵活性：支持选择性保存和加载
- 高效性：支持增量保存，避免内存溢出
"""

import os
import copy
import pickle
import json
import time
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch


class CheckpointManager:
    """
    联邦学习检查点管理器
    
    功能：
    1. 管理训练过程中的各类检查点
    2. 保存和加载训练历史
    3. 支持参数演化分析
    4. 支持多种联邦学习算法
    """
    
    def __init__(
        self,
        save_dir: str,
        experiment_name: str,
        alg: str = 'fedavg',
        save_frequency: int = 5,
        save_client_weights: bool = True,
        save_global_model: bool = True,
        save_aggregation_info: bool = True,
        max_checkpoints: int = -1,
        device: str = 'cpu'
    ):
        """
        初始化检查点管理器
        
        Args:
            save_dir: 保存目录根路径
            experiment_name: 实验名称（用于创建子目录）
            alg: 联邦学习算法类型 ('fedavg', 'fedlora', 'fedsdg')
            save_frequency: 保存频率（每隔多少轮保存一次详细检查点）
            save_client_weights: 是否保存客户端本地权重
            save_global_model: 是否保存全局模型参数
            save_aggregation_info: 是否保存聚合信息
            max_checkpoints: 最大保存的检查点数量（-1 表示无限制）
            device: 保存时使用的设备（建议使用 'cpu' 以节省 GPU 内存）
        """
        self.save_dir = os.path.join(save_dir, experiment_name)
        self.experiment_name = experiment_name
        self.alg = alg
        self.save_frequency = save_frequency
        self.save_client_weights = save_client_weights
        self.save_global_model = save_global_model
        self.save_aggregation_info = save_aggregation_info
        self.max_checkpoints = max_checkpoints
        self.device = device
        
        # 创建保存目录
        self._create_directories()
        
        # 训练历史（内存中的轻量级记录）
        self.training_history = {
            'rounds': [],
            'train_loss': [],
            'train_acc': [],
            'test_acc': [],
            'test_loss': [],
            'local_test_acc': [],
            'local_test_loss': [],
            'comm_volume_mb': [],
            'timestamps': [],
            # LoRA 参数范数历史
            'lora_param_norms': [],
            # 聚合权重历史（仅 FedSDG alignment 模式）
            'aggregation_weights': [],
            'alignment_scores': [],
            # 门控参数历史（仅 FedSDG）
            'gate_values_mean': [],
            'gate_values_std': [],
            'gate_values_min': [],
            'gate_values_max': [],
        }
        
        # 参数演化追踪（用于分析参数更新幅度和方向）
        self.param_evolution = {
            'rounds': [],
            'param_deltas': [],  # 每轮参数更新量
            'update_magnitudes': [],  # 更新幅度
            'update_directions': [],  # 更新方向相似度
        }
        
        # 保存的检查点列表
        self.checkpoint_list = []
        
        print(f"\n{'='*70}")
        print("[CheckpointManager] 初始化完成")
        print(f"{'='*70}")
        print(f"  保存目录: {self.save_dir}")
        print(f"  算法类型: {self.alg}")
        print(f"  保存频率: 每 {self.save_frequency} 轮")
        print(f"  保存客户端权重: {self.save_client_weights}")
        print(f"  保存全局模型: {self.save_global_model}")
        print(f"  保存聚合信息: {self.save_aggregation_info}")
        print(f"{'='*70}\n")
    
    def _create_directories(self):
        """创建保存目录结构"""
        dirs = [
            self.save_dir,
            os.path.join(self.save_dir, 'checkpoints'),
            os.path.join(self.save_dir, 'global_models'),
            os.path.join(self.save_dir, 'client_weights'),
            os.path.join(self.save_dir, 'aggregation'),
            os.path.join(self.save_dir, 'private_states'),  # FedSDG 专用
            os.path.join(self.save_dir, 'param_evolution'),
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    
    def _to_cpu(self, state_dict: Dict) -> Dict:
        """将 state_dict 移动到 CPU"""
        return {k: v.cpu().clone() if isinstance(v, torch.Tensor) else v 
                for k, v in state_dict.items()}
    
    def _get_param_norm(self, state_dict: Dict, key_filter: str = None) -> float:
        """计算参数范数"""
        total_norm = 0.0
        for key, param in state_dict.items():
            if key_filter is None or key_filter in key:
                if isinstance(param, torch.Tensor):
                    total_norm += param.float().norm().item() ** 2
        return np.sqrt(total_norm)
    
    def _compute_param_delta(self, old_state: Dict, new_state: Dict, keys: List[str] = None) -> Dict[str, float]:
        """
        计算参数更新量
        
        Args:
            old_state: 旧的 state_dict
            new_state: 新的 state_dict
            keys: 要计算的参数键（None 表示所有）
            
        Returns:
            Dict[str, float]: {param_name: delta_norm}
        """
        deltas = {}
        keys_to_check = keys or new_state.keys()
        
        for key in keys_to_check:
            if key in old_state and key in new_state:
                old_param = old_state[key].float()
                new_param = new_state[key].float()
                delta = (new_param - old_param).norm().item()
                deltas[key] = delta
        
        return deltas
    
    def _compute_update_direction_similarity(
        self, 
        deltas1: Dict[str, torch.Tensor], 
        deltas2: Dict[str, torch.Tensor]
    ) -> float:
        """
        计算两个更新方向的余弦相似度
        
        用于分析：
        1. 不同轮次更新方向的一致性
        2. 不同客户端更新方向的相似度
        """
        common_keys = set(deltas1.keys()) & set(deltas2.keys())
        if not common_keys:
            return 0.0
        
        vec1 = torch.cat([deltas1[k].flatten().float() for k in common_keys])
        vec2 = torch.cat([deltas2[k].flatten().float() for k in common_keys])
        
        norm1 = vec1.norm()
        norm2 = vec2.norm()
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = (torch.dot(vec1, vec2) / (norm1 * norm2)).item()
        return similarity
    
    def save_round_checkpoint(
        self,
        round_idx: int,
        global_model: torch.nn.Module,
        local_weights: List[Dict] = None,
        local_losses: List[float] = None,
        selected_clients: List[int] = None,
        aggregation_info: Dict = None,
        local_private_states: Dict = None,
        train_loss: float = None,
        train_acc: float = None,
        test_acc: float = None,
        test_loss: float = None,
        local_test_acc: float = None,
        local_test_loss: float = None,
        comm_volume_mb: float = None,
        previous_global_state: Dict = None,
    ):
        """
        保存单轮训练检查点
        
        Args:
            round_idx: 当前轮次索引
            global_model: 全局模型
            local_weights: 客户端本地权重列表
            local_losses: 客户端本地损失列表
            selected_clients: 被选中的客户端 ID 列表
            aggregation_info: 聚合信息（权重、对齐度分数等）
            local_private_states: FedSDG 私有状态
            train_loss, train_acc, test_acc, test_loss: 性能指标
            local_test_acc, local_test_loss: 本地个性化性能
            comm_volume_mb: 累计通信量
            previous_global_state: 上一轮全局模型状态（用于计算更新量）
        """
        timestamp = time.time()
        is_detailed_checkpoint = (round_idx % self.save_frequency == 0) or (round_idx == 0)
        
        # 1. 更新轻量级训练历史（每轮都记录）
        self.training_history['rounds'].append(round_idx)
        self.training_history['train_loss'].append(train_loss)
        self.training_history['train_acc'].append(train_acc)
        self.training_history['test_acc'].append(test_acc)
        self.training_history['test_loss'].append(test_loss)
        self.training_history['local_test_acc'].append(local_test_acc)
        self.training_history['local_test_loss'].append(local_test_loss)
        self.training_history['comm_volume_mb'].append(comm_volume_mb)
        self.training_history['timestamps'].append(timestamp)
        
        # 2. 记录 LoRA 参数范数（FedLoRA/FedSDG）
        if self.alg in ('fedlora', 'fedsdg'):
            global_state = global_model.state_dict()
            lora_norm = self._get_param_norm(global_state, 'lora_')
            self.training_history['lora_param_norms'].append(lora_norm)
        
        # 3. 记录聚合信息
        if aggregation_info is not None:
            if 'weights' in aggregation_info:
                self.training_history['aggregation_weights'].append(aggregation_info['weights'])
            if 'alignment_scores' in aggregation_info:
                self.training_history['alignment_scores'].append(aggregation_info['alignment_scores'])
        
        # 4. 记录门控参数（FedSDG）
        if self.alg == 'fedsdg':
            gate_values = self._extract_gate_values(global_model, local_private_states)
            if gate_values:
                values = list(gate_values.values())
                self.training_history['gate_values_mean'].append(np.mean(values))
                self.training_history['gate_values_std'].append(np.std(values))
                self.training_history['gate_values_min'].append(np.min(values))
                self.training_history['gate_values_max'].append(np.max(values))
        
        # 5. 计算参数演化（如果提供了上一轮状态）
        if previous_global_state is not None:
            current_state = self._to_cpu(global_model.state_dict())
            param_deltas = self._compute_param_delta(previous_global_state, current_state)
            
            # 计算总更新幅度
            total_magnitude = np.sqrt(sum(d**2 for d in param_deltas.values()))
            
            self.param_evolution['rounds'].append(round_idx)
            self.param_evolution['update_magnitudes'].append(total_magnitude)
            
            # 计算与上一轮更新方向的相似度
            if len(self.param_evolution['param_deltas']) > 0:
                prev_deltas = self.param_evolution['param_deltas'][-1]
                current_deltas_tensor = {k: torch.tensor(v) for k, v in param_deltas.items()}
                prev_deltas_tensor = {k: torch.tensor(v) for k, v in prev_deltas.items()}
                direction_sim = self._compute_update_direction_similarity(
                    current_deltas_tensor, prev_deltas_tensor
                )
                self.param_evolution['update_directions'].append(direction_sim)
            
            self.param_evolution['param_deltas'].append(param_deltas)
        
        # 6. 保存详细检查点（按频率）
        if is_detailed_checkpoint:
            checkpoint_path = os.path.join(
                self.save_dir, 'checkpoints', f'round_{round_idx:04d}.pkl'
            )
            
            checkpoint_data = {
                'round_idx': round_idx,
                'timestamp': timestamp,
                'alg': self.alg,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'test_loss': test_loss,
                'local_test_acc': local_test_acc,
                'local_test_loss': local_test_loss,
                'comm_volume_mb': comm_volume_mb,
            }
            
            # 保存全局模型状态
            if self.save_global_model:
                checkpoint_data['global_state'] = self._to_cpu(global_model.state_dict())
            
            # 保存客户端本地权重
            if self.save_client_weights and local_weights is not None:
                checkpoint_data['local_weights'] = [self._to_cpu(w) for w in local_weights]
                checkpoint_data['local_losses'] = local_losses
                checkpoint_data['selected_clients'] = selected_clients
            
            # 保存聚合信息
            if self.save_aggregation_info and aggregation_info is not None:
                checkpoint_data['aggregation_info'] = aggregation_info
            
            # 保存 FedSDG 私有状态
            if self.alg == 'fedsdg' and local_private_states is not None:
                # 将私有状态移动到 CPU
                private_states_cpu = {}
                for client_id, state in local_private_states.items():
                    private_states_cpu[client_id] = self._to_cpu(state)
                checkpoint_data['private_states'] = private_states_cpu
            
            # 保存到文件
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            self.checkpoint_list.append(checkpoint_path)
            
            # 清理旧检查点（如果设置了最大数量）
            if self.max_checkpoints > 0 and len(self.checkpoint_list) > self.max_checkpoints:
                old_checkpoint = self.checkpoint_list.pop(0)
                if os.path.exists(old_checkpoint):
                    os.remove(old_checkpoint)
            
            print(f"  [Checkpoint] Round {round_idx} 检查点已保存: {checkpoint_path}")
    
    def _extract_gate_values(
        self, 
        model: torch.nn.Module, 
        local_private_states: Dict = None
    ) -> Dict[str, float]:
        """
        提取门控参数值
        
        优先使用客户端私有状态中的门控值（聚合平均）
        """
        gate_values = {}
        
        # 优先从私有状态提取
        if local_private_states:
            all_gates = {}
            for client_id, state in local_private_states.items():
                for name, param in state.items():
                    if 'lambda_k_logit' in name:
                        m_k = torch.sigmoid(param).item()
                        layer_name = name.replace('.lambda_k_logit', '')
                        if layer_name not in all_gates:
                            all_gates[layer_name] = []
                        all_gates[layer_name].append(m_k)
            
            # 计算聚合平均
            for layer_name, values in all_gates.items():
                gate_values[layer_name] = np.mean(values)
        else:
            # 从模型中提取
            for name, param in model.named_parameters():
                if 'lambda_k_logit' in name:
                    m_k = torch.sigmoid(param).item()
                    layer_name = name.replace('.lambda_k_logit', '')
                    gate_values[layer_name] = m_k
        
        return gate_values
    
    def save_final(
        self,
        global_model: torch.nn.Module,
        local_private_states: Dict = None,
        final_results: Dict = None,
        args: Any = None,
    ):
        """
        保存最终训练结果
        
        Args:
            global_model: 最终全局模型
            local_private_states: FedSDG 最终私有状态
            final_results: 最终结果字典
            args: 实验配置参数
        """
        print(f"\n{'='*70}")
        print("[CheckpointManager] 保存最终训练结果")
        print(f"{'='*70}")
        
        # 1. 保存最终模型
        final_model_path = os.path.join(self.save_dir, 'final_model.pth')
        torch.save(global_model.state_dict(), final_model_path)
        print(f"  ✓ 最终模型已保存: {final_model_path}")
        
        # 2. 保存训练历史
        history_path = os.path.join(self.save_dir, 'training_history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(self.training_history, f)
        print(f"  ✓ 训练历史已保存: {history_path}")
        
        # 3. 保存参数演化
        evolution_path = os.path.join(self.save_dir, 'param_evolution.pkl')
        with open(evolution_path, 'wb') as f:
            pickle.dump(self.param_evolution, f)
        print(f"  ✓ 参数演化已保存: {evolution_path}")
        
        # 4. 保存 FedSDG 私有状态
        if self.alg == 'fedsdg' and local_private_states is not None:
            private_states_cpu = {}
            for client_id, state in local_private_states.items():
                private_states_cpu[client_id] = self._to_cpu(state)
            
            private_path = os.path.join(self.save_dir, 'final_private_states.pkl')
            with open(private_path, 'wb') as f:
                pickle.dump(private_states_cpu, f)
            print(f"  ✓ FedSDG 私有状态已保存: {private_path}")
        
        # 5. 保存最终结果和配置
        summary = {
            'experiment_name': self.experiment_name,
            'alg': self.alg,
            'training_history': self.training_history,
            'final_results': final_results,
            'args': vars(args) if args else None,
            'checkpoint_list': self.checkpoint_list,
        }
        
        summary_path = os.path.join(self.save_dir, 'experiment_summary.pkl')
        with open(summary_path, 'wb') as f:
            pickle.dump(summary, f)
        print(f"  ✓ 实验摘要已保存: {summary_path}")
        
        # 6. 保存 JSON 格式的轻量级摘要（便于快速查看）
        json_summary = {
            'experiment_name': self.experiment_name,
            'alg': self.alg,
            'total_rounds': len(self.training_history['rounds']),
            'final_train_acc': self.training_history['train_acc'][-1] if self.training_history['train_acc'] else None,
            'final_test_acc': self.training_history['test_acc'][-1] if self.training_history['test_acc'] else None,
            'final_local_acc': self.training_history['local_test_acc'][-1] if self.training_history['local_test_acc'] else None,
            'checkpoint_count': len(self.checkpoint_list),
        }
        
        if final_results:
            json_summary.update({
                'global_test_acc': final_results.get('global_test_acc'),
                'local_avg_acc': final_results.get('local_avg_acc'),
            })
        
        json_path = os.path.join(self.save_dir, 'summary.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_summary, f, indent=2, ensure_ascii=False)
        print(f"  ✓ JSON 摘要已保存: {json_path}")
        
        print(f"\n  所有文件已保存到: {self.save_dir}")
        print(f"{'='*70}\n")
    
    @staticmethod
    def load_checkpoint(checkpoint_path: str) -> Dict:
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点文件路径
            
        Returns:
            检查点数据字典
        """
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def load_training_history(save_dir: str) -> Dict:
        """
        加载训练历史
        
        Args:
            save_dir: 实验保存目录
            
        Returns:
            训练历史字典
        """
        history_path = os.path.join(save_dir, 'training_history.pkl')
        with open(history_path, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def load_param_evolution(save_dir: str) -> Dict:
        """
        加载参数演化数据
        
        Args:
            save_dir: 实验保存目录
            
        Returns:
            参数演化字典
        """
        evolution_path = os.path.join(save_dir, 'param_evolution.pkl')
        with open(evolution_path, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def load_private_states(save_dir: str) -> Dict:
        """
        加载 FedSDG 私有状态
        
        Args:
            save_dir: 实验保存目录
            
        Returns:
            私有状态字典 {client_id: state_dict}
        """
        private_path = os.path.join(save_dir, 'final_private_states.pkl')
        if os.path.exists(private_path):
            with open(private_path, 'rb') as f:
                return pickle.load(f)
        return {}


class TrainingAnalyzer:
    """
    训练历史分析器
    
    提供各种分析功能：
    1. 参数更新分析
    2. 聚合权重分析
    3. 门控参数分析
    4. 客户端差异分析
    """
    
    def __init__(self, save_dir: str):
        """
        初始化分析器
        
        Args:
            save_dir: 实验保存目录
        """
        self.save_dir = save_dir
        self.training_history = CheckpointManager.load_training_history(save_dir)
        self.param_evolution = CheckpointManager.load_param_evolution(save_dir)
        
        # 尝试加载私有状态
        try:
            self.private_states = CheckpointManager.load_private_states(save_dir)
        except FileNotFoundError:
            self.private_states = {}
    
    def get_lora_param_evolution(self) -> Dict[str, List[float]]:
        """
        获取 LoRA 参数范数演化
        
        Returns:
            Dict: {'rounds': [...], 'norms': [...]}
        """
        return {
            'rounds': self.training_history['rounds'],
            'norms': self.training_history.get('lora_param_norms', [])
        }
    
    def get_update_magnitude_evolution(self) -> Dict[str, List[float]]:
        """
        获取参数更新幅度演化
        
        Returns:
            Dict: {'rounds': [...], 'magnitudes': [...]}
        """
        return {
            'rounds': self.param_evolution['rounds'],
            'magnitudes': self.param_evolution['update_magnitudes']
        }
    
    def get_update_direction_consistency(self) -> Dict[str, List[float]]:
        """
        获取更新方向一致性演化
        
        Returns:
            Dict: {'rounds': [...], 'similarities': [...]}
        """
        return {
            'rounds': self.param_evolution['rounds'][1:],  # 从第二轮开始
            'similarities': self.param_evolution.get('update_directions', [])
        }
    
    def get_aggregation_weights_evolution(self) -> Dict[str, List]:
        """
        获取聚合权重演化
        
        Returns:
            Dict: {'rounds': [...], 'weights': [[...], ...]}
        """
        weights = self.training_history.get('aggregation_weights', [])
        rounds_with_weights = [r for i, r in enumerate(self.training_history['rounds']) 
                              if i < len(weights)]
        return {
            'rounds': rounds_with_weights,
            'weights': weights
        }
    
    def get_alignment_scores_evolution(self) -> Dict[str, List]:
        """
        获取对齐度分数演化
        
        Returns:
            Dict: {'rounds': [...], 'scores': [[...], ...]}
        """
        scores = self.training_history.get('alignment_scores', [])
        rounds_with_scores = [r for i, r in enumerate(self.training_history['rounds']) 
                             if i < len(scores)]
        return {
            'rounds': rounds_with_scores,
            'scores': scores
        }
    
    def get_gate_values_evolution(self) -> Dict[str, List[float]]:
        """
        获取门控参数演化
        
        Returns:
            Dict: {
                'rounds': [...],
                'mean': [...],
                'std': [...],
                'min': [...],
                'max': [...]
            }
        """
        return {
            'rounds': self.training_history['rounds'],
            'mean': self.training_history.get('gate_values_mean', []),
            'std': self.training_history.get('gate_values_std', []),
            'min': self.training_history.get('gate_values_min', []),
            'max': self.training_history.get('gate_values_max', [])
        }
    
    def analyze_client_diversity(self, checkpoint_path: str) -> Dict[str, float]:
        """
        分析客户端参数多样性
        
        Args:
            checkpoint_path: 检查点文件路径
            
        Returns:
            Dict: {
                'mean_distance': float,  # 客户端间平均距离
                'max_distance': float,   # 最大距离
                'min_distance': float,   # 最小距离
            }
        """
        checkpoint = CheckpointManager.load_checkpoint(checkpoint_path)
        local_weights = checkpoint.get('local_weights', [])
        
        if len(local_weights) < 2:
            return {'mean_distance': 0, 'max_distance': 0, 'min_distance': 0}
        
        # 计算两两客户端之间的参数距离
        distances = []
        for i in range(len(local_weights)):
            for j in range(i + 1, len(local_weights)):
                dist = self._compute_weight_distance(local_weights[i], local_weights[j])
                distances.append(dist)
        
        return {
            'mean_distance': np.mean(distances),
            'max_distance': np.max(distances),
            'min_distance': np.min(distances)
        }
    
    def _compute_weight_distance(self, w1: Dict, w2: Dict) -> float:
        """计算两个 state_dict 之间的欧氏距离"""
        total_dist = 0.0
        for key in w1.keys():
            if key in w2:
                diff = w1[key].float() - w2[key].float()
                total_dist += diff.norm().item() ** 2
        return np.sqrt(total_dist)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        获取性能摘要
        
        Returns:
            Dict: 性能统计摘要
        """
        history = self.training_history
        
        summary = {
            'total_rounds': len(history['rounds']),
            'final_train_acc': history['train_acc'][-1] if history['train_acc'] else None,
            'final_test_acc': history['test_acc'][-1] if history['test_acc'] else None,
            'final_local_acc': history['local_test_acc'][-1] if history['local_test_acc'] else None,
            'best_test_acc': max(filter(None, history['test_acc'])) if any(history['test_acc']) else None,
            'best_test_round': history['rounds'][np.argmax([x or 0 for x in history['test_acc']])] if any(history['test_acc']) else None,
            'total_comm_mb': history['comm_volume_mb'][-1] if history['comm_volume_mb'] else None,
        }
        
        return summary
    
    def export_for_visualization(self, output_path: str = None) -> Dict:
        """
        导出数据用于可视化
        
        Args:
            output_path: 可选的输出路径（JSON 格式）
            
        Returns:
            Dict: 可视化数据
        """
        viz_data = {
            'training_curves': {
                'rounds': self.training_history['rounds'],
                'train_loss': self.training_history['train_loss'],
                'train_acc': self.training_history['train_acc'],
                'test_acc': self.training_history['test_acc'],
                'local_test_acc': self.training_history['local_test_acc'],
            },
            'lora_evolution': self.get_lora_param_evolution(),
            'update_evolution': self.get_update_magnitude_evolution(),
            'gate_evolution': self.get_gate_values_evolution(),
            'aggregation_weights': self.get_aggregation_weights_evolution(),
            'alignment_scores': self.get_alignment_scores_evolution(),
            'performance_summary': self.get_performance_summary(),
        }
        
        if output_path:
            # 转换为 JSON 可序列化格式
            def convert_to_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(v) for v in obj]
                return obj
            
            serializable_data = convert_to_serializable(viz_data)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)
            print(f"可视化数据已导出: {output_path}")
        
        return viz_data


# ==================== 便捷函数 ====================

def create_checkpoint_manager(args, path_project: str) -> CheckpointManager:
    """
    根据参数创建 CheckpointManager 实例
    
    Args:
        args: 命令行参数
        path_project: 项目根目录
        
    Returns:
        CheckpointManager 实例
    """
    # 生成实验名称
    if args.alg == 'fedlora':
        experiment_name = f"{args.dataset}_{args.model}_{args.alg}_r{args.lora_r}_E{args.epochs}"
    elif args.alg == 'fedsdg':
        experiment_name = f"{args.dataset}_{args.model}_{args.alg}_r{args.lora_r}_E{args.epochs}_{args.server_agg_method}"
    else:
        experiment_name = f"{args.dataset}_{args.model}_{args.alg}_E{args.epochs}"
    
    # 添加时间戳避免覆盖
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    experiment_name = f"{experiment_name}_{timestamp}"
    
    save_dir = os.path.join(path_project, 'save', 'checkpoints')
    
    # 获取保存频率（可以通过参数配置，默认为 5）
    save_frequency = getattr(args, 'save_frequency', 5)
    save_client_weights = getattr(args, 'save_client_weights', True)
    
    return CheckpointManager(
        save_dir=save_dir,
        experiment_name=experiment_name,
        alg=args.alg,
        save_frequency=save_frequency,
        save_client_weights=save_client_weights,
        save_global_model=True,
        save_aggregation_info=True,
    )


if __name__ == '__main__':
    # 测试代码
    print("CheckpointManager 模块测试")
    
    # 创建测试目录
    test_dir = '/tmp/test_checkpoint_manager'
    os.makedirs(test_dir, exist_ok=True)
    
    # 创建管理器
    manager = CheckpointManager(
        save_dir=test_dir,
        experiment_name='test_experiment',
        alg='fedsdg',
        save_frequency=2
    )
    
    # 模拟训练历史
    for i in range(10):
        manager.training_history['rounds'].append(i)
        manager.training_history['train_loss'].append(1.0 - i * 0.05)
        manager.training_history['train_acc'].append(0.5 + i * 0.04)
        manager.training_history['test_acc'].append(0.45 + i * 0.04)
        manager.training_history['test_loss'].append(1.1 - i * 0.05)
        manager.training_history['gate_values_mean'].append(0.5 + (i - 5) * 0.02)
    
    # 保存最终结果
    manager.save_final(
        global_model=None,  # 实际使用时传入模型
        final_results={'global_test_acc': 0.85, 'local_avg_acc': 0.87},
        args=None
    )
    
    print("测试完成！")

