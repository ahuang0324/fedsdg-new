#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FedSDG: Federated Structure-Decoupled Gating Algorithm

Core innovation:
- Dual-path LoRA architecture: Global branch (aggregated) + Private branch (local)
- Learnable gating mechanism (λ_k) to balance global and private contributions
- Alignment-based server aggregation for improved convergence

Components:
1. Server-side: Alignment-based weighted aggregation
2. Client-side: Three-term loss function (task + gate sparsity + private regularization)
3. Model-side: Dual-path LoRA with gating (see fl/models/lora.py)

Reference:
- FedSDG_Design.md for algorithm details
- FedSDG实施技术报告.md for implementation notes
"""

import copy
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Any


def aggregate_fedsdg(
    client_weights: List[Dict[str, torch.Tensor]],
    global_state_dict: Dict[str, torch.Tensor],
    agg_method: str = 'alignment',
    epsilon: float = 1e-8,
    return_info: bool = False
):
    """
    FedSDG server-side aggregation.
    
    Only aggregates global LoRA parameters (lora_A, lora_B) and classification head.
    Private parameters (lora_A_private, lora_B_private) and gate parameters (lambda_k)
    remain on each client and are not aggregated.
    
    Supports two aggregation methods:
    1. 'alignment': Alignment-based weighted aggregation (recommended)
    2. 'fedavg': Uniform-weighted FedAvg aggregation
    
    Args:
        client_weights: List of client state_dicts
        global_state_dict: Full global model state_dict (preserves frozen weights)
        agg_method: Aggregation method ('alignment' or 'fedavg')
        epsilon: Numerical stability parameter
        return_info: Whether to return aggregation info
    
    Returns:
        Updated global state_dict
        If return_info=True: Also returns aggregation_info dict
    """
    # Deep copy global state_dict as base
    w_avg = copy.deepcopy(global_state_dict)
    
    # Aggregation info
    aggregation_info = {
        'algorithm': 'fedsdg',
        'agg_method': agg_method,
        'num_clients': len(client_weights),
        'weights': None,
        'alignment_scores': None,
        'aggregated_keys': [],
        'excluded_keys': [],
    }
    
    # Identify parameter keys to aggregate
    # Include: lora_A, lora_B (global), mlp_head, head
    # Exclude: _private (private branch), lambda_k (gate parameters)
    aggregated_keys = []
    excluded_keys = []
    
    for key in client_weights[0].keys():
        if '_private' in key or 'lambda_k' in key:
            excluded_keys.append(key)
        elif 'lora_' in key or 'mlp_head' in key or 'head' in key:
            aggregated_keys.append(key)
    
    aggregation_info['aggregated_keys'] = aggregated_keys
    aggregation_info['excluded_keys'] = excluded_keys
    
    if len(aggregated_keys) == 0:
        print("  [FedSDG WARNING] No parameters to aggregate!")
        if return_info:
            return w_avg, aggregation_info
        return w_avg
    
    # Compute aggregation weights
    if agg_method == 'alignment':
        weights, weight_stats = _compute_alignment_weights(
            client_weights, global_state_dict, aggregated_keys, epsilon
        )
        aggregation_info['weights'] = weights
        aggregation_info['alignment_scores'] = weight_stats.get('alphas', [])
        aggregation_info['weight_stats'] = weight_stats
        
        # Weighted aggregation
        for key in aggregated_keys:
            weighted_sum = torch.zeros_like(client_weights[0][key], dtype=torch.float32)
            for i, client_w in enumerate(client_weights):
                weighted_sum += weights[i] * client_w[key].float()
            w_avg[key] = weighted_sum.to(client_weights[0][key].dtype)
        
        print(f"  [FedSDG-Alignment] Aggregated {len(aggregated_keys)} parameters, "
              f"excluded {len(excluded_keys)} private/gate parameters")
        print(f"  [FedSDG-Alignment] Weight stats: mean={weight_stats['mean']:.4f}, "
              f"std={weight_stats['std']:.4f}, range=[{weight_stats['min']:.4f}, {weight_stats['max']:.4f}]")
    else:
        # Uniform FedAvg aggregation
        uniform_weight = 1.0 / len(client_weights)
        aggregation_info['weights'] = [uniform_weight] * len(client_weights)
        
        for key in aggregated_keys:
            w_avg[key] = copy.deepcopy(client_weights[0][key])
            for i in range(1, len(client_weights)):
                w_avg[key] += client_weights[i][key]
            w_avg[key] = torch.div(w_avg[key], len(client_weights))
        
        print(f"  [FedSDG-FedAvg] Aggregated {len(aggregated_keys)} parameters, "
              f"excluded {len(excluded_keys)} private/gate parameters")
    
    if return_info:
        return w_avg, aggregation_info
    return w_avg


def _compute_alignment_weights(
    client_weights: List[Dict[str, torch.Tensor]],
    global_state_dict: Dict[str, torch.Tensor],
    lora_keys: List[str],
    epsilon: float = 1e-8
) -> Tuple[List[float], Dict[str, Any]]:
    """
    Compute FedSDG alignment-based aggregation weights.
    
    Algorithm (from FedSDG_Design.md):
    1. Compute update vectors: Δθ_g^(k) = θ_g^(k) - θ_g^(t)
    2. Compute mean update direction: Δ̄ = (1/M) Σ Δθ_g^(k)
    3. Compute alignment scores: α_k = max(0, cos(Δθ_g^(k), Δ̄))
    4. Normalize weights: w_k = α_k / (Σ α_j + ε)
    
    The alignment score measures how well each client's update aligns with
    the average update direction. Clients with updates that deviate significantly
    (possibly due to noisy or adversarial data) receive lower weights.
    
    Args:
        client_weights: List of client state_dicts
        global_state_dict: Current global model state_dict
        lora_keys: List of parameter keys to aggregate
        epsilon: Numerical stability parameter
    
    Returns:
        weights: List of normalized client weights
        weight_stats: Weight statistics dict
    """
    M = len(client_weights)
    
    # Edge cases
    if M == 0:
        return [], {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'num_zero': 0, 'alphas': []}
    
    if M == 1:
        return [1.0], {'mean': 1.0, 'std': 0, 'min': 1.0, 'max': 1.0, 'num_zero': 0, 'alphas': [1.0]}
    
    # Flatten parameters to 1D vector
    def flatten_params(state_dict: Dict, keys: List[str]) -> torch.Tensor:
        tensors = []
        for key in keys:
            if key in state_dict:
                tensors.append(state_dict[key].flatten().float())
        return torch.cat(tensors) if tensors else torch.tensor([])
    
    # Get global parameter vector
    global_flat = flatten_params(global_state_dict, lora_keys)
    
    # Compute update vectors (Δθ_g^(k) = θ_g^(k) - θ_g^(t))
    deltas = []
    for client_w in client_weights:
        client_flat = flatten_params(client_w, lora_keys)
        delta = client_flat - global_flat
        deltas.append(delta)
    
    # Compute mean update direction (Δ̄)
    delta_stack = torch.stack(deltas)
    delta_mean = delta_stack.mean(dim=0)
    
    # Compute alignment scores (α_k = max(0, cos(Δθ_g^(k), Δ̄)))
    alphas = []
    norm_mean = torch.linalg.norm(delta_mean, ord=2)
    
    for delta in deltas:
        numerator = torch.dot(delta, delta_mean)
        norm_delta = torch.linalg.norm(delta, ord=2)
        denominator = norm_delta * norm_mean + epsilon
        
        # Clamp to [0, 1] to handle numerical issues
        alpha = max(0.0, (numerator / denominator).item())
        alphas.append(alpha)
    
    # Normalize weights (w_k = α_k / Σα_j)
    sum_alpha = sum(alphas) + epsilon
    
    if sum_alpha < 1e-6:
        print("  [FedSDG-Alignment] Warning: All alignment scores near 0, falling back to uniform weights")
        weights = [1.0 / M] * M
    else:
        weights = [alpha / sum_alpha for alpha in alphas]
    
    # Statistics
    weight_stats = {
        'mean': float(np.mean(weights)),
        'std': float(np.std(weights)),
        'min': float(min(weights)),
        'max': float(max(weights)),
        'num_zero': sum(1 for w in weights if w < 1e-6),
        'alphas': alphas,
    }
    
    return weights, weight_stats


class FedSDGClientState:
    """
    Manager for FedSDG client private states.
    
    Each client maintains:
    - lora_A_private, lora_B_private: Private LoRA parameters
    - lambda_k_logit: Gate parameter (controls m_k = sigmoid(lambda_k_logit))
    
    These parameters are NOT aggregated and persist across rounds.
    """
    
    def __init__(self):
        self.states: Dict[int, Dict[str, torch.Tensor]] = {}
    
    def save_client_state(self, client_id: int, model: torch.nn.Module) -> None:
        """Save client's private parameters after local training."""
        private_state = {}
        for name, param in model.named_parameters():
            if '_private' in name or 'lambda_k' in name:
                private_state[name] = param.data.clone().cpu()
        self.states[client_id] = private_state
    
    def load_client_state(self, client_id: int, model: torch.nn.Module) -> None:
        """Load client's private parameters before local training."""
        if client_id not in self.states:
            return  # First round, no state to load
        
        current_state = model.state_dict()
        for param_name, param_value in self.states[client_id].items():
            if param_name in current_state:
                current_state[param_name] = param_value.clone().to(
                    current_state[param_name].device
                )
        model.load_state_dict(current_state)
    
    def get_all_states(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """Get all client private states."""
        return self.states
    
    def get_gate_statistics(self) -> Dict[str, float]:
        """Get statistics of gate values (m_k = sigmoid(lambda_k_logit))."""
        all_gates = []
        for client_state in self.states.values():
            for name, param in client_state.items():
                if 'lambda_k_logit' in name:
                    m_k = torch.sigmoid(param).item()
                    all_gates.append(m_k)
        
        if not all_gates:
            return {'mean': 0.5, 'std': 0, 'min': 0.5, 'max': 0.5}
        
        return {
            'mean': float(np.mean(all_gates)),
            'std': float(np.std(all_gates)),
            'min': float(min(all_gates)),
            'max': float(max(all_gates)),
        }


# Backward compatibility alias
def average_weights_fedsdg(w, global_state_dict, agg_method='alignment', 
                           epsilon=1e-8, return_aggregation_info=False):
    """Alias for aggregate_fedsdg (backward compatibility)."""
    result = aggregate_fedsdg(w, global_state_dict, agg_method, epsilon, return_info=True)
    if return_aggregation_info:
        return result
    return result[0]

