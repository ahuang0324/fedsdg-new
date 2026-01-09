#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FedLoRA: Federated Low-Rank Adaptation Algorithm

Communication-efficient federated learning using LoRA (Low-Rank Adaptation).
Only LoRA parameters (lora_A, lora_B matrices) and the classification head
are communicated and aggregated, while the pre-trained backbone remains frozen.

Key features:
- Significant communication savings (typically >99% reduction)
- Preserves pre-trained knowledge
- Compatible with any pre-trained ViT model

For FedSDG (dual-path LoRA with gating), see fedsdg.py
"""

import copy
import torch
from typing import List, Dict, Tuple, Optional, Any


def aggregate_fedlora(
    client_weights: List[Dict[str, torch.Tensor]],
    global_state_dict: Dict[str, torch.Tensor],
    return_info: bool = False
) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, Any]]]:
    """
    FedLoRA aggregation: Uniform-weighted average of LoRA parameters only.
    
    Aggregates only parameters containing:
    - 'lora_' (LoRA matrices A and B)
    - 'mlp_head' or 'head' (classification head)
    
    All other parameters (backbone) remain unchanged.
    
    Args:
        client_weights: List of client state_dicts
        global_state_dict: Full global model state_dict
        return_info: Whether to return aggregation info
    
    Returns:
        Updated global state_dict
        If return_info=True: Also returns aggregation_info dict
    """
    w_avg = copy.deepcopy(global_state_dict)
    
    # Aggregation info
    aggregation_info = {
        'algorithm': 'fedlora',
        'agg_method': 'fedavg',
        'num_clients': len(client_weights),
        'weights': [1.0 / len(client_weights)] * len(client_weights),
        'aggregated_keys': [],
    }
    
    # Identify LoRA parameter keys to aggregate
    lora_keys = [
        key for key in client_weights[0].keys()
        if 'lora_' in key or 'mlp_head' in key or 'head' in key
    ]
    
    aggregation_info['aggregated_keys'] = lora_keys
    
    if len(lora_keys) == 0:
        print("  [FedLoRA WARNING] No LoRA parameters found!")
        if return_info:
            return w_avg, aggregation_info
        return w_avg
    
    # Uniform-weighted aggregation
    for key in lora_keys:
        w_avg[key] = copy.deepcopy(client_weights[0][key])
        for i in range(1, len(client_weights)):
            w_avg[key] += client_weights[i][key]
        w_avg[key] = torch.div(w_avg[key], len(client_weights))
    
    print(f"  [FedLoRA] Aggregated {len(lora_keys)} LoRA parameter keys")
    
    if return_info:
        return w_avg, aggregation_info
    return w_avg


def average_weights_lora(
    w: List[Dict[str, torch.Tensor]],
    global_state_dict: Dict[str, torch.Tensor],
    agg_method: str = 'fedavg',
    epsilon: float = 1e-8,
    return_aggregation_info: bool = False
) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, Any]]]:
    """
    Unified LoRA aggregation function (supports both FedLoRA and FedSDG).
    
    This is the main entry point for LoRA-based aggregation in main.py.
    Routes to appropriate implementation based on agg_method:
    - 'fedavg': Standard FedLoRA (uniform weights)
    - 'alignment': FedSDG alignment-based aggregation
    
    Args:
        w: List of client state_dicts
        global_state_dict: Full global model state_dict
        agg_method: 'fedavg' for FedLoRA, 'alignment' for FedSDG
        epsilon: Numerical stability parameter (for alignment)
        return_aggregation_info: Whether to return aggregation info
    
    Returns:
        Updated global state_dict
        If return_aggregation_info=True: Also returns aggregation_info dict
    """
    if agg_method == 'alignment':
        # Use FedSDG alignment-based aggregation
        from .fedsdg import aggregate_fedsdg
        return aggregate_fedsdg(
            w, global_state_dict, 
            agg_method='alignment', 
            epsilon=epsilon,
            return_info=return_aggregation_info
        )
    else:
        # Standard FedLoRA aggregation
        # Handle FedSDG models (exclude private and gate params)
        has_private = any('_private' in key for key in w[0].keys())
        
        if has_private:
            # FedSDG model with FedAvg aggregation method
            from .fedsdg import aggregate_fedsdg
            return aggregate_fedsdg(
                w, global_state_dict,
                agg_method='fedavg',
                epsilon=epsilon,
                return_info=return_aggregation_info
            )
        else:
            # Pure FedLoRA model
            return aggregate_fedlora(
                w, global_state_dict,
                return_info=return_aggregation_info
            )
