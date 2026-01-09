#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Federated learning aggregation algorithms.

Available algorithms:
- FedAvg: Standard Federated Averaging (all parameters)
- FedLoRA: LoRA parameter aggregation (communication-efficient)
- FedSDG: Structure-Decoupled Gating with alignment-based aggregation
"""

from .fedavg import average_weights
from .fedlora import average_weights_lora, aggregate_fedlora
from .fedsdg import aggregate_fedsdg, FedSDGClientState

__all__ = [
    # FedAvg
    'average_weights',
    # FedLoRA
    'average_weights_lora',
    'aggregate_fedlora',
    # FedSDG
    'aggregate_fedsdg',
    'FedSDGClientState',
]
