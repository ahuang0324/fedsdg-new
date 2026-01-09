#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FedAvg: Federated Averaging Algorithm

Standard federated learning aggregation that averages all model parameters.
"""

import copy
import torch


def average_weights(w):
    """
    Compute the average of model weights from multiple clients.
    
    Args:
        w: List of state_dicts from clients
    
    Returns:
        Averaged state_dict
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

