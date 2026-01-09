#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions for federated learning."""

from .paths import PROJECT_ROOT, DATA_DIR, LOGS_DIR, OUTPUTS_DIR
from .checkpoint import CheckpointManager, create_checkpoint_manager
from .communication import get_communication_stats, print_communication_profile
from .evaluation import test_inference, local_test_inference, evaluate_local_personalization
from .logger import exp_details

__all__ = [
    'PROJECT_ROOT', 'DATA_DIR', 'LOGS_DIR', 'OUTPUTS_DIR',
    'CheckpointManager', 'create_checkpoint_manager',
    'get_communication_stats', 'print_communication_profile',
    'test_inference', 'local_test_inference', 'evaluate_local_personalization',
    'exp_details',
]

