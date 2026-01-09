#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Communication statistics utilities for federated learning.

Provides functions to compute and display communication overhead.
"""


def get_communication_stats(model, alg):
    """
    Calculate communication statistics for federated learning.
    
    Args:
        model: PyTorch model
        alg: Algorithm type ('fedavg', 'fedlora', or 'fedsdg')
    
    Returns:
        dict: Communication statistics
            - total_params: Total model parameters
            - trainable_params: Trainable parameters
            - comm_params: Parameters communicated per round
            - total_size_mb: Total model size (MB)
            - trainable_size_mb: Trainable parameters size (MB)
            - comm_size_mb: Communication size per round (MB)
            - compression_ratio: Compression ratio (%)
    """
    # Total parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Communication parameters
    if alg == 'fedavg':
        # FedAvg: Communicate all parameters
        comm_params = total_params
    elif alg in ('fedlora', 'fedsdg'):
        # FedLoRA/FedSDG: Only communicate global LoRA parameters
        # Exclude private parameters (_private) and gate parameters (lambda_k)
        comm_params = 0
        for name, p in model.named_parameters():
            if p.requires_grad:
                if '_private' not in name and 'lambda_k' not in name:
                    comm_params += p.numel()
    else:
        comm_params = total_params
    
    # Convert to MB (assuming float32, 4 bytes per parameter)
    bytes_per_param = 4
    total_size_mb = (total_params * bytes_per_param) / (1024 ** 2)
    trainable_size_mb = (trainable_params * bytes_per_param) / (1024 ** 2)
    comm_size_mb = (comm_params * bytes_per_param) / (1024 ** 2)
    
    # Compression ratio
    compression_ratio = (comm_params / total_params) * 100 if total_params > 0 else 100
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'comm_params': comm_params,
        'total_size_mb': total_size_mb,
        'trainable_size_mb': trainable_size_mb,
        'comm_size_mb': comm_size_mb,
        'compression_ratio': compression_ratio
    }


def print_communication_profile(comm_stats, args):
    """
    Print formatted communication statistics.
    
    Args:
        comm_stats: Statistics from get_communication_stats()
        args: Command line arguments
    """
    print("\n" + "="*70)
    print("COMMUNICATION PROFILE".center(70))
    print("="*70)
    
    print(f"\n{'Metric':<35} {'Value':>20} {'Unit':>10}")
    print("-"*70)
    
    # Parameter statistics
    print(f"{'Total Parameters':<35} {comm_stats['total_params']:>20,} {'params':>10}")
    print(f"{'Trainable Parameters':<35} {comm_stats['trainable_params']:>20,} {'params':>10}")
    print(f"{'Communication Parameters':<35} {comm_stats['comm_params']:>20,} {'params':>10}")
    
    print("-"*70)
    
    # Size statistics
    print(f"{'Total Model Size':<35} {comm_stats['total_size_mb']:>20.2f} {'MB':>10}")
    print(f"{'Trainable Size':<35} {comm_stats['trainable_size_mb']:>20.2f} {'MB':>10}")
    print(f"{'Communication per Round (1-way)':<35} {comm_stats['comm_size_mb']:>20.2f} {'MB':>10}")
    
    # Two-way communication
    comm_per_round_2way = comm_stats['comm_size_mb'] * 2
    print(f"{'Communication per Round (2-way)':<35} {comm_per_round_2way:>20.2f} {'MB':>10}")
    
    print("-"*70)
    
    # Total communication estimate
    total_rounds = args.epochs
    estimated_total_volume = comm_per_round_2way * total_rounds
    print(f"{'Total Rounds':<35} {total_rounds:>20} {'rounds':>10}")
    print(f"{'Estimated Total Volume':<35} {estimated_total_volume:>20.2f} {'MB':>10}")
    print(f"{'Estimated Total Volume':<35} {estimated_total_volume/1024:>20.2f} {'GB':>10}")
    
    print("-"*70)
    
    # Efficiency metrics
    print(f"{'Compression Ratio':<35} {comm_stats['compression_ratio']:>20.2f} {'%':>10}")
    
    if args.alg == 'fedlora':
        bandwidth_saving = 100 - comm_stats['compression_ratio']
        print(f"{'Bandwidth Saving vs FedAvg':<35} {bandwidth_saving:>20.2f} {'%':>10}")
    
    print("="*70)
    
    # Algorithm-specific notes
    if args.alg == 'fedavg':
        print("\n[FedAvg] Communicating ALL model parameters each round")
    elif args.alg == 'fedlora':
        print("\n[FedLoRA] Communicating ONLY LoRA parameters + classification head")
        print(f"[FedLoRA] Parameter Efficiency: {comm_stats['compression_ratio']:.2f}% of full model")
    elif args.alg == 'fedsdg':
        print("\n[FedSDG] Communicating ONLY Global LoRA parameters (lora_A, lora_B) + classification head")
        print(f"[FedSDG] Private parameters (lora_A_private, lora_B_private, lambda_k) stay local")
        print(f"[FedSDG] Communication Efficiency: {comm_stats['compression_ratio']:.2f}% of full model (same as FedLoRA)")
    
    print("="*70 + "\n")

