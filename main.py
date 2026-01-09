#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main entry point for federated learning experiments.

Supports:
- FedAvg: Federated Averaging
- FedLoRA: Federated Low-Rank Adaptation
- FedSDG: Federated Structure-Decoupled Gating

Usage:
    python main.py --alg fedavg --model cnn --dataset cifar --epochs 100
    python main.py --alg fedlora --model vit --model_variant pretrained --dataset cifar100 --epochs 50
    python main.py --alg fedsdg --model vit --model_variant pretrained --dataset cifar100 --epochs 50
"""

import os
import copy
import time
import pickle
import math
from typing import Dict, List, Any, Optional, Union
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
from fl.data import get_dataset
from fl.models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, ViT, get_pretrained_vit
from fl.models import inject_lora, inject_lora_timm
from fl.algorithms import average_weights, average_weights_lora
from fl.clients import LocalUpdate


if __name__ == '__main__':
    start_time = time.time()

    # Define paths using centralized path management
    path_project = PROJECT_ROOT
    args = args_parser()
    exp_details(args)

    # Generate experiment name for consistent naming across all outputs
    experiment_name = generate_experiment_name(args)
    
    # Setup logging: logs/{algorithm}/{experiment_name}/
    log_dir = get_log_dir(args.alg, experiment_name)
    logger = SummaryWriter(log_dir)
    print(f"\n[Output] TensorBoard logs: {log_dir}")

    # Device setup
    use_cuda = (args.gpu is not None) and (int(args.gpu) >= 0) and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(int(args.gpu))
    device = 'cuda' if use_cuda else 'cpu'

    # Load dataset and user groups (dual evaluation: train + test partitions)
    train_dataset, test_dataset, user_groups, user_groups_test = get_dataset(args)

    # BUILD MODEL
    global_model: nn.Module  # Type declaration for model union
    
    if args.model == 'cnn':
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'vit':
        if hasattr(args, 'model_variant') and args.model_variant == 'pretrained':
            global_model = get_pretrained_vit(
                num_classes=args.num_classes,
                image_size=args.image_size if hasattr(args, 'image_size') else 224,
                pretrained_path=args.pretrained_path if hasattr(args, 'pretrained_path') else None
            )
        else:
            img_size = train_dataset[0][0].shape[-1]
            channels = train_dataset[0][0].shape[0]
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
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set model to train mode and send to device
    global_model.to(device)
    global_model.train()
    print(global_model)
    
    # FedLoRA/FedSDG: Inject LoRA into ViT model
    if args.alg in ('fedlora', 'fedsdg'):
        if args.model != 'vit':
            raise ValueError(
                f"{args.alg.upper()} currently only supports ViT model, but got model='{args.model}'. "
                f"Please use --model vit or switch to --alg fedavg for other models."
            )
        
        is_fedsdg = (args.alg == 'fedsdg')
        
        if hasattr(args, 'model_variant') and args.model_variant == 'pretrained':
            global_model = inject_lora_timm(
                global_model, 
                r=args.lora_r, 
                lora_alpha=args.lora_alpha,
                train_head=bool(args.lora_train_mlp_head),
                is_fedsdg=is_fedsdg
            )
        else:
            print("\n" + "="*60)
            alg_name = "FedSDG" if is_fedsdg else "FedLoRA"
            print(f"[{alg_name}] Injecting {alg_name} into custom ViT model...")
            print("="*60)
            global_model = inject_lora(
                global_model, 
                r=args.lora_r, 
                lora_alpha=args.lora_alpha,
                train_mlp_head=bool(args.lora_train_mlp_head),
                is_fedsdg=is_fedsdg
            )
            print("="*60 + "\n")

    # Copy weights
    global_weights = global_model.state_dict()

    # Communication statistics
    comm_stats = get_communication_stats(global_model, args.alg)
    print_communication_profile(comm_stats, args)
    
    # Log communication info
    logger.add_scalar('info/total_params', comm_stats['total_params'], 0)
    logger.add_scalar('info/trainable_params', comm_stats['trainable_params'], 0)
    logger.add_scalar('info/comm_params_per_round', comm_stats['comm_params'], 0)
    logger.add_scalar('info/comm_size_per_round_MB', comm_stats['comm_size_mb'], 0)
    logger.add_scalar('info/compression_ratio_percent', comm_stats['compression_ratio'], 0)
    
    comm_per_round_2way_mb = comm_stats['comm_size_mb'] * 2
    total_comm_volume_mb = comm_per_round_2way_mb * args.epochs
    logger.add_scalar('info/estimated_total_volume_MB', total_comm_volume_mb, 0)
    logger.add_scalar('info/estimated_total_volume_GB', total_comm_volume_mb / 1024, 0)
    
    # Efficiency metrics
    full_model_size_mb = comm_stats['total_params'] * 4 / (1024 * 1024)
    savings_ratio_percent = (1 - comm_stats['comm_size_mb'] / full_model_size_mb) * 100
    logger.add_scalar('Efficiency/communication_savings_percent', savings_ratio_percent, 0)
    
    log10_comm_size = math.log10(comm_stats['comm_size_mb']) if comm_stats['comm_size_mb'] > 0 else 0
    logger.add_scalar('Efficiency/log10_comm_size_MB', log10_comm_size, 0)
    
    print(f"\n{'='*70}")
    print("[Efficiency Metrics]")
    print(f"{'='*70}")
    print(f"  Full model size: {full_model_size_mb:.2f} MB")
    print(f"  Actual comm size: {comm_stats['comm_size_mb']:.2f} MB")
    print(f"  Communication savings: {savings_ratio_percent:.2f}%")
    print(f"{'='*70}\n")

    # FedSDG: Client private state management
    local_private_states: Optional[Dict[int, Dict[str, torch.Tensor]]] = {} if args.alg == 'fedsdg' else None
    
    if args.alg == 'fedsdg':
        print("\n" + "="*70)
        print("[FedSDG] Client private state management initialized")
        print("="*70)
        print(f"  Each client maintains independent private parameters (lora_A_private, lora_B_private, lambda_k)")
        print(f"  Private parameters do not participate in server aggregation")
        print("="*70 + "\n")
    
    # Checkpoint manager
    checkpoint_manager = None
    if args.enable_checkpoint:
        checkpoint_manager = create_checkpoint_manager(args, path_project, experiment_name)
    
    # Training
    train_loss: List[float] = []
    train_accuracy: List[float] = []
    val_acc_list: List[float] = []
    net_list: List[Any] = []
    cv_loss: List[float] = []
    cv_acc: List[float] = []
    print_every = 2
    val_loss_pre, counter = 0, 0
    
    cumulative_comm_volume_mb = 0
    best_efficiency_score = 0
    best_efficiency_epoch = 0

    for epoch in tqdm(range(args.epochs)):
        start_epoch_time = time.time()
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            # FedSDG: Load client private state
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
            
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(
                model=local_model_copy, global_round=epoch)
            
            # FedSDG: Save client private state
            if args.alg == 'fedsdg' and local_private_states is not None:
                private_state: Dict[str, torch.Tensor] = {}
                for name, param in local_model_copy.named_parameters():
                    if '_private' in name or 'lambda_k' in name:
                        private_state[name] = param.data.clone().cpu()
                local_private_states[idx] = private_state
            
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            
            # FedSDG Debug
            if args.alg == 'fedsdg' and epoch % 5 == 0 and idx == idxs_users[0]:
                print(f"\n[FedSDG Debug - Round {epoch+1}, Client {idx}]")
                lambda_k_values = []
                for name, param in local_model_copy.named_parameters():
                    if 'lambda_k_logit' in name:
                        lambda_k = torch.sigmoid(param).item()
                        lambda_k_values.append(lambda_k)
                if lambda_k_values:
                    print(f"  Lambda_k mean: {np.mean(lambda_k_values):.4f} (range: {min(lambda_k_values):.4f} - {max(lambda_k_values):.4f})")
                print(f"[FedSDG Debug End]\n")

        # Update global weights
        aggregation_info = None
        previous_global_state = copy.deepcopy(global_model.state_dict()) if checkpoint_manager else None
        
        if args.alg in ('fedlora', 'fedsdg'):
            agg_method = args.server_agg_method if args.alg == 'fedsdg' else 'fedavg'
            global_weights, aggregation_info = average_weights_lora(
                local_weights, 
                global_model.state_dict(), 
                agg_method=agg_method,
                return_aggregation_info=True
            )
            
            if args.alg == 'fedsdg' and epoch % 5 == 0:
                print(f"\n[FedSDG Aggregation Debug - Round {epoch+1}]")
                aggregated_keys = [k for k in global_weights.keys() 
                                   if ('lora_' in k or 'head' in k) 
                                   and '_private' not in k 
                                   and 'lambda_k' not in k]
                print(f"  Aggregated parameter keys: {len(aggregated_keys)}")
                print(f"[FedSDG Aggregation Debug End]\n")
        else:
            global_weights = average_weights(local_weights)

        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        
        logger.add_scalar('global/train_loss_avg', loss_avg, epoch)
        logger.add_scalar('client/train_loss_mean', np.mean(local_losses), epoch)
        logger.add_scalar('client/train_loss_var', np.var(local_losses), epoch)
        logger.add_scalar('lr', args.lr, epoch)

        # Evaluate training accuracy (every 5 rounds)
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            list_acc, list_loss = [], []
            global_model.eval()
            for idx in idxs_users:
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[idx], logger=logger)
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

        # Evaluation (every 5 rounds)
        round_test_acc, round_test_loss = None, None
        round_local_acc, round_local_loss = None, None
        
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            # Global test
            test_acc, test_loss = test_inference(args, global_model, test_dataset)
            logger.add_scalar('global/test_acc', test_acc, epoch)
            logger.add_scalar('global/test_loss', test_loss, epoch)
            round_test_acc, round_test_loss = test_acc, test_loss
            
            # Local personalization evaluation
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
            
            # Efficiency metrics
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
        
        # Save checkpoint
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
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

    # Final evaluation
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    
    final_local_acc, final_local_loss, final_client_results = evaluate_local_personalization(
        args=args,
        global_model=global_model,
        test_dataset=test_dataset,
        user_groups_test=user_groups_test,
        local_private_states=local_private_states,
        sample_clients=None
    )

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Global Test Accuracy: {:.2f}%".format(100*test_acc))
    print("|---- Local Personalized Accuracy (Avg): {:.2f}%".format(100*final_local_acc))
    print("|---- Acc Gap (Local - Global): {:.2f}%".format(100*(final_local_acc - test_acc)))

    # Save results to outputs/results/{algorithm}/{experiment_name}.pkl
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
    print(f"[Output] Results saved: {result_path}")

    total_time = time.time() - start_time
    print('\n Total Run Time: {0:0.4f}'.format(total_time))
    
    # Generate summary report
    fedavg_estimated_comm_mb = full_model_size_mb * 2 * args.epochs
    saved_comm_mb = fedavg_estimated_comm_mb - cumulative_comm_volume_mb
    saved_comm_gb = saved_comm_mb / 1024
    savings_multiplier = fedavg_estimated_comm_mb / cumulative_comm_volume_mb if cumulative_comm_volume_mb > 0 else 1
    
    summary_text = f"""
# Federated Learning Experiment Summary

## Configuration
- **Algorithm**: {args.alg.upper()}
- **Model**: {args.model.upper()} ({args.model_variant if hasattr(args, 'model_variant') else 'scratch'})
- **Dataset**: {args.dataset.upper()}
- **Epochs**: {args.epochs}
- **Clients**: {args.num_users}
- **Participation rate**: {args.frac * 100:.1f}%

## Performance
- **Final Train Accuracy**: {train_accuracy[-1] * 100:.2f}%
- **Global Test Accuracy**: {test_acc * 100:.2f}%
- **Local Personalized Accuracy**: {final_local_acc * 100:.2f}%
- **Accuracy Gap (Local - Global)**: {(final_local_acc - test_acc) * 100:+.2f}%

## Communication Efficiency
- **Total Communication**: {cumulative_comm_volume_mb:.2f} MB ({cumulative_comm_volume_mb / 1024:.2f} GB)
- **Communication Savings**: {savings_ratio_percent:.2f}%
- **Savings vs FedAvg**: {savings_multiplier:.2f}x

## Training Time
- **Total Time**: {total_time / 60:.2f} min ({total_time:.2f} sec)
- **Avg Time/Round**: {total_time / args.epochs:.2f} sec

---
*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    logger.add_text('Experiment_Summary', summary_text, 0)
    
    # Save summary to outputs/summaries/{algorithm}/{experiment_name}.txt
    summary_path = get_summary_path(args.alg, experiment_name)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"[Output] Summary saved: {summary_path}")
    
    # Save final model to outputs/models/{algorithm}/{experiment_name}_final.pth
    model_path = get_model_path(args.alg, experiment_name, suffix='final')
    torch.save(global_model.state_dict(), model_path)
    print(f"[Output] Final model saved: {model_path}")
    
    # Save final checkpoint
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

    logger.close()

