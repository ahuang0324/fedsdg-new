#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Local training module for federated learning clients.

Implements local training logic for:
- FedAvg: Standard SGD training
- FedLoRA: LoRA parameter training
- FedSDG: Dual-path training with gate regularization
"""

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..models.lora import get_lora_state_dict


class DatasetSplit(Dataset):
    """Dataset wrapper for client-specific data indices."""

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    """
    Local training handler for federated learning clients.
    
    Handles local training, validation, and inference for each client.
    Supports FedAvg, FedLoRA, and FedSDG algorithms.
    """
    
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        use_cuda = (args.gpu is not None) and (int(args.gpu) >= 0) and torch.cuda.is_available()
        self.device = 'cuda' if use_cuda else 'cpu'
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """Split client data into train, validation, and test sets."""
        idxs = list(idxs)
        np.random.shuffle(idxs)
        # Split: 80% train, 10% val, 10% test
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True,
                                 num_workers=4, pin_memory=True, prefetch_factor=2)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=max(1, int(len(idxs_val)/10)), shuffle=False,
                                 num_workers=2, pin_memory=True)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=max(1, int(len(idxs_test)/10)), shuffle=False,
                                num_workers=2, pin_memory=True)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        """
        Client local training function.
        
        FedSDG Loss (Equation 5 from FedSDG_Design.md):
        Loss = (1/|B|) Σ ℓ(f(x), y) + λ₁ Σ|m_{k,l}| + λ₂ ||θ_{p,k}||²₂
        
        Returns:
            state_dict: Model parameters to upload
            loss: Average training loss
        """
        model.train()
        epoch_loss = []

        # Optimizer configuration
        if self.args.alg == 'fedsdg':
            # FedSDG: Three parameter groups with different learning rates
            global_params = []
            private_params = []
            gate_params = []
            
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if 'lambda_k_logit' in name:
                        gate_params.append(param)
                    elif '_private' in name:
                        private_params.append(param)
                    else:
                        global_params.append(param)
            
            param_groups = [
                {'params': global_params, 'lr': self.args.lr},
                {'params': private_params, 'lr': self.args.lr},
                {'params': gate_params, 'lr': self.args.lr_gate}
            ]
            optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.999), weight_decay=0)
        else:
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            if self.args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(trainable_params, lr=self.args.lr,
                                            momentum=0.5)
            elif self.args.optimizer == 'adam':
                optimizer = torch.optim.Adam(trainable_params, lr=self.args.lr,
                                             weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                logits = model(images)
                
                # Task loss
                task_loss = self.criterion(logits, labels)
                
                if self.args.alg == 'fedsdg':
                    # Gate sparsity penalty (λ₁)
                    gate_penalty = torch.tensor(0.0, device=self.device)
                    gate_count = 0
                    for name, param in model.named_parameters():
                        if 'lambda_k_logit' in name:
                            m_k = torch.sigmoid(param)
                            if self.args.gate_penalty_type == 'bilateral':
                                gate_penalty = gate_penalty + torch.sum(torch.min(m_k, 1 - m_k))
                            else:
                                gate_penalty = gate_penalty + torch.sum(torch.abs(m_k))
                            gate_count += param.numel()
                    
                    # Private parameter L2 regularization (λ₂)
                    private_penalty = torch.tensor(0.0, device=self.device)
                    private_count = 0
                    for name, param in model.named_parameters():
                        if '_private' in name:
                            private_penalty = private_penalty + torch.sum(param ** 2)
                            private_count += param.numel()
                    
                    # Combined loss
                    lambda1 = self.args.lambda1
                    lambda2 = self.args.lambda2
                    loss = task_loss + lambda1 * gate_penalty + lambda2 * private_penalty
                    
                    # Log FedSDG metrics
                    if batch_idx == len(self.trainloader) - 1:
                        global_step_epoch = global_round * self.args.local_ep + iter
                        self.logger.add_scalar('FedSDG/task_loss', task_loss.item(), global_step_epoch)
                        self.logger.add_scalar('FedSDG/gate_penalty_raw', gate_penalty.item(), global_step_epoch)
                        self.logger.add_scalar('FedSDG/gate_penalty_weighted', lambda1 * gate_penalty.item(), global_step_epoch)
                        self.logger.add_scalar('FedSDG/private_penalty_raw', private_penalty.item(), global_step_epoch)
                        self.logger.add_scalar('FedSDG/private_penalty_weighted', lambda2 * private_penalty.item(), global_step_epoch)
                        
                        gate_values = []
                        for name, param in model.named_parameters():
                            if 'lambda_k_logit' in name:
                                gate_values.append(torch.sigmoid(param).item())
                        if gate_values:
                            self.logger.add_scalar('FedSDG/gate_mean', sum(gate_values) / len(gate_values), global_step_epoch)
                            self.logger.add_scalar('FedSDG/gate_min', min(gate_values), global_step_epoch)
                            self.logger.add_scalar('FedSDG/gate_max', max(gate_values), global_step_epoch)
                else:
                    loss = task_loss
                
                loss.backward()
                
                # FedSDG: Gradient clipping
                if self.args.alg == 'fedsdg' and self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.grad_clip)
                
                # Debug gradient for gate parameters
                if self.args.alg == 'fedsdg' and batch_idx == 0 and iter == 0:
                    for name, param in model.named_parameters():
                        if 'lambda_k_logit' in name:
                            grad_val = param.grad.item() if param.grad is not None else 0.0
                            print(f"  [Gradient Debug] {name}: grad={grad_val:.6f}, value={param.data.item():.4f}")
                            break
                
                optimizer.step()

                if self.args.verbose and batch_idx == 0:
                    print('| Global Round : {} | Local Epoch : {} | Starting training...'.format(
                        global_round, iter))
                global_step = (global_round * self.args.local_ep * len(self.trainloader)
                               + iter * len(self.trainloader) + batch_idx)
                self.logger.add_scalar('loss', loss.item(), global_step=global_step)
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # Return appropriate state_dict
        if self.args.alg in ('fedlora', 'fedsdg'):
            return get_lora_state_dict(model), sum(epoch_loss) / len(epoch_loss)
        else:
            return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model, loader='train'):
        """Client inference evaluation."""
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        if loader == 'train':
            dataloader = self.trainloader
        elif loader == 'val':
            dataloader = self.validloader
        elif loader == 'test':
            dataloader = self.testloader
        else:
            raise ValueError(f"Unknown loader: {loader}")

        # FedSDG: Disable private branch during global model evaluation
        original_gate_values = {}
        if self.args.alg == 'fedsdg':
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if 'lambda_k_logit' in name:
                        original_gate_values[name] = param.data.clone()
                        param.data.fill_(-100.0)

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = model(images)
                batch_loss = self.criterion(outputs, labels)
                loss += batch_loss.item()

                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

        # Restore gate values
        if self.args.alg == 'fedsdg':
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in original_gate_values:
                        param.data.copy_(original_gate_values[name])

        accuracy = correct/total
        return accuracy, loss


