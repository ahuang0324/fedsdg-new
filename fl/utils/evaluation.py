#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation utilities for federated learning.

Provides global and local evaluation functions for dual evaluation mechanism.
"""

import copy
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    """Dataset wrapper for specific indices."""

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def test_inference(args, model, test_dataset):
    """
    Global test inference function.
    
    For FedSDG: Disable private branches during global testing by setting
    gate parameters to a very negative value (m_k ≈ 0).
    
    Args:
        args: Command line arguments
        model: Global model
        test_dataset: Test dataset
    
    Returns:
        accuracy: Test accuracy
        loss: Test loss
    """
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    use_cuda = (args.gpu is not None) and (int(args.gpu) >= 0) and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False, num_workers=4, pin_memory=True)

    # FedSDG: Disable private branch during global testing
    original_gate_values = {}
    if args.alg == 'fedsdg':
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'lambda_k_logit' in name:
                    original_gate_values[name] = param.data.clone()
                    param.data.fill_(-100.0)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

    # Restore gate values
    if args.alg == 'fedsdg':
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in original_gate_values:
                    param.data.copy_(original_gate_values[name])

    accuracy = correct/total
    return accuracy, loss


def local_test_inference(args, model, test_dataset, idxs, private_state=None):
    """
    Local personalization test inference function.
    
    Evaluates client performance on their local test set.
    
    Args:
        args: Command line arguments
        model: Model (global or with private parameters loaded)
        test_dataset: Test dataset
        idxs: Local test set indices for this client
        private_state: FedSDG private parameters {param_name: tensor}
    
    Returns:
        accuracy: Local test accuracy
        loss: Local test loss
    """
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    
    use_cuda = (args.gpu is not None) and (int(args.gpu) >= 0) and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    criterion = nn.CrossEntropyLoss().to(device)
    
    local_test_loader = DataLoader(
        DatasetSplit(test_dataset, idxs),
        batch_size=128,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # FedSDG: Load client private parameters
    original_state = None
    if args.alg == 'fedsdg' and private_state is not None:
        original_state = {}
        current_state = model.state_dict()
        
        with torch.no_grad():
            for param_name, param_value in private_state.items():
                if param_name in current_state:
                    original_state[param_name] = current_state[param_name].clone()
                    model.state_dict()[param_name].copy_(param_value.to(device))
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(local_test_loader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()
            
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
    
    # Restore original state
    if args.alg == 'fedsdg' and original_state is not None:
        with torch.no_grad():
            for param_name, param_value in original_state.items():
                model.state_dict()[param_name].copy_(param_value)
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, loss


def evaluate_local_personalization(args, global_model, test_dataset, user_groups_test, 
                                    local_private_states=None, sample_clients=None):
    """
    Evaluate local personalization performance for all (or sampled) clients.
    
    This is Step B of dual evaluation mechanism:
    - Iterate through clients
    - Load their personalized model state (Global + Private for FedSDG)
    - Test on their local test set
    - Compute average local accuracy and loss
    
    Args:
        args: Command line arguments
        global_model: Global model
        test_dataset: Test dataset
        user_groups_test: {client_id: test_indices}
        local_private_states: FedSDG private states {client_id: {param_name: tensor}}
        sample_clients: List of clients to evaluate, None for all
    
    Returns:
        avg_acc: Average local test accuracy
        avg_loss: Average local test loss
        client_results: {client_id: (acc, loss)} detailed results
    """
    global_model.eval()
    
    if sample_clients is None:
        clients_to_eval = list(user_groups_test.keys())
    else:
        clients_to_eval = sample_clients
    
    client_results = {}
    total_acc, total_loss = 0.0, 0.0
    
    use_cuda = (args.gpu is not None) and (int(args.gpu) >= 0) and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    
    for client_id in clients_to_eval:
        test_idxs = user_groups_test[client_id]
        
        if len(test_idxs) == 0:
            continue
        
        # Prepare model based on algorithm
        if args.alg == 'fedsdg' and local_private_states is not None and client_id in local_private_states:
            # FedSDG: Deep copy and load private parameters
            local_model = copy.deepcopy(global_model)
            private_state = local_private_states[client_id]
            
            with torch.no_grad():
                current_state = local_model.state_dict()
                for param_name, param_value in private_state.items():
                    if param_name in current_state:
                        current_state[param_name] = param_value.to(device)
                local_model.load_state_dict(current_state)
            
            acc, loss = local_test_inference(args, local_model, test_dataset, test_idxs)
            
            del local_model
        else:
            # FedAvg / FedLoRA: Use global model directly
            acc, loss = local_test_inference(args, global_model, test_dataset, test_idxs)
        
        client_results[client_id] = (acc, loss)
        total_acc += acc
        total_loss += loss
    
    num_clients = len(client_results)
    avg_acc = total_acc / num_clients if num_clients > 0 else 0.0
    avg_loss = total_loss / num_clients if num_clients > 0 else 0.0
    
    return avg_acc, avg_loss, client_results


