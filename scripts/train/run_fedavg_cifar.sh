#!/bin/bash
# FedAvg with CNN - CIFAR-10 baseline
# Standard federated averaging with full model communication

# ==================== Configuration ====================
ALG="fedavg"
MODEL="cnn"
DATASET="cifar"
EPOCHS=100
NUM_USERS=100
FRAC=0.1
LOCAL_EP=5
LOCAL_BS=64
LR=0.01
OPTIMIZER="sgd"
DIRICHLET_ALPHA=0.5
GPU=0
# =======================================================

LOG_SUBDIR="${ALG}_${MODEL}_${DATASET}_E${EPOCHS}_lr${LR}_alpha${DIRICHLET_ALPHA}"

echo "=========================================="
echo "FedAvg Training - CIFAR-10 (CNN)"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Algorithm: ${ALG}"
echo "  - Model: ${MODEL}"
echo "  - Dataset: ${DATASET}"
echo "  - Epochs: ${EPOCHS}"
echo "  - Clients: ${NUM_USERS}"
echo "  - Participation rate: $(echo "scale=0; ${FRAC} * 100" | bc)%"
echo "  - Local Epochs: ${LOCAL_EP}"
echo "  - Local Batch Size: ${LOCAL_BS}"
echo "  - Learning rate: ${LR}"
echo "  - Dirichlet alpha: ${DIRICHLET_ALPHA}"
echo "  - GPU: ${GPU}"
echo ""
echo "=========================================="
echo ""

# Change to project root directory
cd "$(dirname "$0")/../.."

python main.py \
    --alg ${ALG} \
    --model ${MODEL} \
    --dataset ${DATASET} \
    --epochs ${EPOCHS} \
    --num_users ${NUM_USERS} \
    --frac ${FRAC} \
    --local_ep ${LOCAL_EP} \
    --local_bs ${LOCAL_BS} \
    --lr ${LR} \
    --optimizer ${OPTIMIZER} \
    --dirichlet_alpha ${DIRICHLET_ALPHA} \
    --gpu ${GPU} \
    --log_subdir ${LOG_SUBDIR}

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""

