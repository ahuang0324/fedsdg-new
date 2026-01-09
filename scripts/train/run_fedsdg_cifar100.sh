#!/bin/bash
# FedSDG with Pretrained ViT - CIFAR-100 with offline preprocessed data
# Dual-path federated learning with global + private branches

# ==================== Configuration ====================
ALG="fedsdg"
MODEL="vit"
MODEL_VARIANT="pretrained"
DATASET="cifar100"
NUM_CLASSES=100
IMAGE_SIZE=224
EPOCHS=100
NUM_USERS=100
FRAC=0.1
LOCAL_EP=5
LOCAL_BS=128
LR=0.001
LR_GATE=0.01
OPTIMIZER="adam"
LORA_R=8
LORA_ALPHA=16
DIRICHLET_ALPHA=0.1
GPU=0
LAMBDA1=0.01
LAMBDA2=0.0001
GATE_PENALTY_TYPE="bilateral"
AGG_METHOD="fedavg"  # or "alignment"
# =======================================================

LOG_SUBDIR="${ALG}_${MODEL_VARIANT}_${MODEL}_${DATASET}_E${EPOCHS}_lr${LR}_lrgate${LR_GATE}_alpha${DIRICHLET_ALPHA}_bs${LOCAL_BS}_l1${LAMBDA1}_gate${GATE_PENALTY_TYPE}_${AGG_METHOD}"
OFFLINE_DATA_ROOT="./data/preprocessed/"

export HF_ENDPOINT=https://hf-mirror.com

echo "=========================================="
echo "FedSDG Training - CIFAR-100 (Pretrained + Offline Data)"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Algorithm: ${ALG} (Dual-path: Global + Private branches)"
echo "  - Model: ViT-Tiny (timm pretrained, ImageNet-21k)"
echo "  - Dataset: ${DATASET} (offline preprocessed ${IMAGE_SIZE}x${IMAGE_SIZE})"
echo "  - Epochs: ${EPOCHS}"
echo "  - Clients: ${NUM_USERS}"
echo "  - Participation rate: $(echo "scale=0; ${FRAC} * 100" | bc)%"
echo "  - Local Epochs: ${LOCAL_EP}"
echo "  - Local Batch Size: ${LOCAL_BS}"
echo "  - LoRA rank: ${LORA_R}"
echo "  - LoRA alpha: ${LORA_ALPHA}"
echo "  - Learning rate: ${LR}"
echo "  - Gate learning rate: ${LR_GATE}"
echo "  - Lambda1 (gate penalty): ${LAMBDA1}"
echo "  - Lambda2 (private L2): ${LAMBDA2}"
echo "  - Gate penalty type: ${GATE_PENALTY_TYPE}"
echo "  - Server aggregation: ${AGG_METHOD}"
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
    --model_variant ${MODEL_VARIANT} \
    --dataset ${DATASET} \
    --image_size ${IMAGE_SIZE} \
    --use_offline_data \
    --offline_data_root ${OFFLINE_DATA_ROOT} \
    --epochs ${EPOCHS} \
    --num_users ${NUM_USERS} \
    --frac ${FRAC} \
    --local_ep ${LOCAL_EP} \
    --local_bs ${LOCAL_BS} \
    --lr ${LR} \
    --lr_gate ${LR_GATE} \
    --optimizer ${OPTIMIZER} \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lambda1 ${LAMBDA1} \
    --lambda2 ${LAMBDA2} \
    --gate_penalty_type ${GATE_PENALTY_TYPE} \
    --server_agg_method ${AGG_METHOD} \
    --dirichlet_alpha ${DIRICHLET_ALPHA} \
    --gpu ${GPU} \
    --log_subdir ${LOG_SUBDIR}

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Output locations:"
echo "  - TensorBoard logs: ./logs/${LOG_SUBDIR}/"
echo "  - Experiment summary: ./save/summaries/${DATASET}_${MODEL}_${ALG}_E${EPOCHS}_summary.txt"
echo "  - Final model: ./save/models/${DATASET}_${MODEL}_final.pth"
echo ""
echo "View TensorBoard:"
echo "  tensorboard --logdir=./logs/${LOG_SUBDIR}"
echo ""


