#!/bin/bash
# FedAvg with Pretrained ViT - CIFAR-100 (离线预处理数据)
# 全量参数训练的联邦学习 (作为 FedLoRA/FedSDG 的对比基准)

# ==================== 参数配置 ====================
ALG="fedavg"
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
LR=0.0003
OPTIMIZER="adam"
DIRICHLET_ALPHA=0.1
GPU=2
OFFLINE_DATA_ROOT="./datasets/preprocessed"
# =======================================================

export HF_ENDPOINT=https://hf-mirror.com

echo "=========================================="
echo "FedAvg 训练 - CIFAR-100 (预训练 + 离线数据)"
echo "=========================================="
echo ""
echo "训练配置："
echo "  - 算法: ${ALG} (全量参数训练)"
echo "  - 模型: ViT-Tiny (timm 预训练, ImageNet-21k)"
echo "  - 数据集: ${DATASET} (离线预处理 ${IMAGE_SIZE}x${IMAGE_SIZE})"
echo "  - 训练轮次: ${EPOCHS}"
echo "  - 客户端数量: ${NUM_USERS}"
echo "  - 参与率: $(echo "scale=0; ${FRAC} * 100" | bc)%"
echo "  - 本地 Epoch: ${LOCAL_EP}"
echo "  - 本地 Batch Size: ${LOCAL_BS}"
echo "  - 学习率: ${LR}"
echo "  - 优化器: ${OPTIMIZER}"
echo "  - Dirichlet Alpha: ${DIRICHLET_ALPHA}"
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
    --optimizer ${OPTIMIZER} \
    --dirichlet_alpha ${DIRICHLET_ALPHA} \
    --gpu ${GPU}

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo ""
echo "FedAvg 特点："
echo "  - 训练所有参数 (~5.7M)"
echo "  - 通信开销较大 (传输完整模型)"
echo "  - 作为 FedLoRA/FedSDG 的对比基准"
echo ""