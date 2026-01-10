#!/bin/bash
# FedSDG with Pretrained ViT - CIFAR-100 (离线预处理数据)
# 双路架构 + 门控机制的联邦学习

# ==================== 参数配置 ====================
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
GPU=2
LAMBDA1=0.01
LAMBDA2=0.0001
GATE_PENALTY_TYPE="bilateral"
AGG_METHOD="alignment"
OFFLINE_DATA_ROOT="./datasets/preprocessed"
# =======================================================

export HF_ENDPOINT=https://hf-mirror.com

echo "=========================================="
echo "FedSDG 训练 - CIFAR-100 (预训练 + 离线数据)"
echo "=========================================="
echo ""
echo "训练配置："
echo "  - 算法: ${ALG} (双路架构 + 门控机制)"
echo "  - 模型: ViT-Tiny (timm 预训练, ImageNet-21k)"
echo "  - 数据集: ${DATASET} (离线预处理 ${IMAGE_SIZE}x${IMAGE_SIZE})"
echo "  - 训练轮次: ${EPOCHS}"
echo "  - 客户端数量: ${NUM_USERS}"
echo "  - 参与率: $(echo "scale=0; ${FRAC} * 100" | bc)%"
echo "  - 本地 Epoch: ${LOCAL_EP}"
echo "  - 本地 Batch Size: ${LOCAL_BS}"
echo "  - LoRA 秩: ${LORA_R}"
echo "  - LoRA Alpha: ${LORA_ALPHA}"
echo "  - 学习率: ${LR}"
echo "  - 门控学习率: ${LR_GATE}"
echo "  - Lambda1 (门控惩罚): ${LAMBDA1}"
echo "  - Lambda2 (私有L2): ${LAMBDA2}"
echo "  - 门控惩罚类型: ${GATE_PENALTY_TYPE}"
echo "  - 服务器聚合: ${AGG_METHOD}"
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
    --lr_gate ${LR_GATE} \
    --optimizer ${OPTIMIZER} \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lambda1 ${LAMBDA1} \
    --lambda2 ${LAMBDA2} \
    --gate_penalty_type ${GATE_PENALTY_TYPE} \
    --server_agg_method ${AGG_METHOD} \
    --dirichlet_alpha ${DIRICHLET_ALPHA} \
    --gpu ${GPU}

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo ""
echo "FedSDG 特点："
echo "  - 全局分支: 参与服务器聚合"
echo "  - 私有分支: 仅本地更新，学习客户端特定模式"
echo "  - 门控参数: 自动学习全局/私有权重"
echo "  - 通信量与 FedLoRA 一致"
echo ""
