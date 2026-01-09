# FedLoRA 实现文档

## 概述

FedLoRA 是一种基于 LoRA (Low-Rank Adaptation) 的联邦学习算法，通过参数高效微调技术减少通信开销和计算成本。

## 核心特性

### 1. LoRA 层实现 (`models.py`)

**LoRALayer 类**：
- 实现公式：`h = Wx + (α/r) * BAx`
- `W`：原始冻结权重矩阵
- `A`：低秩矩阵 (in_features × r)，使用 Kaiming 初始化
- `B`：低秩矩阵 (r × out_features)，初始化为 0
- `α/r`：缩放因子

**inject_lora 函数**：
- 针对 ViT 的 6 层 TransformerEncoderLayer
- 替换每层的 `self_attn.out_proj`（注意力输出投影）
- 替换每层的 `linear2`（FFN 第二层）
- 冻结所有骨干参数，仅训练 LoRA 参数和 mlp_head

### 2. 选择性聚合 (`utils.py`)

**average_weights_lora 函数**：
- 仅聚合包含 `lora_` 关键词的参数
- 保留冻结的骨干权重不变
- 减少通信开销（仅传输 LoRA 参数）

### 3. 配置参数 (`options.py`)

新增参数：
- `--alg`：算法选择 (`fedavg` 或 `fedlora`)
- `--lora_r`：LoRA 秩，默认 8
- `--lora_alpha`：LoRA 缩放参数，默认 16
- `--lora_train_mlp_head`：是否训练分类头，默认 1 (True)

### 4. 训练流程 (`federated_main.py`)

**FedLoRA 流程**：
1. 构建 ViT 模型
2. 注入 LoRA 层（冻结骨干，开放 LoRA 参数）
3. 客户端本地训练（仅更新 LoRA 参数）
4. 服务器端选择性聚合（仅聚合 LoRA 参数）
5. 下发更新后的全局模型

## 使用方法

### 基础用法

```bash
python src/federated_main.py \
    --alg fedlora \
    --model vit \
    --dataset cifar \
    --epochs 50 \
    --lora_r 8 \
    --lora_alpha 16 \
    --gpu 0
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--alg` | 算法类型 (fedavg/fedlora) | fedavg |
| `--lora_r` | LoRA 秩 | 8 |
| `--lora_alpha` | LoRA 缩放因子 | 16 |
| `--lora_train_mlp_head` | 是否训练分类头 | 1 |
| `--lr` | 学习率 | 0.01 |
| `--local_ep` | 本地训练轮数 | 10 |
| `--local_bs` | 本地批大小 | 10 |
| `--dirichlet_alpha` | Dirichlet 分布参数 (Non-IID) | 0.5 |

### 对比实验

**FedLoRA vs FedAvg**：

```bash
# FedLoRA（参数高效）
python src/federated_main.py --alg fedlora --model vit --dataset cifar --lora_r 8

# FedAvg（全量训练，Baseline）
python src/federated_main.py --alg fedavg --model vit --dataset cifar
```

## 参数效率分析

以 ViT (dim=128, depth=6) 为例：

- **总参数**：~200K
- **LoRA 参数** (r=8)：~15K (约 7.5%)
- **通信开销**：减少约 92.5%

## 实现细节

### 1. LoRA 注入位置

针对 ViT 的每个 TransformerEncoderLayer：
- `self_attn.out_proj`：注意力输出投影层
- `linear2`：FFN 的第二个线性层

每层注入 2 个 LoRA 层，共 6 层 × 2 = 12 个 LoRA 层。

### 2. 参数冻结策略

```python
# 1. 冻结整个模型
model.requires_grad_(False)

# 2. 仅开放 LoRA 参数
# LoRALayer 内部的 lora_A 和 lora_B 自动设置为可训练

# 3. 可选：开放分类头
if train_mlp_head:
    model.mlp_head.requires_grad_(True)
```

### 3. 聚合逻辑

```python
if args.alg == 'fedlora':
    # 选择性聚合：仅聚合 LoRA 参数
    global_weights = average_weights_lora(local_weights, global_model.state_dict())
else:
    # 全量聚合：聚合所有参数
    global_weights = average_weights(local_weights)
```

## 兼容性

- ✅ 完全兼容原有 FedAvg 训练路径
- ✅ 支持所有原有配置参数
- ✅ 支持 MNIST、Fashion-MNIST、CIFAR-10 数据集
- ⚠️ 目前仅支持 ViT 模型（CNN 模型可扩展）

## 输出文件

训练结果保存在 `../save/` 目录：

1. **训练日志**：`../logs/{log_subdir}/`
2. **训练曲线**：`../save/objects/{dataset}_{model}_{alg}_...pkl`
3. **最终模型**：`../save/models/{dataset}_{model}_{alg}_final_r[{r}]_lalpha[{alpha}].pth`
4. **数据分布热力图**：`../save/objects/client_class_heatmap_...png`

## 扩展建议

### 1. 支持更多模型

可以扩展 `inject_lora` 函数以支持 CNN、ResNet 等模型：

```python
def inject_lora_cnn(model, r=8, lora_alpha=16):
    # 针对 CNN 的全连接层注入 LoRA
    pass
```

### 2. 动态 LoRA 秩

根据层的重要性动态调整 LoRA 秩：

```python
# 浅层使用较小的秩，深层使用较大的秩
lora_ranks = [4, 4, 8, 8, 16, 16]
```

### 3. 个性化 LoRA

为每个客户端维护独立的 LoRA 参数（FedSDG 方向）：

```python
# 服务器维护全局 LoRA + 每个客户端的个性化 LoRA
global_lora = {...}
client_lora = {client_id: {...} for client_id in range(num_clients)}
```

## 注意事项

1. **仅支持 ViT**：当前实现仅支持 ViT 模型，使用 CNN 需设置 `--alg fedavg`
2. **学习率调整**：LoRA 训练通常需要较小的学习率（建议 0.001-0.0001）
3. **秩的选择**：r=8 是常用值，可根据任务复杂度调整（4-32）
4. **内存占用**：FedLoRA 显著减少内存占用和通信开销

## 参考文献

- LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)
- Communication-Efficient Learning of Deep Networks from Decentralized Data (McMahan et al., 2017)

## 作者

实现日期：2026-01-06
用途：FedSDG 研究的 Baseline 对照组
