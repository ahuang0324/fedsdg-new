# CIFAR-100 数据集使用指南

## 概述

本项目已完整支持 CIFAR-100 数据集，包括离线预处理和联邦学习训练。CIFAR-100 相比 CIFAR-10 更具挑战性（100个类别 vs 10个类别），更适合评估模型的泛化能力和 LoRA 的性能。

## 主要特性

- ✅ **离线预处理**：预先将图像调整为 224x224，避免训练时重复 Resize
- ✅ **内存映射**：使用 numpy memmap，支持多进程共享内存，降低 CPU 和内存消耗
- ✅ **完整兼容**：与现有 CIFAR-10 代码完全兼容，无需修改核心逻辑
- ✅ **自动配置**：自动设置 100 个类别，无需手动指定

## 快速开始

### 1. 数据预处理

首先运行预处理脚本，将 CIFAR-100 数据集下载并预处理为 224x224 的 memmap 格式：

```bash
cd src
python3 preprocess_cifar100.py
```

**预处理过程**：
- 自动下载 CIFAR-100 数据集（约 170 MB）
- 将图像从 32x32 调整为 224x224（适配预训练 ViT）
- 保存为 numpy memmap 格式（约 2.4 GB）
- 验证数据完整性

**输出路径**（与 CIFAR-10 共享同一目录）：
- 训练集图像：`../data/preprocessed/cifar100_224x224/train_images.npy`
- 训练集标签：`../data/preprocessed/cifar100_224x224/train_labels.npy`
- 测试集图像：`../data/preprocessed/cifar100_224x224/test_images.npy`
- 测试集标签：`../data/preprocessed/cifar100_224x224/test_labels.npy`

### 2. 验证预处理数据

运行验证脚本确保数据处理正确：

```bash
python3 verify_cifar100.py
```

**验证内容**：
- ✓ 文件存在性检查
- ✓ 数据加载测试
- ✓ 数据形状和类型验证
- ✓ OfflineCIFAR100 类测试
- ✓ DataLoader 性能测试

### 3. 训练模型

#### 3.1 FedAvg（全量参数训练）

```bash
bash run_fedavg_pretrained_cifar100.sh
```

**关键参数**：
- 数据集：CIFAR-100（100个类别）
- 算法：FedAvg（训练所有参数）
- 模型：预训练 ViT-Tiny（5.7M 参数）
- 学习率：0.0001
- 通信开销：~5.7M 参数/轮

**预期性能**：
- 准确率：60-75%（CIFAR-100 比 CIFAR-10 更难）
- 每轮时间：~30-50秒
- GPU 利用率：显著提升（数据加载不再是瓶颈）

#### 3.2 FedLoRA（参数高效训练）

```bash
bash run_fedlora_pretrained_cifar100.sh
```

**关键参数**：
- 数据集：CIFAR-100（100个类别）
- 算法：FedLoRA（仅训练 LoRA 参数）
- 模型：预训练 ViT-Tiny + LoRA
- LoRA 秩：r=8
- LoRA alpha：16
- 学习率：0.0003
- 通信开销：~200K 参数/轮（仅 3.5%）

**预期性能**：
- 准确率：55-70%（略低于 FedAvg，但通信开销大幅降低）
- 每轮时间：~25-40秒
- 通信效率：减少 96.5% 的通信开销

## 文件结构

```
Federated-Learning-PyTorch/
├── src/
│   ├── preprocess_cifar100.py          # CIFAR-100 预处理脚本
│   ├── verify_cifar100.py              # CIFAR-100 验证脚本
│   ├── offline_dataset.py              # 离线数据集类（包含 OfflineCIFAR100）
│   ├── run_fedavg_pretrained_cifar100.sh   # FedAvg 训练脚本
│   ├── run_fedlora_pretrained_cifar100.sh  # FedLoRA 训练脚本
│   ├── utils.py                        # 数据加载工具（已支持 CIFAR-100）
│   ├── options.py                      # 参数解析（支持 cifar10/cifar100 选择）
│   └── ...
├── data/
│   ├── cifar/                          # 原始 CIFAR 数据（CIFAR-10 和 CIFAR-100 共享）
│   └── preprocessed/                   # 预处理后的数据（统一目录）
│       ├── cifar10_224x224/            # CIFAR-10 预处理数据
│       │   ├── train_images.npy
│       │   ├── train_labels.npy
│       │   ├── test_images.npy
│       │   └── test_labels.npy
│       └── cifar100_224x224/           # CIFAR-100 预处理数据
│           ├── train_images.npy
│           ├── train_labels.npy
│           ├── test_images.npy
│           └── test_labels.npy
└── logs/                               # TensorBoard 日志
```

## 技术细节

### 数据预处理优化

1. **离线 Resize**：预处理时完成图像调整，训练时零 CPU 开销
2. **Memmap 格式**：内存映射文件，支持多进程共享，降低内存占用
3. **标准化延迟**：仅在 DataLoader 中进行标准化，保持灵活性

### 数据加载优化

1. **多进程加载**：`num_workers=4`，并行加载数据
2. **内存固定**：`pin_memory=True`，加速 GPU 传输
3. **预取机制**：`prefetch_factor=2`，提前准备数据

### 联邦学习配置

1. **Non-IID 划分**：使用 Dirichlet 分布（alpha=0.5）模拟数据异构性
2. **客户端采样**：每轮随机选择 10% 的客户端（10/100）
3. **本地训练**：每个客户端训练 5 个 epoch

## 性能对比

| 指标 | CIFAR-10 | CIFAR-100 |
|------|----------|-----------|
| 类别数 | 10 | 100 |
| 训练样本 | 50,000 | 50,000 |
| 测试样本 | 10,000 | 10,000 |
| FedAvg 准确率 | 75-90% | 60-75% |
| FedLoRA 准确率 | 70-85% | 55-70% |
| 难度 | 简单 | 中等 |

## FedAvg vs FedLoRA 对比

| 维度 | FedAvg | FedLoRA |
|------|--------|---------|
| 训练参数 | 5.7M（100%） | ~200K（3.5%） |
| 通信开销 | 高（5.7M/轮） | 低（200K/轮） |
| 准确率 | 高 | 略低（-5%） |
| 内存占用 | 高 | 低 |
| 适用场景 | 高性能需求 | 资源受限场景 |

## 常见问题

### Q1: 预处理需要多长时间？
A: 约 2-5 分钟，取决于 CPU 性能。首次运行会下载数据集（~170 MB）。

### Q2: 预处理数据占用多少空间？
A: 约 2.4 GB（训练集 ~2.0 GB + 测试集 ~0.4 GB）。

### Q3: 如何切换回 CIFAR-10？
A: 修改训练脚本中的 `--dataset cifar100` 为 `--dataset cifar`，并更新 `--offline_data_root` 路径。

### Q4: 是否支持在线数据加载（不预处理）？
A: 支持。移除 `--use_offline_data` 参数即可使用在线加载，但会增加 CPU 负载。

### Q5: 如何调整数据异构性？
A: 修改 `--dirichlet_alpha` 参数：
- alpha < 1：高度异构（Non-IID）
- alpha = 1：中等异构
- alpha > 10：接近 IID

## 下一步建议

1. **实验对比**：在 CIFAR-10 和 CIFAR-100 上对比 FedAvg 和 FedLoRA 的性能
2. **参数调优**：尝试不同的 LoRA 秩（r=4, 8, 16）和学习率
3. **数据异构性**：测试不同的 Dirichlet alpha 值（0.1, 0.5, 1.0, 10.0）
4. **可视化分析**：使用 TensorBoard 查看训练曲线和客户端数据分布热力图

## 参考资料

- CIFAR-100 数据集：https://www.cs.toronto.edu/~kriz/cifar.html
- LoRA 论文：https://arxiv.org/abs/2106.09685
- ViT 论文：https://arxiv.org/abs/2010.11929
