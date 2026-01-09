# FedLoRA 训练准确率下降问题诊断报告

## 问题现象

从 TensorBoard 观察到：
- **test_acc**: 从 ~18.5% 下降到 ~16%
- **test_loss**: 从 ~171 上升到 ~180
- **train_acc_avg**: 从 ~18.5% 下降到 ~16%
- **train_loss_avg**: 从 ~1.9 下降到 ~1.68

## 问题分析

### 1. 已修复：LoRA 初始化问题

**原问题**：
```python
nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)  # 不正确的初始化
```

**修复后**：
```python
nn.init.normal_(self.lora_A, mean=0.0, std=1.0/r**0.5)  # 遵循 LoRA 论文
```

### 2. 主要问题：学习率过大

**当前配置**：
- 学习率：0.001
- 优化器：Adam
- LoRA 秩：8
- LoRA alpha：16

**问题**：
- 对于 LoRA 微调，学习率 0.001 偏大
- LoRA 参数量很小（仅 3.8%），但学习率与全量训练相同
- Adam 优化器会自适应调整学习率，但初始学习率仍然重要

### 3. 可能的次要问题

#### a) LoRA 缩放因子
- 当前：alpha=16, r=8, scaling=2.0
- 这个缩放可能导致 LoRA 的贡献过大

#### b) 训练不稳定
- 10 个客户端（frac=0.1 × 100）
- 每个客户端本地训练 5 轮
- 可能导致客户端漂移

## 解决方案

### 方案 1：降低学习率（推荐）

```bash
# 将学习率降低到 0.0001 或 0.0003
--lr 0.0001
```

### 方案 2：调整 LoRA 超参数

```bash
# 减小 alpha 或增大 r
--lora_r 16 --lora_alpha 16  # scaling = 1.0
# 或
--lora_r 8 --lora_alpha 8    # scaling = 1.0
```

### 方案 3：使用学习率预热和衰减

在代码中添加学习率调度器（需要修改 update.py）

### 方案 4：减少本地训练轮数

```bash
--local_ep 3  # 从 5 降到 3
```

## 推荐的实验配置

### 配置 A：保守配置（最稳定）
```bash
python3 federated_main.py \
    --alg fedlora \
    --model vit \
    --dataset cifar \
    --epochs 80 \
    --lr 0.0001 \
    --optimizer adam \
    --lora_r 8 \
    --lora_alpha 8 \
    --local_ep 3 \
    --gpu 0
```

### 配置 B：中等配置
```bash
python3 federated_main.py \
    --alg fedlora \
    --model vit \
    --dataset cifar \
    --epochs 80 \
    --lr 0.0003 \
    --optimizer adam \
    --lora_r 8 \
    --lora_alpha 16 \
    --local_ep 5 \
    --gpu 0
```

### 配置 C：SGD 优化器（更稳定）
```bash
python3 federated_main.py \
    --alg fedlora \
    --model vit \
    --dataset cifar \
    --epochs 80 \
    --lr 0.01 \
    --optimizer sgd \
    --momentum 0.9 \
    --lora_r 8 \
    --lora_alpha 16 \
    --local_ep 5 \
    --gpu 0
```

## 代码修复总结

### 已修复
1. ✅ LoRA 初始化：从 Kaiming 改为正态分布（std=1/√r）
2. ✅ 设备兼容性：确保 LoRA 参数在正确设备上
3. ✅ 属性代理：添加 weight 和 bias 属性

### 建议修改
1. 🔧 降低学习率到 0.0001-0.0003
2. 🔧 考虑调整 LoRA 缩放因子
3. 🔧 可选：添加学习率调度器

## 理论分析

### 为什么学习率过大会导致准确率下降？

1. **梯度爆炸**：LoRA 参数很小，大学习率导致参数更新幅度过大
2. **优化不稳定**：跳过最优解，在损失函数表面震荡
3. **模型崩溃**：参数更新到错误的区域，难以恢复

### LoRA 的特殊性

- LoRA 只训练 3.8% 的参数
- 这些参数对输出的影响通过 scaling factor (α/r) 放大
- 因此需要更小心地控制学习率

## 下一步行动

1. **立即行动**：使用配置 A 重新训练
2. **对比实验**：同时运行配置 B 和 C
3. **监控指标**：关注前 10 轮的 loss 和 acc 变化
4. **调整策略**：如果仍不稳定，进一步降低学习率

## 参考

- LoRA 论文推荐学习率：1e-4 到 3e-4
- 联邦学习通常需要更小的学习率
- ViT 模型对学习率敏感
