# FedSDG 实现审计与修复技术报告

## 一、审计概述

### 1.1 审计目标

验证 FedSDG 算法实现是否符合 `FedSDG_Design.md` 设计规范，特别是：
- **Equation 3**: 门控权重计算 `m_{k,l} = σ(a_{k,l})`
- **Equation 4**: 残差分解适应 `θ̃_{k,l} = θ_{g,l} + m_{k,l} · θ_{p,k,l}`
- **Equation 5**: 客户端优化目标 `Loss = TaskLoss + λ₁ Σ|m_{k,l}| + λ₂ ||θ_{p,k}||²₂`

### 1.2 审计范围

| 文件 | 审计内容 |
|------|----------|
| `src/update.py` | LocalUpdate.update_weights() 损失函数 |
| `src/models.py` | LoRALayer 前向传播、门控初始化 |
| `src/options.py` | 命令行参数定义 |
| `src/federated_main.py` | 训练流程、聚合逻辑 |
| `src/utils.py` | 聚合函数 |

---

## 二、发现的问题

### 2.1 🔴 严重问题：缺失 Equation 5 正则化项

**问题描述**：

原始 `update_weights()` 方法仅计算任务损失，完全缺失设计文档中定义的两个正则化项：

```python
# 原始代码 (update.py:85)
loss = self.criterion(logits, labels)  # 仅任务损失
```

**设计规范要求** (Equation 5)：

```
Loss = (1/|B|) Σ ℓ(f(x), y) + λ₁ Σ|m_{k,l}| + λ₂ ||θ_{p,k}||²₂
       └──────────────────┘   └──────────┘   └────────────┘
       Task Loss              L1 Gate        L2 Private
```

**影响**：
- λ₁ L1 门控惩罚缺失 → 门控参数无法学习稀疏化，核心创新失效
- λ₂ L2 私有惩罚缺失 → 私有参数可能无限制增长，过拟合风险

---

### 2.2 🟡 中等问题：优化器 weight_decay 误用

**问题描述**：

```python
# 原始代码 (update.py:75-76)
optimizer = torch.optim.Adam(trainable_params, lr=self.args.lr,
                             weight_decay=1e-4)
```

**问题**：
- `weight_decay=1e-4` 作用于**所有可训练参数**
- 论文的 λ₂ **仅作用于私有参数**
- 这会错误地惩罚全局 LoRA 参数和分类头

---

### 2.3 🟡 中等问题：门控参数初始化偏差

**问题描述**：

```python
# 原始代码 (models.py:77)
self.lambda_k_logit = nn.Parameter(torch.tensor([-2.0]))
# sigmoid(-2.0) ≈ 0.12，即 88% 全局 + 12% 私有
```

**设计规范要求**：

```
a_{k,l} = 0 → m_{k,l} = σ(0) = 0.5
```

**影响**：初始化偏向全局分支，不符合"无偏起点"的设计原则。

---

### 2.4 🟡 中等问题：前向传播公式偏差

**问题描述**：

```python
# 原始代码 (models.py:117-118)
# 加权插值形式
lora_output = (global_output * (1 - lambda_k) + private_output * lambda_k) * self.scaling
```

**设计规范要求** (Equation 4)：

```
θ̃_{k,l} = θ_{g,l} + m_{k,l} · θ_{p,k,l}  # 加性残差形式
```

**影响**：
- 原始实现：`(1-m) * global + m * private`（插值）
- 设计要求：`global + m * private`（残差）
- 语义差异：残差形式确保全局分支始终贡献

---

### 2.5 🟡 中等问题：缺失命令行参数

**问题描述**：`options.py` 缺少 `--lambda1` 和 `--lambda2` 参数定义。

---

## 三、修复方案

### 3.1 修复 Equation 5 损失函数 (`src/update.py`)

**修复内容**：

```python
def update_weights(self, model, global_round):
    """
    客户端本地训练函数
    
    FedSDG 算法核心实现 (Equation 5 from FedSDG_Design.md):
    Loss = (1/|B|) Σ ℓ(f(x), y) + λ₁ Σ|m_{k,l}| + λ₂ ||θ_{p,k}||²₂
    """
    # ... 优化器配置 ...
    
    for iter in range(self.args.local_ep):
        for batch_idx, (images, labels) in enumerate(self.trainloader):
            # 基础任务损失
            task_loss = self.criterion(logits, labels)
            
            if self.args.alg == 'fedsdg':
                # ========== λ₁ L1 门控稀疏性惩罚 ==========
                gate_penalty = torch.tensor(0.0, device=self.device)
                for name, param in model.named_parameters():
                    if 'lambda_k_logit' in name:
                        m_k = torch.sigmoid(param)
                        gate_penalty += torch.sum(torch.abs(m_k))
                
                # ========== λ₂ L2 私有参数正则化 ==========
                private_penalty = torch.tensor(0.0, device=self.device)
                for name, param in model.named_parameters():
                    if '_private' in name:
                        private_penalty += torch.sum(param ** 2)
                
                # ========== 组合总损失 (Equation 5) ==========
                loss = task_loss + self.args.lambda1 * gate_penalty + self.args.lambda2 * private_penalty
            else:
                loss = task_loss
```

**关键改动**：
1. 添加 λ₁ L1 门控惩罚计算
2. 添加 λ₂ L2 私有参数惩罚计算
3. 仅在 `args.alg == 'fedsdg'` 时应用正则化
4. FedSDG 模式下禁用优化器 `weight_decay`

---

### 3.2 修复门控参数初始化 (`src/models.py`)

**修复内容**：

```python
# 修复后 (models.py:86)
self.lambda_k_logit = nn.Parameter(torch.tensor([0.0]))
# sigmoid(0.0) = 0.5，即 50% 全局 + 50% 私有
```

**符合设计规范**：训练开始时共享和私有组件等权重。

---

### 3.3 修复前向传播公式 (`src/models.py`)

**修复内容**：

```python
def forward(self, x):
    """
    FedSDG 模式实现 Equation 4:
    θ̃_{k,l} = θ_{g,l} + m_{k,l} · θ_{p,k,l}
    """
    original_output = self.original_layer(x)
    
    if self.is_fedsdg:
        m_k = torch.sigmoid(self.lambda_k_logit)
        global_output = x @ self.lora_A @ self.lora_B
        private_output = x @ self.lora_A_private @ self.lora_B_private
        
        # Equation 4: 加性残差形式
        lora_output = (global_output + m_k * private_output) * self.scaling
    else:
        lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling
    
    return original_output + lora_output
```

---

### 3.4 添加命令行参数 (`src/options.py`)

**修复内容**：

```python
# FedSDG 专用参数
parser.add_argument('--lambda1', type=float, default=1e-3,
                    help='FedSDG: L1 门控稀疏性惩罚系数 λ₁')
parser.add_argument('--lambda2', type=float, default=1e-4,
                    help='FedSDG: L2 私有参数正则化系数 λ₂')
```

---

## 四、验证结果

### 4.1 测试套件执行结果

```
🎉 所有测试通过！FedSDG 实现符合设计规范！

总结：
  ✓ LoRALayer 双路架构工作正常
  ✓ 私有参数过滤功能正确
  ✓ 通信量与 FedLoRA 一致
  ✓ 客户端私有状态管理正常
  ✓ 前向和反向传播正常
  ✓ 门控参数初始化符合规范 (a_{k,l}=0 → m_{k,l}=0.5)
  ✓ Equation 5 损失函数组件计算正确
  ✓ Equation 4 前向传播实现正确（加性残差形式）
```

### 4.2 关键验证点

| 验证项 | 预期值 | 实际值 | 状态 |
|--------|--------|--------|------|
| 门控 logit 初始化 | 0.0 | 0.0 | ✅ |
| 门控权重初始化 | 0.5 | 0.5 | ✅ |
| gate_penalty (4层) | 2.0 | 2.0 | ✅ |
| 通信量一致性 | FedLoRA = FedSDG | 11,530 = 11,530 | ✅ |
| Equation 4 极端情况 | m=0 仅全局, m=1 全局+私有 | 通过 | ✅ |

---

## 五、使用指南

### 5.1 运行 FedSDG 训练

```bash
python federated_main.py \
    --alg fedsdg \
    --model vit \
    --model_variant pretrained \
    --dataset cifar100 \
    --epochs 50 \
    --lambda1 1e-3 \
    --lambda2 1e-4 \
    --lora_r 8 \
    --lora_alpha 16
```

### 5.2 推荐超参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--lambda1` | 1e-3 ~ 5e-4 | L1 门控稀疏性惩罚，越大越稀疏 |
| `--lambda2` | 1e-4 ~ 1e-3 | L2 私有参数惩罚，越大私有参数越小 |
| `--lora_r` | 8 | LoRA 秩 |
| `--lora_alpha` | 16 | LoRA 缩放因子 |

### 5.3 训练日志解读

训练时会输出损失分解信息：

```
[FedSDG Loss] task=2.3000, gate_penalty=2.0000 (x0.001=0.002000), 
              private_penalty=756.77 (x0.0001=0.075677), total=2.3777
```

**预期行为**：
- `gate_penalty` 应逐渐下降（门控稀疏化）
- `private_penalty` 应保持较小（< 0.1）
- 训练后大部分 `m_{k,l}` 应 < 0.1 或 > 0.9

---

## 六、修改文件清单

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `src/update.py` | 核心修复 | 实现 Equation 5 完整损失函数 |
| `src/models.py` | 修复 | 门控初始化 + Equation 4 前向传播 |
| `src/options.py` | 新增 | `--lambda1`, `--lambda2` 参数 |
| `src/test_fedsdg.py` | 增强 | 添加 Equation 4/5 验证测试 |

---

## 七、总结

本次审计发现并修复了 FedSDG 实现中的 **4 个关键偏差**：

1. **Equation 5 正则化缺失**（严重）→ 已实现完整损失函数
2. **weight_decay 误用**（中等）→ FedSDG 模式下禁用
3. **门控初始化偏差**（中等）→ 修正为 0.0（m=0.5）
4. **前向传播公式偏差**（中等）→ 修正为加性残差形式

修复后的实现完全符合 `FedSDG_Design.md` 设计规范，所有测试通过。

---

*报告生成时间: 2026-01-08*
*审计人: Cascade AI*
