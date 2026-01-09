# New page

## 算法概述

### 核心思想

FedSDG (Federated Structure-Decoupled Gating) 是一个联邦微调框架，将个性化建模为**结构化的、可学习的偏差**，从共享模型中自适应偏离。

**关键创新**：

- **结构解耦**：将模型适应分解为共享组件和客户端特定组件

- **可学习门控**：引入轻量级的块级门控来调节两者的交互

- **层级个性化**：不同层可以表现出不同程度的个性化

- **低通信开销**：只传输共享参数更新，私有参数保持本地

**适用场景**：

- 强数据异构性

- 有限通信预算

- 大型预训练 Transformer 模型

- 需要隐私保护的联邦学习

---

## 问题设置

### 联邦学习场景

- **客户端数量**：K 个客户端

- **本地数据**：客户端 k 拥有数据集 $D_k = {(x_i, y_i)}_{i=1}^{n_k}$，来自分布 $P_k$

- **数据异构性**：$P_k$ 在客户端间非独立同分布（non-IID）

- **通信轮次**：每轮 t 有子集 $S_t \subseteq {1, ..., K}$ 的 M 个客户端参与

- **预训练模型**：L 层 Transformer，冻结的骨干参数 $\theta_{bb}$，任务头参数 $\theta_{head}$

### 参数定义

**全局共享参数** $\theta_g \in \mathbb{R}^{d_g}$：

- 代表共享适应组件

- 跨客户端同步和聚合

- 包含 LoRA 参数和任务头

- 也称为 $\theta_{comm}$（通信参数）

**客户端私有参数** $\theta_{p,k} \in \mathbb{R}^{d_p}$：

- 代表客户端特定适应组件

- 仅在本地存储和更新

- 永不传输到服务器

- 捕获对客户端有用但不一定与全局目标对齐的残差方向

**门控参数**：

- **门控 logit**：$a_{k,l} \in \mathbb{R}$（每个客户端 k、每层 l）

- **门控权重**：$m_{k,l} \in [0, 1]$，通过 sigmoid 函数计算

---

## 核心设计

### 结构解耦参数化

FedSDG 将个性化建模为从共享模型的**结构化自适应偏差**，而非独立模型或硬参数分区。

**设计原则**：

- 客户端应共享共同的表示框架

- 仅在必要时才偏离共享模型

- 偏差应该是结构化的和可控的

### 块级可学习门控

**门控定义（Equation 3）**：

```plaintext
m_{k,l} = σ(a_{k,l})
```

其中：

- $m_{k,l}$：客户端 k 在层 l 的门控权重，范围 [0, 1]

- $a_{k,l}$：可学习的门控 logit，范围 (-∞, +∞)

- $σ(\cdot)$：sigmoid 函数

**门控粒度**：

- 每个 Transformer 块一个门控

- 控制该块中注意力和前馈网络的适应

- 避免细粒度门控带来的不稳定性和过拟合风险

### 残差分解适应

**有效参数（Equation 4）**：

```plaintext
θ̃_{k,l} = θ_{g,l} + m_{k,l} · θ_{p,k,l}
```

其中：

- $\theta_{g,l}$：层 l 的共享适应参数

- $\theta_{p,k,l}$：客户端 k 在层 l 的私有适应参数

- $m_{k,l}$：调节偏差幅度的门控权重

**加性形式含义**：

- 将个性化建模为共享结构的**残差扰动**

- 共享组件 $\theta_{g,l}$ 捕获跨客户端广泛有用的方向

- 私有组件 $\theta_{p,k,l}$ 捕获客户端特定的偏差

- 门控 $m_{k,l}$ 使每层在全局对齐和本地特化之间连续变化

**极端情况**：

- $m_{k,l} = 0$：层 l 仅遵循共享适应（完全全局）

- $m_{k,l} = 1$：应用完整的客户端特定残差（完全个性化）

- $0 < m_{k,l} < 1$：从共享模型部分偏离（混合模式）

---

## 关键公式

### Equation 3: 门控权重

```plaintext
m_{k,l} = σ(a_{k,l})
```

**实现**：

- 在代码中，`a_{k,l}` 对应参数名包含 `lambda_k_logit`

- 通过 `torch.sigmoid(param)` 计算 `m_{k,l}`

### Equation 4: 有效参数

```plaintext
θ̃_{k,l} = θ_{g,l} + m_{k,l} · θ_{p,k,l}
```

**实现**：

- 在前向传播中动态构建

- 共享 LoRA 参数与私有 LoRA 参数的加权组合

### Equation 5: 客户端优化目标（核心）

```plaintext
min_{θ_{p,k}, a_k, θ_g} (1/|B_k|) Σ ℓ(f(x; θ̃_k), y) + λ₁ Σ_{l=1}^L |m_{k,l}| + λ₂ ||θ_{p,k}||²₂
                         └─────────────────────┘   └──────────────┘   └──────────┘
                         Task Loss                 L1 Gate Penalty    L2 Private Penalty
```

**三个组成部分**：

1. **任务损失**：$(1/|B_k|) \sum_{(x,y) \in B_k} \ell(f(x; \tilde{\theta}_k), y)$

   - 标准交叉熵损失

   - 驱动模型拟合本地数据

2. **λ₁ 正则化**：$\lambda_1 \sum_{l=1}^L |m_{k,l}|$

   - L1 惩罚门控参数

   - 鼓励稀疏激活（大部分门控接近 0 或 1）

   - **核心创新**：实现层级选择性个性化

3. **λ₂ 正则化**：$\lambda_2 ||\theta_{p,k}||_2^2$

   - L2 惩罚私有 LoRA 参数

   - 限制残差适应的容量

   - 防止私有分支过拟合和吸收全局有用信息

---

## 优化过程

### 客户端本地更新

**更新规则（Equations 6-8）**：

```plaintext
θ_g ← θ_g - η_g ∇_{θ_g} L_k         (更新共享参数)
θ_{p,k} ← θ_{p,k} - η_p ∇_{θ_{p,k}} L_k   (更新私有参数)
a_k ← a_k - η_m ∇_{a_k} L_k         (更新门控 logit)
```

**关键点**：

- 三组参数联合优化

- 只有 $\theta_g$ 会被同步到服务器

- $\theta_{p,k}$ 和 $a_k$ 保持本地

- 执行 U 步本地 SGD（对应 E 个本地 epoch）

**梯度计算**：

- 任务损失梯度：通过反向传播自动计算

- L1 门控惩罚梯度：$\lambda_1 \cdot \text{sign}(m_{k,l}) \cdot m_{k,l}(1-m_{k,l})$

- L2 私有惩罚梯度：$2\lambda_2 \cdot \theta_{p,k}$

### 客户端上传

**上传内容（Equation 9）**：

```plaintext
Δθ_g^{(k)} = θ_g^{(k)} - θ_g^{(t)}
```

**通信量**：

- 每轮通信 $2M \cdot d_g$ 个标量（上传 + 下载）

- 私有参数和门控参数**不通信**

### 服务器聚合

**对齐加权（Equation 10）**：

```plaintext
α_k = max(0, <Δθ_g^{(k)}, Δ̄> / (||Δθ_g^{(k)}||₂ · ||Δ̄||₂ + ε))
w_k = α_k / (Σ_{j∈S_t} α_j + ε)
```

其中 $\bar{\Delta} = \frac{1}{M} \sum_{k \in S_t} \Delta\theta_g^{(k)}$

**聚合更新（Equation 11）**：

```plaintext
θ_g^{(t+1)} = θ_g^{(t)} + Σ_{k∈S_t} w_k Δθ_g^{(k)}
```

**聚合策略**：

- 基于余弦相似度的加权

- 抑制与全局共识方向不一致的更新

- 减少强异构性下的冲突和噪声影响

---

## 实现细节

### LoRA 实例化

**共享参数 $\theta_g$**：

- 插入到每个 Transformer 块的：

  - 注意力输出投影

  - 前馈网络输出投影

- 包含所有 LoRA 参数 + 任务头

**私有参数 $\theta_{p,k}$**：

- 与共享 LoRA 相同的结构

- 本地维护的额外 LoRA 分支

- 确保两者在相同低维子空间操作

**门控参数**：

- 每个块一个标量 gate

- 共 L 个门控参数（L 为 Transformer 层数）

- 控制该块中注意力和前馈的适应

### 超参数设置

**联邦学习参数**：

- 客户端数量：K ∈ {50, 100}

- 每轮采样率：M/K = 0.1

- 本地 epoch：E ∈ {1, 2}

- 数据异构性：Dirichlet α ∈ {0.1, 0.3, 1.0}

**LoRA 配置**：

- 秩：r = 8

- 缩放因子：α = 16

- 冻结预训练骨干

**学习率**：

- 共享参数：η_g = 10⁻³

- 私有参数：η_p = 10⁻³

- 门控参数：η_m ∈ {5×10⁻³, 10⁻²}

**正则化系数**：

- L1 门控惩罚：λ₁ ∈ {10⁻⁴, 5×10⁻⁴, 10⁻³}

- L2 私有惩罚：λ₂ ∈ {10⁻⁴, 10⁻³}

**优化器**：

- Adam，(β₁, β₂) = (0.9, 0.999)

- 梯度裁剪：norm = 1.0

- 无学习率预热

### 初始化

**门控 logit 初始化**：

```plaintext
a_{k,l} = 0  →  m_{k,l} = σ(0) = 0.5
```

**含义**：

- 训练开始时，共享和私有组件等权重

- 无偏向的起点

- 让优化过程自然地学习个性化程度

**随机种子固定**：

- 客户端采样

- 数据分区

- 参数初始化

- 确保可重复性和公平比较

---

## 参数命名约定

### 代码中的参数名称

**门控参数**：

- 名称包含：`lambda_k_logit`

- 类型：标量张量

- 范围：(-∞, +∞)

- 通过 `torch.sigmoid()` 转换为门控权重

**私有 LoRA 参数**：

- 名称包含：`_private` 后缀

- 示例：

  - `encoder.blocks.0.attn.qkv.lora_A_private`

  - `encoder.blocks.0.attn.qkv.lora_B_private`

  - `encoder.blocks.0.mlp.fc1.lora_A_private`

  - `encoder.blocks.0.mlp.fc1.lora_B_private`

**共享 LoRA 参数**：

- 无 `_private` 后缀的 LoRA 参数

- 示例：

  - `encoder.blocks.0.attn.qkv.lora_A`

  - `encoder.blocks.0.attn.qkv.lora_B`

**任务头**：

- 通常包含在 $\theta_g$ 中

- 可选择性地移到 $\theta_{p,k}$ 以实现任务特定个性化

---

## 正则化机制

### λ₁：L1 门控稀疏性惩罚

**作用**：

```plaintext
λ₁ Σ_{l=1}^L |m_{k,l}|
```

**目的**：

- 鼓励门控权重接近 0 或 1（稀疏激活）

- 让大部分层依赖全局知识

- 仅少数关键层激活私有知识

- 实现**选择性动态门控**的核心机制

**实现**：

```python
gate_penalty = 0.0
for name, param in model.named_parameters():
    if 'lambda_k_logit' in name:
        m_k = torch.sigmoid(param)
        gate_penalty += torch.sum(torch.abs(m_k))

loss += lambda1 * gate_penalty
```

**预期效果**：

- 训练后：50%-80% 的门控 m_{k,l} < 0.1（接近关闭）

- 少数关键层 m_{k,l} > 0.9（接近完全激活）

- 不应出现大量门控停留在 0.5 附近（密集模式）

### λ₂：L2 私有参数正则化

**作用**：

```plaintext
λ₂ ||θ_{p,k}||²₂
```

**目的**：

- 限制私有 LoRA 参数的模长

- 防止私有分支过拟合本地数据

- 确保私有分支只学习必要的客户端特定偏差

- 防止私有参数"抢占"全局参数的学习能力

**实现**：

```python
private_penalty = 0.0
for name, param in model.named_parameters():
    if '_private' in name:
        private_penalty += torch.sum(param ** 2)

loss += lambda2 * private_penalty
```

**预期效果**：

- 私有参数模长适中：||θ_{p,k}|| ≈ 0.1 × ||θ_g||

- 避免私有参数无限制增长

- 保持全局-私有的功能分离

### 正则化强度关系

**典型设置**：

- λ₁ ≫ λ₂（L1 惩罚通常更强）

- 原因：门控稀疏性是核心目标，而私有参数约束是辅助手段

**错误理解**：

- ❌ 优化器的 `weight_decay` ≠ 论文的 λ₂

- ❌ `weight_decay` 作用于所有参数

- ✅ λ₂ **仅作用于私有参数**

---

## 关键 Takeaways

### Takeaway 3.1（结构解耦）

Equation 4 定义了纯共享和完全个性化适应之间的**连续且结构化的插值**：

- $m_{k,l} = 0$：层 l 仅遵循共享适应

- $m_{k,l} = 1$：应用完整的客户端特定残差

- $0 < m_{k,l} < 1$：从共享模型部分偏离

- 初始化为 $m_{k,l} = 0.5$，提供无偏起点

**额外客户端存储**：

- $\theta_{p,k}$：与共享 LoRA 相同大小

- L 个标量门控 logit

- 总开销可忽略不计

### Takeaway 3.2（优化协议）

客户端在结构正则化下联合优化三组参数：

- 共享参数 $\theta_g$

- 客户端特定残差 $\theta_{p,k}$

- 门控参数 $a_k$

**仅传输共享更新**：

- 上传：$\Delta\theta_g^{(k)}$

- 下载：$\theta_g^{(t+1)}$

- 私有参数和门控永不传输

**服务器聚合**：

- 使用对齐加权抑制冲突或纯客户端特定方向

- 保护全局模型免受强异构性影响

**优点**：

- 保护隐私

- 限制通信开销

- 强制共享与个性化组件的清晰分离

### Takeaway 3.3（实现细节）

**初始化策略**：

- $a_{k,l} = 0 \rightarrow m_{k,l} = 0.5$（训练开始时）

- 梯度裁剪：norm = 1.0

- 固定随机种子（采样、分区、初始化）

**兼容性**：

- 可作为 FedAvg 的最小修改实现

- 仅同步 $\theta_g$

- $(\theta_{p,k}, a_k)$ 保持本地

- 前向传播中动态构建 $\tilde{\theta}_{k,l}$

**可重复性**：

- 确保观察到的效果归因于设计而非随机性

- 便于与其他方法公平比较

---

## 验证标准

### 代码层面

**必须检查**：

- [ ]  损失函数包含三项：`task_loss + λ₁*gate_penalty + λ₂*private_penalty`

- [ ]  门控惩罚正确计算：`Σ |sigmoid(lambda_k_logit)|`

- [ ]  私有惩罚正确计算：`Σ (param_with_'_private')²`

- [ ]  仅在 `args.alg == 'fedsdg'` 时应用正则化

- [ ]  优化器不使用 `weight_decay`（手动实现正则化）

### 运行时验证

**训练日志检查**：

- 初始值（L=6 层）：

  - `gate_penalty ≈ 3.0`（因为初始 m_{k,l} ≈ 0.5）

  - `private_penalty ≈ 0.0`（因为私有参数初始化为 0）

- 惩罚项数值范围合理：

  - `gate_penalty` 应逐渐下降（稀疏化）

  - `private_penalty` 应保持较小（< 0.1）

**训练后检查**：

- 门控稀疏性：

  ```python
  for name, param in model.named_parameters():
      if 'lambda_k_logit' in name:
          m_k = torch.sigmoid(param)
          print(f"{name}: {m_k.item():.4f}")
  ```

- 预期：大部分 m_{k,l} < 0.1 或 > 0.9，不应聚集在 0.5 附近

---

## 实现清单

### 必需修改

**文件 1：**`src/update.py`

- 添加 λ₁ 和 λ₂ 参数读取

- 移除优化器的 `weight_decay`

- 计算门控 L1 惩罚

- 计算私有 L2 惩罚

- 修改损失为三项之和

**文件 2：**`src/options.py`

- 添加 `--lambda1` 参数

- 添加 `--lambda2` 参数

### 调试建议

**添加打印语句**：

```python
if batch_idx % 10 == 0:
    print(f"task_loss={task_loss.item():.4f}, "
          f"gate_penalty={gate_penalty.item():.4f}, "
          f"private_penalty={private_penalty.item():.4f}, "
          f"total_loss={loss.item():.4f}")
```

**训练后分析**：

```python
# 检查门控分布
gate_values = []
for name, param in model.named_parameters():
    if 'lambda_k_logit' in name:
        gate_values.append(torch.sigmoid(param).item())
print(f"Gate statistics: min={min(gate_values):.4f}, "
      f"max={max(gate_values):.4f}, "
      f"mean={sum(gate_values)/len(gate_values):.4f}")
```