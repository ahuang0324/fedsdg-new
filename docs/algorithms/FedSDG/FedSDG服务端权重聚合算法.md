# FedSDG服务端聚合算法规范文档

## 算法概述

### 核心思想

FedSDG的服务端聚合算法是一种基于对齐度加权的联邦学习聚合方法。该算法的核心目标是：**在数据异构性强的联邦学习场景下，通过计算客户端更新与全局更新方向的对齐度，自适应地调整各客户端更新的权重，从而抑制冲突性更新或纯客户端特定的偏差**。

### 设计动机

在强数据异构性（non-IID）环境下，不同客户端的本地梯度更新可能存在显著差异：

- 某些客户端的更新方向与全局优化方向一致，应获得更高权重

- 某些客户端的更新可能是客户端特定的噪声或残差，应被抑制

- 传统的均匀加权（如FedAvg）无法区分这些情况，导致全局模型性能下降

FedSDG通过**对齐度计算**和**动态权重分配**机制解决这一问题。

### 与传统FedAvg的区别

| 特性 | FedAvg | FedSDG |
| --- | --- | --- |
| 权重分配 | 均匀或基于数据量 | 基于更新对齐度 |
| 异构性处理 | 被动接受所有更新 | 主动抑制冲突更新 |
| 收敛稳定性 | 在强non-IID下不稳定 | 通过对齐加权提升稳定性 |

---

## 数学符号与定义

### 基本符号

- $K$：总客户端数量

- $t$：当前通信轮次

- $S_t$：第 $t$ 轮参与训练的客户端集合，$|S_t| = M$

- $\theta_g^{(t)}$：第 $t$ 轮的全局共享参数

- $\theta_g^{(k)}$：客户端 $k$ 完成本地训练后的共享参数

- $\Delta\theta_g^{(k)}$：客户端 $k$ 上传的参数更新

### 参数更新

客户端 $k$ 上传的更新定义为：

$$\Delta\theta_g^{(k)} = \theta_g^{(k)} - \theta_g^{(t)}$$

这个更新向量包含了客户端在本地数据上执行 $U$ 步SGD后对共享参数的修改。

---

## 对齐度加权机制

### 核心公式

服务端聚合算法的核心是计算每个客户端更新的对齐度权重。完整的权重计算分为两步：

#### 步骤1：计算对齐度分数

对于每个参与客户端 $k \in S_t$，首先计算所有更新的平均方向：

$$\bar{\Delta} = \frac{1}{M} \sum_{k \in S_t} \Delta\theta_g^{(k)}$$

然后计算客户端 $k$ 的对齐度分数：

$$\alpha_k = \max\left(0, \frac{\langle \Delta\theta_g^{(k)}, \bar{\Delta} \rangle}{|\Delta\theta_g^{(k)}|_2 \cdot |\bar{\Delta}|_2 + \epsilon}\right)$$

其中：

- $\langle \cdot, \cdot \rangle$ 表示向量内积

- $|\cdot|_2$ 表示L2范数

- $\epsilon$ 是数值稳定性参数（建议值：$\epsilon = 10^{-8}$）

- $\max(0, \cdot)$ 确保对齐度非负

**物理意义**：

- 分子：客户端更新与平均更新的内积，衡量方向一致性

- 分母：归一化项，使对齐度不受更新幅度影响

- $\alpha_k \in [0, 1]$：接近1表示高度对齐，接近0表示冲突或正交

#### 步骤2：归一化权重

将对齐度分数归一化为聚合权重：

$$w_k = \frac{\alpha_k}{\sum_{j \in S_t} \alpha_j + \epsilon}$$

确保权重满足：

- $w_k \geq 0$

- $\sum_{k \in S_t} w_k = 1$

### 特殊情况处理

**退化情况**：当所有 $\alpha_k = 0$ 时（所有更新与平均方向正交或相反）

- 理论上：这是测度为零的事件

- 实践中：通过 $\epsilon$ 项避免除零

- 回退策略：如果 $\sum \alpha_j < 10^{-6}$，使用均匀权重 $w_k = 1/M$

---

## 服务端聚合公式

### 加权聚合

使用计算得到的权重 $w_k$，更新全局参数：

$$\theta_g^{(t+1)} = \theta_g^{(t)} + \sum_{k \in S_t} w_k \cdot \Delta\theta_g^{(k)}$$

**等价形式**：

$$\theta_g^{(t+1)} = \sum_{k \in S_t} w_k \cdot \theta_g^{(k)} + \left(1 - \sum_{k \in S_t} w_k\right) \cdot \theta_g^{(t)}$$

但由于 $\sum w_k = 1$，简化为：

$$\theta_g^{(t+1)} = \sum_{k \in S_t} w_k \cdot \theta_g^{(k)}$$

### 与FedAvg的对比

- **FedAvg**：$\theta_g^{(t+1)} = \frac{1}{M} \sum_{k \in S_t} \theta_g^{(k)}$（均匀权重）

- **FedSDG**：$\theta_g^{(t+1)} = \sum_{k \in S_t} w_k \cdot \theta_g^{(k)}$（对齐度加权）



---

## 实现细节与最佳实践

### 1. 数值稳定性

#### epsilon参数选择

```python
# 推荐配置
epsilon = 1e-8  # PyTorch默认浮点精度
```

#### 范数计算注意事项

```python
# 避免
norm = torch.sqrt(torch.sum(tensor ** 2))  # 可能数值不稳定

# 推荐
norm = torch.linalg.norm(tensor, ord=2)  # 使用库函数
```

### 2. 内存优化

对于大模型（如Transformer），参数可能很大。优化策略：

```python
# 避免：存储所有更新
all_deltas = [theta_k - theta_t for theta_k in client_params]

# 推荐：流式计算
delta_mean = torch.zeros_like(theta_t)
for theta_k in client_params:
    delta_mean += (theta_k - theta_t) / M
```

### 3. 分布式计算

在多GPU环境下：

```python
# 使用torch.distributed进行高效聚合
import torch.distributed as dist

# 收集所有客户端更新
gathered_deltas = [torch.zeros_like(delta) for _ in range(M)]
dist.all_gather(gathered_deltas, local_delta)

# 在主节点执行聚合
if rank == 0:
    theta_g_new = aggregate(theta_g, gathered_deltas)
```

### 4. 权重监控

记录每轮的权重分布以诊断问题：

```python
# 记录权重统计信息
weight_stats = {
    'mean': np.mean(list(weights.values())),
    'std': np.std(list(weights.values())),
    'min': min(weights.values()),
    'max': max(weights.values()),
    'num_zero': sum(w < 1e-6 for w in weights.values())
}
```

### 5. 边界情况处理

```python
def safe_aggregate(theta_g_t, client_updates, epsilon=1e-8):
    """安全的聚合函数，处理所有边界情况"""
    
    # 情况1：没有客户端
    if len(client_updates) == 0:
        return theta_g_t, {}
    
    # 情况2：只有一个客户端
    if len(client_updates) == 1:
        return list(client_updates.values())[0], {0: 1.0}
    
    # 情况3：正常聚合
    return fedsdg_aggregate(theta_g_t, client_updates, epsilon)
```

---

## 输入输出规范

### 函数签名（Python）

```python
def fedsdg_server_aggregate(
    global_params: Dict[str, torch.Tensor],
    client_params: List[Dict[str, torch.Tensor]],
    epsilon: float = 1e-8,
    return_weights: bool = True
) -> Union[Dict[str, torch.Tensor], 
           Tuple[Dict[str, torch.Tensor], List[float]]]:
    """
    FedSDG服务端聚合算法
    
    参数:
        global_params: 当前全局参数字典
            格式: {'layer_name': tensor}
        client_params: 客户端参数列表
            格式: [{'layer_name': tensor}, ...]
        epsilon: 数值稳定性参数
        return_weights: 是否返回客户端权重
    
    返回:
        如果return_weights=False:
            聚合后的全局参数
        如果return_weights=True:
            (聚合后的全局参数, 客户端权重列表)
    """
```

### 数据类型要求

| 参数 | 类型 | 形状要求 | 值域 |
| --- | --- | --- | --- |
| `global_params` | Dict[str, Tensor] | 每个tensor形状一致 | 任意实数 |
| `client_params` | List[Dict[str, Tensor]] | 与global_params key和shape一致 | 任意实数 |
| `epsilon` | float | 标量 | (0, 0.01] |
| 返回的权重 | List[float] | 长度=len(client_params) | [0, 1]且和为1 |

---

## 数值示例演练

### 示例场景

假设：

- 3个客户端参与本轮训练

- 参数是2维向量（简化演示）

- 当前全局参数：$\theta_g^{(t)} = [1.0, 2.0]$

### 步骤1：接收客户端更新

```plaintext
客户端1上传: θ_g^(1) = [1.5, 2.3]
客户端2上传: θ_g^(2) = [1.6, 2.4]
客户端3上传: θ_g^(3) = [0.5, 1.8]
```

### 步骤2：计算更新向量

```plaintext
Δθ_g^(1) = [1.5, 2.3] - [1.0, 2.0] = [0.5, 0.3]
Δθ_g^(2) = [1.6, 2.4] - [1.0, 2.0] = [0.6, 0.4]
Δθ_g^(3) = [0.5, 1.8] - [1.0, 2.0] = [-0.5, -0.2]
```

### 步骤3：计算平均更新

```plaintext
Δ̄ = (1/3) * ([0.5, 0.3] + [0.6, 0.4] + [-0.5, -0.2])
   = (1/3) * [0.6, 0.5]
   = [0.2, 0.167]
```

### 步骤4：计算对齐度

**客户端1**：

```plaintext
分子 = <[0.5, 0.3], [0.2, 0.167]>
     = 0.5*0.2 + 0.3*0.167
     = 0.1 + 0.0501 = 0.1501

分母 = ||[0.5, 0.3]|| * ||[0.2, 0.167]|| + ε
     = 0.583 * 0.261 + 1e-8
     = 0.152

α_1 = max(0, 0.1501/0.152) = 0.987
```

**客户端2**：

```plaintext
分子 = <[0.6, 0.4], [0.2, 0.167]>
     = 0.6*0.2 + 0.4*0.167
     = 0.120 + 0.0668 = 0.1868

分母 = ||[0.6, 0.4]|| * ||[0.2, 0.167]|| + ε
     = 0.721 * 0.261 + 1e-8
     = 0.188

α_2 = max(0, 0.1868/0.188) = 0.994
```

**客户端3**：

```plaintext
分子 = <[-0.5, -0.2], [0.2, 0.167]>
     = -0.5*0.2 + (-0.2)*0.167
     = -0.1 - 0.0334 = -0.1334

α_3 = max(0, -0.1334/...) = 0  （负值被截断）
```

### 步骤5：归一化权重

```plaintext
Σα = 0.987 + 0.994 + 0 = 1.981

w_1 = 0.987 / 1.981 = 0.498
w_2 = 0.994 / 1.981 = 0.502
w_3 = 0 / 1.981 = 0
```

**解释**：客户端3的更新与平均方向相反，被完全抑制。

### 步骤6：聚合

```plaintext
θ_g^(t+1) = 0.498 * [1.5, 2.3] + 0.502 * [1.6, 2.4] + 0 * [0.5, 1.8]
          = [0.747, 1.146] + [0.803, 1.205] + [0, 0]
          = [1.550, 2.351]
```

### 对比FedAvg

如果使用FedAvg（均匀权重）：

```plaintext
θ_g^(t+1) = (1/3) * ([1.5, 2.3] + [1.6, 2.4] + [0.5, 1.8])
          = (1/3) * [3.6, 6.5]
          = [1.200, 2.167]
```

**观察**：FedSDG的结果 [1.550, 2.351] 更偏向客户端1和2，因为客户端3的更新被识别为冲突并被抑制。

---

## 实现检查清单

在实现FedSDG服务端聚合时，确保满足以下要求：

### 功能性检查

- [ ]  正确计算参数更新向量 $\Delta\theta_g^{(k)}$

- [ ]  正确计算平均更新 $\bar{\Delta}$

- [ ]  正确实现向量内积和L2范数

- [ ]  对齐度公式实现正确（包括max(0,·)）

- [ ]  权重归一化正确（和为1）

- [ ]  加权聚合公式正确

### 数值稳定性检查

- [ ]  使用epsilon避免除零

- [ ]  处理所有对齐度为0的退化情况

- [ ]  使用稳定的范数计算方法

- [ ]  避免浮点数溢出/下溢

### 边界情况检查

- [ ]  处理0个客户端

- [ ]  处理1个客户端

- [ ]  处理所有更新相同的情况

- [ ]  处理所有更新正交的情况

### 性能检查

- [ ]  避免不必要的参数复制

- [ ]  使用原地操作减少内存

- [ ]  支持GPU加速

- [ ]  大模型场景下内存效率

### 可观测性检查

- [ ]  记录每轮的权重分布

- [ ]  记录对齐度统计信息

- [ ]  记录退化情况发生次数

- [ ]  提供调试模式输出中间结果

---

## 常见问题与解决方案

### Q1: 权重全为0或接近0怎么办？

**原因**：所有客户端更新都与平均方向负相关。

**解决方案**：



### Q2: 某些客户端权重异常大

**原因**：可能是数值精度问题或更新幅度过大。

**解决方案**：

- 使用梯度裁剪限制更新幅度

- 检查epsilon是否太小

- 添加权重上限：`w_k = min(w_k, max_weight)`

### Q3: 聚合后参数NaN

**排查步骤**：

1. 检查客户端上传的参数是否包含NaN

2. 检查范数计算是否为0（导致除零）

3. 检查epsilon设置是否合理

4. 添加断言验证中间结果


### Q4: 收敛速度慢

**可能原因**：

- 过度抑制某些客户端

- epsilon设置过大

**调优建议**：

- 降低epsilon到1e-10

- 监控权重熵：`H = -Σ w_k log(w_k)`，低熵表示权重集中

### Q5: 与论文结果不一致

**检查项**：

- 确认epsilon值一致

- 确认是否使用了梯度裁剪

- 确认客户端本地更新步数U一致

- 确认数据分区方法一致（Dirichlet α值）

---



```python
import torch
from typing import Dict, List, Tuple

def fedsdg_aggregate(
    global_params: Dict[str, torch.Tensor],
    client_params_list: List[Dict[str, torch.Tensor]],
    epsilon: float = 1e-8
) -> Tuple[Dict[str, torch.Tensor], List[float]]:
    """
    FedSDG服务端聚合实现
    """
    M = len(client_params_list)
    if M == 0:
        return global_params, []
    
    # 展平参数以便向量运算
    def flatten_params(params_dict):
        return torch.cat([p.flatten() for p in params_dict.values()])
    
    # 步骤1: 计算更新向量
    global_flat = flatten_params(global_params)
    deltas = []
    for client_params in client_params_list:
        client_flat = flatten_params(client_params)
        delta = client_flat - global_flat
        deltas.append(delta)
    
    # 步骤2: 计算平均更新
    delta_mean = torch.stack(deltas).mean(dim=0)
    
    # 步骤3: 计算对齐度
    alphas = []
    for delta in deltas:
        numerator = torch.dot(delta, delta_mean)
        norm_delta = torch.linalg.norm(delta, ord=2)
        norm_mean = torch.linalg.norm(delta_mean, ord=2)
        denominator = norm_delta * norm_mean + epsilon
        
        alpha = max(0.0, (numerator / denominator).item())
        alphas.append(alpha)
    
    # 步骤4: 归一化权重
    sum_alpha = sum(alphas) + epsilon
    weights = [alpha / sum_alpha for alpha in alphas]
    
    # 步骤5: 加权聚合（参数空间）
    aggregated_params = {}
    for key in global_params.keys():
        weighted_sum = sum(
            w * client_params[key] 
            for w, client_params in zip(weights, client_params_list)
        )
        aggregated_params[key] = weighted_sum
    
    return aggregated_params, weights
```

---

## 总结

FedSDG服务端聚合算法的关键要点：

1. **核心机制**：基于余弦相似度的对齐度加权

2. **关键公式**：

   - 对齐度：$\alpha_k = \max(0, \frac{\langle \Delta\theta_g^{(k)}, \bar{\Delta} \rangle}{|\Delta\theta_g^{(k)}|_2 \cdot |\bar{\Delta}|_2 + \epsilon})$

   - 权重：$w_k = \frac{\alpha_k}{\sum_j \alpha_j + \epsilon}$

   - 聚合：$\theta_g^{(t+1)} = \sum_k w_k \cdot \theta_g^{(k)}$

3. **实现要点**：数值稳定性、边界情况、内存效率

4. **监控指标**：权重分布、对齐度统计、退化情况频率

该算法设计简洁、易于实现，并在强non-IID场景下提供了显著的性能改进。