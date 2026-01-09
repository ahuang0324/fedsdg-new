# Configuration Files

每个算法一个配置文件，数据集作为内部选项。

## 目录结构

```
configs/
├── fedavg.yaml     # FedAvg 配置
├── fedlora.yaml    # FedLoRA 配置
├── fedsdg.yaml     # FedSDG 配置 (核心方法)
└── README.md
```

## 使用方法

### 基本用法

```bash
# 使用默认数据集 (cifar100)
python main.py --config configs/fedsdg.yaml

# 指定数据集 (覆盖配置文件)
python main.py --config configs/fedsdg.yaml --dataset cifar

# 覆盖其他参数
python main.py --config configs/fedsdg.yaml --epochs 50 --lr 0.0005
```

### 优先级

命令行参数 > 配置文件 > 默认值

### 快速启动

```bash
# FedAvg
python main.py --config configs/fedavg.yaml

# FedLoRA  
python main.py --config configs/fedlora.yaml

# FedSDG (推荐)
python main.py --config configs/fedsdg.yaml
```

## 配置文件结构

```yaml
algorithm: fedsdg           # 算法名称

dataset: cifar100           # 默认数据集
datasets:                   # 数据集预设
  cifar100:
    num_classes: 100
    model: vit
    ...

lora:                       # LoRA 参数
  r: 8
  alpha: 16
  
fedsdg:                     # FedSDG 特有参数
  lambda1: 0.01
  lambda2: 0.001
  ...

federated:                  # 联邦学习参数
  num_users: 100
  frac: 0.1
  ...

training:                   # 训练参数
  epochs: 100
  lr: 0.001
  ...
```

## FedSDG 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lambda1` | 0.01 | 门控稀疏性惩罚 (L1 on m_k) |
| `lambda2` | 0.001 | 私有参数正则化 (L2 on θ_private) |
| `lr_gate` | 0.01 | 门控参数学习率 |
| `gate_penalty_type` | unilateral | 惩罚类型 |
| `server_agg_method` | alignment | 聚合方法 |
| `grad_clip` | 1.0 | 梯度裁剪 |

## 扩展配置

如需添加新算法，创建新的配置文件并遵循相同结构即可。
