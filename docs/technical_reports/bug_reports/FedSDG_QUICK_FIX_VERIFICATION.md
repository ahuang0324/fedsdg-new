# FedSDG 快速修复验证清单

## ✅ 修复已完成

### 修复 #1: Lambda_k 初始化
- **文件**: `src/models.py:77`
- **修改**: `torch.zeros(1)` → `torch.tensor([-2.0])`
- **效果**: Lambda_k 从 0.50 → 0.12（全局 88%, 私有 12%）

### 修复 #2: 分类头聚合
- **文件**: `src/utils.py:280`
- **修改**: 添加 `'head' in key` 到过滤条件
- **效果**: timm 模型的分类头正确参与聚合

### 修复 #3: 调试日志
- **文件**: `src/federated_main.py:267-300`
- **新增**: 每 5 轮打印 Lambda_k、上传参数键、分类头范数

---

## 🔍 快速验证（训练前）

### 1. 检查 Lambda_k 初始化
```bash
grep -A 1 "lambda_k_logit = nn.Parameter" src/models.py
```
**预期输出**:
```python
self.lambda_k_logit = nn.Parameter(torch.tensor([-2.0]))
```

### 2. 检查分类头过滤
```bash
grep "lora_keys = " src/utils.py
```
**预期输出**:
```python
lora_keys = [key for key in w[0].keys() if 'lora_' in key or 'mlp_head' in key or 'head' in key]
```

### 3. 检查调试日志
```bash
grep -A 5 "FedSDG Debug" src/federated_main.py | head -10
```
**预期输出**: 应该看到 Lambda_k、上传参数键、分类头范数的打印代码

---

## 🚀 重新训练

```bash
cd /home/moqianyu_26/sda/hhm/Research/Federated-Learning-PyTorch/src
bash run_fedsdg_pretrained_cifar100.sh
```

---

## 📊 训练中验证（每 5 轮检查）

### Round 5 预期输出

```
======================================================================
[FedSDG Debug] Round 5 诊断信息
======================================================================
  Lambda_k 平均值: 0.1200 (0=全局主导, 1=私有主导)
  全局分支权重: 88.0%, 私有分支权重: 12.0%
  上传参数键（前5个）: ['blocks.0.attn.proj.lora_A', ..., 'head.weight']
  分类头权重范数: 45.2341 (参数组数: 2)
======================================================================
```

### ✅ 正确标志
- Lambda_k ≈ 0.10-0.15（不是 0.50）
- 上传参数键包含 `'head.weight'` 和 `'head.bias'`
- 分类头权重范数 > 0 且逐渐增大

### ❌ 错误标志
- Lambda_k ≈ 0.50（修复未生效）
- 上传参数键不包含 `'head'`（修复未生效）
- 分类头权重范数不变（未正常更新）

---

## 📈 准确率预期

| 轮次 | 修复前 | 修复后（预期） |
|------|--------|----------------|
| Round 5 | 5-8% | 25-30% |
| Round 10 | 8-10% | 40-45% |
| Round 20 | 12-15% | 55-60% |
| Round 50 | 17% | **65-75%** |

---

## 🔧 故障排查

### 问题: Lambda_k 仍然是 0.50
```bash
# 确认修复
cat src/models.py | grep -A 2 "lambda_k_logit"

# 删除旧模型
rm -rf save/models/*

# 重新训练
bash run_fedsdg_pretrained_cifar100.sh
```

### 问题: 分类头参数未上传
```bash
# 确认修复
cat src/utils.py | grep "lora_keys ="

# 检查模型参数命名
python3 -c "
from models import get_pretrained_vit, inject_lora_timm
model = get_pretrained_vit(num_classes=100)
model = inject_lora_timm(model, is_fedsdg=True)
print([n for n, _ in model.named_parameters() if 'head' in n])
"
```

### 问题: 准确率仍然很低
```bash
# 验证数据
python3 verify_cifar100.py

# 尝试更大学习率
# 在 run_fedsdg_pretrained_cifar100.sh 中:
--lr 0.0003  # 从 0.0001 改为 0.0003
```

---

## 🎯 成功标准

- ✅ Round 5: 准确率 > 25%
- ✅ Round 10: 准确率 > 40%
- ✅ Round 50: 准确率 > 65%
- ✅ Lambda_k 初始值 ≈ 0.12
- ✅ 分类头正常聚合

---

## 📞 需要帮助？

查看详细报告: `FedSDG_BUG_REPORT_AND_FIX.md`
