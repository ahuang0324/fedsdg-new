# 项目文档目录

本目录包含了 Federated Learning PyTorch 项目的所有技术文档，按照功能和类型进行了分类组织。

## 📁 目录结构

```
docs/
├── algorithms/                    # 算法设计与实现
│   ├── FedSDG/                   # FedSDG 算法相关
│   │   ├── FedSDG_Design.md                      # 算法设计规范
│   │   ├── FedSDG_Implementation_Report.md       # 实现审计报告
│   │   ├── FedSDG实施技术报告.md                 # 技术实施报告
│   │   └── FedSDG服务端权重聚合算法.md           # 权重聚合算法
│   ├── FedLoRA/                  # FedLoRA 算法相关
│   │   └── FedLoRA 实现文档.md                   # 实现文档
│   └── FedAvg/                   # FedAvg 算法相关
│
├── user_guides/                   # 用户指南
│   ├── data_preprocessing/       # 数据预处理
│   │   ├── CIFAR100_使用指南.md                  # CIFAR100 数据集使用
│   │   ├── 离线数据预处理使用指南.md             # 离线预处理使用
│   │   └── 离线数据预处理实现总结.md             # 离线预处理实现
│   ├── pretrained_models/        # 预训练模型
│   │   ├── 预训练 ViT 模型使用指南.md            # ViT 模型使用
│   │   ├── 预训练权重下载指南.md                 # 权重下载指南
│   │   └── 预训练 ViT 实施总结.md                # ViT 实施总结
│   └── features/                 # 功能特性
│       ├── 模型保存与分析系统使用指南.md         # 模型保存与分析
│       └── 通信量统计功能使用指南.md             # 通信量统计
│
└── technical_reports/             # 技术报告
    ├── bug_reports/              # Bug 报告与修复
    │   ├── FedSDG_BUG_REPORT_AND_FIX.md          # FedSDG Bug 报告
    │   ├── FedSDG_QUICK_FIX_VERIFICATION.md      # FedSDG 快速修复验证
    │   └── FedLoRA 训练准确率下降问题诊断报告.md # FedLoRA 诊断报告
    ├── optimization/             # 优化报告
    │   ├── FedLoRA 代码优化报告.md               # FedLoRA 优化
    │   ├── 通信效率优化实施报告.md               # 通信效率优化
    │   └── 通信量统计功能实施报告.md             # 通信量统计实施
    ├── performance_analysis/     # 性能分析
    │   ├── FedSDG_性能崩溃根因分析与修复报告.md  # FedSDG 性能分析
    │   └── FedAvg与FedLoRA训练时间分析报告.md    # 训练时间对比
    └── infrastructure/           # 基础设施与运维
        └── Git仓库大文件问题诊断与修复报告.md    # Git 仓库管理问题
```

## 📖 文档分类说明

### 1. algorithms/ - 算法设计与实现
包含各种联邦学习算法的设计文档、实现细节和技术规范。

- **FedSDG**: Federated Structure-Decoupled Gating 算法
- **FedLoRA**: Federated Low-Rank Adaptation 算法
- **FedAvg**: Federated Averaging 算法

### 2. user_guides/ - 用户指南
面向用户的使用指南和操作手册。

- **data_preprocessing**: 数据预处理相关指南
- **pretrained_models**: 预训练模型使用指南
- **features**: 系统功能特性使用说明

### 3. technical_reports/ - 技术报告
技术问题分析、优化方案和性能评估报告。

- **bug_reports**: Bug 修复报告和问题诊断
- **optimization**: 代码优化和性能改进
- **performance_analysis**: 性能分析和对比研究
- **infrastructure**: 基础设施、运维和开发环境相关问题

## 🔍 快速导航

### 新手入门
1. [README.md](../README.md) - 项目总体说明
2. [CIFAR100_使用指南.md](user_guides/data_preprocessing/CIFAR100_使用指南.md) - 数据集准备
3. [预训练权重下载指南.md](user_guides/pretrained_models/预训练权重下载指南.md) - 模型准备

### 算法研究
1. [FedSDG_Design.md](algorithms/FedSDG/FedSDG_Design.md) - FedSDG 设计规范
2. [FedLoRA 实现文档.md](algorithms/FedLoRA/FedLoRA%20实现文档.md) - FedLoRA 实现
3. [FedAvg与FedLoRA训练时间分析报告.md](technical_reports/performance_analysis/FedAvg与FedLoRA训练时间分析报告.md) - 算法对比

### 问题排查
1. [FedSDG_BUG_REPORT_AND_FIX.md](technical_reports/bug_reports/FedSDG_BUG_REPORT_AND_FIX.md)
2. [FedLoRA 训练准确率下降问题诊断报告.md](technical_reports/bug_reports/FedLoRA%20训练准确率下降问题诊断报告.md)
3. [Git仓库大文件问题诊断与修复报告.md](technical_reports/infrastructure/Git仓库大文件问题诊断与修复报告.md) - 仓库管理问题

### 性能优化
1. [通信效率优化实施报告.md](technical_reports/optimization/通信效率优化实施报告.md)
2. [FedLoRA 代码优化报告.md](technical_reports/optimization/FedLoRA%20代码优化报告.md)

## 📝 文档维护规范

1. **新增文档**: 根据文档类型放入相应的子目录
2. **命名规范**: 使用清晰描述性的文件名
3. **更新记录**: 在文档中注明最后更新日期
4. **交叉引用**: 使用相对路径链接相关文档

## 📧 联系方式

如有文档相关问题，请提交 Issue 或 Pull Request。


