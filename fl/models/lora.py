#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LoRA (Low-Rank Adaptation) implementation for federated learning.

Supports:
- Standard LoRA for FedLoRA
- Dual-path LoRA for FedSDG (global + private branches)
"""

import torch
from torch import nn
from .vit import ViT


class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) 层实现
    支持两种模式：
    1. 标准 LoRA (is_fedsdg=False): h = Wx + (alpha/r) * B * A * x
    2. FedSDG 双路架构 (is_fedsdg=True): 
       h = Wx + scaling * [(Global_Path) * (1 - lambda_k) + (Private_Path) * lambda_k]
    
    其中:
    - W: 原始冻结权重 (in_features × out_features)
    - A/B: 低秩矩阵（全局分支）
    - A_private/B_private: 低秩矩阵（私有分支，仅 FedSDG）
    - lambda_k: 可学习的门控参数（仅 FedSDG）
    - r: LoRA 秩
    - alpha: 缩放因子
    """
    def __init__(self, original_layer, r=8, lora_alpha=16, is_fedsdg=False):
        super().__init__()
        # 保存原始线性层（冻结）
        self.original_layer = original_layer
        # 冻结原始权重
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # LoRA 参数
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r  # 缩放因子 alpha/r
        self.is_fedsdg = is_fedsdg  # FedSDG 模式标志
        
        # ========== 全局分支（Global Path）==========
        # 低秩分解矩阵 A 和 B（在 FedSDG 中作为全局分支）
        # A: (in_features, r) - 使用正态分布初始化（遵循 LoRA 论文）
        self.lora_A = nn.Parameter(torch.zeros(in_features, r))
        # B: (r, out_features) - 初始化为 0，确保初始时 LoRA 不影响输出
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        
        # 初始化 A 矩阵：使用正态分布，标准差为 1/sqrt(r)
        nn.init.normal_(self.lora_A, mean=0.0, std=1.0/r**0.5)
        # B 矩阵保持为 0（已经是 0）
        
        # ========== FedSDG 专用：私有分支（Private Path）==========
        if self.is_fedsdg:
            # ==================== 私有参数初始化策略 ====================
            # 
            # 设计原则:
            # 1. 私有分支需要能够接收梯度，让门控参数 lambda_k 能够学习
            # 2. 初始时私有分支贡献应该较小，但不能为零
            # 
            # 关键问题：如果 B_private = 0，则：
            # - private_output = x @ A_private @ B_private = 0
            # - ∂Loss/∂lambda_k = ∂Loss/∂output * private_output = 0
            # - 门控参数永远不会更新！
            # 
            # 解决方案：给 B_private 一个小的非零初始化
            # - A_private: 正态分布 N(0, 1/sqrt(r))
            # - B_private: 小的正态分布 N(0, 0.01)
            # 
            # 这确保:
            # - 初始时 private_output 很小但非零
            # - 门控参数可以从一开始就接收梯度
            # - 私有分支可以正常学习
            # ==========================================================
            
            # 私有低秩矩阵（不参与服务器聚合）
            self.lora_A_private = nn.Parameter(torch.zeros(in_features, r))
            self.lora_B_private = nn.Parameter(torch.zeros(r, out_features))
            
            # 初始化 A_private：使用正态分布
            nn.init.normal_(self.lora_A_private, mean=0.0, std=1.0/r**0.5)
            # 初始化 B_private：使用小的正态分布，确保门控参数能接收梯度
            nn.init.normal_(self.lora_B_private, mean=0.0, std=0.01)
            
            # ==================== 门控参数初始化 (FedSDG_Design.md 规范) ====================
            # 门控参数 lambda_k_logit (a_{k,l})：控制全局/私有分支的权重
            # 
            # 根据设计文档 Equation 3: m_{k,l} = σ(a_{k,l})
            # 初始化策略: a_{k,l} = 0 → m_{k,l} = σ(0) = 0.5
            # 
            # 含义:
            # - 训练开始时，共享和私有组件等权重（各占 50%）
            # - 提供无偏向的起点
            # - 让优化过程自然地学习个性化程度
            # 
            # 使用 sigmoid 激活确保 lambda_k ∈ [0, 1]
            # ==========================================================================
            self.lambda_k_logit = nn.Parameter(torch.tensor([0.0]))
        
        # 保存 in_features 和 out_features 供外部访问
        self.in_features = in_features
        self.out_features = out_features
    
    @property
    def weight(self):
        """代理属性：返回原始层的 weight，用于兼容 PyTorch 内部调用"""
        return self.original_layer.weight
    
    @property
    def bias(self):
        """代理属性：返回原始层的 bias，用于兼容 PyTorch 内部调用"""
        return self.original_layer.bias
        
    def forward(self, x):
        """
        前向传播
        
        FedSDG 模式实现 Equation 4 (FedSDG_Design.md):
        θ̃_{k,l} = θ_{g,l} + m_{k,l} · θ_{p,k,l}
        
        这是**加性残差形式**，将个性化建模为共享结构的残差扰动：
        - θ_{g,l}: 共享适应参数（全局 LoRA）
        - θ_{p,k,l}: 客户端特定适应参数（私有 LoRA）
        - m_{k,l}: 门控权重，调节偏差幅度
        
        极端情况：
        - m_{k,l} = 0: 仅使用共享适应（完全全局）
        - m_{k,l} = 1: 共享 + 完整私有残差（完全个性化）
        - 0 < m_{k,l} < 1: 从共享模型部分偏离（混合模式）
        """
        # 原始输出: Wx (+ b)
        original_output = self.original_layer(x)
        
        if self.is_fedsdg:
            # ==================== FedSDG 模式：残差分解适应 (Equation 4) ====================
            # 计算门控参数 m_{k,l} = σ(a_{k,l}) ∈ [0, 1]
            m_k = torch.sigmoid(self.lambda_k_logit)
            
            # 全局分支输出: x @ θ_{g,l}
            global_output = x @ self.lora_A @ self.lora_B
            
            # 私有分支输出: x @ θ_{p,k,l}
            private_output = x @ self.lora_A_private @ self.lora_B_private
            
            # ========== Equation 4: θ̃_{k,l} = θ_{g,l} + m_{k,l} · θ_{p,k,l} ==========
            # 有效 LoRA 输出 = 全局输出 + 门控权重 * 私有输出
            # 这是加性残差形式，而非加权插值
            # 
            # 含义：
            # - 全局分支始终贡献（基础适应）
            # - 私有分支作为残差扰动，由门控调节幅度
            # - m_k=0 时，仅使用全局分支
            # - m_k=1 时，全局 + 完整私有残差
            lora_output = (global_output + m_k * private_output) * self.scaling
            # =========================================================================
        else:
            # ========== 标准 LoRA 模式：单路计算 ==========
            # LoRA 输出: (alpha/r) * x @ A @ B
            # lora_A 和 lora_B 是 nn.Parameter，会自动跟随模型移动到正确设备
            lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling
        
        # 组合输出
        return original_output + lora_output


def inject_lora(model, r=8, lora_alpha=16, train_mlp_head=True, is_fedsdg=False):
    """
    将 LoRA 层注入到 ViT 模型中
    
    手术位置：
    - 针对 ViT 的 6 层 TransformerEncoderLayer
    - 替换每层中的 self_attn.out_proj (注意力输出投影)
    - 替换每层中的 linear2 (FFN 的第二个线性层)
    
    参数:
        model: ViT 模型实例
        r: LoRA 秩，默认 8
        lora_alpha: LoRA 缩放参数，默认 16
        train_mlp_head: 是否训练分类头，默认 True
        is_fedsdg: 是否启用 FedSDG 双路架构，默认 False
    
    返回:
        注入 LoRA 后的模型
    """
    if not isinstance(model, ViT):
        raise ValueError("inject_lora 目前仅支持 ViT 模型")
    
    # 0. 获取模型所在设备（用于确保 LoRA 参数在正确设备上）
    device = next(model.parameters()).device
    
    # 1. 冻结整个模型的所有参数
    model.requires_grad_(False)
    
    # 2. 遍历 TransformerEncoder 的所有层，注入 LoRA
    for layer_idx, encoder_layer in enumerate(model.transformer.layers):
        # 2.1 替换 self_attn.out_proj (注意力输出投影层)
        original_out_proj = encoder_layer.self_attn.out_proj
        lora_out_proj = LoRALayer(original_out_proj, r=r, lora_alpha=lora_alpha, is_fedsdg=is_fedsdg)
        lora_out_proj.to(device)  # 将 LoRA 层移动到模型所在设备
        encoder_layer.self_attn.out_proj = lora_out_proj
        
        # 2.2 替换 linear2 (FFN 的第二个线性层)
        original_linear2 = encoder_layer.linear2
        lora_linear2 = LoRALayer(original_linear2, r=r, lora_alpha=lora_alpha, is_fedsdg=is_fedsdg)
        lora_linear2.to(device)  # 将 LoRA 层移动到模型所在设备
        encoder_layer.linear2 = lora_linear2
        
        mode_str = "FedSDG" if is_fedsdg else "LoRA"
        print(f"  [{mode_str}] 已注入第 {layer_idx} 层: out_proj 和 linear2")
    
    # 3. 可选：开放分类头的梯度更新
    if train_mlp_head:
        for param in model.mlp_head.parameters():
            param.requires_grad = True
        print(f"  [LoRA] mlp_head 参数已解冻用于训练")
    
    # 4. 统计可训练参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [LoRA] 总参数: {total_params:,} | 可训练参数: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # 5. 验证参数冻结：确保只有 LoRA 参数、mlp_head 和 FedSDG 门控参数可训练
    trainable_param_names = [name for name, p in model.named_parameters() if p.requires_grad]
    for name in trainable_param_names:
        if not ('lora_' in name or 'mlp_head' in name or 'lambda_k' in name):
            raise RuntimeError(
                f"参数冻结验证失败：发现非 LoRA/FedSDG 参数 '{name}' 是可训练的！"
                f"这违背了 LoRA 参数高效微调的原则。"
            )
    print(f"  [LoRA] 参数冻结验证通过：仅 LoRA 参数和 mlp_head 可训练")
    
    return model


def inject_lora_timm(model, r=8, lora_alpha=16, train_head=True, is_fedsdg=False):
    """
    为 timm 预训练 ViT 模型注入 LoRA 层
    
    与手写 ViT 的主要区别:
        - timm 使用 'blocks' 而非 'layers'
        - 注意力层命名: blocks[i].attn.proj (而非 self_attn.out_proj)
        - FFN 命名: blocks[i].mlp.fc2 (而非 linear2)
        - 分类头命名: 'head' (而非 'mlp_head')
    
    参数:
        model: timm 创建的 ViT 模型
        r: LoRA 秩
        lora_alpha: LoRA 缩放因子
        train_head: 是否训练分类头
        is_fedsdg: 是否启用 FedSDG 双路架构，默认 False
    
    返回:
        注入 LoRA 后的模型
    """
    mode_str = "FedSDG" if is_fedsdg else "LoRA"
    print("\n" + "="*60)
    print(f"[{mode_str} Injection - timm ViT] 开始注入 {mode_str}...")
    print("="*60)
    
    # 0. 获取模型所在设备
    device = next(model.parameters()).device
    
    # 1. 冻结整个模型
    model.requires_grad_(False)
    print("  [LoRA] 已冻结所有参数")
    
    # 2. 遍历 Transformer blocks，注入 LoRA
    # timm ViT 结构: model.blocks[i].attn.proj 和 model.blocks[i].mlp.fc2
    num_blocks = len(model.blocks)
    
    for block_idx, block in enumerate(model.blocks):
        # 2.1 替换注意力输出投影层 (attn.proj)
        if hasattr(block.attn, 'proj'):
            original_proj = block.attn.proj
            lora_proj = LoRALayer(original_proj, r=r, lora_alpha=lora_alpha, is_fedsdg=is_fedsdg)
            lora_proj.to(device)
            block.attn.proj = lora_proj
            print(f"  [{mode_str}] Block {block_idx}: 已注入 attn.proj")
        
        # 2.2 替换 FFN 第二层 (mlp.fc2)
        if hasattr(block.mlp, 'fc2'):
            original_fc2 = block.mlp.fc2
            lora_fc2 = LoRALayer(original_fc2, r=r, lora_alpha=lora_alpha, is_fedsdg=is_fedsdg)
            lora_fc2.to(device)
            block.mlp.fc2 = lora_fc2
            print(f"  [{mode_str}] Block {block_idx}: 已注入 mlp.fc2")
    
    # 3. 可选：开放分类头的梯度更新
    if train_head and hasattr(model, 'head'):
        for param in model.head.parameters():
            param.requires_grad = True
        print(f"  [{mode_str}] 分类头 'head' 参数已解冻用于训练")
    
    # 4. 统计可训练参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [{mode_str}] 总参数: {total_params:,} | 可训练参数: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # 5. 验证参数冻结
    trainable_param_names = [name for name, p in model.named_parameters() if p.requires_grad]
    for name in trainable_param_names:
        if not ('lora_' in name or 'head' in name or 'lambda_k' in name):
            raise RuntimeError(
                f"参数冻结验证失败：发现非 LoRA/FedSDG 参数 '{name}' 是可训练的！"
            )
    print(f"  [{mode_str}] 参数冻结验证通过：仅 {mode_str} 参数和 head 可训练")
    print("="*60 + "\n")
    
    return model


def get_lora_state_dict(model):
    """
    提取模型中所有 LoRA 相关的参数（包含 'lora_' 关键词的参数）
    用于 FedLoRA 和 FedSDG 的选择性聚合
    
    FedSDG 特殊处理：
    - 排除私有参数（包含 '_private' 的参数）
    - 排除门控参数（包含 'lambda_k' 的参数）
    - 仅上传全局分支参数（lora_A, lora_B）
    
    参数:
        model: 注入了 LoRA 的模型
    
    返回:
        仅包含 LoRA 全局参数的 state_dict（FedSDG 的私有参数被过滤）
    """
    lora_state_dict = {}
    for name, param in model.named_parameters():
        # 检查是否为 LoRA 相关参数或分类头
        if 'lora_' in name or 'mlp_head' in name or 'head' in name:
            # FedSDG 过滤：排除私有参数和门控参数
            if '_private' in name or 'lambda_k' in name:
                # 跳过私有参数，不参与服务器聚合
                continue
            # 添加全局参数到返回字典
            lora_state_dict[name] = param.data.clone()
    return lora_state_dict


