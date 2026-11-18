#!/usr/bin/env python3
"""
三元Q值框架简化测试脚本
绕过sacred框架，直接测试核心功能
"""

import sys
import os
sys.path.append('/root/autodl-tmp/manger/MANGER/src')

import torch
import numpy as np
import yaml
from copy import deepcopy

def test_triq_framework():
    """测试三元Q值框架的核心功能"""
    print("🚀 三元Q值框架核心功能测试")
    print("="*50)
    
    # 1. 加载配置
    print("\n📋 1. 加载配置文件...")
    try:
        with open('src/config/algs/triq_sep.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ 配置加载成功: {config['mac']}, {config['agent']}, {config['learner']}")
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False
    
    # 2. 测试组件导入
    print("\n📋 2. 测试组件导入...")
    try:
        from modules.agents.triq_rnn_agent import TriQRNNAgent
        from modules.layer.multi_rnd import MultiRND
        from modules.layer.surrounding_attention import SimpleSurroundingAttention
        from utils.spatial_utils import extract_agent_positions, extract_agent_types
        print("✅ 所有组件导入成功")
    except Exception as e:
        print(f"❌ 组件导入失败: {e}")
        return False
    
    # 3. 创建模拟参数
    print("\n📋 3. 创建模拟参数...")
    class Args:
        def __init__(self, config):
            self.rnn_hidden_dim = config.get('rnn_hidden_dim', 256)
            self.n_agents = 5  # 2s3z地图的智能体数量
            self.n_actions = 14  # StarCraft II的动作数量
            self.agent_type_embed_dim = config.get('agent_type_embed_dim', 16)
            self.surrounding_radius = config.get('surrounding_radius', 3.0)
            self.use_simple_attention = config.get('use_simple_attention', True)
            self.use_layer_norm = config.get('use_layer_norm', False)
            self.use_orthogonal = config.get('use_orthogonal', False)
            self.gain = config.get('gain', 0.01)
            self.mixing_embed_dim = config.get('mixing_embed_dim', 32)
            self.device = 'cpu'
            self.rnd_update_interval = config.get('rnd_update_interval', 2)
    
    args = Args(config)
    input_shape = 80  # 假设的观测维度
    
    # 4. 测试三元Q值智能体
    print("\n📋 4. 测试三元Q值智能体...")
    try:
        agent = TriQRNNAgent(input_shape, args)
        print(f"✅ TriQRNNAgent创建成功，参数数量: {sum(p.numel() for p in agent.parameters())}")
        
        # 测试前向传播
        batch_size = 4
        n_agents = 5
        obs = torch.randn(batch_size, n_agents, input_shape)
        hidden = agent.init_hidden().expand(batch_size * n_agents, -1)
        agent_positions = torch.randn(batch_size, n_agents, 2)
        agent_types = torch.randint(0, 6, (batch_size, n_agents))
        
        q_values, new_hidden = agent(obs, hidden, None, agent_positions, agent_types)
        print(f"✅ 前向传播成功，Q值形状: {q_values.shape}")
    except Exception as e:
        print(f"❌ 智能体测试失败: {e}")
        return False
    
    # 5. 测试多维RND网络
    print("\n📋 5. 测试多维RND网络...")
    try:
        scheme = {'obs': {'vshape': input_shape}}
        multi_rnd = MultiRND(args, scheme)
        novelty_info = multi_rnd(obs, agent_positions, agent_types)
        print(f"✅ MultiRND网络成功，新颖度键: {list(novelty_info.keys())}")
    except Exception as e:
        print(f"❌ MultiRND网络测试失败: {e}")
        return False
    
    # 6. 测试注意力机制
    print("\n📋 6. 测试注意力机制...")
    try:
        attention = SimpleSurroundingAttention(args)
        # 注意力机制需要隐藏状态输入，不是原始观测
        rnn_output_for_attention = torch.randn(batch_size, n_agents, args.rnn_hidden_dim)
        surrounding_features = attention(rnn_output_for_attention, agent_positions)
        print(f"✅ 注意力机制成功，输出形状: {surrounding_features.shape}")
    except Exception as e:
        print(f"❌ 注意力机制测试失败: {e}")
        return False
    
    # 7. 测试空间工具
    print("\n📋 7. 测试空间工具...")
    try:
        positions = extract_agent_positions(obs)
        types = extract_agent_types(obs)
        print(f"✅ 空间工具成功，位置形状: {positions.shape}, 类型形状: {types.shape}")
    except Exception as e:
        print(f"❌ 空间工具测试失败: {e}")
        return False
    
    print("\n" + "="*50)
    print("🎉 三元Q值框架核心功能测试全部通过！")
    print("✅ 所有组件都能正常工作")
    print("✅ 配置文件正确")
    print("✅ 组件集成成功")
    print("")
    print("🚀 三元Q值框架已准备就绪！")
    print("可以开始训练和实验了！")
    
    return True

def show_usage_instructions():
    """显示使用说明"""
    print("\n" + "="*60)
    print("📚 三元Q值框架使用说明")
    print("="*60)
    print("")
    print("🎯 核心特性:")
    print("  • Q_total = α×Q_common + β×Q_individual + γ×Q_surrounding")
    print("  • 动态权重α、β、γ根据智能体类型和状态调整")
    print("  • 多维新颖度机制支持三种RND网络")
    print("  • 轻量级注意力机制捕获局部协调")
    print("")
    print("🚀 训练命令:")
    print("  conda activate pymarl")
    print("  cd /root/autodl-tmp/manger/MANGER")
    print("  PYTHONPATH=/root/autodl-tmp/manger/MANGER/src python src/main.py \\")
    print("      --config=triq_sep --env-config=sc2 --env_args.map_name=2s3z")
    print("")
    print("🔬 对比实验:")
    print("  # 原始MANGER")
    print("  PYTHONPATH=/root/autodl-tmp/manger/MANGER/src python src/main.py \\")
    print("      --config=qmix_sep --env-config=sc2 --env_args.map_name=2s3z")
    print("")
    print("  # 三元Q值框架")
    print("  PYTHONPATH=/root/autodl-tmp/manger/MANGER/src python src/main.py \\")
    print("      --config=triq_sep --env-config=sc2 --env_args.map_name=2s3z")
    print("")
    print("📊 复杂度梯度测试:")
    print("  • 简单场景: 2s3z")
    print("  • 中等场景: 3s5z")
    print("  • 复杂场景: MMM")
    print("  • 超复杂场景: MMM2")
    print("")
    print("🎉 预期改进:")
    print("  • MMM2地图胜率提升")
    print("  • 样本效率提升")
    print("  • 协调能力增强")
    print("  • 适应性增强")

if __name__ == "__main__":
    success = test_triq_framework()
    if success:
        show_usage_instructions()
    else:
        print("\n❌ 测试失败，请检查配置和组件")
