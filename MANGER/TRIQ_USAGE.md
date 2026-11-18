# 三元Q值框架使用指南

## 🚀 快速开始

### 环境准备
```bash
# 激活conda环境
conda activate pymarl

# 确保在正确的目录
cd /root/autodl-tmp/manger/MANGER
```

### 基本使用方法

三元Q值框架的使用方法与原始MANGER完全相同，只需要将配置从`qmix_sep`改为`triq_sep`：

#### StarCraft II 环境
```bash
# 基本训练命令
PYTHONPATH=/root/autodl-tmp/manger/MANGER/src python src/main.py --config=triq_sep --env-config=sc2

# 指定地图训练
PYTHONPATH=/root/autodl-tmp/manger/MANGER/src python src/main.py --config=triq_sep --env-config=sc2 --env_args.map_name=2s3z
PYTHONPATH=/root/autodl-tmp/manger/MANGER/src python src/main.py --config=triq_sep --env-config=sc2 --env_args.map_name=MMM2
```

#### Google Research Football 环境
```bash
PYTHONPATH=/root/autodl-tmp/manger/MANGER/src python src/main.py --config=triq_sep --env-config=gfootball
```

## 🔬 对比实验

### 性能对比
```bash
# 原始MANGER方法
PYTHONPATH=/root/autodl-tmp/manger/MANGER/src python src/main.py --config=qmix_sep --env-config=sc2 --env_args.map_name=2s3z

# 三元Q值框架
PYTHONPATH=/root/autodl-tmp/manger/MANGER/src python src/main.py --config=triq_sep --env-config=sc2 --env_args.map_name=2s3z
```

### 复杂度梯度测试
```bash
# 简单场景：2s3z
PYTHONPATH=/root/autodl-tmp/manger/MANGER/src python src/main.py --config=triq_sep --env-config=sc2 --env_args.map_name=2s3z --t_max=100000

# 中等场景：3s5z  
PYTHONPATH=/root/autodl-tmp/manger/MANGER/src python src/main.py --config=triq_sep --env-config=sc2 --env_args.map_name=3s5z --t_max=200000

# 复杂场景：MMM
PYTHONPATH=/root/autodl-tmp/manger/MANGER/src python src/main.py --config=triq_sep --env-config=sc2 --env_args.map_name=MMM --t_max=500000

# 超复杂场景：MMM2
PYTHONPATH=/root/autodl-tmp/manger/MANGER/src python src/main.py --config=triq_sep --env-config=sc2 --env_args.map_name=MMM2 --t_max=1000000
```

## 🛠️ 高级配置

### 自定义参数
```bash
# 调整三元Q值参数
PYTHONPATH=/root/autodl-tmp/manger/MANGER/src python src/main.py \
    --config=triq_sep \
    --env-config=sc2 \
    --env_args.map_name=2s3z \
    --surrounding_radius=5.0 \
    --attention_dim=128 \
    --attention_heads=8 \
    --agent_type_embed_dim=32
```

### 训练参数调整
```bash
# 调整训练参数
PYTHONPATH=/root/autodl-tmp/manger/MANGER/src python src/main.py \
    --config=triq_sep \
    --env-config=sc2 \
    --env_args.map_name=2s3z \
    --t_max=1000000 \
    --batch_size=128 \
    --lr=0.001 \
    --rnd_lr=0.0005
```

## 📊 快速验证

### 使用测试脚本
```bash
# 运行完整测试
PYTHONPATH=/root/autodl-tmp/manger/MANGER/src python scripts/test_triq.py --test_type both

# 快速测试
PYTHONPATH=/root/autodl-tmp/manger/MANGER/src python scripts/test_triq.py --test_type complexity --quick
```

### 手动验证
```bash
# 验证组件导入
PYTHONPATH=/root/autodl-tmp/manger/MANGER/src python -c "
from modules.agents.triq_rnn_agent import TriQRNNAgent
from modules.layer.multi_rnd import MultiRND
print('✅ 三元Q值框架组件导入成功！')
"
```

## 🎯 核心特性

### 三元Q值架构
- **Q_common**: 全局协调策略（医疗兵高度协调）
- **Q_individual**: 个体策略（陆战队独立行动）
- **Q_surrounding**: 局部协调策略（小群体配合）

### 动态权重系统
- α、β、γ系数根据智能体类型和状态动态调整
- 解决原始MANGER固定权重λ=0.5的问题

### 多维新颖度机制
- 三个独立的RND网络
- 新颖度协调和融合
- 支持多维新颖度引导训练

### 轻量级注意力机制
- 局部邻域特征聚合
- 支持Q_surrounding计算

## 📈 预期改进

1. **MMM2地图胜率提升**: 解决原始MANGER胜率不足千分之一的问题
2. **样本效率提升**: 通过多维新颖度引导的样本重用
3. **协调能力增强**: 局部群体配合能力提升
4. **适应性增强**: 不同智能体类型的差异化策略

## 🔧 故障排除

### 常见问题
1. **导入错误**: 确保使用`PYTHONPATH=/root/autodl-tmp/manger/MANGER/src`
2. **环境问题**: 确保激活了`conda activate pymarl`
3. **配置问题**: 确保配置文件`src/config/algs/triq_sep.yaml`存在

### 调试模式
```bash
# 启用调试输出
PYTHONPATH=/root/autodl-tmp/manger/MANGER/src python src/main.py \
    --config=triq_sep \
    --env-config=sc2 \
    --env_args.map_name=2s3z \
    --t_max=1000 \
    --test_interval=100 \
    --log_interval=50
```

## 📚 文件结构

```
MANGER/
├── src/
│   ├── config/algs/triq_sep.yaml          # 三元Q值配置
│   ├── modules/agents/triq_rnn_agent.py   # 三元智能体
│   ├── learners/triq_learner.py           # 三元学习器
│   ├── controllers/triq_controller.py     # 三元控制器
│   ├── modules/layer/multi_rnd.py         # 多维RND网络
│   ├── modules/layer/surrounding_attention.py # 注意力机制
│   └── utils/spatial_utils.py             # 空间工具
├── scripts/test_triq.py                   # 测试脚本
└── TRIQ_USAGE.md                          # 本使用指南
```

## 🎉 开始使用

现在您可以开始使用三元Q值框架了！建议从简单的2s3z地图开始测试，然后逐步测试更复杂的地图。

```bash
# 立即开始训练
conda activate pymarl
cd /root/autodl-tmp/manger/MANGER
PYTHONPATH=/root/autodl-tmp/manger/MANGER/src python src/main.py --config=triq_sep --env-config=sc2 --env_args.map_name=2s3z
```
