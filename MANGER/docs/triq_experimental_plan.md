# TriQ算法实验计划

## 📋 实验概述

本文档详细描述TriQ算法的实验设计，参考AAAI 2025论文"Novelty-Guided Data Reuse for Efficient and Diversified Multi-Agent Reinforcement Learning"的实验方法论。

### 实验目标

1. **性能验证**：验证TriQ相对于QMIX_SEP和其他baseline的性能提升
2. **消融研究**：分析TriQ各组件（Q_common, Q_individual, Q_surrounding, 多维新颖度）的贡献
3. **复杂度梯度**：验证算法在不同复杂度环境下的鲁棒性
4. **可扩展性**：验证算法在更大规模场景下的有效性

---

## 🎯 实验一：主性能对比实验

### 1.1 实验设置

#### 基准算法
- **TriQ** (我们的方法)
- **QMIX_SEP** (AAAI 2025论文方法)
- **QMIX** (经典baseline)
- **IQL** (独立Q学习)
- **VDN** (价值分解网络)

#### 测试环境
| 环境名称 | 地图 | 我方智能体 | 敌方智能体 | 难度 | 特点 |
|---------|------|----------|----------|------|------|
| MMM2_3 | MMM2 | 3 Marines + 3 Marauders | 3 Marines + 3 Marauders | 3 | 中等规模对称对战 |
| MMM2_5 | MMM2 | 3 Marines + 3 Marauders | 3 Marines + 3 Marauders | 5 | 中等规模对称对战(困难) |
| MMM | MMM | 4 Marines + 5 Marauders | 4 Marines + 5 Marauders | 5 | 大规模对称对战 |
| MMM_7 | MMM | 4 Marines + 5 Marauders | 4 Marines + 5 Marauders | 7 | 大规模对称对战(非常困难) |

#### 实验参数
```yaml
# 训练参数（所有算法统一）
t_max: 18200000          # 总时间步
batch_size: 512          # 批次大小
buffer_size: 50000       # 回放缓冲区
lr: 0.0005              # 学习率
gamma: 0.999             # 折扣因子
epsilon_start: 1.0       # 起始探索率
epsilon_finish: 0.005    # 结束探索率
epsilon_anneal_time: 2000000  # 探索衰减时间

# 评估参数
test_interval: 2000     # 每2000步评估一次
test_nepisode: 20        # 每次评估20个episode
evaluation_episodes: 200  # 最终评估200个episode
```

### 1.2 评估指标

#### 主要指标
1. **胜率 (Win Rate)**: 平均获胜概率
2. **样本效率 (Sample Efficiency)**: 达到50%胜率所需时间步
3. **最终性能 (Final Performance)**: 最后200个episode的平均胜率

#### 次要指标
4. **训练稳定性**: 胜率曲线的方差
5. **探索效率**: 遇到新状态的比例
6. **协调质量**: 协作动作的匹配度

#### 统计显著性
- 每种配置运行**5次独立实验**（不同随机种子）
- 报告平均值±标准差
- 进行t检验确定统计显著性（p < 0.05）

### 1.3 实验设计

#### 实验矩阵
```
算法     ×  环境配置     ×  重复次数 = 总实验数
5个算法  ×  4个环境配置  ×  5次     = 100次实验

环境配置说明：
- MMM2_3: MMM2地图, difficulty=3
- MMM2_5: MMM2地图, difficulty=5
- MMM_5: MMM地图, difficulty=5
- MMM_7: MMM地图, difficulty=7
```

#### 运行计划
```bash
# 示例：TriQ在MMM2_3环境下的5次运行
for seed in {1..5}; do
    python src/main.py \
        --config=triq_sep \
        --env-config=sc2 \
        --env_args.map_name=MMM2 \
        --env_args.difficulty=3 \
        --seed=$seed \
        --name=triq_mmm2_3_seed_$seed
done
```

### 1.4 预期结果

#### 假设
- **H1**: TriQ在所有测试环境下胜率显著高于QMIX_SEP
- **H2**: TriQ样本效率提升至少20%
- **H3**: TriQ在复杂环境（MMM_7）下优势更明显

#### 预期数据表格
| 环境 | 算法 | 最终胜率(%) | 样本效率(10^6步) | 稳定性 |
|------|------|------------|-----------------|--------|
| MMM2_3 | IQL | X±Y | Z | W |
| MMM2_3 | QMIX | X±Y | Z | W |
| MMM2_3 | QMIX_SEP | X±Y | Z | W |
| MMM2_3 | **TriQ** | **X±Y** | **Z** | **W** |
| ... | ... | ... | ... | ... |

---

## 🔬 实验二：消融研究 (Ablation Study)

### 2.1 研究目标

验证TriQ各组件的必要性和贡献：
- Q_common（全局协调）
- Q_individual（个体策略）
- Q_surrounding（局部协调）
- 动态权重机制
- 多维新颖度机制

### 2.2 消融变体

```yaml
# 变体1: TriQ-NoCommon (移除全局协调Q值)
remove_q_common: true
α_weight: 0.0

# 变体2: TriQ-NoIndividual (移除个体策略Q值)
remove_q_individual: true
β_weight: 0.0

# 变体3: TriQ-NoSurrounding (移除局部协调Q值)
remove_q_surrounding: true
γ_weight: 0.0

# 变体4: TriQ-FixedWeights (固定权重α=β=γ=1/3)
use_dynamic_weights: false
fixed_weights: [0.333, 0.333, 0.334]

# 变体5: TriQ-SingleRND (单一RND网络)
use_multi_rnd: false
use_single_rnd: true

# 变体6: TriQ-NoNovelty (无新颖度引导)
use_novelty_guided_training: false

# 变体7: TriQ-NoAttention (无注意力机制)
use_attention: false
```

### 2.3 实验设计

- **环境**: 选择MMM2_5（中等复杂度，能体现效果）
- **每种变体**: 运行3次（考虑计算资源）
- **评估**: 使用相同的评估指标

### 2.4 预期结果

```markdown
## 消融实验结果

| 变体 | 移除组件 | 性能损失 | 关键发现 |
|------|---------|---------|---------|
| TriQ-Full | 无 | 0% | baseline |
| TriQ-NoCommon | Q_common | X% | 验证全局协调重要性 |
| TriQ-NoIndividual | Q_individual | Y% | 验证个体策略重要性 |
| TriQ-NoSurrounding | Q_surrounding | Z% | 验证局部协调重要性 |
| TriQ-FixedWeights | 动态权重 | W% | 验证动态权重有效性 |
| TriQ-SingleRND | 多RND | V% | 验证多维新颖度必要 |
| TriQ-NoNovelty | 新颖度引导 | U% | 验证样本重用效果 |
| TriQ-NoAttention | 注意力 | T% | 验证注意力机制贡献 |
```

---

## 📊 实验三：复杂度梯度测试

### 3.1 实验设计

测试TriQ在不同复杂度环境下的鲁棒性：

#### 智能体数量梯度
```yaml
scenarios:
  - name: "small_scale"
    map: "3m"
    friendly_agents: 3
    enemy_agents: 3
    description: "小规模 (3 Marines vs 3 Marines)"
    
  - name: "medium_scale"
    map: "MMM2"
    friendly_agents: 6 (3 Marines + 3 Marauders)
    enemy_agents: 6 (3 Marines + 3 Marauders)
    description: "中规模 (3 Marines + 3 Marauders vs 3 Marines + 3 Marauders)"
    
  - name: "large_scale"
    map: "MMM"
    friendly_agents: 9 (4 Marines + 5 Marauders)
    enemy_agents: 9 (4 Marines + 5 Marauders)
    description: "大规模 (4 Marines + 5 Marauders vs 4 Marines + 5 Marauders)"
    
  - name: "extra_large_scale"
    map: "8m_vs_9m"
    friendly_agents: 8
    enemy_agents: 9
    description: "超大规模 (8 Marines vs 9 Marines)"
```

#### 地图复杂度梯度
```yaml
complexity_levels:
  - difficulty: 1
    description: "简单 (AI主动送死)"
  
  - difficulty: 3
    description: "中等 (标准AI)"
  
  - difficulty: 5
    description: "困难 (AI具备基本策略)"
  
  - difficulty: 7
    description: "非常困难 (AI策略优化)"
```

### 3.2 评估指标

关注算法在不同复杂度下的**性能下降率**：
```
性能下降率 = (最高性能 - 当前性能) / 最高性能 × 100%
```

理想情况：TriQ应该展现**最小的性能下降率**。

---

## 🔄 实验四：新颖度机制验证

### 4.1 实验目标

验证TriQ的多维新颖度机制的实际效果：
1. 多维新颖度是否提供了更多样化的探索
2. 新颖度引导的训练是否提高了样本效率
3. RND损失作为辅助目标是否稳定了训练

### 4.2 实验设计

#### 4.2.1 新颖度追踪
追踪训练过程中的新颖度变化：
```python
# 记录以下指标
novelty_metrics = {
    'common_novelty': [],      # 全局协调新颖度
    'individual_novelty': [],   # 个体策略新颖度
    'surrounding_novelty': [],  # 局部协调新颖度
    'fused_novelty': [],       # 融合新颖度
    'update_frequency': [],     # 新颖度引导的更新频率
}
```

#### 4.2.2 样本重用分析
```python
# 分析数据重用效果
reuse_metrics = {
    'reuse_ratio': [],          # 数据重用比例
    'reuse_quality': [],        # 重用数据的TD误差
    'efficiency_gain': [],      # 效率提升
}
```

### 4.3 可视化需求

1. **新颖度曲线**: 三种新颖度随时间的变化
2. **重用热图**: 哪些样本被重用，重用频率
3. **探索轨迹**: 智能体在状态空间的探索轨迹

---

## 📈 实验五：动态权重分析

### 5.1 实验目标

验证TriQ的动态权重机制：
- 权重的变化是否符合智能体类型的特性
- 在不同博弈阶段，权重是否自适应调整

### 5.2 分析内容

#### 5.2.1 权重演化
```python
# 追踪权重变化
weight_stats = {
    'alpha_history': [],      # α（全局协调）权重历史
    'beta_history': [],        # β（个体策略）权重历史
    'gamma_history': [],        # γ（局部协调）权重历史
    'weight_variance': [],     # 权重方差（稳定性）
    'agent_type_weights': {}, # 不同类型智能体的权重
}
```

#### 5.2.2 博弈阶段分析
```python
# 分析不同阶段的权重
phases = {
    'early_game': {'alpha': X, 'beta': Y, 'gamma': Z},
    'mid_game': {'alpha': X, 'beta': Y, 'gamma': Z},
    'late_game': {'alpha': X, 'beta': Y, 'gamma': Z},
}
```

### 5.3 验证实验

**对照实验**：固定权重 vs 动态权重
```yaml
experiments:
  - name: "TriQ-DynamicWeights"
    use_dynamic_weights: true
    weight_predictor: "learned"
    
  - name: "TriQ-FixedWeights"
    use_dynamic_weights: false
    fixed_weights: [0.5, 0.3, 0.2]  # 经验最优值
```

---

## 🔍 实验六：注意力机制分析

### 6.1 实验目标

验证局部协调注意力机制的有效性：
- 注意力权重是否正确聚焦到邻近智能体
- 局部协调是否提升了团队协作

### 6.2 可视化需求

#### 6.2.1 注意力热图
```
对于每个智能体i，可视化它对其他智能体的注意力权重
AttentionMatrix[i, j] = 权重(i关注j)
```

#### 6.2.2 邻域分析
```
分析r_surrounding = 6.0的设置是否合理
- 邻域大小是否合适
- 是否应该自适应调整
```

### 6.3 对照实验

```yaml
experiments:
  - name: "TriQ-Attention"
    use_surrounding_attention: true
    surrounding_radius: 6.0
    
  - name: "TriQ-NoAttention"
    use_surrounding_attention: false
    
  - name: "TriQ-DifferentRadius"
    surrounding_radius: 3.0  # 更小的邻域
    
  - name: "TriQ-LargerRadius"
    surrounding_radius: 9.0  # 更大的邻域
```

---

## 📐 实验七：收敛性分析

### 7.1 实验目标

从数学角度验证TriQ的收敛性：
- 与QMIX_SEP对比收敛速度
- 与QMIX对比收敛稳定性
- 验证数学推导中的收敛性定理

### 7.2 评估指标

```python
convergence_metrics = {
    'convergence_step': [],      # 达到收敛的时间步
    'convergence_rate': [],      # 收敛速度
    'final_q_values': [],        # 最终Q值
    'td_error_magnitude': [],    # TD误差大小
    'weight_stability': [],      # 权重稳定性
}
```

### 7.3 理论验证

验证数学推导中的关键结论：
1. **单调性**: ∂Q_tot/∂q_triq ≥ 0
2. **最优性**: Q_star = max Q_triq
3. **收敛性**: lim Q_t = Q_star

---

## 📝 实验八：超参数敏感性分析

### 8.1 关键超参数

```yaml
# 需要测试的超参数
hyperparameters:
  - learning_rates: [0.0002, 0.0005, 0.001]
  - rnd_lrs: [0.0002, 0.0005, 0.001]
  - attention_dims: [32, 64, 128]
  - attention_heads: [2, 4, 8]
  - surrounding_radii: [3.0, 6.0, 9.0]
  - alpha_initial: [0.1, 0.3, 0.5]
  - beta_initial: [0.2, 0.4, 0.6]
  - gamma_initial: [0.2, 0.4, 0.6]
```

### 8.2 实验设计

使用网格搜索或贝叶斯优化找到最优超参数组合。

---

## 🗓️ 实验执行计划

### 时间表

| 周次 | 实验内容 | 预计时间 |
|------|---------|---------|
| 1-2周 | 主性能对比实验 | 80 GPU小时 |
| 3周 | 消融研究 | 60 GPU小时 |
| 4周 | 复杂度梯度测试 | 40 GPU小时 |
| 5周 | 新颖度机制验证 | 30 GPU小时 |
| 6周 | 动态权重分析 | 20 GPU小时 |
| 7周 | 注意力机制分析 | 20 GPU小时 |
| 8周 | 收敛性分析 | 20 GPU小时 |
| 9-10周 | 超参数调优 + 重新运行最优配置 | 80 GPU小时 |

**总计**: 约330 GPU小时

### 资源配置

- **GPU**: 单卡或多卡（V100/A100）
- **并行**: 同时运行多个环境的独立实验
- **存储**: 预计需要100GB存储空间（模型checkpoints + logs）

---

## 📊 实验记录与报告

### 数据记录格式

```json
{
  "experiment_id": "exp_MMM2_5_triq_seed_1",
  "algorithm": "TriQ",
  "environment": "MMM2_5",
  "seed": 1,
  "config": {
    "learning_rate": 0.0005,
    "rnd_lr": 0.0002,
    ...
  },
  "results": {
    "win_rate": 0.85,
    "sample_efficiency": 5.2e6,
    "final_performance": 0.90,
    ...
  },
  "metrics": {
    "losses": [...],
    "q_values": [...],
    "weights": [...],
    "novelty": [...],
    ...
  }
}
```

### 可视化需求

1. **训练曲线**: Win rate vs Time steps
2. **性能对比**: Bar charts comparing algorithms
3. **消融研究**: Heatmaps showing component contributions
4. **注意力可视化**: Attention weights over time
5. **权重演化**: α, β, γ weights over training

---

## 🎯 成功标准

### 必须满足的标准

1. ✅ **性能提升**: TriQ胜率 > QMIX_SEP胜率 + 5%（所有环境）
2. ✅ **样本效率**: TriQ达到50%胜率的步数 < QMIX_SEP × 0.8
3. ✅ **统计显著性**: p < 0.05（至少5次独立运行）
4. ✅ **鲁棒性**: 在所有测试环境下都优于baseline

### 期望达到的标准

5. ⭐ **消融研究**: 所有组件都有显著贡献（性能损失 > 10%）
6. ⭐ **复杂度鲁棒**: 在extra_large环境下仍保持优势
7. ⭐ **样本效率**: 样本效率提升 > 30%

---

## 🚀 实验启动脚本

### 主性能对比
```bash
# 批量运行脚本
bash scripts/run_main_comparison.sh
```

### 消融研究
```bash
bash scripts/run_ablation_study.sh
```

### 复杂度梯度
```bash
bash scripts/run_complexity_gradient.sh
```

---

## 📚 参考文献

1. AAAI 2025: "Novelty-Guided Data Reuse for Efficient and Diversified Multi-Agent Reinforcement Learning"
2. QMIX: "Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning"
3. RND: "Exploration by Random Network Distillation"

---

## 📝 附录

### A. 当前实验状态

已完成的实验：
- ✅ MMM2_3: TriQ (1次运行)
- ✅ MMM2_5: TriQ (2次运行，已完成)
- ⏳ MMM_5: 进行中

需要补充的实验：
- ⚠️ MMM2_3: 需要再运行4次（达到5次重复）
- ⚠️ MMM2_5: 需要再运行3次（达到5次重复）
- ⚠️ MMM (difficulty 5): 需要完整运行5次
- ⚠️ MMM (difficulty 7): 需要运行5次
- ⚠️ 所有baseline算法：需要完整运行

### B. 实验优先级

**高优先级**（必须完成）：
1. 主性能对比实验（5个算法 × 4个环境 × 5次重复）
2. 消融研究（7个变体 × 3次重复）

**中优先级**（强烈建议）：
3. 复杂度梯度测试
4. 动态权重分析
5. 注意力机制分析

**低优先级**（可选）：
6. 超参数敏感性分析
7. 收敛性理论验证

---

**文档版本**: 1.0  
**最后更新**: 2024-12-19  
**维护者**: TriQ开发团队

