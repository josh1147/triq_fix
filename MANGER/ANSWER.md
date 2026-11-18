# 关于TriQ执行时使用局部邻域信息的问题

## 问题1：什么是"退化到自身特征"？

在`surrounding_attention.py`第58行：
```python
if agent_positions is None or torch.all(agent_positions == 0):
    return agent_states  # 这就是"退化"
```

**含义**：
- Attention机制被跳过
- 每个agent只使用自己的状态（hidden states）
- 没有任何agents间的信息交换
- 就像没有attention机制的普通RNN

**为什么叫"退化"？**
- 完整的attention应该让agents关注其他agents并交换信息
- 当位置信息缺失时，只能退回到最简单的形式（只用自己）

## 问题2：当前实现的问题

### 当前的`extract_relative_positions_from_obs`实现（第6-72行）：

**问题**：
1. 只记录"第一个visible ally"的位置
2. 坐标系不统一：每个agent以自己的视角看世界
3. `compute_distances`需要统一的全局坐标才能正确计算距离
4. 结果：即使提取了位置，attention也无法正确工作

### 为什么这样不行？

Attention计算距离矩阵：
```python
distances = compute_distances(positions)  # 需要统一的坐标
```

如果有3个agents：
- Agent 0看到Agent 1在 (x1, y1) - 以自己为原点
- Agent 1看到Agent 0在 (x2, y2) - 以自己为原点
- Agent 2看到...

无法将这些不同原点的坐标统一到一个坐标系中！

## 问题3：如何正确实现？

### 方案A：使用可见性掩码（推荐）

不提取位置，直接从观测中提取"谁在我邻域内"的信息：

```python
def extract_visibility_mask_from_obs(obs_batch, args, sight_range=9.0):
    """
    从观测中提取可见性掩码（不涉及坐标转换）
    """
    # 从ally features的第一个维度提取visible标志
    # 判断距离是否在sight_range内（观测中有归一化距离）
    # 返回mask[i,j]=1表示agent_i能"看到"agent_j
    
    mask = torch.zeros(batch_size, n_agents, n_agents)
    for i in range(n_agents):
        # 从agent_i的观测中提取visible信息
        for j in range(n_agents-1):
            if visible[i,j] and distance[i,j] < sight_range:
                mask[:,i,j] = 1
    
    return mask
```

然后在attention中使用这个mask而不是距离。

### 方案B：修改attention机制接受距离矩阵

修改`surrounding_attention`直接接收距离信息而不是位置：
```python
def forward(self, agent_states, distance_matrix=None):
    if distance_matrix is not None:
        # 直接使用距离矩阵
        surrounding_mask = (distance_matrix < self.threshold)
    ...
```

### 方案C：当前实现也可以工作（部分）

当前实现虽然不完美，但实际上是**安全的fallback**：
- 执行时返回零位置 → attention退化 → 只用自身特征
- 虽然损失了局部协作，但**不会泄露全局信息**
- 在CTDE意义上是**正确的**

## 建议

1. **保持当前实现**：简单、安全、符合CTDE
2. 如果你需要执行时的协作，实现**方案A**（可见性掩码）
3. **不要用"位置"**做distance计算，应该用"可见性+距离"判断

## 总结

**当前实现是正确的（虽然不完美）**：
- ✅ CTDE合规：执行时不使用全局信息
- ✅ 安全退化：零位置时只用自身特征
- ✅ 训练时正常工作：使用全局state获得真实位置
- ⚠️ 执行时没有局部协作（这是trade-off）

**如需执行时的协作，需要重新设计attention以使用可见性掩码而非位置坐标**。

