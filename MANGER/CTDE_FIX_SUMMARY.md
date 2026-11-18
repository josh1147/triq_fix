# TriQ CTDE修复总结

## 修复内容

此次修复使TriQ算法符合CTDE（集中式训练-分散式执行）原则，确保注意力机制不会在执行时泄露视野外的信息。

## 修改的文件

### 1. `src/utils/spatial_utils.py`
**修改的函数**: `extract_agent_positions`

**改进内容**:
- 新增参数：`state_batch`, `test_mode`, `args`
- **训练模式**：从全局`state`中提取真实位置信息
- **执行模式**：返回全零位置（无法从局部观测提取全局位置）
- 从state中正确解析ally坐标，恢复绝对位置

**关键点**:
```python
# 训练时使用全局state获取位置
ally_state = state_batch[:, :ally_state_size].view(batch_size, n_agents, nf_al)
relative_x = ally_state[:, :, 2]  # 相对X坐标
relative_y = ally_state[:, :, 3]  # 相对Y坐标
positions[:, :, 0] = relative_x * max_dist_x + center_x
positions[:, :, 1] = relative_y * max_dist_y + center_y
```

### 2. `src/controllers/triq_controller.py`
**修改的函数**: `forward`, `_extract_positions`

**改进内容**:
- `forward`方法中传递`test_mode`和`state_batch`给位置提取
- `_extract_positions`方法支持接收`state_batch`和`test_mode`参数
- 正确的CTDE流程：训练用state，执行用obs

**关键代码**:
```python
# CTDE: 训练时使用state，测试时只使用obs
state_batch = ep_batch.get("state")[:, t] if "state" in ep_batch else None
agent_positions = self._extract_positions(ep_batch, t, state_batch, test_mode)
```

### 3. `src/modules/layer/surrounding_attention.py`
**修改的类**: `SurroundingAttention`, `SimpleSurroundingAttention`

**改进内容**:
- 处理零位置情况：当位置全为0时，退化到只使用自身特征
- 检查距离矩阵有效性，防止除零错误
- 确保CTDE执行时注意力不泄露信息

**关键代码**:
```python
# 如果没有位置信息或全部为0，退化处理（CTDE执行模式）
if agent_positions is None or torch.all(agent_positions == 0):
    return agent_states

# 检查是否有有效位置
if torch.all(distances == 0):
    return agent_states
```

### 4. `src/config/algs/triq_sep.yaml`
**新增配置**:
```yaml
extract_agent_positions: True  # 训练时使用全局state，执行时使用零位置
extract_agent_types: True  # 从观测中提取agent类型
```

## CTDE工作原理

### 训练时 (test_mode=False)
1. 从全局`state_batch`中提取真实的agent位置
2. 使用位置信息计算局部邻域（radius=6.0）
3. Attention机制关注距离在6.0范围内的agents
4. 学习局部协调策略

### 执行时 (test_mode=True)
1. 无法访问全局state，位置提取返回全零
2. Attention检测到零位置，退化到只使用自身特征
3. 每个agent只依赖自己的局部观测
4. 完全符合分散式执行要求

## 优势

1. **符合CTDE原则**：训练和执行使用不同的信息源
2. **零泄露保证**：执行时完全无法获得全局信息
3. **平滑退化**：位置信息缺失时自动退化为非注意力版本
4. **正确提取**：修复了之前位置提取错误的问题

## 验证方法

运行训练和测试，确认：
- 训练时无warnings
- 测试时行为正常
- 性能不会因为CTDE合规而显著下降

## 注意事项

- StarCraft环境必须提供state信息
- `args`对象需要包含地图参数（map_x, map_y, max_distance_x, max_distance_y等）
- 如果参数缺失，函数会安全地返回零位置

