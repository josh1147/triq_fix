# 开启 obs_terrain_height=True 后需要修改的算法

## 总结

当开启 `obs_terrain_height=True` 后，观察空间会增加 **9个高度特征**（`n_obs_height = 9`），这些特征会被添加到移动特征（move_feats）部分。

## 自动适配的部分 ✅

### 1. 观察空间大小获取（自动适配）
- **位置**：`src/envs/starcraft/StarCraft2Env.py` - `get_obs_size()`
- **说明**：环境已经自动计算了包含高度特征的观察空间大小
  ```python
  def get_obs_move_feats_size(self):
      move_feats = self.n_actions_move  # 4
      if self.obs_pathing_grid:
          move_feats += self.n_obs_pathing  # +8
      if self.obs_terrain_height:
          move_feats += self.n_obs_height  # +9
      return move_feats
  ```

### 2. Scheme 初始化（自动适配）
- **位置**：`src/run/run.py` 等运行脚本
- **说明**：通过 `env_info["obs_shape"]` 自动获取正确的观察空间大小，所有算法都会自动使用

### 3. 神经网络输入层（自动适配）
- **位置**：所有 Agent 和 Controller
- **说明**：通过 `scheme["obs"]["vshape"]` 获取输入维度，会自动适配

## 需要手动修改的部分 ⚠️

### 1. **`src/utils/spatial_utils.py` - `build_visibility_and_distance_masks_from_obs()`**

**问题**：硬编码了移动特征大小为4，没有考虑路径网格和高度特征。

**当前代码**（第27行）：
```python
move_feats_size = 4  # n_actions_move
```

**需要修改为**：
```python
# 计算移动特征的实际大小
move_feats_size = 4  # n_actions_move (基础4个方向)
if getattr(args, 'obs_pathing_grid', False):
    move_feats_size += 8  # n_obs_pathing
if getattr(args, 'obs_terrain_height', False):
    move_feats_size += 9  # n_obs_height
```

**影响范围**：
- 所有使用 `build_visibility_and_distance_masks_from_obs()` 的算法
- 特别是 TriQ 相关算法（`src/learners/triq_learner.py`）

### 2. **`src/learners/CDS_QMIX.py`（可选检查）**

**位置**：第38、40、42、44、48、50、52、54行

**说明**：虽然使用了 `args.obs_shape`，但需要确认这个值是否从 `env_info` 正确传递。建议检查：
- `args.obs_shape` 是否等于 `env_info["obs_shape"]`
- 如果不一致，需要修改初始化代码

**需要检查的代码**：
```python
# 检查这些地方是否正确使用了动态的 obs_shape
args.rnn_hidden_dim + args.obs_shape + args.n_actions
```

### 3. **特征提取函数（如果存在硬编码）**

**位置**：`src/utils/spatial_utils.py` - `extract_agent_types()`

**说明**：第160行从 `obs_dim - unit_type_bits` 开始提取类型信息，这个逻辑是正确的（从末尾开始），但需要确认：

- 如果观察结构发生变化，需要验证提取逻辑的准确性

## 具体修改步骤

### 步骤1：修复 `spatial_utils.py`

修改 `build_visibility_and_distance_masks_from_obs()` 函数，使其正确计算移动特征大小：

```python
def build_visibility_and_distance_masks_from_obs(obs_batch, args, sight_range: float = 9.0):
    # ... 前面的代码 ...
    
    # 计算移动特征的实际大小（支持路径网格和高度特征）
    move_feats_size = 4  # n_actions_move (基础4个方向)
    if getattr(args, 'obs_pathing_grid', False):
        move_feats_size += 8  # n_obs_pathing
    if getattr(args, 'obs_terrain_height', False):
        move_feats_size += 9  # n_obs_height
    
    # ... 后续代码保持不变 ...
```

### 步骤2：验证 CDS_QMIX

检查 `src/learners/CDS_QMIX.py` 中 `args.obs_shape` 的来源，确保它正确反映了包含高度特征的观察空间大小。

### 步骤3：测试

1. 开启 `obs_terrain_height: True` 运行一个简单的算法（如 QMIX）
2. 检查观察空间大小是否正确（应该增加9）
3. 测试 TriQ 算法，确保特征提取正常工作

## 影响算法列表

### 必须修改
- ✅ **TriQ 相关算法**（使用 `build_visibility_and_distance_masks_from_obs()`）

### 建议检查
- ⚠️ **CDS_QMIX**（使用了 `args.obs_shape`，需要确认来源）
- ⚠️ **所有使用自定义特征提取的算法**

### 不需要修改（自动适配）
- ✅ **QMIX**
- ✅ **VDN**
- ✅ **IQL**
- ✅ **MADDPG**
- ✅ **所有基于 BasicMAC 的算法**

## 注意事项

1. **向后兼容性**：修改后的代码应该同时支持 `obs_terrain_height=False` 和 `True` 的情况
2. **路径网格特征**：如果同时开启 `obs_pathing_grid=True`，移动特征会增加8+9=17个特征
3. **预训练模型**：如果加载预训练模型，需要确保模型架构与新的观察空间大小匹配

