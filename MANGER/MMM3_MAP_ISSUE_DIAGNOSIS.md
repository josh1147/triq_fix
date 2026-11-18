# MMM3 地图问题诊断和解决方案

## 🔍 问题分析

根据最新日志 `logs/triq_training_20251122_152422.txt`，主要错误是：

1. **StarCraft II 进程崩溃**（段错误 signal 11）
   - 多次出现 `Error signal: '11'`
   - SC2 进程在 `join_game` 阶段崩溃

2. **连接错误**
   - `ConnectionError: Error during join_game: Connection already closed. SC2 probably crashed.`
   - 发生位置：`StarCraft2Env._launch()` → `self._controller.join_game(join)`

## ✅ 已完成的修复

1. **地图文件已复制**
   - ✅ 已将 `MMM3.SC2Map` 复制到 `/root/StarCraftII/Maps/SMAC_Maps/`
   - ✅ 已将 `MMM3.SC2Map` 复制到 `/root/autodl-tmp/StarCraftII/Maps/SMAC_Maps/`

2. **地图注册配置**
   - ✅ 已在 `src/envs/starcraft/smac_maps.py` 中注册 MMM3 地图
   - ✅ 配置信息：
     - `n_agents: 13`
     - `n_enemies: 16`
     - `limit: 200`
     - `a_race: "T"`
     - `b_race: "T"`
     - `unit_type_bits: 3`
     - `map_type: "MMM"`

3. **脚本修复**
   - ✅ 修复了 `run_mymap.sh` 中的参数传递（添加 `with` 关键字）
   - ✅ 修复了 `spatial_utils.py` 中的高度特征支持

## ⚠️ 可能的问题

### 1. 地图配置与实际地图文件不匹配
- **问题**：代码中配置的 13 agents vs 16 enemies 可能与地图文件中的实际单位数量不匹配
- **解决**：需要验证 MMM3.SC2Map 文件中的实际单位配置

### 2. 地图文件格式问题
- **问题**：MMM3.SC2Map 文件可能损坏或格式不正确
- **解决**：需要检查地图文件是否与 MMM2 格式一致

### 3. SC2 版本兼容性
- **问题**：地图文件可能与当前 SC2 版本不兼容
- **解决**：确保使用兼容的 SC2 版本（Base75689）

### 4. 资源限制
- **问题**：13 vs 16 的大规模战斗可能导致资源不足
- **解决**：考虑减少并行运行的实例数

## 🔧 下一步建议

### 1. 验证地图文件
检查 MMM3.SC2Map 文件：
```bash
# 比较文件大小
ls -lh smac_maps/MMM*.SC2Map

# 检查文件是否损坏
file smac_maps/MMM3.SC2Map
```

### 2. 减少并行运行
如果问题持续，尝试减少并行运行的实例数：
```bash
# 在 run_mymap.sh 或配置中减少 batch_size_run
# 例如从 16 减少到 4
```

### 3. 验证配置匹配
确保 MMM3 地图的实际单位配置与代码中的配置一致：
- 13 个友方单位（Marine + Marauder + Medivac）
- 16 个敌方单位

### 4. 测试单个实例
先用单个实例测试，看是否是资源问题：
```bash
# 临时修改配置，设置 batch_size_run=1
python src/main.py --config=triq_sep_standard --env-config=sc2 with env_args.map_name=MMM3 batch_size_run=1
```

## 📋 创建的文件

1. **install_mmm3_map.sh** - 自动安装MMM3地图到SC2目录的脚本
2. **test_mmm3_map.py** - 地图配置测试脚本（需要在有smac环境时运行）

## 🎯 立即可以尝试的解决方案

如果SC2仍然崩溃，可能是地图文件本身的问题。建议：

1. **检查地图文件来源**：确认 MMM3.SC2Map 是否正确生成或下载
2. **对比MMM2**：查看MMM2地图的配置，确保MMM3配置格式一致
3. **简化测试**：先用更少的单位数测试（例如修改为 6 vs 9），看是否崩溃

## 📝 当前状态

✅ **已完成**：
- 地图文件已复制到SC2目录
- 地图已在代码中注册
- 运行脚本已修复

⚠️ **待验证**：
- SC2能否正常加载MMM3地图
- 地图配置是否与实际文件匹配

