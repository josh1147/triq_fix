#!/bin/bash

# MMM3地图运行脚本
# 自动清理残留进程并运行训练
# 支持多screen并行运行（只清理triq_sep相关进程）

echo "🚀 启动MMM3地图训练..."

# 1. 清理triq_sep相关的残留进程（不影响qmix_sep等其他配置）
echo "🧹 清理triq_sep残留进程..."
if [ -f "./cleanup_simple.sh" ]; then
    ./cleanup_simple.sh triq_sep_standard
else
    echo "⚠️  cleanup_simple.sh 不存在，跳过清理步骤"
fi

# 2. 等待一下确保清理完成
sleep 2

# 3. 运行三元Q值框架（使用MMM3地图）
echo "🎯 启动MMM3地图训练..."
echo "📍 地图配置: MMM3 (13 agents vs 16 enemies)"
#echo "📍 已启用地形高度特征: obs_terrain_height=True"
echo "📍 已开启路径网格信息: obs_pathing_grid=True"
python src/main.py --config=triq_sep_standard --env-config=sc2 with env_args.map_name=MMM3 env_args.obs_terrain_height=False env_args.obs_pathing_grid=True batch_size_run=16 "$@"

