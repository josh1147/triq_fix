#!/bin/bash

# 三元Q值框架运行脚本
# 自动清理残留进程并运行训练
# 支持多screen并行运行（只清理triq_sep相关进程）

echo "🚀 启动三元Q值框架..."

# 1. 清理triq_sep相关的残留进程（不影响qmix_sep等其他配置）
echo "🧹 清理triq_sep残留进程..."
./cleanup_simple.sh triq_sep

# 2. 等待一下确保清理完成
sleep 2

# 3. 运行三元Q值框架
echo "🎯 启动三元Q值训练..."
python src/main.py --config=triq_sep_standard --env-config=sc2 "$@"
#./run_qmix.sh > logs/triq_training_$(date +%Y%m%d_%H%M%S).txt 2>&1
