#!/bin/bash

# QMIX_SEP框架运行脚本
# 自动清理残留进程并运行训练
# 支持多screen并行运行（只清理qmix_sep相关进程）

echo "🚀 启动QMIX_SEP框架..."

# 1. 清理qmix_sep相关的残留进程（不影响triq_sep等其他配置）
echo "🧹 清理qmix_sep残留进程..."
./cleanup_simple.sh qmix_sep

# 2. 等待一下确保清理完成
sleep 2

# 3. 运行QMIX_SEP框架
echo "🎯 启动qmix_sep训练..."
python src/main.py --config=qmix_sep --env-config=sc2 "$@"
