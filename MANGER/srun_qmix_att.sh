#!/bin/bash

# 顺序运行 qmix_att 在 3s5z / 5m6m / MMM2 上的训练，并记录日志

set -e

echo "🚀 启动 QMIX-Att 批量任务..."

# 1. 清理残留
echo "🧹 清理 qmix_att 残留进程..."
./cleanup_simple.sh qmix_att
sleep 2

run_task () {
  MAP_NAME=$1
  LOG_NAME=$2
  echo "🎯 运行 qmix_att @ ${MAP_NAME} ..."
  python src/main.py \
    --config=qmix_att \
    --env-config=sc2 \
    with env_args.map_name=${MAP_NAME} \
    > logs/${LOG_NAME} 2>&1
}

#run_task 3s5z    qmix_att_3s5z_$(date +%Y%m%d_%H%M%S).log
run_task 5m_vs_6m    qmix_att_5m_vs_6m_$(date +%Y%m%d_%H%M%S).log
run_task MMM2    qmix_att_MMM2_$(date +%Y%m%d_%H%M%S).log

echo "✅ 全部任务完成，日志已写入 logs/ 目录。"

