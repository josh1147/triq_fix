#!/bin/bash

# 简单清理脚本
# 支持按配置选择性清理，避免多个screen并行运行时互相干扰
# 用法: ./cleanup_simple.sh [config_name]
#   如果指定config_name，只清理该配置的进程
#   如果不指定，清理所有python src/main.py进程（向后兼容）

CONFIG_NAME="$1"

echo "🧹 清理残留进程..."

if [ -z "$CONFIG_NAME" ]; then
    # 向后兼容：如果没有指定配置，清理所有python进程
    # 清理Python训练进程
    python_pids=$(ps aux | grep 'python src/main.py' | grep -v grep | awk '{print $2}')
    if [ ! -z "$python_pids" ]; then
        echo "发现Python进程: $python_pids"
        pkill -f "python src/main.py"
        sleep 1
        # 强制杀死残留进程
        remaining_pids=$(ps aux | grep 'python src/main.py' | grep -v grep | awk '{print $2}')
        if [ ! -z "$remaining_pids" ]; then
            kill -9 $remaining_pids 2>/dev/null
        fi
        echo "✅ 清理Python训练进程"
    else
        echo "✅ 没有Python训练进程需要清理"
    fi
    
    # 清理SC2进程（共享的，谨慎清理）
    # 注意：如果多个配置并行运行，不要清理SC2进程
    # 只在真正需要时（如脚本退出后）才清理
    # pkill -f SC2
    # echo "✅ 清理SC2进程"
else
    # 只清理指定配置的进程
    echo "📋 清理配置: $CONFIG_NAME 的相关进程"
    # 使用更精确的匹配，确保只匹配指定的配置名
    # 匹配模式: --config=$CONFIG_NAME 或 --config=$CONFIG_NAME (后面可以有空格或其他参数)
    python_pids=$(ps aux | grep "python.*src/main.py" | grep "\--config=$CONFIG_NAME" | grep -v grep | awk '{print $2}')
    if [ ! -z "$python_pids" ]; then
        echo "发现 $CONFIG_NAME 进程: $python_pids"
        for pid in $python_pids; do
            kill $pid 2>/dev/null
        done
        sleep 1
        # 强制杀死残留进程
        remaining_pids=$(ps aux | grep "python.*src/main.py" | grep "\--config=$CONFIG_NAME" | grep -v grep | awk '{print $2}')
        if [ ! -z "$remaining_pids" ]; then
            echo "强制杀死残留进程: $remaining_pids"
            for pid in $remaining_pids; do
                kill -9 $pid 2>/dev/null
            done
        fi
        echo "✅ 清理 $CONFIG_NAME 训练进程"
    else
        echo "✅ 没有 $CONFIG_NAME 训练进程需要清理"
    fi
    
    # SC2进程不清理（可能被其他配置共享）
    # echo "ℹ️  保留SC2进程（可能被其他配置使用）"
fi

# 清理临时文件（仅在清理所有进程时执行，避免干扰并行运行的配置）
if [ -z "$CONFIG_NAME" ]; then
    rm -rf /tmp/sc-* 2>/dev/null
    echo "✅ 清理临时文件"
fi

echo "🎉 清理完成！"
