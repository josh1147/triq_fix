#!/bin/bash

# =============================================================================
# 清理残留进程脚本
# 用于清理MANGER训练过程中产生的残留Python和SC2进程
# =============================================================================

echo "🧹 开始清理残留进程..."

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 显示当前进程数量
echo -e "${BLUE}📊 清理前的进程状态:${NC}"
echo "Python训练进程数量: $(ps aux | grep 'python src/main.py' | grep -v grep | wc -l)"
echo "SC2进程数量: $(ps aux | grep SC2 | grep -v grep | wc -l)"

# 检查是否有残留进程
python_count=$(ps aux | grep 'python src/main.py' | grep -v grep | wc -l)
sc2_count=$(ps aux | grep SC2 | grep -v grep | wc -l)

if [ $python_count -eq 0 ] && [ $sc2_count -eq 0 ]; then
    echo -e "${GREEN}✅ 没有发现残留进程，系统很干净！${NC}"
    exit 0
fi

echo -e "${YELLOW}⚠️  发现残留进程，开始清理...${NC}"

# 1. 清理Python训练进程
echo -e "${BLUE}🔍 清理Python训练进程...${NC}"
python_pids=$(ps aux | grep 'python src/main.py' | grep -v grep | awk '{print $2}')
if [ ! -z "$python_pids" ]; then
    echo "发现Python进程: $python_pids"
    pkill -f "python src/main.py"
    sleep 2
    # 强制杀死仍在运行的进程
    remaining_pids=$(ps aux | grep 'python src/main.py' | grep -v grep | awk '{print $2}')
    if [ ! -z "$remaining_pids" ]; then
        echo -e "${RED}强制杀死残留Python进程: $remaining_pids${NC}"
        kill -9 $remaining_pids 2>/dev/null
    fi
    echo -e "${GREEN}✅ Python训练进程清理完成${NC}"
else
    echo -e "${GREEN}✅ 没有Python训练进程需要清理${NC}"
fi

# 2. 清理SC2进程
echo -e "${BLUE}🔍 清理SC2进程...${NC}"
sc2_pids=$(ps aux | grep SC2 | grep -v grep | awk '{print $2}')
if [ ! -z "$sc2_pids" ]; then
    echo "发现SC2进程: $sc2_pids"
    pkill -f SC2
    sleep 3
    # 强制杀死仍在运行的进程
    remaining_sc2_pids=$(ps aux | grep SC2 | grep -v grep | awk '{print $2}')
    if [ ! -z "$remaining_sc2_pids" ]; then
        echo -e "${RED}强制杀死残留SC2进程: $remaining_sc2_pids${NC}"
        kill -9 $remaining_sc2_pids 2>/dev/null
    fi
    echo -e "${GREEN}✅ SC2进程清理完成${NC}"
else
    echo -e "${GREEN}✅ 没有SC2进程需要清理${NC}"
fi

# 3. 清理临时文件
echo -e "${BLUE}🧽 清理临时文件...${NC}"
temp_dirs=$(ls /tmp/ | grep "sc-" 2>/dev/null | wc -l)
if [ $temp_dirs -gt 0 ]; then
    echo "发现 $temp_dirs 个SC2临时目录"
    rm -rf /tmp/sc-* 2>/dev/null
    echo -e "${GREEN}✅ 临时文件清理完成${NC}"
else
    echo -e "${GREEN}✅ 没有临时文件需要清理${NC}"
fi

# 4. 显示清理后的状态
echo -e "${BLUE}📊 清理后的进程状态:${NC}"
final_python_count=$(ps aux | grep 'python src/main.py' | grep -v grep | wc -l)
final_sc2_count=$(ps aux | grep SC2 | grep -v grep | wc -l)
echo "Python训练进程数量: $final_python_count"
echo "SC2进程数量: $final_sc2_count"

# 5. 显示系统资源状态
echo -e "${BLUE}💾 系统资源状态:${NC}"
echo "内存使用情况:"
free -h | grep -E "(Mem|Swap)"

echo "GPU使用情况:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | while read line; do
        echo "  GPU: $line"
    done
else
    echo "  nvidia-smi 不可用"
fi

# 6. 清理结果总结
if [ $final_python_count -eq 0 ] && [ $final_sc2_count -eq 0 ]; then
    echo -e "${GREEN}🎉 清理完成！所有残留进程已成功清理${NC}"
    echo -e "${GREEN}✅ 系统已准备好运行新的训练任务${NC}"
else
    echo -e "${YELLOW}⚠️  仍有部分进程未能清理，可能需要手动检查${NC}"
fi

echo -e "${BLUE}📝 使用说明:${NC}"
echo "  运行训练: python src/main.py --config=triq_sep --env-config=sc2"
echo "  清理进程: ./cleanup_processes.sh"
echo "  检查进程: ps aux | grep -E '(python.*main.py|SC2)'"
