#!/bin/bash
# 安装MMM3地图到SC2的Maps目录

echo "🔧 安装MMM3地图到StarCraft II..."

# 检测SC2路径
SC2PATHS=(
    "$HOME/StarCraftII"
    "/root/StarCraftII"
    "/root/autodl-tmp/StarCraftII"
)

MAP_SOURCE="smac_maps/MMM3.SC2Map"

if [ ! -f "$MAP_SOURCE" ]; then
    echo "❌ 错误: 找不到源地图文件: $MAP_SOURCE"
    exit 1
fi

# 查找SC2安装目录
SC2PATH=""
for path in "${SC2PATHS[@]}"; do
    if [ -d "$path/Maps" ]; then
        SC2PATH="$path"
        echo "✅ 找到SC2安装目录: $SC2PATH"
        break
    fi
done

if [ -z "$SC2PATH" ]; then
    echo "❌ 错误: 找不到StarCraft II安装目录"
    echo "   请确保SC2已安装，或手动设置SC2PATH环境变量"
    exit 1
fi

MAP_DIR="$SC2PATH/Maps/SMAC_Maps"

# 创建目录（如果不存在）
if [ ! -d "$MAP_DIR" ]; then
    echo "📁 创建SMAC_Maps目录: $MAP_DIR"
    mkdir -p "$MAP_DIR"
fi

# 复制地图文件
echo "📋 复制MMM3.SC2Map到: $MAP_DIR"
cp "$MAP_SOURCE" "$MAP_DIR/"

if [ $? -eq 0 ]; then
    echo "✅ MMM3地图安装成功！"
    echo "   - 源文件: $MAP_SOURCE"
    echo "   - 目标: $MAP_DIR/MMM3.SC2Map"
    ls -lh "$MAP_DIR/MMM3.SC2Map"
else
    echo "❌ 复制失败"
    exit 1
fi

# 验证安装
if [ -f "$MAP_DIR/MMM3.SC2Map" ]; then
    file_size=$(stat -f%z "$MAP_DIR/MMM3.SC2Map" 2>/dev/null || stat -c%s "$MAP_DIR/MMM3.SC2Map" 2>/dev/null)
    echo ""
    echo "📊 地图信息:"
    echo "   - 文件大小: $(($file_size / 1024)) KB"
    echo "   - 文件路径: $MAP_DIR/MMM3.SC2Map"
    echo ""
    echo "✅ 安装完成！现在可以运行训练了。"
else
    echo "❌ 验证失败: 地图文件未正确安装"
    exit 1
fi

