#!/bin/bash
# 检查PyTorch安装状态

echo "📊 检查PyTorch安装状态..."
echo ""

# 检查安装进程
if ps aux | grep -E "install_pytorch|pip install" | grep -v grep > /dev/null; then
    echo "⏳ 安装正在进行中..."
    echo ""
    echo "📋 安装日志（最后20行）:"
    tail -20 pytorch_install.log 2>/dev/null || echo "  日志文件尚未生成"
else
    echo "✅ 安装进程已完成或未运行"
    echo ""
    echo "📋 完整安装日志:"
    cat pytorch_install.log 2>/dev/null || echo "  无日志文件"
    echo ""
    echo "🔍 验证PyTorch安装:"
    source /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh
    conda activate pymarl
    python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA版本: {torch.version.cuda}')
    print(f'GPU名称: {torch.cuda.get_device_name(0)}')
    print(f'GPU计算能力: {torch.cuda.get_device_capability(0)}')
    try:
        x = torch.randn(2, 3).cuda()
        print('✅ GPU测试成功！')
    except Exception as e:
        print(f'❌ GPU测试失败: {e}')
" 2>&1
fi

