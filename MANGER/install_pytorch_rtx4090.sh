#!/bin/bash
# 安装支持RTX 4090 D (sm_89)的PyTorch版本

set -e

echo "🔧 开始安装支持RTX 4090 D的PyTorch..."

# 激活conda环境
source /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh
conda activate pymarl

echo "📦 当前PyTorch版本:"
python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "  PyTorch未安装或无法导入"

echo ""
echo "🗑️  卸载旧版本PyTorch..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || echo "  无需卸载（可能已卸载）"

echo ""
echo "📥 安装PyTorch 2.0+ (CUDA 12.1, 支持sm_89)..."
echo "   这可能需要几分钟，请耐心等待..."

# 使用pip安装（更可靠）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "✅ 安装完成！验证安装..."

python -c "
import torch
print(f'✅ PyTorch版本: {torch.__version__}')
print(f'✅ CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ CUDA版本: {torch.version.cuda}')
    print(f'✅ GPU名称: {torch.cuda.get_device_name(0)}')
    print(f'✅ GPU计算能力: {torch.cuda.get_device_capability(0)}')
    # 检查是否支持sm_89
    try:
        # 尝试创建一个简单的tensor来测试
        x = torch.randn(2, 3).cuda()
        print('✅ GPU测试成功！PyTorch可以正常使用GPU')
    except Exception as e:
        print(f'❌ GPU测试失败: {e}')
else:
    print('❌ CUDA不可用')
"

echo ""
echo "🎉 安装完成！现在可以运行训练脚本了。"

