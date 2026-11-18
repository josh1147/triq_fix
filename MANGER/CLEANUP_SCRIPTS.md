# 清理脚本使用说明

本文档介绍如何使用清理脚本来清理MANGER训练过程中产生的残留进程。

## 📁 脚本文件

| 文件名 | 类型 | 描述 |
|--------|------|------|
| `cleanup_processes.sh` | Bash脚本 | 完整功能的清理脚本，包含详细输出 |
| `cleanup_simple.sh` | Bash脚本 | 简化版清理脚本，快速清理 |
| `cleanup_processes.py` | Python脚本 | Python版本的清理脚本，功能完整 |

## 🚀 使用方法

### 1. Bash脚本 (推荐)

#### 完整版清理脚本
```bash
./cleanup_processes.sh
```

**功能特点:**
- ✅ 显示清理前后的进程状态
- ✅ 彩色输出，易于阅读
- ✅ 显示系统资源状态 (内存、GPU)
- ✅ 清理临时文件
- ✅ 详细的清理日志

#### 简化版清理脚本
```bash
./cleanup_simple.sh
```

**功能特点:**
- ✅ 快速清理，无冗余输出
- ✅ 适合自动化脚本使用
- ✅ 轻量级，执行速度快

### 2. Python脚本

```bash
# 普通模式
python cleanup_processes.py

# 安静模式 (减少输出)
python cleanup_processes.py --quiet
```

**功能特点:**
- ✅ 跨平台兼容
- ✅ 使用psutil库，更精确的进程管理
- ✅ 支持安静模式
- ✅ 详细的系统信息显示

## 🔧 清理内容

所有脚本都会清理以下内容：

1. **Python训练进程**: 所有 `python src/main.py` 相关进程
2. **SC2进程**: 所有StarCraft II相关进程
3. **临时文件**: `/tmp/sc-*` 目录下的所有临时文件

## 📊 示例输出

### 完整版脚本输出
```
🧹 开始清理残留进程...
📊 清理前的进程状态:
Python训练进程数量: 16
SC2进程数量: 63
⚠️  发现残留进程，开始清理...
🔍 清理Python训练进程...
✅ Python训练进程清理完成
🔍 清理SC2进程...
✅ SC2进程清理完成
🧽 清理临时文件...
✅ 临时文件清理完成
📊 清理后的进程状态:
Python训练进程数量: 0
SC2进程数量: 0
💾 系统资源状态:
内存使用情况: 755Gi total, 87Gi used, 660Gi available
GPU使用情况: GPU: 1, 24564, 0
🎉 清理完成！所有残留进程已成功清理
✅ 系统已准备好运行新的训练任务
```

## 🛠️ 故障排除

### 1. 权限问题
如果遇到权限错误，确保脚本有执行权限：
```bash
chmod +x cleanup_processes.sh
chmod +x cleanup_simple.sh
chmod +x cleanup_processes.py
```

### 2. Python依赖
Python脚本需要psutil库：
```bash
pip install psutil
```

### 3. 手动清理
如果脚本无法清理某些进程，可以手动清理：
```bash
# 查看进程
ps aux | grep -E "(python.*main.py|SC2)"

# 手动杀死进程
kill -9 <PID>

# 或者使用pkill
pkill -f "python src/main.py"
pkill -f SC2
```

## 🔄 自动化使用

### 在训练脚本中使用
```bash
#!/bin/bash
# 训练前清理
./cleanup_simple.sh

# 运行训练
python src/main.py --config=triq_sep --env-config=sc2

# 训练后清理 (如果需要)
./cleanup_simple.sh
```

### 定时清理
```bash
# 添加到crontab，每小时清理一次
0 * * * * /path/to/MANGER/cleanup_simple.sh
```

## ⚠️ 注意事项

1. **数据安全**: 清理脚本只会清理训练相关的进程，不会影响其他重要进程
2. **运行环境**: 确保在正确的conda环境中运行Python脚本
3. **权限要求**: 某些进程可能需要sudo权限才能清理
4. **网络连接**: 清理SC2进程可能会中断正在进行的网络连接

## 📞 支持

如果遇到问题，可以：
1. 检查进程状态: `ps aux | grep -E "(python.*main.py|SC2)"`
2. 查看系统资源: `free -h` 和 `nvidia-smi`
3. 手动清理残留进程
4. 重启系统 (最后手段)

---

**推荐使用顺序**: `cleanup_processes.sh` → `cleanup_simple.sh` → `cleanup_processes.py`
