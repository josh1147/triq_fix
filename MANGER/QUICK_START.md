# 三元Q值框架快速使用指南

## 🚀 快速开始

### 1. 激活环境
```bash
conda activate pymarl
```

### 2. 清理残留进程 (推荐)
```bash
./cleanup_processes.sh
```

### 3. 运行三元Q值框架
```bash
python src/main.py --config=triq_sep --env-config=sc2
```

## 🛠️ 常用命令

### 训练命令
```bash
# 默认MMM地图
python src/main.py --config=triq_sep --env-config=sc2

# 指定地图
python src/main.py --config=triq_sep --env-config=sc2 --env_args.map_name=2s3z

# 限制训练步数
python src/main.py --config=triq_sep --env-config=sc2 --t_max=10000
```

### 清理命令
```bash
# 完整清理 (推荐)
./cleanup_processes.sh

# 快速清理
./cleanup_simple.sh

# Python版本
python cleanup_processes.py
```

### 检查命令
```bash
# 检查进程
ps aux | grep -E "(python.*main.py|SC2)"

# 检查系统资源
free -h
nvidia-smi
```

## 🎯 三元Q值框架特点

- **三元Q值结构**: `Q_total = α×Q_common + β×Q_individual + γ×Q_surrounding`
- **动态权重**: 根据情境自动调整权重
- **多维新颖度**: 三个独立的RND网络
- **局部协调**: 注意力机制捕获邻近智能体影响
- **完全兼容**: 与MANGER框架无缝集成

## 📊 成功运行标志

看到以下输出表示框架运行成功：
```
Multi-RND Size: 44.323K
t_env: 705 / 4200000
Estimated time left: 1 hours, 48 seconds
```

## 🔧 故障排除

### 问题1: "Killed" 错误
**解决方案**: 运行清理脚本
```bash
./cleanup_processes.sh
```

### 问题2: 进程残留
**解决方案**: 手动清理
```bash
pkill -f "python src/main.py"
pkill -f SC2
```

### 问题3: 内存不足
**解决方案**: 检查系统资源
```bash
free -h
nvidia-smi
```

## 📁 重要文件

- `src/config/algs/triq_sep.yaml` - 三元Q值配置
- `src/modules/agents/triq_rnn_agent.py` - 三元Q值智能体
- `src/learners/triq_learner.py` - 三元Q值训练器
- `cleanup_processes.sh` - 清理脚本

## 🎉 恭喜！

您的三元Q值框架已经成功实现并可以运行！
