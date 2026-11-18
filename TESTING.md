# 快速测试

三条命令，按顺序。

```bash
cd manger_fix1

python check_env.py     # 先看环境是否有阻塞问题
python verify_all.py    # 跑全部离线检查
python diagnose_u.py    # 可选：看 U 的分布
```

`verify_all.py` 退出码为 0 即全部通过。加 `-v` 可看每个脚本的完整输出。

## 这些检查覆盖什么

| 脚本 | 内容 | 需要 SC2 |
| --- | --- | --- |
| `check_env.py` | 依赖、Python/NumPy 兼容性、项目能否导入 | 否 |
| `verify_fix1.py` | 观测分段、三路 RND 输入宽度与 one-hot 语义、`_score`、`U`、权重、梯度隔离、掩码、兵种提取 | 否 |
| `verify_gradients.py` | 注意力特征缓存、权重形状校验、Local 损失不回流 Q 网络、TD 不碰 RND | 否 |
| `verify_train_smoke.py` | 用合成 batch 跑通完整 `TriQLearner.train()` 三步 | 否 |
| `diagnose_u.py` | `U` 的分布，不是断言，只输出统计 | 否 |

`verify_train_smoke.py` 用 `importlib` 直接从文件加载被测模块，绕开 `envs` 包对 SC2 的导入链。

## 装环境

分成两份，因为离线检查不需要 SC2。

```bash
# 本地 / 任意机器，只跑检查
uv venv --python 3.10
uv pip install -r requirements.txt

# 线上训练机，额外装 SC2 相关
uv pip install -r requirements-sc2.txt
```

torch 故意不锁版本，按机器的 CUDA 装：

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 两个必须知道的坑

**`numpy` 必须 `<2`。** `src/components/action_selectors.py` 里有 `from matplotlib.pyplot import xcorr`，这个名字是 matplotlib 从 numpy 转出的，而 numpy 2.0 删掉了 `np.xcorr`。用了 numpy 2，控制器导入直接失败，什么都跑不起来。

**`collections.Mapping` 在 Python 3.10 已移除。** `src/main.py` 第 59 行用它做 `isinstance` 判断。照现在这样写，只能在 3.9 上跑。`check_env.py` 会报这一条。要留在 3.10+，把那一行改成 `collections.abc.Mapping`。

## 线上首轮

离线检查通过后，真正跑训练：

```bash
python src/main.py --config=triq_sep --env-config=sc2 \
    --env_args.map_name=6h_vs_8z --t_max=200000
```

重点看这几个日志键：

| 键 | 期望 |
| --- | --- |
| `u_mean` | 0 到 3 之间，实测约 0.1 到 0.3 |
| `n_extra` | 批次内最大更新次数。**长期为 0 说明额外更新没触发** |
| `loss_td` | 稳定下降，不发散 |
| `rnd_loss_com` / `ind` / `loc` | 三路都在下降 |
| `s_ind_mean` | 若长期接近 0，说明 Individual 那一路饱和了 |

关于 `u_mean` 偏低：这是方案固有的数学性质，不是 bug。详见 `IMPLEMENTATION.md` 和 `openspec/changes/triq-fix-1/design.md` 的 R6b。

## 常见问题

**`check_env.py` 报 sacred / smac 缺失。**
本地没装这两个属正常，`check_env.py` 会把它们归到 "Needed on the training box"，不影响离线检查的通过判定。

**`verify_train_smoke.py` 报 `No module named 'smac'`。**
不该发生——它用 `importlib` 绕过了 SC2 导入链。若出现，说明 `MANGER/src` 路径没被正确加入 `sys.path`，检查脚本顶部的 `SRC` 变量。

**想单独确认某个断言。**
每个脚本都可以独立运行，例如 `python verify_fix1.py`，失败时会打印具体是哪条断言。
