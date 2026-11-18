# triq-fix-1 实施说明

源码目录：`manger_fix1/MANGER/src`
原始压缩包：`wrriting/manger3-3.tar`，MD5 `CE2ED77CE4B60C133DA531A6DD34DE65`，未修改。

## 改了什么

### 环境元数据

- `envs/starcraft/StarCraft2Env.py`：新增 `get_obs_slices()` 与覆写的 `get_env_info()`，发布 `n_enemies`、`unit_type_bits`、`shield_bits_*`、`obs_slices`。四项之和必须等于 `get_obs_size()`，否则抛错。`obs_timestep_number=True` 时直接拒绝启动，因为该开关会让观测比 `get_obs_size()` 多一维，分段契约不成立。
- `run/run.py`：把这些字段写入 `args.obs_shape` / `args.n_enemies` / `args.unit_type_bits` / `args.obs_slices`，并把 `slices`、`unit_type_bits` 挂到 `scheme["obs"]`。

### 掩码与兵种

- `utils/spatial_utils.py`
  - `build_visibility_and_distance_masks_from_obs` 改用 `args.obs_slices` 定位友军段，不再从 `obs_dim` 反推、也不再依赖 `n_enemies` 的缺省值。缺 `obs_slices` 时抛 `ValueError`。
  - `extract_agent_types` 签名改为 `(obs_batch, unit_type_bits=0, obs_slices=None)`，`unit_type_bits <= 0` 返回 `None`；有兵种位时按 `own` 段真实偏移取 one-hot。
- `controllers/triq_controller.py`
  - `_extract_types` 显式传 `unit_type_bits` 与 `obs_slices`。
  - `use_simple_attention=True` 时打印一次警告（该实现把 0 号智能体的邻域掩码广播给全体）。
  - `TriQMACV2.forward` 补上 `attention_masks`，此前它不传掩码，与主路径不一致。

### RND

- `modules/layer/rnd.py`：`RND(args, scheme=None, input_size=None)`，三路可以有不同输入宽度。
- `modules/layer/multi_rnd.py`：重写。
  - 三个共享网络：Common 吃整段 `obs`（70 维），Individual 吃 `[move; own; onehot(id)]`（11 维），Local 吃注意力特征（256 维）。
  - 删除 `novelty_coordinator`、`novelty_fusion`；`AdaptiveMultiRND` 保留名字但一实例化就抛错，避免被误用。
  - `losses()` 按 `filled` 掩码归一化。
  - `_score()` 在**最后一维**（智能体维）求均值与样本标准差，截断到 `[0,3]`。
  - `update_counts()` 先平均再取一次整，上限取 `max_extra_training`。
  - `branch_weights()` 返回 `softmax(s)`，已 detach。

### Q 网络与控制流

- `modules/agents/triq_rnn_agent.py`：`forward` 增加 `branch_weights`；缓存 `self.last_surrounding`；删除 `α(1-0.3U)` 那套缩放与 `update_metric` 的 reshape/截断逻辑。
- `controllers/triq_controller.py`：`forward` 增加 `branch_weights` 并透传，缓存 `self.last_c`。
- `learners/triq_learner.py`：重写训练主循环。
  - 删除 `target_multi_rnd`、`update_multi_rnd`、`standard_tri_training`、`novelty_guided_training`、`additional_training_step`、`compute_update_metrics`、`record_q_stats`。
  - 新增 `_td_step`（统一 TD，可选 `branch_weights` 与 `collect_loc_error`）、`_rnd_errors`、`_update_rnd`、`_record_novelty_stats`。
  - Local 误差在 TD 前向循环里逐时间步算，只累积 `[B,T,N]` 标量，不缓存整份 `[B,T,N,256]` 特征（省约 237 MB）。
  - 额外更新按 `W[:, t]` 切片传入，修掉旧代码整块传入导致的「所有时间步共用第 0 步权重」。
  - `load_models` 对 `multi_rnd` 用 `strict=False`，打印缺失/多余键。

### 配置

- `triq_sep.yaml`、`triq_sep_standard.yaml`、`triq_sep_1.yaml`、`triq_MMM2_5.yaml`、`record_triq_sep_MMM7.yaml`：移除从未被读取的 `weight_update_freq`、`weight_memory_size`、`novelty_threshold`。`max_extra_training` 现在真正被 `update_counts` 使用。

## 本地验证

三个脚本，均不需要 StarCraft：

```bash
cd manger_fix1
python verify_fix1.py          # 40 项：分段、输入宽度、one-hot 语义、_score、U、权重、梯度隔离、掩码、类型
python verify_gradients.py     # 8 项：注意力缓存、权重形状校验、Local 损失不回流 Q 网络、TD 不碰 RND
python verify_train_smoke.py   # 端到端 train() 冒烟：3 步训练、更新节奏、日志键、U 范围
python diagnose_u.py           # 诊断脚本：U 的分布，不是断言
```

全部通过。`verify_train_smoke.py` 用 `importlib` 直接从文件加载被测模块，绕开 SC2 的导入链。

## 实施中发现并修掉的问题

1. **`_score` 的统计轴写错。** 最初用 `dim=1`，而 learner 传入的是 `[B,T,N]`，`dim=1` 是时间维。已改为 `dim=-1`，并补了三维形状的回归测试。
2. **统计掩码维度错配。** `_record_novelty_stats` 里对 `[B,T,1]` 掩码多做了一次 `squeeze(-1)`，导致与 `[B,T,N]` 张量广播失败。同时分母少乘了 `n_agents`。
3. **日志触发 `requires_grad` 警告。** 统计在 `no_grad` 下计算并对张量 detach。

只有端到端冒烟测试才能发现第 1、2 条，单元测试当时用的是二维张量。

## U 的分布（重要）

诊断结果显示 `U` 几乎只在 0 和 1 之间取值：

| 场景 | U 均值 | P(U=0..3) |
| --- | --- | --- |
| 独立噪声 | 0.169 | 0.83 / 0.17 / 0.00 / 0.00 |
| 单路明显突出 | 0.333 | 0.83 / 0.00 / 0.17 / 0.00 |
| 20% 时刻全体新颖 | 0.163 | 0.84 / 0.16 / 0.00 / 0.00 |

这是方案固有的数学性质，不是实现缺陷：`U ≥ 2` 要求三路分数之和 ≥ 6，而每路是同一时刻 6 个智能体的 z 值，`clip(0,3)` 之后至多一两个能超过 1。已确认先按现状跑一轮观察。**上线后请重点看 `n_extra`：若长期为 0，说明额外更新几乎没触发。**

## 上线前

- 打包 `manger_fix1` 上传，保留原目录以便回滚。
- 建议先用小 `t_max`（如 2e5）短跑，确认启动无报错、`u_mean` / `n_extra` / `loss_td` 曲线正常。
- 旧 checkpoint 不能复用：`multi_rnd` 的参数集合变了。
- 修好 `n_enemies` 后 `vis_mask` 从「敌人可见性」变成真正的队友可见性，`Q^loc` 的行为会明显变化，旧曲线不可与新结果直接对比。
