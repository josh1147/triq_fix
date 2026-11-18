import torch as th
import numpy as np
from components.action_selectors import REGISTRY as action_REGISTRY
from modules.agents import REGISTRY as agent_REGISTRY
from .basic_controller import BasicMAC
from utils.spatial_utils import extract_agent_positions, extract_agent_types, build_visibility_and_distance_masks_from_obs

class TriQMAC(BasicMAC):
    """
    三元Q值多智能体控制器
    支持三元Q值智能体和位置/类型信息传递
    """
    def __init__(self, scheme, groups, args):
        # 位置和类型信息提取标志
        self.extract_positions = getattr(args, 'extract_agent_positions', False)
        self.extract_types = getattr(args, 'extract_agent_types', False)

        # 简化注意力把单个智能体的邻域掩码广播给全体，语义与多头版本不同。
        use_simple = getattr(args, 'use_simple_attention', True)
        if use_simple:
            print("[TriQMAC] WARNING: use_simple_attention=True. "
                  "SimpleSurroundingAttention broadcasts agent 0's neighbourhood "
                  "mask to every agent, so the local branch does not see "
                  "per-agent neighbours. Set use_simple_attention=False for the "
                  "multi-head implementation.")

        # 确保继承父类的属性
        self.obs_agent_id = getattr(args, 'obs_agent_id', True)
        
        # 手动初始化基础属性（避免父类调用错误的_get_input_shape）
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.save_probs = getattr(self.args, 'save_probs', False)

        self.hidden_states = None
        # 最近一次前向的注意力特征，供 Local RND 使用
        self.last_c = None
    
    def _build_agents(self, input_shape):
        """
        构建智能体网络
        Args:
            input_shape: 输入形状
        """
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
        
    def forward(self, ep_batch, t, update_metric=None, test_mode=False, branch_weights=None):
        """
        前向传播
        Args:
            ep_batch: 经验批次
            t: 时间步
            update_metric: 已废弃，保留仅为兼容旧调用
            test_mode: 测试模式
            branch_weights: [batch, n_agents, 3] 额外更新的三路权重；
                None 表示标准 TD，使用门控权重
        Returns:
            agent_outs: 智能体输出
        """
        # 提取智能体输入
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        # 统一取得目标设备，供后续张量对齐
        try:
            target_device = agent_inputs.device
        except AttributeError:
            target_device = None
        
        # 提取位置和类型信息
        agent_positions = None
        agent_types = None
        
        if self.extract_positions:
            # CTDE: 训练时使用state，测试时只使用obs
            state_batch = ep_batch.get("state")[:, t] if "state" in ep_batch else None
            agent_positions = self._extract_positions(ep_batch, t, state_batch, test_mode)
            # 设备对齐到batch所在设备（与agent_inputs一致）
            if target_device is not None and agent_positions is not None and agent_positions.device != target_device:
                agent_positions = agent_positions.to(target_device)
        
        if self.extract_types:
            agent_types = self._extract_types(ep_batch, t)
            # 设备对齐
            if target_device is not None and agent_types is not None and agent_types.device != target_device:
                agent_types = agent_types.to(target_device)
        
        # 设置测试模式
        if test_mode:
            self.agent.eval()
        
        # 基于观测构造可见性/距离掩码（train/test一致）
        obs_t = ep_batch["obs"][:, t]
        vis_mask, dist_norm = build_visibility_and_distance_masks_from_obs(obs_t, self.args)

        # 调试：注意力掩码/距离的轻量级监控（默认关闭）
        if getattr(self.args, 'debug_attention', False):
            # 每隔若干时间步打印一次，避免刷屏
            if (t % getattr(self.args, 'debug_attention_interval', 25)) == 0:
                with th.no_grad():
                    try:
                        vm_mean = vis_mask.float().mean().item()
                        # 仅在可见位置上统计距离
                        visible_mask = (vis_mask > 0)
                        if visible_mask.any():
                            dn_visible = dist_norm.masked_select(visible_mask)
                            dn_mean = dn_visible.mean().item()
                            dn_min = dn_visible.min().item()
                            dn_max = dn_visible.max().item()
                        else:
                            # 空可见集时，避免打印 NaN，统一显示为 0
                            dn_mean = 0.0
                            dn_min = 0.0
                            dn_max = 0.0
                        # 一致性检查：不可见处距离应为0
                        inconsistent = th.any((vis_mask == 0) & (dist_norm > 0)).item()
                        # 可选：半径裁剪后的比例（若设置了surrounding_radius）
                        radius = getattr(self.args, 'surrounding_radius', None)
                        radius_ratio = None
                        if radius is not None:
                            thr = radius / 9.0
                            within_radius = (dist_norm <= thr) & visible_mask
                            radius_ratio = within_radius.float().mean().item() if visible_mask.any() else 0.0
                        print(f"[ATTDEBUG] t={t} vis_mean={vm_mean:.4f} dist(mean/min/max)={dn_mean:.4f}/{dn_min:.4f}/{dn_max:.4f} inconsistent={(bool(inconsistent))}"
                              + (f" radius_within_ratio={radius_ratio:.4f}" if radius_ratio is not None else ""))
                    except Exception as _:
                        pass

        # 智能体前向传播（传入注意力掩码）
        agent_outs, self.hidden_states = self.agent(
            agent_inputs,
            self.hidden_states,
            update_metric=update_metric,
            agent_positions=None,
            agent_types=agent_types,
            attention_masks=(vis_mask, dist_norm),
            branch_weights=branch_weights
        )

        # 缓存注意力特征，供 Local RND 使用。梯度截断由调用方负责，
        # 因为标准 TD 那一趟仍需要保留梯度。
        self.last_c = getattr(self.agent, 'last_surrounding', None)
        
        # 软最大处理（如果需要）
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # 将不可用动作的logits设为很小的值
                agent_outs = agent_outs.reshape(ep_batch.batch_size * self.n_agents, -1)
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)
    
    def _extract_positions(self, ep_batch, t, state_batch=None, test_mode=False):
        """
        从经验批次中提取位置信息（CTDE版本）
        Args:
            ep_batch: 经验批次
            t: 时间步
            state_batch: 全局状态数据（训练时可用）
            test_mode: 是否为测试模式
        Returns:
            agent_positions: 智能体位置 [batch_size, n_agents, 2]
        """
        # 获取观测数据
        obs = ep_batch["obs"][:, t]
        
        # 提取位置信息（传递test_mode和state，以及args）
        agent_positions = extract_agent_positions(obs, state_batch, test_mode, self.args)
        
        return agent_positions
    
    def _extract_types(self, ep_batch, t):
        """
        从经验批次中提取类型信息
        Args:
            ep_batch: 经验批次
            t: 时间步
        Returns:
            agent_types: 智能体类型 [batch_size, n_agents]，或 None（地图无兵种位）
        """
        # 获取观测数据
        obs = ep_batch["obs"][:, t]

        # 显式传入兵种位数与观测分段：同质地图返回 None，门控走常数分支
        unit_type_bits = getattr(self.args, 'unit_type_bits', 0)
        obs_slices = getattr(self.args, 'obs_slices', None)
        agent_types = extract_agent_types(
            obs, unit_type_bits=unit_type_bits, obs_slices=obs_slices)

        return agent_types
    
    def _build_inputs(self, batch, t):
        """
        构建智能体输入
        Args:
            batch: 经验批次
            t: 时间步
        Returns:
            inputs: 智能体输入
        """
        # 获取可用动作
        avail_actions = batch["avail_actions"][:, t]
        
        # 获取观测数据
        inputs = batch["obs"][:, t]
        
        # 添加last action信息（如果启用）
        if self.args.obs_last_action:
            if t == 0:
                inputs = th.cat([inputs, th.zeros_like(batch["actions_onehot"][:, t])], dim=-1)
            else:
                inputs = th.cat([inputs, batch["actions_onehot"][:, t-1]], dim=-1)
        
        # 添加agent ID信息（如果启用）
        if self.obs_agent_id:
            inputs = th.cat([inputs, th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(batch.batch_size, -1, -1)], dim=-1)
        
        return inputs
    
    def _get_input_shape(self, scheme):
        """
        计算输入形状
        Args:
            scheme: 数据模式
        Returns:
            input_shape: 输入形状
        """
        input_shape = scheme["obs"]["vshape"]
        
        # 添加last action维度
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        
        # 添加agent ID维度
        if self.obs_agent_id:
            input_shape += self.n_agents
        
        return input_shape
    
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        """
        选择动作
        Args:
            ep_batch: 经验批次
            t_ep: 时间步
            t_env: 环境时间步
            bs: 批次切片
            test_mode: 测试模式
        Returns:
            chosen_actions: 选择的动作
        """
        # 只对选定的批次元素选择动作
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions
    
    def init_hidden(self, batch_size):
        """
        初始化隐藏状态
        Args:
            batch_size: 批次大小
        """
        self.hidden_states = self.agent.init_hidden()
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

class TriQMACV2(TriQMAC):
    """
    三元Q值控制器的增强版本
    包含更多高级功能
    """
    def __init__(self, scheme, groups, args):
        super(TriQMACV2, self).__init__(scheme, groups, args)
        
        # 位置和类型信息缓存
        self.position_cache = {}
        self.type_cache = {}
        self.cache_size = 1000
        
        # 统计信息
        self.stats = {
            'position_extraction_time': [],
            'type_extraction_time': [],
            'forward_time': []
        }
    
    def forward(self, ep_batch, t, update_metric=None, test_mode=False, branch_weights=None):
        """增强版前向传播"""
        import time
        start_time = time.time()
        
        # 构建输入
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        
        # 提取位置信息（带缓存）
        agent_positions = self._extract_positions_cached(ep_batch, t)
        
        # 提取类型信息（带缓存）
        agent_types = self._extract_types_cached(ep_batch, t)

        # 与主路径一致：用同一套观测构造可见性/距离掩码
        obs_t = ep_batch["obs"][:, t]
        vis_mask, dist_norm = build_visibility_and_distance_masks_from_obs(obs_t, self.args)
        
        # 智能体前向传播
        agent_outs, self.hidden_states = self.agent(
            agent_inputs,
            self.hidden_states,
            update_metric=update_metric,
            agent_positions=agent_positions,
            agent_types=agent_types,
            attention_masks=(vis_mask, dist_norm),
            branch_weights=branch_weights
        )

        self.last_c = getattr(self.agent, 'last_surrounding', None)
        
        # 软最大处理（如果需要）
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # 将不可用动作的logits设为很小的值
                agent_outs = agent_outs.reshape(ep_batch.batch_size * self.n_agents, -1)
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
        
        # 记录统计信息
        forward_time = time.time() - start_time
        self.stats['forward_time'].append(forward_time)
        
        # 限制统计历史大小
        if len(self.stats['forward_time']) > self.cache_size:
            self.stats['forward_time'].pop(0)
        
        return agent_outs
    
    def _extract_positions_cached(self, ep_batch, t):
        """带缓存的位置信息提取"""
        import time
        start_time = time.time()
        
        # 生成缓存键
        cache_key = f"{ep_batch.batch_size}_{t}_{id(ep_batch)}"
        
        # 检查缓存
        if cache_key in self.position_cache:
            return self.position_cache[cache_key]
        
        # 提取位置信息
        agent_positions = self._extract_positions(ep_batch, t)
        
        # 缓存结果
        if len(self.position_cache) < self.cache_size:
            self.position_cache[cache_key] = agent_positions
        
        # 记录时间
        extraction_time = time.time() - start_time
        self.stats['position_extraction_time'].append(extraction_time)
        
        if len(self.stats['position_extraction_time']) > self.cache_size:
            self.stats['position_extraction_time'].pop(0)
        
        return agent_positions
    
    def _extract_types_cached(self, ep_batch, t):
        """带缓存的类型信息提取"""
        import time
        start_time = time.time()
        
        # 生成缓存键
        cache_key = f"{ep_batch.batch_size}_{t}_{id(ep_batch)}"
        
        # 检查缓存
        if cache_key in self.type_cache:
            return self.type_cache[cache_key]
        
        # 提取类型信息
        agent_types = self._extract_types(ep_batch, t)
        
        # 缓存结果
        if len(self.type_cache) < self.cache_size:
            self.type_cache[cache_key] = agent_types
        
        # 记录时间
        extraction_time = time.time() - start_time
        self.stats['type_extraction_time'].append(extraction_time)
        
        if len(self.stats['type_extraction_time']) > self.cache_size:
            self.stats['type_extraction_time'].pop(0)
        
        return agent_types
    
    def get_stats(self):
        """获取统计信息"""
        stats = {}
        
        if self.stats['forward_time']:
            stats['avg_forward_time'] = np.mean(self.stats['forward_time'])
            stats['max_forward_time'] = np.max(self.stats['forward_time'])
            stats['min_forward_time'] = np.min(self.stats['forward_time'])
        
        if self.stats['position_extraction_time']:
            stats['avg_position_extraction_time'] = np.mean(self.stats['position_extraction_time'])
        
        if self.stats['type_extraction_time']:
            stats['avg_type_extraction_time'] = np.mean(self.stats['type_extraction_time'])
        
        stats['cache_size'] = len(self.position_cache) + len(self.type_cache)
        
        return stats
    
    def clear_cache(self):
        """清空缓存"""
        self.position_cache.clear()
        self.type_cache.clear()
        
        for key in self.stats:
            self.stats[key].clear()
