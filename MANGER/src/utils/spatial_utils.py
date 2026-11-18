import torch
import numpy as np
import math


def build_visibility_and_distance_masks_from_obs(obs_batch, args, sight_range: float = 9.0):
    """
    从局部观测构造注意力可用的可见性掩码与距离矩阵（不依赖全局坐标）。

    Args:
        obs_batch: [B, n_agents, obs_dim]
        args: 需包含 n_agents、obs_slices，以及 obs_all_health / shield_bits_* /
              unit_type_bits / obs_last_action / n_actions（用于校验分段长度）
        sight_range: 视野半径（SC2固定为9）

    Returns:
        vis_mask: [B, n_agents, n_agents]，第 i 行仅填充 agent i 能看到的队友 j（i!=j）
        dist_norm: [B, n_agents, n_agents]，agent i 看到 j 的归一化距离（0~1），不可见为0

    友军段起点由 obs_slices 直接给出。此前用 obs_dim 反推，而推导又依赖
    args.n_enemies；该字段缺失时 ally_start 会退化为移动段长度，导致从敌方
    段读取“队友可见性”。
    """
    B, n_agents, obs_dim = obs_batch.shape
    device = obs_batch.device
    dtype = obs_batch.dtype

    vis_mask = torch.zeros(B, n_agents, n_agents, device=device, dtype=dtype)
    dist_norm = torch.zeros(B, n_agents, n_agents, device=device, dtype=dtype)

    slices = getattr(args, 'obs_slices', None)
    if slices is None:
        raise ValueError(
            "args.obs_slices is required to locate the ally block. "
            "It is published by StarCraft2Env.get_env_info and wired in run.py.")

    expected = slices["move"] + slices["enemy"] + slices["ally"] + slices["own"]
    if expected != obs_dim:
        raise ValueError(
            "Observation slices sum to {} but obs_dim is {}".format(expected, obs_dim))

    unit_type_bits = getattr(args, 'unit_type_bits', 0)
    obs_all_health = getattr(args, 'obs_all_health', True)
    obs_last_action = getattr(args, 'obs_last_action', False)
    n_actions = getattr(args, 'n_actions', 0)
    shield_bits_ally = getattr(args, 'shield_bits_ally', 0)

    # 友军每一行的宽度。
    ally_base_size = 4  # visible, distance, relative_x, relative_y
    if obs_all_health:
        ally_base_size += 1 + shield_bits_ally
    ally_base_size += unit_type_bits
    if obs_last_action:
        ally_base_size += n_actions

    ally_start = slices["move"] + slices["enemy"]
    if ally_base_size <= 0 or slices["ally"] % ally_base_size != 0:
        raise ValueError(
            "Ally segment {} is not a multiple of the per-unit width {}"
            .format(slices["ally"], ally_base_size))
    ally_count = min(n_agents - 1, slices["ally"] // ally_base_size)

    for i in range(n_agents):
        a_obs = obs_batch[:, i, :]  # [B, obs_dim]
        ally_k = 0
        for j in range(n_agents):
            if j == i:
                continue
            if ally_k >= ally_count:
                break
            base = ally_start + ally_k * ally_base_size
            # ally features: [visible, distance, rel_x, rel_y, ...]
            visible = a_obs[:, base + 0]
            dist = a_obs[:, base + 1]

            # 只记录可见的（距离已按 sight_range 归一化）
            vis_mask[:, i, j] = (visible > 0.5).to(dtype)
            # 确保不可见处为0
            dist_norm[:, i, j] = torch.where(visible > 0.5, dist, torch.zeros_like(dist))

            ally_k += 1

    # 自身不可见，置0（对角线）
    vis_mask.diagonal(dim1=1, dim2=2).zero_()
    dist_norm.diagonal(dim1=1, dim2=2).zero_()

    return vis_mask, dist_norm


def extract_global_positions_from_state(obs_batch, state_batch, args):
    """训练模式下的全局位置提取"""
    batch_size, n_agents, obs_dim = obs_batch.shape
    positions = torch.zeros(batch_size, n_agents, 2, device=obs_batch.device, dtype=obs_batch.dtype)
    
    try:
        # 计算state的结构尺寸
        shield_bits_ally = getattr(args, 'shield_bits_ally', 0)
        unit_type_bits = getattr(args, 'unit_type_bits', 0)
        n_enemies = getattr(args, 'n_enemies', 0)
        shield_bits_enemy = getattr(args, 'shield_bits_enemy', 0)
        
        nf_al = 4 + shield_bits_ally + unit_type_bits  # ally特征数量
        nf_en = 3 + shield_bits_enemy + unit_type_bits  # enemy特征数量
        ally_state_size = n_agents * nf_al
        
        # 将state转换为float32（如果输入是numpy）
        if isinstance(state_batch, np.ndarray):
            state_batch = torch.from_numpy(state_batch).float().to(obs_batch.device)
        
        # 从state中提取ally的相对坐标（在nf_al的第2、3维）
        ally_state = state_batch[:, :ally_state_size].view(batch_size, n_agents, nf_al)
        
        # 获取地图参数
        map_x = getattr(args, 'map_x', 64)
        map_y = getattr(args, 'map_y', 64)
        max_dist_x = getattr(args, 'max_distance_x', 64)
        max_dist_y = getattr(args, 'max_distance_y', 64)
        
        center_x = map_x / 2
        center_y = map_y / 2
        
        # 恢复绝对位置：state中的坐标是相对于地图中心的归一化坐标
        relative_x = ally_state[:, :, 2]  # 索引2是相对X
        relative_y = ally_state[:, :, 3]  # 索引3是相对Y
        
        positions[:, :, 0] = relative_x * max_dist_x + center_x
        positions[:, :, 1] = relative_y * max_dist_y + center_y
        
    except Exception as e:
        print(f"Warning: Failed to extract positions from state: {e}")
        return positions
    
    return positions


def extract_agent_positions(obs_batch, state_batch=None, test_mode=False, args=None):
    """
    兼容接口的占位函数（保留但不再使用全局坐标）。
    注意：建议使用 build_visibility_and_distance_masks_from_obs 构造注意力掩码。
    这里始终返回零位置，促使注意力依赖可见性/距离掩码而非坐标。
    """
    B, n_agents, _ = obs_batch.shape
    return torch.zeros(B, n_agents, 2, device=obs_batch.device, dtype=obs_batch.dtype)


def extract_agent_types(obs_batch, unit_type_bits=0, obs_slices=None):
    """
    从观测数据中提取智能体类型信息。

    Args:
        obs_batch: 观测批次数据 [batch_size, n_agents, obs_dim]
        unit_type_bits: 单位类型位数；为 0 时表示该地图没有兵种位
        obs_slices: 观测分段长度（含 move/enemy/ally/own），用于定位自身段。
            为 None 时回退到“观测末尾 unit_type_bits 位”的旧行为。

    Returns:
        agent_types: [batch_size, n_agents]，或 None（当地图没有兵种位时）
    """
    if unit_type_bits is None or unit_type_bits <= 0:
        # 同质地图没有兵种位。返回 None 让门控走常数分支，而不是从
        # 其它特征推出一个没有意义的类型。
        return None

    batch_size, n_agents, obs_dim = obs_batch.shape

    if obs_slices is not None:
        own_start = obs_dim - obs_slices["own"]
        type_start_idx = own_start + (obs_slices["own"] - unit_type_bits)
    else:
        type_start_idx = obs_dim - unit_type_bits

    if type_start_idx < 0 or type_start_idx + unit_type_bits > obs_dim:
        raise ValueError(
            "Agent type bits [{}, {}) fall outside obs_dim {}".format(
                type_start_idx, type_start_idx + unit_type_bits, obs_dim))

    type_features = obs_batch[:, :, type_start_idx:type_start_idx + unit_type_bits]
    agent_types = torch.argmax(type_features, dim=-1).to(torch.long)

    return agent_types


def compute_distances(positions):
    """计算智能体间距离矩阵"""
    return torch.cdist(positions, positions, p=2)


def build_surrounding_mask(distances, threshold=3.0):
    """构建局部邻域掩码"""
    surrounding_mask = (distances < threshold).to(distances.dtype)
    # 排除自己（对角线置零）
    surrounding_mask.diagonal(dim1=1, dim2=2).zero_()
    return surrounding_mask


def get_agent_type_names(agent_types):
    """将智能体类型ID转换为名称"""
    type_mapping = {
        0: 'marine',      # 陆战队
        1: 'marauder',    # 重机枪兵
        2: 'medic',       # 医疗兵
        3: 'stalker',     # 追猎者
        4: 'zealot',      # 狂热者
        5: 'colossus',    # 巨像
    }
    
    type_names = []
    for batch in agent_types:
        batch_names = []
        for agent_type in batch:
            batch_names.append(type_mapping.get(agent_type.item(), 'unknown'))
        type_names.append(batch_names)
    
    return type_names


def normalize_positions(positions, map_size=(32, 32)):
    """标准化位置坐标到[0, 1]范围"""
    normalized_positions = positions.clone()
    normalized_positions[:, :, 0] = positions[:, :, 0] / map_size[0]
    normalized_positions[:, :, 1] = positions[:, :, 1] / map_size[1]
    return normalized_positions


def compute_surrounding_centroid(positions, surrounding_mask):
    """计算局部邻域的中心位置"""
    batch_size, n_agents, _ = positions.shape
    centroids = torch.zeros_like(positions)
    
    for i in range(n_agents):
        neighbors_mask = surrounding_mask[:, i, :]
        neighbors_count = torch.sum(neighbors_mask, dim=1, keepdim=True)
        neighbors_count = torch.clamp(neighbors_count, min=1)
        
        neighbors_positions = positions * neighbors_mask.unsqueeze(-1)
        centroids[:, i] = torch.sum(neighbors_positions, dim=1) / neighbors_count
    
    return centroids
