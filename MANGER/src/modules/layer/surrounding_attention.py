import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.spatial_utils import compute_distances, build_surrounding_mask

class SurroundingAttention(nn.Module):
    """
    轻量级注意力机制，用于计算局部邻域内的智能体交互
    """
    def __init__(self, args):
        super(SurroundingAttention, self).__init__()
        self.args = args
        self.distance_threshold = getattr(args, 'surrounding_radius', 3.0)
        self.attention_dim = getattr(args, 'attention_dim', 64)
        self.n_heads = getattr(args, 'attention_heads', 4)
        self.hidden_dim = args.rnn_hidden_dim
        
        # 确保attention_dim能被n_heads整除
        assert self.attention_dim % self.n_heads == 0
        self.head_dim = self.attention_dim // self.n_heads
        
        # 查询、键、值投影
        self.query = nn.Linear(self.hidden_dim, self.attention_dim)
        self.key = nn.Linear(self.hidden_dim, self.attention_dim)
        self.value = nn.Linear(self.hidden_dim, self.attention_dim)
        
        # 输出投影
        self.output_proj = nn.Linear(self.attention_dim, self.hidden_dim)
        
        # 位置编码
        self.position_encoding = nn.Linear(2, self.attention_dim)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化网络权重"""
        for module in [self.query, self.key, self.value, self.output_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
            
    def forward(self, agent_states, agent_positions=None, vis_mask=None, dist_norm=None):
        """
        前向传播
        Args:
            agent_states: 智能体状态 [batch_size, n_agents, hidden_dim]
            agent_positions: 智能体位置 [batch_size, n_agents, 2]（可选，若不提供则使用掩码）
            vis_mask: 可见性掩码 [batch_size, n_agents, n_agents]（可选）
            dist_norm: 归一化距离 [batch_size, n_agents, n_agents]（可选）
        Returns:
            surrounding_features: 局部邻域特征 [batch_size, n_agents, hidden_dim]
        """
        batch_size, n_agents, hidden_dim = agent_states.shape
        
        # 优先使用基于观测的可见性/距离掩码（推荐，train/test 一致）
        if vis_mask is not None:
            surrounding_mask = vis_mask
            # 防止整行全0导致softmax(-inf)→NaN：允许最少自注意
            surrounding_mask = surrounding_mask.clone()
            surrounding_mask.diagonal(dim1=1, dim2=2).fill_(1)
            if surrounding_mask.device != agent_states.device:
                surrounding_mask = surrounding_mask.to(agent_states.device)
            # 不使用位置编码，避免引入全局坐标依赖
            pos_encoding = 0
        else:
            # 回退到基于坐标的掩码（仅当提供了坐标）
            if agent_positions is None or torch.all(agent_positions == 0):
                return agent_states
            if agent_positions.device != agent_states.device:
                agent_positions = agent_positions.to(agent_states.device)
            distances = compute_distances(agent_positions)
            surrounding_mask = build_surrounding_mask(distances, self.distance_threshold)
            if torch.all(distances == 0):
                return agent_states
            if surrounding_mask.device != agent_states.device:
                surrounding_mask = surrounding_mask.to(agent_states.device)
            pos_encoding = self.position_encoding(agent_states.new_zeros(batch_size, n_agents, 2))
        
        # 查询、键、值投影
        Q = self.query(agent_states) + pos_encoding  # [batch_size, n_agents, attention_dim]
        K = self.key(agent_states) + pos_encoding    # [batch_size, n_agents, attention_dim]
        V = self.value(agent_states)                 # [batch_size, n_agents, attention_dim]
        
        # 多头注意力
        Q = Q.view(batch_size, n_agents, self.n_heads, self.head_dim).transpose(1, 2)  # [batch_size, n_heads, n_agents, head_dim]
        K = K.view(batch_size, n_agents, self.n_heads, self.head_dim).transpose(1, 2)  # [batch_size, n_heads, n_agents, head_dim]
        V = V.view(batch_size, n_agents, self.n_heads, self.head_dim).transpose(1, 2)  # [batch_size, n_heads, n_agents, head_dim]
        
        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [batch_size, n_heads, n_agents, n_agents]
        
        # 应用邻域掩码（可见性）
        surrounding_mask_expanded = surrounding_mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        attention_scores = attention_scores.masked_fill(surrounding_mask_expanded == 0, float('-inf'))
        
        # 额外稳健性：若某行全为 -inf，改为全0，softmax→均匀或自注意
        all_neg_inf = torch.isinf(attention_scores).all(dim=-1, keepdim=True)
        attention_scores = torch.where(all_neg_inf, attention_scores.new_zeros(attention_scores.shape), attention_scores)
        
        # Softmax归一化
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 加权聚合
        surrounding_features = torch.matmul(attention_weights, V)  # [batch_size, n_heads, n_agents, head_dim]
        
        # 合并多头
        surrounding_features = surrounding_features.transpose(1, 2).contiguous().view(batch_size, n_agents, self.attention_dim)
        
        # 输出投影
        surrounding_features = self.output_proj(surrounding_features)
        
        # 残差连接和层归一化
        surrounding_features = self.layer_norm(surrounding_features + agent_states)
        
        return surrounding_features

class SimpleSurroundingAttention(nn.Module):
    """
    简化版注意力机制，用于快速验证
    """
    def __init__(self, args):
        super(SimpleSurroundingAttention, self).__init__()
        self.args = args
        self.distance_threshold = getattr(args, 'surrounding_radius', 3.0)
        self.hidden_dim = getattr(args, 'rnn_hidden_dim', 256)
        
        # 简化的注意力机制
        self.attention = nn.Linear(self.hidden_dim, 1)
        self.output_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        
    def forward(self, agent_states, agent_positions):
        """
        简化版前向传播
        Args:
            agent_states: 智能体状态 [batch_size, n_agents, hidden_dim]
            agent_positions: 智能体位置 [batch_size, n_agents, 2]
        Returns:
            surrounding_features: 局部邻域特征 [batch_size, n_agents, hidden_dim]
        """
        batch_size, n_agents, hidden_dim = agent_states.shape
        
        # 如果没有位置信息或全部为0，退化处理（CTDE执行模式）
        if agent_positions is None or torch.all(agent_positions == 0):
            return agent_states
        
        # 确保位置信息与状态在同一设备
        if agent_positions.device != agent_states.device:
            agent_positions = agent_positions.to(agent_states.device)
        
        # 计算距离矩阵和邻域掩码
        distances = compute_distances(agent_positions)
        surrounding_mask = build_surrounding_mask(distances, self.distance_threshold)
        
        # 检查是否有有效位置
        if torch.all(distances == 0):
            return agent_states
        
        # 掩码设备对齐
        if surrounding_mask.device != agent_states.device:
            surrounding_mask = surrounding_mask.to(agent_states.device)
        
        # 计算注意力权重 - 需要先reshape到正确的形状
        agent_states_flat = agent_states.view(-1, self.hidden_dim)  # [batch_size * n_agents, hidden_dim]
        attention_weights_flat = self.attention(agent_states_flat).squeeze(-1)  # [batch_size * n_agents]
        attention_weights = attention_weights_flat.view(batch_size, n_agents)  # [batch_size, n_agents]
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        # 应用邻域掩码
        attention_weights = attention_weights * surrounding_mask[:, 0, :]  # 使用第一个智能体的邻域掩码作为示例
        
        # 加权聚合邻域特征
        surrounding_features = torch.matmul(attention_weights.unsqueeze(1), agent_states).squeeze(1)  # [batch_size, hidden_dim]
        surrounding_features = surrounding_features.unsqueeze(1).expand(-1, n_agents, -1)  # [batch_size, n_agents, hidden_dim]
        
        # 输出投影
        surrounding_features = self.output_proj(surrounding_features)
        
        return surrounding_features
