import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm
from modules.layer.surrounding_attention import SurroundingAttention, SimpleSurroundingAttention
from utils.spatial_utils import extract_agent_positions, extract_agent_types

class TriQRNNAgent(nn.Module):
    """
    三元Q值RNN智能体
    结合全局协调(Q_common)、个体策略(Q_individual)和局部协调(Q_surrounding)
    """
    def __init__(self, input_shape, args):
        super(TriQRNNAgent, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.rnn_hidden_dim = args.rnn_hidden_dim
        
        # 基础特征提取网络 - 动态适应输入维度
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        
        # 三元Q值头
        self.q_common = nn.Linear(args.rnn_hidden_dim, args.n_actions)  # 全局共享策略
        self.q_individual = nn.ModuleList([
            nn.Linear(args.rnn_hidden_dim, args.n_actions) for _ in range(args.n_agents)
        ])  # 个体策略
        self.q_surrounding = nn.Linear(args.rnn_hidden_dim, args.n_actions)  # 局部协调策略
        
        # 动态权重预测器
        self.agent_type_embed_dim = getattr(args, 'agent_type_embed_dim', 16)
        self.agent_type_embedding = nn.Embedding(10, self.agent_type_embed_dim)  # 支持最多10种智能体类型
        
        self.weight_predictor = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim + self.agent_type_embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # α, β, γ
            nn.Softmax(dim=-1)
        )
        
        # 注意力机制
        self.use_simple_attention = getattr(args, 'use_simple_attention', True)
        if self.use_simple_attention:
            self.surrounding_attention = SimpleSurroundingAttention(args)
        else:
            self.surrounding_attention = SurroundingAttention(args)
        
        # 层归一化
        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.rnn_hidden_dim)
        else:
            self.layer_norm = None
        
        # 正交初始化
        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.q_common, gain=args.gain)
            orthogonal_init_(self.q_surrounding, gain=args.gain)
            for q_ind in self.q_individual:
                orthogonal_init_(q_ind, gain=args.gain)
        
        # 权重初始化
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for module in [self.q_common, self.q_surrounding]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
            
        for q_ind in self.q_individual:
            nn.init.xavier_uniform_(q_ind.weight)
            nn.init.constant_(q_ind.bias, 0)
    
    def init_hidden(self):
        """初始化隐藏状态"""
        return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()
    
    def forward(self, inputs, hidden_state, update_metric=None, agent_positions=None, agent_types=None, attention_masks=None, branch_weights=None):
        """
        前向传播
        Args:
            inputs: 输入观测 [batch_size, n_agents, input_shape]
            hidden_state: 隐藏状态
            update_metric: 已废弃，保留仅为兼容旧调用；不再影响权重
            agent_positions: 智能体位置 [batch_size, n_agents, 2]
            agent_types: 智能体类型 [batch_size, n_agents]，None 表示地图无兵种位
            attention_masks: (vis_mask, dist_norm)
            branch_weights: [batch_size, n_agents, 3]，额外更新使用的三路权重。
                为 None 时走门控权重（标准 TD）。
        Returns:
            q_total: 组合后的Q值 [batch_size, n_agents, n_actions]
            rnn_output: RNN输出 [batch_size, n_agents, rnn_hidden_dim]
        """
        b, a, e = inputs.size()
        
        # 基础特征提取
        inputs = inputs.view(-1, e)
        
        # 动态适应输入维度
        if e != self.input_shape:
            # 如果输入维度不匹配，使用自适应层
            if not hasattr(self, 'adaptive_fc1') or self.adaptive_fc1.in_features != e:
                self.adaptive_fc1 = nn.Linear(e, self.rnn_hidden_dim).to(inputs.device)
            x = F.relu(self.adaptive_fc1(inputs), inplace=True)
        else:
            x = F.relu(self.fc1(inputs), inplace=True)
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        rnn_output = self.rnn(x, h_in)
        
        # 层归一化
        if self.layer_norm is not None:
            rnn_output = self.layer_norm(rnn_output)
        
        # 重塑回原始形状
        rnn_output = rnn_output.reshape(b, a, -1)
        
        # 计算三元Q值
        q_common = self.q_common(rnn_output)  # [batch_size, n_agents, n_actions]
        
        # 个体Q值
        q_individual_list = []
        for i, q_ind_head in enumerate(self.q_individual):
            q_individual_list.append(q_ind_head(rnn_output[:, i, :]))  # [batch_size, n_actions]
        q_individual = th.stack(q_individual_list, dim=1)  # [batch_size, n_agents, n_actions]
        
        # 局部协调Q值
        vis_mask = None
        dist_norm = None
        if attention_masks is not None:
            vis_mask, dist_norm = attention_masks
        surrounding_features = self.surrounding_attention(rnn_output, None, vis_mask=vis_mask, dist_norm=dist_norm)
        q_surrounding = self.q_surrounding(surrounding_features)

        # 缓存注意力特征，供 Local RND 使用（调用方负责 detach）
        self.last_surrounding = surrounding_features

        # 三路权重：额外更新用外部分数，标准训练用门控
        if branch_weights is None:
            alpha, beta, gamma = self.compute_dynamic_weights(rnn_output, agent_types)
        else:
            bw = branch_weights
            if bw.dim() != 3 or bw.size(-1) != 3:
                raise ValueError(
                    "branch_weights must be [batch, n_agents, 3], got {}"
                    .format(tuple(bw.shape)))
            if bw.size(0) != b or bw.size(1) != a:
                raise ValueError(
                    "branch_weights shape {} does not match inputs [{}, {}, ...]"
                    .format(tuple(bw.shape), b, a))
            alpha = bw[..., 0]
            beta = bw[..., 1]
            gamma = bw[..., 2]

        q_total = (alpha.unsqueeze(-1) * q_common
                   + beta.unsqueeze(-1) * q_individual
                   + gamma.unsqueeze(-1) * q_surrounding)

        return q_total, rnn_output
    
    def compute_dynamic_weights(self, rnn_output, agent_types):
        """
        计算动态权重 α, β, γ
        Args:
            rnn_output: RNN输出 [batch_size, n_agents, rnn_hidden_dim]
            agent_types: 智能体类型 [batch_size, n_agents]
        Returns:
            alpha, beta, gamma: 动态权重 [batch_size, n_agents]
        """
        batch_size, n_agents, hidden_dim = rnn_output.shape
        
        if agent_types is not None:
            # 获取智能体类型嵌入
            agent_type_embed = self.agent_type_embedding(agent_types)  # [batch_size, n_agents, embed_dim]
            
            # 组合特征
            combined_features = th.cat([rnn_output, agent_type_embed], dim=-1)
            
            # 预测权重
            weights = self.weight_predictor(combined_features)  # [batch_size, n_agents, 3]
            
            alpha = weights[:, :, 0]
            beta = weights[:, :, 1] 
            gamma = weights[:, :, 2]
        else:
            # 如果没有类型信息，使用默认权重
            alpha = th.ones(batch_size, n_agents) * 0.33
            beta = th.ones(batch_size, n_agents) * 0.33
            gamma = th.ones(batch_size, n_agents) * 0.34
            
            if rnn_output.is_cuda:
                alpha = alpha.cuda()
                beta = beta.cuda()
                gamma = gamma.cuda()
        
        return alpha, beta, gamma
    
    def get_base_weights_by_type(self, agent_types):
        """
        根据智能体类型获取基础权重
        Args:
            agent_types: 智能体类型 [batch_size, n_agents]
        Returns:
            base_weights: 基础权重 [batch_size, n_agents, 3]
        """
        batch_size, n_agents = agent_types.shape
        base_weights = th.zeros(batch_size, n_agents, 3)
        
        if agent_types.is_cuda:
            base_weights = base_weights.cuda()
        
        # 类型特定的基础权重
        type_weights = {
            0: [0.2, 0.6, 0.2],  # marine: 更多个体行动
            1: [0.2, 0.2, 0.6],  # marauder: 更多局部配合
            2: [0.6, 0.2, 0.2],  # medic: 更多全局协调
            3: [0.3, 0.4, 0.3],  # stalker: 平衡
            4: [0.2, 0.5, 0.3],  # zealot: 个体+局部
            5: [0.4, 0.3, 0.3],  # colossus: 全局+其他
        }
        
        for b in range(batch_size):
            for a in range(n_agents):
                agent_type = agent_types[b, a].item()
                if agent_type in type_weights:
                    base_weights[b, a] = th.tensor(type_weights[agent_type])
                else:
                    base_weights[b, a] = th.tensor([0.33, 0.33, 0.34])  # 默认权重
        
        return base_weights

class TriQRNNAgentV2(TriQRNNAgent):
    """
    三元Q值智能体的改进版本，包含更多高级功能
    """
    def __init__(self, input_shape, args):
        super(TriQRNNAgentV2, self).__init__(input_shape, args)
        
        # 添加位置编码
        self.position_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        # 增强的权重预测器
        self.enhanced_weight_predictor = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim + self.agent_type_embed_dim + 16, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1)
        )
        
        # 权重历史记录
        self.weight_history = []
        self.max_history_size = 1000
        
    def forward(self, inputs, hidden_state, update_metric=None, agent_positions=None, agent_types=None):
        """增强版前向传播"""
        b, a, e = inputs.size()
        
        # 基础特征提取
        inputs = inputs.view(-1, e)
        x = F.relu(self.fc1(inputs), inplace=True)
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        rnn_output = self.rnn(x, h_in)
        
        if self.layer_norm is not None:
            rnn_output = self.layer_norm(rnn_output)
        
        rnn_output = rnn_output.reshape(b, a, -1)
        
        # 计算三元Q值
        q_common = self.q_common(rnn_output)
        
        # 个体Q值
        q_individual_list = []
        for i, q_ind_head in enumerate(self.q_individual):
            q_individual_list.append(q_ind_head(rnn_output[:, i, :]))
        q_individual = th.stack(q_individual_list, dim=1)
        
        # 局部协调Q值
        if agent_positions is not None:
            # 位置编码
            pos_encoding = self.position_encoder(agent_positions)
            # 将rnn_output重塑为[batch_size, n_agents, hidden_dim]用于注意力机制
            rnn_output_for_attention = rnn_output.reshape(b, a, -1)
            surrounding_features = self.surrounding_attention(rnn_output_for_attention, agent_positions)
            # 将注意力输出重塑回[batch_size * n_agents, hidden_dim]
            surrounding_features = surrounding_features.reshape(-1, self.rnn_hidden_dim)
            q_surrounding = self.q_surrounding(surrounding_features)
        else:
            q_surrounding = self.q_surrounding(rnn_output)
        
        # 增强的动态权重计算
        alpha, beta, gamma = self.compute_enhanced_weights(rnn_output, agent_types, agent_positions)
        
        # 记录权重历史
        self.record_weight_history(alpha, beta, gamma)
        
        # 组合Q值
        q_total = alpha.unsqueeze(-1) * q_common + beta.unsqueeze(-1) * q_individual + gamma.unsqueeze(-1) * q_surrounding
        
        return q_total, rnn_output
    
    def compute_enhanced_weights(self, rnn_output, agent_types, agent_positions):
        """增强版权重计算"""
        batch_size, n_agents, hidden_dim = rnn_output.shape
        
        # 基础特征
        features = [rnn_output]
        
        # 添加类型特征
        if agent_types is not None:
            agent_type_embed = self.agent_type_embedding(agent_types)
            features.append(agent_type_embed)
        else:
            features.append(th.zeros(batch_size, n_agents, self.agent_type_embed_dim))
        
        # 添加位置特征
        if agent_positions is not None:
            pos_encoding = self.position_encoder(agent_positions)
            features.append(pos_encoding)
        else:
            features.append(th.zeros(batch_size, n_agents, 16))
        
        # 组合特征
        combined_features = th.cat(features, dim=-1)
        
        # 预测权重
        weights = self.enhanced_weight_predictor(combined_features)
        
        alpha = weights[:, :, 0]
        beta = weights[:, :, 1]
        gamma = weights[:, :, 2]
        
        return alpha, beta, gamma
    
    def record_weight_history(self, alpha, beta, gamma):
        """记录权重历史"""
        weight_info = {
            'alpha': alpha.mean().item(),
            'beta': beta.mean().item(), 
            'gamma': gamma.mean().item()
        }
        self.weight_history.append(weight_info)
        
        # 限制历史大小
        if len(self.weight_history) > self.max_history_size:
            self.weight_history.pop(0)
    
    def get_weight_statistics(self):
        """获取权重统计信息"""
        if not self.weight_history:
            return {}
        
        alphas = [w['alpha'] for w in self.weight_history]
        betas = [w['beta'] for w in self.weight_history]
        gammas = [w['gamma'] for w in self.weight_history]
        
        return {
            'alpha_mean': np.mean(alphas),
            'alpha_std': np.std(alphas),
            'beta_mean': np.mean(betas),
            'beta_std': np.std(betas),
            'gamma_mean': np.mean(gammas),
            'gamma_std': np.std(gammas)
        }
