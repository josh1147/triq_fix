import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.nmix import Mixer
from modules.mixers.vdn import VDNMixer
from modules.mixers.qatten import QattenMixer
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
from utils.spatial_utils import build_visibility_and_distance_masks_from_obs
import torch as th
from torch.optim import RMSprop, Adam
import numpy as np
from utils.th_utils import get_parameters_num
from modules.layer.multi_rnd import MultiRND

class TriQLearner:
    """
    三元Q值学习器
    基于MANGER的SEPQLearner扩展，支持三元Q值训练和多维新颖度机制
    """
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.scheme = scheme
        
        # 多维RND网络
        self.multi_rnd = MultiRND(args, scheme)
        
        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda else 'cpu')
        self.params = list(mac.parameters())
        
        # 混合器选择
        if args.mixer == "qatten":
            self.mixer = QattenMixer(args)
        elif args.mixer == "vdn":
            self.mixer = VDNMixer()
        elif args.mixer == "qmix":
            self.mixer = Mixer(args)
        else:
            raise "mixer error"
        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())
        
        # 优化器
        self.rnd_optimiser = Adam(params=self.multi_rnd.parameters(), 
                                 lr=args.rnd_lr, 
                                 weight_decay=getattr(args, "weight_decay", 0))
        
        print('Mixer Size: ')
        print(get_parameters_num(self.mixer.parameters()))
        print('Multi-RND Size: ')
        print(get_parameters_num(self.multi_rnd.parameters()))
        
        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params, 
                                 lr=args.lr, 
                                 weight_decay=getattr(args, "weight_decay", 0))
        else:
            self.optimiser = RMSprop(params=self.params, 
                                   lr=args.lr, 
                                   alpha=0.9, 
                                   eps=args.optim_eps)
        
        # 目标网络
        self.target_mac = copy.deepcopy(mac)
        
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.train_t = 0
        
        # 优先级经验回放
        self.use_per = getattr(self.args, 'use_per', False)
        self.return_priority = getattr(self.args, "return_priority", False)
        if self.use_per:
            self.priority_max = float('-inf')
            self.priority_min = float('inf')
        
        # 三元Q值统计
        self.q_stats = {
            'weight_alpha_mean': [],
            'weight_beta_mean': [],
            'weight_gamma_mean': []
        }

        # 新颖度与额外训练统计
        self.s_common_mean = 0.0
        self.s_individual_mean = 0.0
        self.s_local_mean = 0.0
        self.u_mean = 0.0
        self.u_max = 0
        self.n_extra = 0
        self.extra_training_ratio = 0.0
        self.extra_training_avg = 0.0
        self.extra_training_intensity = 0.0

        self.rnd_loss = th.zeros(())
        self.rnd_loss_details = {}
        self.L_td = th.zeros(())
        self.grad_norm = 0.0

        # 注意力监控指标
        self.att_vis_mean = 0.0
        self.att_dist_visible_mean = 0.0
        self.att_dist_visible_min = 0.0
        self.att_dist_visible_max = 0.0
        
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, per_weight=None):
        """
        训练函数
        Args:
            batch: 经验批次
            t_env: 环境时间步
            episode_num: 回合数
            per_weight: 优先级权重
        """
        # 提取基本信息
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # 1. 标准 TD。顺带收集每步的 Local RND 误差（标量，不是整份特征）
        L_td, grad_norm, local_errors = self._td_step(
            batch, rewards, actions, terminated, mask, avail_actions,
            per_weight, branch_weights=None, collect_loc_error=True)

        # 2. 三路误差与 RND 更新
        errors = self._rnd_errors(batch, local_errors)
        if self.multi_rnd.should_update():
            self.rnd_loss_details = self._update_rnd(batch, errors)

        # 3. 分数、额外更新次数、三路权重
        valid = batch["filled"][:, :, 0].float().unsqueeze(-1)  # [B, T, 1]
        scores = self.multi_rnd.scores(errors, mask=valid)
        update_counts = self.multi_rnd.update_counts(
            scores, max_extra=getattr(self.args, 'max_extra_training', 3))
        weights = self.multi_rnd.branch_weights(scores)  # [B, T, N, 3]

        self._record_novelty_stats(scores, update_counts, weights, valid, obs=batch["obs"])

        # 4. 额外 TD：先递减再训练，次数等于 batch 内最大计数
        n_extra = 0
        while update_counts.max() > 0:
            update_counts = update_counts - (update_counts > 0).long()
            self._td_step(
                batch, rewards, actions, terminated, mask, avail_actions,
                per_weight, branch_weights=weights, collect_loc_error=False)
            n_extra += 1
        self.n_extra = n_extra

        # 5. 更新目标网络
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        # 6. 记录统计信息
        self.log_training_stats(t_env)

        # 7. 返回优先级信息
        return self.compute_priority_info(rewards, actions, mask, scores)

    # ------------------------------------------------------------------
    # RND
    # ------------------------------------------------------------------
    def _rnd_errors(self, batch, local_errors):
        """Compute the three branch errors.

        Common and Individual only need ``batch["obs"]`` and are evaluated in
        one batch. Local uses the attention feature, which is only available
        during the Q forward pass, so its errors were accumulated there.
        """
        obs = batch["obs"]                       # [B, T, N, D]
        B, T, N, D = obs.shape
        M = B * T

        common_in, individual_in = self.multi_rnd.build_inputs(
            obs.reshape(M, N, D))
        e_common, e_individual = self.multi_rnd.branch_errors(
            common_in, individual_in)

        return {
            'common': e_common.view(B, T, N),
            'individual': e_individual.view(B, T, N),
            'local': local_errors,
        }

    def _update_rnd(self, batch, errors):
        """Back-propagate the summed branch losses, throttled by the interval."""
        mask = batch["filled"][:, :, 0].float().reshape(-1, 1)   # [B*T, 1]
        flat = {k: v.reshape(-1, v.size(-1)) for k, v in errors.items()}
        losses = self.multi_rnd.losses(flat, mask)

        self.rnd_optimiser.zero_grad()
        (losses['common'] + losses['individual'] + losses['local']).backward()
        self.rnd_optimiser.step()

        self.rnd_loss = (losses['common'] + losses['individual'] + losses['local']).detach()
        return {k: v.detach() for k, v in losses.items()}

    # ------------------------------------------------------------------
    # TD
    # ------------------------------------------------------------------
    def _td_step(self, batch, rewards, actions, terminated, mask, avail_actions,
                 per_weight, branch_weights=None, collect_loc_error=False):
        """One full TD update.

        Args:
            branch_weights: ``[B, T, N, 3]``; ``None`` selects the gate weights.
            collect_loc_error: when true, evaluate the Local RND error at each
                timestep from the detached attention feature. Accumulating
                scalars avoids keeping ``[B, T, N, hidden]`` resident.

        Returns:
            ``(loss, grad_norm, local_errors)`` where ``local_errors`` is
            ``[B, T, N]`` or ``None``.
        """
        self.mac.agent.train()
        mac_out = []
        local_errors = [] if collect_loc_error else None
        self.mac.init_hidden(batch.batch_size)

        for t in range(batch.max_seq_length):
            w_t = None if branch_weights is None else branch_weights[:, t]
            agent_outs = self.mac.forward(
                batch, t=t, update_metric=None, branch_weights=w_t)
            mac_out.append(agent_outs)
            if collect_loc_error:
                c_t = self.mac.last_c
                if c_t is None:
                    raise RuntimeError(
                        "mac.last_c is None; the controller did not cache the "
                        "attention feature.")
                local_errors.append(
                    self.multi_rnd.local_errors(c_t.detach()).detach())

        mac_out = th.stack(mac_out, dim=1)
        if collect_loc_error:
            local_errors = th.stack(local_errors, dim=1)

        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        with th.no_grad():
            self.target_mac.agent.train()
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)

            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)

            target_mac_out = th.stack(target_mac_out, dim=1)

            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"])

            if getattr(self.args, 'q_lambda', False):
                qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                qvals = self.target_mixer(qvals, batch["state"])
                targets = build_q_lambda_targets(rewards, terminated, mask, target_max_qvals, qvals,
                                                 self.args.gamma, self.args.td_lambda)
            else:
                targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals,
                                                  self.args.n_agents, self.args.gamma,
                                                  self.args.td_lambda)

        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

        td_error = (chosen_action_qvals - targets.detach())
        td_error2 = 0.5 * td_error.pow(2)

        mask = mask.expand_as(td_error2)
        masked_td_error = td_error2 * mask

        if self.use_per:
            per_weight = th.from_numpy(per_weight).unsqueeze(-1).to(device=self.device)
            masked_td_error = masked_td_error.sum(1) * per_weight

        loss = L_td = masked_td_error.sum() / mask.sum()

        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.L_td = L_td.detach()
        self.grad_norm = grad_norm

        return L_td.detach(), grad_norm, local_errors

    # ------------------------------------------------------------------
    # stats
    # ------------------------------------------------------------------
    def _record_novelty_stats(self, scores, update_counts, weights, valid, obs=None):
        """``valid`` is ``[B, T, 1]`` and broadcasts over the agent axis."""
        with th.no_grad():
            self._record_novelty_stats_impl(scores, update_counts, weights, valid, obs)

    def _record_novelty_stats_impl(self, scores, update_counts, weights, valid, obs):
        m = valid.to(dtype=scores['common'].dtype)
        n_agents = scores['common'].size(-1)
        denom = (m.sum() * n_agents).clamp(min=1.0)

        def masked_mean(x):
            # x is [B, T, N]; broadcast the [B, T, 1] mask over agents
            return float((x.detach() * m).sum() / denom)

        self.s_common_mean = masked_mean(scores['common'])
        self.s_individual_mean = masked_mean(scores['individual'])
        self.s_local_mean = masked_mean(scores['local'])
        self.u_mean = masked_mean(update_counts.to(scores['common'].dtype))
        self.u_max = int(update_counts.max().item())

        # 与旧的 extra_training_* 指标对齐：占比、均值、相对上限的强度
        max_extra = max(1, int(getattr(self.args, 'max_extra_training', 3)))
        self.extra_training_ratio = float((update_counts > 0).float().mean().item())
        self.extra_training_avg = self.u_mean
        self.extra_training_intensity = self.u_mean / max_extra

        for i, name in enumerate(('alpha', 'beta', 'gamma')):
            self.q_stats['weight_{}_mean'.format(name)].append(
                float(weights[..., i].detach().mean().item()))

        max_stats = 1000
        for key in self.q_stats:
            while len(self.q_stats[key]) > max_stats:
                self.q_stats[key].pop(0)

        self._record_attention_stats(obs)

    def _record_attention_stats(self, obs_data):
        """Visibility/distance monitors kept from the 3-3 branch."""
        if obs_data is None:
            return
        import torch as _th
        if len(obs_data.shape) == 4:
            B, T, n_agents, obs_dim = obs_data.shape
            steps = int(min(5, T))
            if steps <= 0:
                return
            idxs = _th.linspace(0, T - 1, steps=steps).round().long().unique()
            vis_vals, dn_means, dn_mins, dn_maxs = [], [], [], []
            for ti in idxs.tolist():
                obs_ti = obs_data[:, ti]
                vis_mask, dist_norm = build_visibility_and_distance_masks_from_obs(
                    obs_ti, self.args)
                vis_vals.append(vis_mask.float().mean())
                visible_mask = (vis_mask > 0)
                if visible_mask.any():
                    dn_visible = dist_norm.masked_select(visible_mask)
                    dn_means.append(dn_visible.mean())
                    dn_mins.append(dn_visible.min())
                    dn_maxs.append(dn_visible.max())
                else:
                    dn_means.append(_th.tensor(0.0, device=obs_ti.device))
                    dn_mins.append(_th.tensor(0.0, device=obs_ti.device))
                    dn_maxs.append(_th.tensor(0.0, device=obs_ti.device))
            self.att_vis_mean = float(_th.stack(vis_vals).mean().item())
            self.att_dist_visible_mean = float(_th.stack(dn_means).mean().item())
            self.att_dist_visible_min = float(_th.stack(dn_mins).mean().item())
            self.att_dist_visible_max = float(_th.stack(dn_maxs).mean().item())
        else:
            vis_mask, dist_norm = build_visibility_and_distance_masks_from_obs(
                obs_data, self.args)
            self.att_vis_mean = float(vis_mask.float().mean().item())
            visible_mask = (vis_mask > 0)
            if visible_mask.any():
                dn_visible = dist_norm.masked_select(visible_mask)
                self.att_dist_visible_mean = float(dn_visible.mean().item())
                self.att_dist_visible_min = float(dn_visible.min().item())
                self.att_dist_visible_max = float(dn_visible.max().item())
            else:
                self.att_dist_visible_mean = 0.0
                self.att_dist_visible_min = 0.0
                self.att_dist_visible_max = 0.0

    def extract_positions_from_batch(self, batch):
        """从批次中提取位置信息"""
        # 回退为占位实现：不从batch中提取位置
        return None

    def extract_types_from_batch(self, batch):
        """从批次中提取类型信息"""
        # 类型由控制器按 scheme/args 中的元数据提取
        return None

    def log_training_stats(self, t_env):
        """记录训练统计信息"""
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", float(self.L_td.item()), t_env)
            self.logger.log_stat("grad_norm", self.grad_norm, t_env)
            self.logger.log_stat("rnd_loss", float(self.rnd_loss.item()), t_env)
            for key, value in self.rnd_loss_details.items():
                self.logger.log_stat("rnd_loss_{}".format(key), float(value.item()), t_env)

            self.logger.log_stat("s_com_mean", self.s_common_mean, t_env)
            self.logger.log_stat("s_ind_mean", self.s_individual_mean, t_env)
            self.logger.log_stat("s_loc_mean", self.s_local_mean, t_env)
            self.logger.log_stat("u_mean", self.u_mean, t_env)
            self.logger.log_stat("u_max", float(self.u_max), t_env)
            self.logger.log_stat("n_extra", float(self.n_extra), t_env)

            self.logger.log_stat("extra_training_ratio", self.extra_training_ratio, t_env)
            self.logger.log_stat("extra_training_avg", self.extra_training_avg, t_env)
            self.logger.log_stat("extra_training_intensity", self.extra_training_intensity, t_env)

            self.logger.log_stat("att_vis_mean", self.att_vis_mean, t_env)
            self.logger.log_stat("att_dist_visible_mean", self.att_dist_visible_mean, t_env)
            self.logger.log_stat("att_dist_visible_min", self.att_dist_visible_min, t_env)
            self.logger.log_stat("att_dist_visible_max", self.att_dist_visible_max, t_env)

            if self.q_stats['weight_alpha_mean']:
                self.logger.log_stat("weight_alpha_mean",
                                     np.mean(self.q_stats['weight_alpha_mean'][-100:]), t_env)
                self.logger.log_stat("weight_beta_mean",
                                     np.mean(self.q_stats['weight_beta_mean'][-100:]), t_env)
                self.logger.log_stat("weight_gamma_mean",
                                     np.mean(self.q_stats['weight_gamma_mean'][-100:]), t_env)

            self.log_stats_t = t_env

    def compute_priority_info(self, rewards, actions, mask, scores):
        """计算优先级信息"""
        info = {}

        if self.use_per:
            if self.return_priority:
                info["td_errors_abs"] = rewards.sum(1).detach().to('cpu')
                self.priority_max = max(th.max(info["td_errors_abs"]).item(), self.priority_max)
                self.priority_min = min(th.min(info["td_errors_abs"]).item(), self.priority_min)
                info["td_errors_abs"] = (info["td_errors_abs"] - self.priority_min) / \
                                      (self.priority_max - self.priority_min + 1e-5)
            else:
                # 使用三路分数的平均作为优先级
                fused = self.multi_rnd.fused_score(scores)
                info["td_errors_abs"] = fused.mean(dim=2).detach().to('cpu')

        return info
    
    def _update_targets(self):
        """更新目标网络"""
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target networks")
    
    def cuda(self):
        """移动到GPU"""
        self.mac.cuda()
        self.target_mac.cuda()
        self.multi_rnd.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
    
    def save_models(self, path):
        """保存模型"""
        self.mac.save_models(path)
        th.save(self.multi_rnd.state_dict(), f"{path}/multi_rnd.th")
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), f"{path}/mixer.th")
        th.save(self.optimiser.state_dict(), f"{path}/opt.th")
        th.save(self.rnd_optimiser.state_dict(), f"{path}/rnd_opt.th")

    def load_models(self, path):
        """加载模型。

        The parameter set of ``multi_rnd`` changed when the fusion and
        coordinator networks were removed, so old checkpoints cannot be loaded
        strictly. Missing and unexpected keys are reported instead of raised.
        """
        self.mac.load_models(path)
        self.target_mac.load_models(path)

        state = th.load(f"{path}/multi_rnd.th",
                        map_location=lambda storage, loc: storage)
        result = self.multi_rnd.load_state_dict(state, strict=False)
        if result.missing_keys or result.unexpected_keys:
            self.logger.console_logger.info(
                "multi_rnd loaded with missing={} unexpected={}".format(
                    list(result.missing_keys), list(result.unexpected_keys)))

        if self.mixer is not None:
            self.mixer.load_state_dict(th.load(f"{path}/mixer.th", 
                                             map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load(f"{path}/opt.th", 
                                             map_location=lambda storage, loc: storage))
        self.rnd_optimiser.load_state_dict(th.load(f"{path}/rnd_opt.th",
                                                 map_location=lambda storage, loc: storage))
