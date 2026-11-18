import torch as th
import torch.nn as nn

from .rnd import RND


class MultiRND(nn.Module):
    """Three branch-wise RND predictors with distinct inputs.

    All three are shared networks. Identity is expressed through the input
    vector rather than through separate per-agent predictors, so that at any
    single timestep the ``n_agents`` errors come from the same target/predictor
    pair and are therefore comparable.

    Branches:
        common      the whole local observation ``o_i``
        individual  ``[move_i ; own_i ; onehot(i)]``
        local       the attention feature ``c_i``, gradient-detached
    """

    def __init__(self, args, scheme):
        super(MultiRND, self).__init__()
        self.args = args
        self.scheme = scheme
        self.n_agents = args.n_agents
        self.obs_dim = int(scheme['obs']['vshape'])

        slices = scheme['obs'].get('slices', None)
        if slices is None:
            raise ValueError(
                "scheme['obs']['slices'] is required. It is published by "
                "StarCraft2Env.get_env_info and wired in run.py.")
        self.slices = slices
        total = slices['move'] + slices['enemy'] + slices['ally'] + slices['own']
        if total != self.obs_dim:
            raise ValueError(
                "Observation slices sum to {} but obs_dim is {}".format(
                    total, self.obs_dim))

        self.ind_dim = int(slices['move'] + slices['own'] + self.n_agents)
        self.loc_dim = int(args.rnn_hidden_dim)

        self.rnd_common = RND(args, input_size=self.obs_dim)
        self.rnd_individual = RND(args, input_size=self.ind_dim)
        self.rnd_local = RND(args, input_size=self.loc_dim)

        self.update_freq = 0
        self.update_interval = getattr(args, 'rnd_update_interval', 2)
        self.max_extra = getattr(args, 'max_extra_training', 3)

        # Cached identity matrix; keeps the one-hot ordering in one place.
        self.register_buffer(
            'agent_ids',
            th.eye(self.n_agents),
            persistent=False)

    # ------------------------------------------------------------------
    # inputs
    # ------------------------------------------------------------------
    def build_inputs(self, obs):
        """Split a flattened observation batch into branch inputs.

        Args:
            obs: ``[M, n_agents, obs_dim]`` where ``M`` is ``batch * time``.

        Returns:
            ``(common, individual)``, both ``[M, n_agents, *]``.

        The local branch needs the attention feature, which is produced during
        the Q forward pass, so it is not built here.
        """
        s = self.slices
        a0, a1 = 0, s['move']
        c1 = s['move'] + s['enemy']
        c2 = c1 + s['ally']

        move = obs[..., a0:a1]
        own = obs[..., c2:c2 + s['own']]

        ids = self.agent_ids.to(device=obs.device, dtype=obs.dtype)
        ids = ids.unsqueeze(0).expand(obs.size(0), -1, -1)

        individual = th.cat([move, own, ids], dim=-1)
        return obs, individual

    # ------------------------------------------------------------------
    # errors and losses
    # ------------------------------------------------------------------
    def branch_errors(self, common_in, individual_in):
        """Errors for the two observation-only branches."""
        return (self.rnd_common(common_in),
                self.rnd_individual(individual_in))

    def local_errors(self, c):
        """Errors for the attention branch. ``c`` must already be detached."""
        return self.rnd_local(c)

    def losses(self, errors, mask):
        """Masked mean of each branch error.

        Args:
            errors: dict with keys ``common``/``individual``/``local``, each
                ``[M, n_agents]``.
            mask: ``[M, 1]``, 1 for real timesteps.

        Padding positions are excluded from both the sum and the denominator;
        folding them into a plain ``mean`` would bias the predictors toward
        whatever the buffer leaves in the padded slots.
        """
        m = mask.to(dtype=errors['common'].dtype)
        denom = m.sum().clamp(min=1.0)
        return {
            k: (v * m).sum() / denom
            for k, v in errors.items()
        }

    def should_update(self):
        self.update_freq += 1
        if self.update_freq >= self.update_interval:
            self.update_freq = 0
            return True
        return False

    # ------------------------------------------------------------------
    # novelty -> scores -> reuse count / weights
    # ------------------------------------------------------------------
    @staticmethod
    def _score(errors, mask=None):
        """Clip normalised branch error into ``[0, 3]``.

        Mirrors MANGER's single-RND rule: mean and sample standard deviation
        (``n_agents - 1`` in the denominator) are taken across agents at each
        timestep, never across time or episodes.

        Args:
            errors: ``[..., n_agents]``; the last axis must be the agent axis.
            mask: optional, broadcastable to ``errors`` with 1 on real
                timesteps; padded positions are forced to zero so they cannot
                contribute to the reuse-count distribution.
        """
        mu = errors.mean(dim=-1, keepdim=True)
        sigma = errors.std(dim=-1, keepdim=True)
        s = ((errors - mu) / (sigma + 1e-8)).clamp(0.0, 3.0)
        if mask is not None:
            s = s * mask.to(dtype=s.dtype)
        return s

    def scores(self, errors, mask=None):
        return {k: self._score(v, mask) for k, v in errors.items()}

    def fused_score(self, scores):
        """Arithmetic mean of the three branch scores, before flooring."""
        return (scores['common'] + scores['individual'] + scores['local']) / 3.0

    def update_counts(self, scores, max_extra=None):
        """Reuse count ``U``.

        Averaging happens before the single floor so that ``(3, 0, 0)`` and
        ``(1, 1, 1)`` stay distinguishable during the average.

        Returns:
            Integer tensor, same shape as the scores, in ``[0, max_extra]``.
        """
        cap = self.max_extra if max_extra is None else int(max_extra)
        fused = self.fused_score(scores)
        return th.floor(fused).clamp(0, cap).to(th.int)

    def branch_weights(self, scores):
        """Per-agent softmax over the three branch scores.

        Returns:
            ``[M, n_agents, 3]`` with rows summing to one. All-zero scores give
            one third each, which is what softmax already produces.
        """
        stacked = th.stack(
            [scores['common'], scores['individual'], scores['local']], dim=-1)
        return th.softmax(stacked.detach(), dim=-1)


class AdaptiveMultiRND(nn.Module):
    """Deprecated placeholder kept only for import compatibility.

    The TriQ learner instantiates :class:`MultiRND`; this class never ran and is
    removed from the active code path. See ``openspec/changes/triq-fix-1``.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "AdaptiveMultiRND is not part of the TriQ training path.")
