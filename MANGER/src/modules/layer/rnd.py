import torch
import torch.nn as nn


class RND(nn.Module):
    """Random Network Distillation with an explicit input width.

    The three TriQ branches consume vectors of different lengths (full
    observation, ``move + own + one-hot id``, attention features), so the width
    cannot be read from ``scheme['obs']['vshape']`` for all of them. ``scheme``
    is kept as a fallback for callers that only have the observation shape.
    """

    def __init__(self, args, scheme=None, input_size=None):
        super(RND, self).__init__()
        if input_size is None:
            if scheme is None:
                raise ValueError("RND needs either input_size or scheme")
            input_size = scheme['obs']['vshape']
        self.input_size = int(input_size)
        output_size = args.mixing_embed_dim
        self.update_freq = 0
        self.target = nn.Sequential(
            nn.Linear(self.input_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size)
        ).to(args.device)
        self.predictor = nn.Sequential(
            nn.Linear(self.input_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size)
        ).to(args.device)
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, x, requires_grad=True):
        target_output = self.target(x)
        predict_output = self.predictor(x)
        intrinsic_reward = ((target_output - predict_output) ** 2).sum(-1)
        if not requires_grad:
            return intrinsic_reward.detach()
        return intrinsic_reward
