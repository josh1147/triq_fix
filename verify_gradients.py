"""Gradient-isolation checks that need a real TriQRNNAgent + MultiRND pair.

StarCraft is not required: attention masks are synthesised from the metadata
contract, exactly as the controller would build them.
"""
import os
import sys
from types import SimpleNamespace as SN

import torch as th

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'MANGER', 'src'))

from modules.agents.triq_rnn_agent import TriQRNNAgent   # noqa: E402
from modules.layer.multi_rnd import MultiRND             # noqa: E402
from utils.spatial_utils import (                        # noqa: E402
    build_visibility_and_distance_masks_from_obs,
)

N_AGENTS, N_ENEMIES = 6, 8
SLICES = {"move": 4, "enemy": 8 * 5, "ally": 5 * 5, "own": 1}
OBS_DIM = sum(SLICES.values())
HIDDEN = 256
N_ACTIONS = 6 + N_ENEMIES
AGENT_IN = OBS_DIM + N_ACTIONS + N_AGENTS      # obs + last action + id

failures = []


def check(name, cond, detail=""):
    print("  {}  {} {}".format("PASS" if cond else "FAIL", name, detail if not cond else ""))
    if not cond:
        failures.append(name)


def make_args():
    return SN(
        n_agents=N_AGENTS, n_actions=N_ACTIONS, rnn_hidden_dim=HIDDEN,
        mixing_embed_dim=32, rnd_update_interval=2, max_extra_training=3,
        device='cpu', unit_type_bits=0, obs_slices=SLICES,
        obs_all_health=True, shield_bits_ally=0, obs_last_action=False,
        agent_type_embed_dim=16, attention_dim=32, attention_heads=2,
        use_simple_attention=False, surrounding_radius=6.0,
        use_layer_norm=False, use_orthogonal=False, gain=0.01,
    )


def make_scheme():
    return {"obs": {"vshape": OBS_DIM, "group": "agents",
                    "slices": SLICES, "unit_type_bits": 0}}


def make_batch(B, seed=0):
    g = th.Generator().manual_seed(seed)
    obs = th.rand(B, N_AGENTS, OBS_DIM, generator=g)
    a0 = SLICES["move"] + SLICES["enemy"]
    for k in range(N_AGENTS - 1):
        obs[..., a0 + k * 5] = (th.rand(B, N_AGENTS, generator=g) > 0.5).float()
    inputs = th.rand(B, N_AGENTS, AGENT_IN, generator=g)
    return obs, inputs


print("\n[1] agent forward exposes the attention feature")
args = make_args()
scheme = make_scheme()
agent = TriQRNNAgent(AGENT_IN, args)
m = MultiRND(args, scheme)

B = 4
obs, inputs = make_batch(B)
vis, dist = build_visibility_and_distance_masks_from_obs(obs, args)
hidden = agent.init_hidden().expand(B, N_AGENTS, -1)

q_total, rnn_out = agent(inputs, hidden, attention_masks=(vis, dist))
check("q_total shape", tuple(q_total.shape) == (B, N_AGENTS, N_ACTIONS), tuple(q_total.shape))
check("last_surrounding cached", hasattr(agent, 'last_surrounding'))
check("attention feature shape",
      tuple(agent.last_surrounding.shape) == (B, N_AGENTS, HIDDEN),
      tuple(agent.last_surrounding.shape))

print("\n[2] branch_weights replace the gate")
w = th.softmax(th.rand(B, N_AGENTS, 3, generator=th.Generator().manual_seed(1)), dim=-1)
q_w, _ = agent(inputs, hidden, attention_masks=(vis, dist), branch_weights=w)
check("weighted forward returns q", tuple(q_w.shape) == (B, N_AGENTS, N_ACTIONS))

try:
    agent(inputs, hidden, attention_masks=(vis, dist), branch_weights=th.rand(B, 3))
    check("bad weight shape rejected", False)
except ValueError:
    check("bad weight shape rejected", True)

print("\n[3] local RND errors do not leak into the Q network")
for p in agent.parameters():
    p.grad = None

e_local = m.local_errors(agent.last_surrounding.detach())
loss_local = e_local.mean()
loss_local.backward()
main_grads = [p.grad for p in agent.parameters() if p.grad is not None]
check("no Q-network gradients from local RND loss", len(main_grads) == 0,
      "{} tensors received gradient".format(len(main_grads)))
check("local predictor did receive gradient",
      any(p.grad is not None for p in m.rnd_local.parameters()))

print("\n[4] TD loss does not touch RND predictors")
for p in m.parameters():
    p.grad = None
q_td, _ = agent(inputs, hidden, attention_masks=(vis, dist))
q_td.pow(2).mean().backward()
rnd_grads = [p.grad for p in m.parameters() if p.grad is not None]
check("no RND gradients from a TD-like loss", len(rnd_grads) == 0,
      "{} tensors received gradient".format(len(rnd_grads)))

print("\n" + "=" * 60)
if failures:
    print("FAILED ({}): {}".format(len(failures), failures))
    sys.exit(1)
print("ALL GRADIENT CHECKS PASSED")
