"""Static checks for the triq-fix-1 changes.

Runs without StarCraft: it fabricates the observation metadata that
StarCraft2Env.get_env_info would publish for 6h_vs_8z and exercises the
MultiRND contract, the score/count/weight maths, and gradient isolation.

Run from anywhere:
    python verify_fix1.py
"""
import os
import sys
from types import SimpleNamespace as SN

import torch as th

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'MANGER', 'src'))

from modules.layer.multi_rnd import MultiRND          # noqa: E402
from utils.spatial_utils import (                     # noqa: E402
    build_visibility_and_distance_masks_from_obs,
    extract_agent_types,
)

# 6h_vs_8z: Zerg, unit_type_bits=0, obs_all_health=True, no last action
N_AGENTS = 6
N_ENEMIES = 8
SLICES = {"move": 4, "enemy": 8 * 5, "ally": 5 * 5, "own": 1}
OBS_DIM = sum(SLICES.values())          # 70
HIDDEN = 256
MIX_EMBED = 32

failures = []


def check(name, cond, detail=""):
    if cond:
        print("  PASS  {}".format(name))
    else:
        print("  FAIL  {}  {}".format(name, detail))
        failures.append(name)


def make_args():
    return SN(
        n_agents=N_AGENTS,
        n_actions=6 + N_ENEMIES,
        rnn_hidden_dim=HIDDEN,
        mixing_embed_dim=MIX_EMBED,
        rnd_update_interval=2,
        max_extra_training=3,
        device='cpu',
        unit_type_bits=0,
        obs_slices=SLICES,
        obs_all_health=True,
        shield_bits_ally=0,
        obs_last_action=False,
    )


def make_scheme():
    return {"obs": {"vshape": OBS_DIM, "group": "agents",
                    "slices": SLICES, "unit_type_bits": 0}}


def make_obs(B, T=None, seed=0):
    g = th.Generator().manual_seed(seed)
    shape = (B, T, N_AGENTS, OBS_DIM) if T else (B, N_AGENTS, OBS_DIM)
    obs = th.rand(shape, generator=g)
    # ally segment: column 0 of each ally row is the visibility flag
    a0 = SLICES["move"] + SLICES["enemy"]
    row = 5
    for k in range(N_AGENTS - 1):
        base = a0 + k * row
        obs[..., base] = (th.rand(shape[:-1], generator=g) > 0.5).float()
    return obs


print("\n[1] observation slices")
check("slices sum to obs_dim", sum(SLICES.values()) == OBS_DIM,
      "{} != {}".format(sum(SLICES.values()), OBS_DIM))

print("\n[2] MultiRND construction and branch input widths")
args = make_args()
scheme = make_scheme()
m = MultiRND(args, scheme)
check("common input width", m.rnd_common.input_size == OBS_DIM,
      m.rnd_common.input_size)
expected_ind = SLICES["move"] + SLICES["own"] + N_AGENTS
check("individual input width", m.rnd_individual.input_size == expected_ind,
      "{} != {}".format(m.rnd_individual.input_size, expected_ind))
check("local input width", m.rnd_local.input_size == HIDDEN,
      m.rnd_local.input_size)
check("no fusion module", not hasattr(m, 'novelty_fusion'))
check("no coordinator module", not hasattr(m, 'novelty_coordinator'))

print("\n[3] branch inputs")
B, T = 4, 7
obs = make_obs(B, T)
M = B * T
common_in, ind_in = m.build_inputs(obs.reshape(M, N_AGENTS, OBS_DIM))
check("common shape", tuple(common_in.shape) == (M, N_AGENTS, OBS_DIM),
      tuple(common_in.shape))
check("individual shape", tuple(ind_in.shape) == (M, N_AGENTS, expected_ind),
      tuple(ind_in.shape))

ids = ind_in[..., -N_AGENTS:]
check("id block is one-hot", th.allclose(ids.sum(-1), th.ones(M, N_AGENTS)))
check("id block diagonal matches agent index",
      th.allclose(ids, th.eye(N_AGENTS).expand(M, -1, -1)))

move_ref = obs.reshape(M, N_AGENTS, OBS_DIM)[..., :SLICES["move"]]
own_start = OBS_DIM - SLICES["own"]
own_ref = obs.reshape(M, N_AGENTS, OBS_DIM)[..., own_start:]
check("move block copied", th.allclose(ind_in[..., :SLICES["move"]], move_ref))
check("own block copied",
      th.allclose(ind_in[..., SLICES["move"]:SLICES["move"] + SLICES["own"]],
                  own_ref))

ally_start = SLICES["move"] + SLICES["enemy"]
ind_width = SLICES["move"] + SLICES["own"]
check("individual width equals move+own+id",
      ind_in.shape[-1] == ind_width + N_AGENTS)
enemy_block = obs.reshape(M, N_AGENTS, OBS_DIM)[..., ally_start:ally_start + SLICES["enemy"]]
check("enemy segment does not appear verbatim in individual input",
      not any(th.allclose(ind_in[..., :ind_width], enemy_block[..., :ind_width])
              for _ in [0]))

print("\n[4] _score behaviour")
equal = th.ones(3, N_AGENTS)
s_eq = MultiRND._score(equal)
check("identical errors give zero scores", th.allclose(s_eq, th.zeros_like(s_eq)))

one_high = th.ones(1, N_AGENTS)
one_high[0, 0] = 10.0
s_one = MultiRND._score(one_high)
check("only the outlier is positive",
      s_one[0, 0] > 0 and th.allclose(s_one[0, 1:], th.zeros(N_AGENTS - 1)),
      s_one)
check("scores stay within [0,3]", float(s_one.min()) >= 0 and float(s_one.max()) <= 3)

masked = MultiRND._score(one_high.expand(2, -1),
                         mask=th.tensor([[1.0], [0.0]]))
check("mask zeroes padded rows", th.allclose(masked[1], th.zeros(N_AGENTS)))

print("\n[4b] _score on the [B, T, N] layout used by the learner")
Bt, Tt = 3, 5
err3 = th.rand(Bt, Tt, N_AGENTS, generator=th.Generator().manual_seed(2))
s3 = MultiRND._score(err3)
check("3-D scores keep the shape", tuple(s3.shape) == (Bt, Tt, N_AGENTS), tuple(s3.shape))
# standardisation must happen across agents, so each (b,t) row of raw errors
# with only one outlier should produce exactly one positive score there
one_hot_err = th.ones(Bt, Tt, N_AGENTS)
one_hot_err[..., 0] = 5.0
s_hot = MultiRND._score(one_hot_err)
check("outlier detected on the agent axis for 3-D input",
      bool((s_hot[..., 0] > 0).all()) and bool((s_hot[..., 1:] == 0).all()))
# if the axis were wrong (time instead of agents) the pattern would not hold
check("score invariance across time slices",
      th.allclose(s_hot[0, 0], s_hot[1, 3]))

print("\n[5] update_counts")
all_high = {k: th.full((1, N_AGENTS), 3.0) for k in ('common', 'individual', 'local')}
check("all three at 3 -> 3", int(m.update_counts(all_high).max()) == 3)

one_path = {'common': th.full((1, N_AGENTS), 3.0),
            'individual': th.zeros(1, N_AGENTS),
            'local': th.zeros(1, N_AGENTS)}
check("single path at 3 -> 1", int(m.update_counts(one_path).max()) == 1)

zeros = {k: th.zeros(1, N_AGENTS) for k in ('common', 'individual', 'local')}
check("all zero -> 0", int(m.update_counts(zeros).max()) == 0)
check("cap respects max_extra_training",
      int(m.update_counts(all_high, max_extra=2).max()) == 2)

print("\n[6] branch_weights")
w_zero = m.branch_weights(zeros)
check("zero scores -> uniform", th.allclose(w_zero, th.full_like(w_zero, 1 / 3)))
w_equal = m.branch_weights({'common': th.full((1, N_AGENTS), 1.5),
                            'individual': th.full((1, N_AGENTS), 1.5),
                            'local': th.full((1, N_AGENTS), 1.5)})
check("equal scores -> uniform", th.allclose(w_equal, th.full_like(w_equal, 1 / 3)))
check("weights sum to one", th.allclose(w_zero.sum(-1), th.ones(1, N_AGENTS)))
check("weights are detached", not w_zero.requires_grad)

print("\n[7] gradient isolation between predictors")
errors = {
    'common': m.rnd_common(common_in),
    'individual': m.rnd_individual(ind_in),
    'local': m.rnd_local(th.zeros(M, N_AGENTS, HIDDEN)),
}
losses = m.losses({k: v.reshape(M * N_AGENTS, 1) for k, v in errors.items()},
                  mask=th.ones(M * N_AGENTS, 1))
losses['common'].backward(retain_graph=True)
grads = {name: [p.grad for p in mod.parameters()]
         for name, mod in (('common', m.rnd_common),
                           ('individual', m.rnd_individual),
                           ('local', m.rnd_local))}
check("common receives gradient", any(g is not None for g in grads['common']))
check("individual untouched by common loss",
      all(g is None for g in grads['individual']))
check("local untouched by common loss", all(g is None for g in grads['local']))

print("\n[8] masks from observation metadata")
obs_flat = make_obs(B)
vm, dn = build_visibility_and_distance_masks_from_obs(obs_flat, args)
check("mask shape", tuple(vm.shape) == (B, N_AGENTS, N_AGENTS))
check("diagonal forced to zero", th.allclose(vm.diagonal(dim1=1, dim2=2),
                                             th.zeros(B, N_AGENTS)))
a0 = SLICES["move"] + SLICES["enemy"]
ally_ref = obs_flat[..., a0 + 0]
check("visibility follows the ally segment",
      th.allclose(vm[:, 0, 1], (ally_ref[:, 0] > 0.5).float()))
enemy_ref = obs_flat[..., SLICES["move"] + 0]
check("visibility ignores the enemy segment",
      not th.allclose(vm[:, 0, 1], (enemy_ref[:, 0] > 0.5).float()))

print("\n[9] missing metadata is rejected")
try:
    build_visibility_and_distance_masks_from_obs(
        obs_flat, SN(n_agents=N_AGENTS, n_actions=14))
    check("missing obs_slices raises", False)
except ValueError as e:
    check("missing obs_slices raises", 'obs_slices' in str(e), str(e))

print("\n[10] agent types")
check("homogeneous map -> None", extract_agent_types(obs_flat, unit_type_bits=0) is None)
hetero_slices = {"move": 4, "enemy": 8 * 6, "ally": 5 * 6, "own": 4}
hetero_dim = sum(hetero_slices.values())
h_obs = th.zeros(2, N_AGENTS, hetero_dim)
own_start = hetero_dim - hetero_slices["own"]
# own layout is [health, type0, type1, type2]; type bits start after health
own_type_start = own_start + 1
for i in range(N_AGENTS):
    h_obs[:, i, own_type_start + i % 3] = 1.0
t = extract_agent_types(h_obs, unit_type_bits=3, obs_slices=hetero_slices)
check("heterogeneous types read from own segment",
      t is not None and t.tolist() == [[i % 3 for i in range(N_AGENTS)]] * 2,
      None if t is None else t.tolist())

print("\n" + "=" * 60)
if failures:
    print("FAILED ({}): {}".format(len(failures), failures))
    sys.exit(1)
print("ALL CHECKS PASSED")
