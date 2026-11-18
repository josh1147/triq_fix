"""End-to-end smoke test of TriQLearner.train() with a synthetic batch.

StarCraft is unavailable locally, so the runner is bypassed and an EpisodeBatch
is filled with random data shaped exactly like the real buffer. The SC2 import
chain is bypassed too: the modules under test are loaded straight from their
files, with only the small helpers they import actually available.

This exercises the whole path (standard TD -> RND errors -> scores/counts/
weights -> extra TD -> target sync -> logging).
"""
import importlib.util
import os
import sys
import types
from types import SimpleNamespace as SN

import torch as th

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MANGER', 'src')
sys.path.insert(0, SRC)

failures = []


def check(name, cond, detail=""):
    print("  {}  {}".format("PASS" if cond else "FAIL", name))
    if not cond:
        print("        {}".format(detail))
        failures.append(name)


def load(name, path, register=True):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    if register:
        sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --- minimal package scaffolding so relative imports resolve ---------------
def make_pkg(name, path):
    pkg = types.ModuleType(name)
    pkg.__path__ = [path]
    sys.modules[name] = pkg
    return pkg


# `modules`, `modules.layer`, `utils`, `components`, `controllers`, `learners`
# must exist as packages, but their __init__ files pull in SC2, so we create
# empty package objects instead of importing the real ones.
for pkg, sub in (
    ('modules', 'modules'),
    ('modules.layer', r'modules\layer'),
    ('modules.mixers', r'modules\mixers'),
    ('modules.agents', r'modules\agents'),
    ('utils', 'utils'),
    ('components', 'components'),
    ('controllers', 'controllers'),
    ('learners', 'learners'),
):
    make_pkg(pkg, os.path.join(SRC, sub))

# stub envs.matrix_game (imported by learners/__init__ -> q_learner)
mg = types.ModuleType('envs.matrix_game')
mg.print_matrix_status = lambda *a, **k: None
sys.modules['envs'] = make_pkg('envs', os.path.join(SRC, 'envs'))
sys.modules['envs.matrix_game'] = mg

# real helper modules
load('utils.th_utils', os.path.join(SRC, r'utils\th_utils.py'))
rl_utils = load('utils.rl_utils', os.path.join(SRC, r'utils\rl_utils.py'))
spatial = load('utils.spatial_utils', os.path.join(SRC, r'utils\spatial_utils.py'))
episode_buffer = load('components.episode_buffer',
                      os.path.join(SRC, r'components\episode_buffer.py'))

# mixers
load('modules.mixers.nmix', os.path.join(SRC, r'modules\mixers\nmix.py'))
load('modules.mixers.vdn', os.path.join(SRC, r'modules\mixers\vdn.py'))

# qatten may import further deps; stub it since mixer='qmix'
qatten = types.ModuleType('modules.mixers.qatten')
qatten.QattenMixer = object
sys.modules['modules.mixers.qatten'] = qatten

# rnd layers
load('modules.layer.rnd', os.path.join(SRC, r'modules\layer\rnd.py'))
load('modules.layer.surrounding_attention',
     os.path.join(SRC, r'modules\layer\surrounding_attention.py'))
load('modules.layer.multi_rnd', os.path.join(SRC, r'modules\layer\multi_rnd.py'))
load('modules.agents.triq_rnn_agent',
     os.path.join(SRC, r'modules\agents\triq_rnn_agent.py'))

# controllers: basic_controller imports action selectors
acts = types.ModuleType('components.action_selectors')
acts.REGISTRY = {'epsilon_greedy': lambda args: types.SimpleNamespace(
    select_action=lambda *a, **k: None)}


class _BaseMAC(object):
    def __init__(self, *a, **k):
        pass

    def init_hidden(self, batch_size):
        pass

    def parameters(self):
        return self.agent.parameters()

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        pass

    def load_models(self, path):
        pass

    def load_state(self, other):
        pass


bc = types.ModuleType('controllers.basic_controller')
bc.BasicMAC = _BaseMAC
sys.modules['components.action_selectors'] = acts
sys.modules['controllers.basic_controller'] = bc

agents_pkg = sys.modules['modules.agents']
agents_pkg.REGISTRY = {'triq_rnn': sys.modules['modules.agents.triq_rnn_agent'].TriQRNNAgent}

load('controllers.triq_controller',
     os.path.join(SRC, r'controllers\triq_controller.py'))
load('learners.triq_learner', os.path.join(SRC, r'learners\triq_learner.py'))

EpisodeBatch = episode_buffer.EpisodeBatch
TriQMAC = sys.modules['controllers.triq_controller'].TriQMAC
TriQLearner = sys.modules['learners.triq_learner'].TriQLearner
get_parameters_num = sys.modules['utils.th_utils'].get_parameters_num

# ---------------------------------------------------------------------------
N_AGENTS, N_ENEMIES = 6, 8
N_ACTIONS = 6 + N_ENEMIES
SLICES = {"move": 4, "enemy": 8 * 5, "ally": 5 * 5, "own": 1}
OBS_DIM = sum(SLICES.values())
HIDDEN = 64
EP_LIMIT = 20
BATCH = 8
STATE_DIM = N_ENEMIES * 3 + N_AGENTS * 4 + N_AGENTS * N_ACTIONS


class FakeLogger:
    def __init__(self):
        self.values = {}

    def log_stat(self, key, value, t):
        self.values[key] = value

    class _Console:
        def info(self, msg):
            pass

    console_logger = _Console()


def make_args():
    return SN(
        n_agents=N_AGENTS, n_actions=N_ACTIONS, rnn_hidden_dim=HIDDEN,
        mixing_embed_dim=32, hypernet_embed=64, state_shape=STATE_DIM,
        rnd_update_interval=2, max_extra_training=3, device='cpu',
        unit_type_bits=0, obs_slices=SLICES, obs_all_health=True,
        shield_bits_ally=0, obs_last_action=False, obs_agent_id=True,
        agent_type_embed_dim=16, attention_dim=32, attention_heads=2,
        use_simple_attention=False, surrounding_radius=6.0,
        use_layer_norm=False, use_orthogonal=False, gain=0.01,
        extract_agent_positions=False, extract_agent_types=True,
        mixer='qmix', learner='triq_learner', agent='triq_rnn',
        mac='triq_mac', optimizer='adam', lr=1e-3, rnd_lr=1e-3,
        gamma=0.99, td_lambda=0.6, q_lambda=False, grad_norm_clip=10,
        use_per=False, return_priority=False, use_cuda=False,
        learner_log_interval=1, target_update_interval=2,
        debug_attention=False, save_probs=False,
        action_selector='epsilon_greedy', agent_output_type='q',
    )


def make_scheme():
    return {
        "state": {"vshape": STATE_DIM},
        "obs": {"vshape": OBS_DIM, "group": "agents",
                "slices": SLICES, "unit_type_bits": 0},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (N_ACTIONS,), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }


def make_batch(scheme, groups, seed=0):
    g = th.Generator().manual_seed(seed)
    L = EP_LIMIT + 1
    b = EpisodeBatch(scheme, groups, BATCH, L)

    b.data.transition_data["obs"][:] = th.rand(BATCH, L, N_AGENTS, OBS_DIM, generator=g)
    a0 = SLICES["move"] + SLICES["enemy"]
    for k in range(N_AGENTS - 1):
        b.data.transition_data["obs"][..., a0 + k * 5] = (
            th.rand(BATCH, L, N_AGENTS, generator=g) > 0.5).float()

    b.data.transition_data["state"][:] = th.rand(BATCH, L, STATE_DIM, generator=g)
    b.data.transition_data["actions"][:] = th.randint(
        0, N_ACTIONS, (BATCH, L, N_AGENTS, 1), generator=g)
    avail = th.ones(BATCH, L, N_AGENTS, N_ACTIONS, dtype=th.int)
    avail[..., -1] = 0
    b.data.transition_data["avail_actions"][:] = avail
    b.data.transition_data["reward"][:] = th.rand(BATCH, L, 1, generator=g)
    b.data.transition_data["terminated"][:] = 0
    filled = th.zeros(BATCH, L, 1)
    filled[:, :L - 1] = 1
    filled[0, 10:] = 0
    b.data.transition_data["filled"][:] = filled

    # actions_onehot comes from the preprocess transform; build it by hand
    onehot = th.nn.functional.one_hot(
        b.data.transition_data["actions"].squeeze(-1), N_ACTIONS).float()
    if "actions_onehot" not in b.data.transition_data:
        b.data.transition_data["actions_onehot"] = th.zeros(
            BATCH, L, N_AGENTS, N_ACTIONS)
    b.data.transition_data["actions_onehot"][:] = onehot
    return b


print("\n[setup]")
args = make_args()
scheme = make_scheme()
groups = {"agents": N_AGENTS}
batch = make_batch(scheme, groups)
check("batch obs shape",
      tuple(batch["obs"].shape) == (BATCH, EP_LIMIT + 1, N_AGENTS, OBS_DIM),
      tuple(batch["obs"].shape))

mac = TriQMAC(batch.scheme, groups, args)
logger = FakeLogger()
learner = TriQLearner(mac, batch.scheme, logger, args)
print("        main params: {}".format(get_parameters_num(learner.params)))
print("        rnd  params: {}".format(
    get_parameters_num(learner.multi_rnd.parameters())))

print("\n[1] three consecutive train steps")
try:
    for step in range(3):
        learner.train(batch, t_env=1000 * (step + 1), episode_num=step)
        print("        step {}: loss={:.4f} u_mean={:.3f} n_extra={} rnd={:.4f}".format(
            step, float(learner.L_td), learner.u_mean, learner.n_extra,
            float(learner.rnd_loss)))
    check("train() runs without error", True)
except Exception as exc:
    import traceback
    traceback.print_exc()
    check("train() runs without error", False, repr(exc))

print("\n[2] RND update cadence")
learner.multi_rnd.update_freq = 0
flags = [learner.multi_rnd.should_update() for _ in range(4)]
check("interval=2 toggles on the second call", flags == [False, True, False, True], flags)

print("\n[3] logged statistics")
expected = {"loss_td", "u_mean", "u_max", "n_extra", "s_com_mean", "s_ind_mean",
            "s_loc_mean", "rnd_loss", "att_vis_mean"}
missing = expected - set(logger.values)
check("expected keys logged", not missing, "missing {}".format(missing))
print("        keys: {}".format(sorted(logger.values)))

print("\n[4] U and weights are sane")
check("u_mean within [0,3]", 0.0 <= learner.u_mean <= 3.0, learner.u_mean)
check("n_extra within [0,max_extra]",
      0 <= learner.n_extra <= args.max_extra_training, learner.n_extra)

print("\n[5] gradients")
check("RND params got gradient",
      any(p.grad is not None for p in learner.multi_rnd.parameters()))
check("main params got gradient",
      any(p.grad is not None for p in learner.params))

print("\n" + "=" * 60)
if failures:
    print("FAILED ({}): {}".format(len(failures), failures))
    sys.exit(1)
print("SMOKE TEST PASSED")
