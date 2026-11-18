"""Environment self-check: run this first, before anything else.

Catches the compatibility traps that break this codebase on modern Python and
NumPy, and reports what is missing without letting a single failure hide the
other findings.

Severity is scoped by what the check is for:
  BLOCKING   breaks even the offline checks in verify_*.py
  TRAINING   only needed on the GPU box that talks to StarCraft

    python check_env.py
"""
import importlib
import os
import platform
import sys

OK = "  OK   "
WARN = "  WARN "
FAIL = "  FAIL "

# Project source root, needed both by the import checks and by the source-level
# compatibility scan below.
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MANGER', 'src')

blocking = []
training = []
warnings = []


def line(status, msg):
    print("{}{}".format(status, msg))


def module_status(name, severity='training'):
    """Import a module and record the outcome at the given severity."""
    try:
        mod = importlib.import_module(name)
    except Exception as exc:
        line(FAIL, "{}: {}".format(name, type(exc).__name__))
        entry = (name, str(exc))
        if severity == 'blocking':
            blocking.append(entry)
        else:
            training.append(entry)
        return None
    version = getattr(mod, '__version__', '')
    line(OK, "{}{}".format(name, " " + version if version else ""))
    return mod


print("=" * 68)
print("TriQ environment check")
print("=" * 68)

print("\n[interpreter]")
line(OK, "python {}".format(platform.python_version()))
line(OK, "{}".format(platform.platform()))

print("\n[needed by the offline checks]")
torch = module_status('torch', 'blocking')
numpy = module_status('numpy', 'blocking')
module_status('yaml', 'blocking')
module_status('matplotlib', 'blocking')
module_status('pygame', 'blocking')

print("\n[needed to actually train]  (src/main.py, sacred runner, SC2 envs)")
module_status('sacred')
module_status('absl')
module_status('cloudpickle')
module_status('pysc2')
module_status('s2clientprotocol')
module_status('smac')

print("\n[optional]")
try:
    importlib.import_module('tensorboard_logger')
    line(OK, "tensorboard_logger")
except Exception:
    line(WARN, "tensorboard_logger missing (only needed if use_tensorboard=true)")
    warnings.append("tensorboard_logger")

try:
    importlib.import_module('gym')
    line(OK, "gym")
except Exception:
    line(WARN, "gym missing (only needed for the football env)")
    warnings.append("gym")

# ---------------------------------------------------------------------------
# The two traps that bite specifically on modern stacks.
# ---------------------------------------------------------------------------
print("\n[compatibility traps]")

import collections
# The interpreter losing the alias only matters if the source still uses it.
# Checking the source too avoids a false alarm after main.py was fixed.
main_py = os.path.join(SRC, 'main.py')
source_ok = False
if os.path.exists(main_py):
    with open(main_py, encoding='utf-8', errors='ignore') as fh:
        src = fh.read()
    source_ok = 'collections.abc.Mapping' in src

if hasattr(collections, 'Mapping'):
    line(OK, "collections.Mapping exists")
elif source_ok:
    line(OK, "collections.Mapping gone, but src/main.py already uses "
              "collections.abc.Mapping")
else:
    line(FAIL, "collections.Mapping is gone and src/main.py still uses it")
    print("         main.py does `isinstance(v, collections.Mapping)` inside")
    print("         recursive_dict_update. On 3.10+ that raises AttributeError")
    print("         before training starts. The offline checks do not touch")
    print("         main.py, so they still run.")
    print("         Fix: import collections.abc and use collections.abc.Mapping.")
    training.append(("collections.Mapping", "src/main.py still uses the old alias"))

if numpy is not None:
    major = int(numpy.__version__.split('.')[0])
    if major >= 2:
        line(FAIL, "numpy {} is >= 2.0".format(numpy.__version__))
        print("         src/components/action_selectors.py does")
        print("         `from matplotlib.pyplot import xcorr`.")
        print("         matplotlib re-exports that from numpy, and numpy 2.0")
        print("         deleted np.xcorr. Every controller import then dies.")
        print("         Fix: pin numpy<2 (already in requirements.txt).")
        blocking.append(("numpy", "must be <2 for pyplot.xcorr"))
    else:
        line(OK, "numpy {} (<2, pyplot.xcorr available)".format(numpy.__version__))

if torch is not None:
    line(OK if torch.cuda.is_available() else WARN,
         "cuda available: {}".format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        line(OK, "device: {}".format(torch.cuda.get_device_name(0)))
    else:
        warnings.append("no CUDA (fine for offline checks)")

# ---------------------------------------------------------------------------
# Can the modules under test even be imported?
# ---------------------------------------------------------------------------
print("\n[project imports]  (offline subset; no SC2 needed)")
sys.path.insert(0, SRC)
for mod in ('modules.layer.rnd', 'modules.layer.multi_rnd',
            'utils.spatial_utils'):
    try:
        importlib.import_module(mod)
        line(OK, mod)
    except Exception as exc:
        line(FAIL, "{}: {}".format(mod, type(exc).__name__))
        blocking.append((mod, str(exc)))

print("\n" + "=" * 68)
if blocking:
    print("BLOCKING the offline checks ({}):".format(len(blocking)))
    for name, why in blocking:
        print("  - {}: {}".format(name, why))

if training:
    print("\nNeeded on the training box ({}):".format(len(training)))
    for name, why in training:
        print("  - {}: {}".format(name, why))
    print("  Install with: uv pip install -r requirements-sc2.txt")

if blocking:
    print("\nResolve the blocking items first.")
    sys.exit(1)

print("\nOffline checks can run.")
if warnings:
    print("Warnings: {}".format(", ".join(warnings)))
print("\nNext: python verify_all.py")

