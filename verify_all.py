"""Run every offline check in sequence and summarise.

These checks never touch StarCraft: they cover the RND contract, the score and
reuse maths, gradient isolation, and a full TriQLearner.train() step on a
synthetic batch. Real training still has to happen on the GPU box.

    python verify_all.py            # run everything
    python verify_all.py -v         # also stream each script's output

Exit code is 0 only if every check passes.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    ("check_env.py", "environment and compatibility"),
    ("verify_fix1.py", "RND inputs, scores, U, weights, masks, types"),
    ("verify_gradients.py", "gradient isolation"),
    ("verify_train_smoke.py", "end-to-end train() on a synthetic batch"),
]


def run(script, verbose):
    path = os.path.join(HERE, script)
    if not os.path.exists(path):
        return None, "missing"
    proc = subprocess.run(
        [sys.executable, path],
        cwd=HERE,
        capture_output=not verbose,
        text=True)
    out = "" if verbose else (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out


def main():
    verbose = '-v' in sys.argv or '--verbose' in sys.argv

    print("=" * 70)
    print("TriQ offline verification")
    print("=" * 70)

    results = []
    for script, purpose in SCRIPTS:
        print("\n>>> {}   ({})".format(script, purpose))
        code, out = run(script, verbose)
        if code is None:
            print("    SKIPPED: file not found")
            results.append((script, "missing", ""))
            continue
        status = "PASS" if code == 0 else "FAIL"
        print("    {}".format(status))
        if code != 0 and not verbose and out:
            # surface only the failing lines so the summary stays readable
            for line in out.splitlines():
                if 'FAIL' in line or 'Error' in line or 'error' in line:
                    print("      {}".format(line.strip()))
        results.append((script, status, out))

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    failed = [r for r in results if r[1] != "PASS"]
    for script, status, _ in results:
        print("  {:28s} {}".format(script, status))

    if failed:
        print("\n{} of {} checks did not pass.".format(len(failed), len(results)))
        if not verbose:
            print("Re-run with -v to see full output.")
        return 1

    print("\nAll offline checks passed.")
    print("Real training still has to be validated on the GPU box:")
    print("  python src/main.py --config=triq_sep --env-config=sc2 \\")
    print("      --env_args.map_name=6h_vs_8z --t_max=200000")
    return 0


if __name__ == '__main__':
    sys.exit(main())
