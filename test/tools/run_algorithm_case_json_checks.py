import argparse
import os
import subprocess
import sys
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
TEST_DIR = TOOLS_DIR.parent
PROJECT_ROOT = TEST_DIR.parent
TEST_FILE = TEST_DIR / "test_algorithm_case_json_workflows.py"
HEATMAP_SCRIPT = TOOLS_DIR / "heatmaps_from_json.py"


CHECKS = {
    "metrics": [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        f"{TEST_FILE}::test_all_json_reconstructed_tree_metrics_match_stored_values",
    ],
    "determinism": [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        f"{TEST_FILE}::test_all_json_reconstruction_is_deterministic_against_stored_tree",
    ],
    "true-tree-matrix": [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        f"{TEST_FILE}::test_all_json_true_tree_recomputes_true_tree_distance_matrix",
    ],
    "cnp2cnp-matrix": [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        f"{TEST_FILE}::test_all_json_biopsies_recompute_cnp2cnp_matrix",
    ],
    "heatmap": [
        sys.executable,
        str(HEATMAP_SCRIPT),
    ],
}

DEFAULT_CHECKS = ["metrics", "determinism", "heatmap"]
ALL_CHECKS = ["metrics", "determinism", "true-tree-matrix", "cnp2cnp-matrix", "heatmap"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen JSON legacy algorithm checks after adding or changing "
            "a reconstruction algorithm."
        )
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Also run true-tree and cnp2cnp distance-matrix replay checks.",
    )
    parser.add_argument(
        "--check",
        action="append",
        choices=ALL_CHECKS,
        help=(
            "Run only the named check. Can be passed multiple times. "
            "Defaults to metrics, determinism, and heatmap."
        ),
    )
    parser.add_argument(
        "--skip-heatmap",
        action="store_true",
        help="Do not regenerate heatmap/ranking outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without running them.",
    )
    return parser.parse_args()


def selected_checks(args):
    checks = args.check or (ALL_CHECKS if args.all else DEFAULT_CHECKS)
    if args.skip_heatmap:
        checks = [check for check in checks if check != "heatmap"]
    return checks


def command_text(command):
    return " ".join(str(part) for part in command)


def run_check(name, command):
    print(f"\n== {name} ==", flush=True)
    print(command_text(command), flush=True)
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "ctbf_mpl_config"))
    env.setdefault("XDG_CACHE_HOME", str(Path("/tmp") / "ctbf_xdg_cache"))
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    return subprocess.run(command, cwd=PROJECT_ROOT, env=env).returncode


def main():
    args = parse_args()
    checks = selected_checks(args)

    if args.dry_run:
        for name in checks:
            print(command_text(CHECKS[name]))
        return 0

    failures = []
    for name in checks:
        returncode = run_check(name, CHECKS[name])
        if returncode:
            failures.append((name, returncode))
            break

    if failures:
        name, returncode = failures[0]
        print(f"\nFailed: {name} exited with {returncode}")
        return returncode

    print("\nAll selected JSON legacy checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
