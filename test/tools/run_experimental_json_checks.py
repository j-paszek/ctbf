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
        f"{TEST_FILE}::test_selected_extra_json_reconstructed_tree_metrics_match_stored_values",
    ],
    "determinism": [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        f"{TEST_FILE}::test_selected_extra_json_reconstruction_is_deterministic_against_stored_tree",
    ],
}

DEFAULT_CHECKS = ["metrics", "determinism"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run frozen JSON checks for non-canonical stored rows such as "
            "experimental algorithms or biopsy-preset benchmark rows."
        )
    )
    parser.add_argument(
        "--algorithm-name",
        action="append",
        required=True,
        help="Stored result row to validate. Can be passed multiple times.",
    )
    parser.add_argument(
        "--check",
        action="append",
        choices=DEFAULT_CHECKS + ["heatmap"],
        help="Run only the named check. Defaults to metrics and determinism.",
    )
    parser.add_argument(
        "--heatmap-output-file",
        type=Path,
        default=PROJECT_ROOT / "test" / "heatmaps_side_by_side_experimental.png",
        help="Output file used when heatmap check is enabled.",
    )
    parser.add_argument(
        "--skip-heatmap",
        action="store_true",
        help="Do not regenerate a heatmap for the selected rows.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without running them.",
    )
    return parser.parse_args()


def selected_checks(args):
    checks = args.check or list(DEFAULT_CHECKS)
    if args.skip_heatmap:
        checks = [check for check in checks if check != "heatmap"]
    return checks


def command_text(command):
    return " ".join(str(part) for part in command)


def run_check(name, command, extra_algorithm_names):
    print(f"\n== {name} ==", flush=True)
    print(command_text(command), flush=True)
    env = os.environ.copy()
    env["CTBF_JSON_EXTRA_ALGORITHM_NAMES"] = ",".join(extra_algorithm_names)
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "ctbf_mpl_config"))
    env.setdefault("XDG_CACHE_HOME", str(Path("/tmp") / "ctbf_xdg_cache"))
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    return subprocess.run(command, cwd=PROJECT_ROOT, env=env).returncode


def main():
    args = parse_args()
    checks = selected_checks(args)

    commands = {name: list(CHECKS[name]) for name in checks if name in CHECKS}
    if "heatmap" in checks:
        command = [
            sys.executable,
            str(HEATMAP_SCRIPT),
        ]
        for name in args.algorithm_name:
            command.extend(["--algorithm-name", name])
        command.extend(["--output-file", str(args.heatmap_output_file)])
        commands["heatmap"] = command

    if args.dry_run:
        for name in checks:
            print(command_text(commands[name]))
        return 0

    failures = []
    for name in checks:
        returncode = run_check(name, commands[name], args.algorithm_name)
        if returncode:
            failures.append((name, returncode))
            break

    if failures:
        name, returncode = failures[0]
        print(f"\nFailed: {name} exited with {returncode}")
        return returncode

    print("\nAll selected experimental JSON checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
