import argparse
import os
import subprocess
import sys
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
TEST_DIR = TOOLS_DIR.parent
PROJECT_ROOT = TEST_DIR.parent


def command_text(command):
    return " ".join(str(part) for part in command)


def base_env(cases_root, extra_algorithm_names):
    env = os.environ.copy()
    env["CTBF_ALGORITHM_CASES_ROOT"] = str(Path(cases_root))
    env.pop("CTBF_JSON_EXTRA_ALGORITHM_NAMES", None)
    if extra_algorithm_names:
        env["CTBF_JSON_EXTRA_ALGORITHM_NAMES"] = ",".join(extra_algorithm_names)
    return env


def build_commands(include_extra):
    commands = [
        (
            "evaluator-exact-grf",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                str(TEST_DIR / "test_evaluator_grf.py"),
                str(
                    TEST_DIR
                    / "test_freeze_algorithm_variant_cases.py::test_metric_summary_stores_exact_ext_grf_and_legacy_set_similarity"
                ),
            ],
        ),
        (
            "frozen-json-metrics",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                str(
                    TEST_DIR
                    / "test_algorithm_case_json_workflows.py::test_all_json_reconstructed_tree_metrics_match_stored_values"
                ),
            ],
        ),
    ]
    if include_extra:
        commands.append(
            (
                "extra-json-metrics",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    str(
                        TEST_DIR
                        / "test_algorithm_case_json_workflows.py::test_selected_extra_json_reconstructed_tree_metrics_match_stored_values"
                    ),
                ],
            )
        )
    return commands


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the GRF metric gate: exact-GRF evaluator tests plus frozen JSON "
            "metric consistency checks."
        )
    )
    parser.add_argument(
        "--cases-root",
        type=Path,
        default=PROJECT_ROOT / "test" / "data" / "algorithm_cases",
        help="Frozen JSON cases root. Can point at a scratch refreshed copy.",
    )
    parser.add_argument(
        "--algorithm-name",
        action="append",
        default=None,
        help="Optional non-canonical result row to include in the extra JSON metric check.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    extra_algorithm_names = args.algorithm_name or []
    commands = build_commands(include_extra=bool(extra_algorithm_names))
    env = base_env(args.cases_root, extra_algorithm_names)

    for name, command in commands:
        print(f"\n== {name} ==")
        print(command_text(command))
        if args.dry_run:
            continue
        result = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
        if result.returncode:
            print(f"\nFailed: {name} exited with {result.returncode}")
            return result.returncode

    print("\nGRF metric gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
