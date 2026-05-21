#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path
import sys

from networkx.readwrite import json_graph

TOOLS_DIR = Path(__file__).resolve().parent
TEST_DIR = TOOLS_DIR.parent
PROJECT_ROOT = TEST_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from freeze_algorithm_variant_cases import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    EXT_GRF_METRIC_FIELD,
    LEGACY_GRF_SET_SIMILARITY_FIELD,
    json_ready,
    load_json,
    metric_summary,
)


DEFAULT_MODES = ("full_cnp", "biopsy_guided_top")


def _node_link_graph(data):
    try:
        return json_graph.node_link_graph(data, directed=True, edges="links")
    except TypeError:
        return json_graph.node_link_graph(data, directed=True)


def _variant_dirs(cases_root, selected_variants):
    root = Path(cases_root)
    if selected_variants:
        for variant in selected_variants:
            variant_dir = root / variant
            if variant_dir.exists():
                yield variant, variant_dir
        return

    for variant_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        yield variant_dir.name, variant_dir


def _seed_dirs(variant_dir, selected_seeds, limit):
    candidates = [
        path
        for path in variant_dir.iterdir()
        if path.is_dir() and path.name.isdigit() and (path / "input.json").exists()
    ]
    candidates = sorted(candidates, key=lambda path: int(path.name))
    if selected_seeds is not None:
        selected = set(selected_seeds)
        candidates = [path for path in candidates if int(path.name) in selected]
    if limit is not None:
        candidates = candidates[:limit]
    return candidates


def iter_result_files(cases_root, variants=None, seeds=None, modes=None, algorithms=None, limit=None):
    modes = tuple(modes or DEFAULT_MODES)
    algorithm_filter = set(algorithms or ())
    for variant, variant_dir in _variant_dirs(cases_root, variants):
        for seed_dir in _seed_dirs(variant_dir, seeds, limit):
            input_file = seed_dir / "input.json"
            for mode in modes:
                mode_dir = seed_dir / mode
                if not mode_dir.exists():
                    continue
                for result_file in sorted(mode_dir.glob("*.json")):
                    if algorithm_filter and result_file.stem not in algorithm_filter:
                        continue
                    yield variant, int(seed_dir.name), mode, result_file.stem, input_file, result_file


def refresh_result_metrics(input_file, result_file, dry_run=False):
    input_case = load_json(input_file)
    result = load_json(result_file)
    true_tree = _node_link_graph(input_case["true_tree"])
    reconstructed_tree = _node_link_graph(result["reconstructed_tree"])
    old_metrics = result.get("metrics", {})
    new_metrics = metric_summary(true_tree, reconstructed_tree)

    row = {
        "variant": result.get("variant"),
        "seed": result.get("seed"),
        "mode": result.get("mode"),
        "algorithm": result.get("algorithm", result_file.stem),
        "old_grf": old_metrics.get("grf", ""),
        "new_grf": new_metrics["grf"],
        "old_ext_grf": old_metrics.get(EXT_GRF_METRIC_FIELD, ""),
        "new_ext_grf": new_metrics[EXT_GRF_METRIC_FIELD],
        "old_legacy_grf": old_metrics.get(LEGACY_GRF_SET_SIMILARITY_FIELD, ""),
        "new_legacy_grf": new_metrics[LEGACY_GRF_SET_SIMILARITY_FIELD],
    }
    row["grf_changed"] = (
        row["old_grf"] == ""
        or abs(float(row["old_grf"]) - float(row["new_grf"])) > 1e-12
    )
    row["legacy_differs_from_exact"] = (
        abs(float(row["new_legacy_grf"]) - float(row["new_grf"])) > 1e-12
    )

    if not dry_run:
        result["metrics"] = new_metrics
        with open(result_file, "w") as f:
            json.dump(json_ready(result), f, indent=2)
            f.write("\n")
    return row


def write_summary(path, rows):
    rows = list(rows)
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Recompute stored frozen JSON result metrics from input.json true trees "
            "and stored reconstructed trees. This does not rerun simulation, "
            "distance computation, or reconstruction."
        )
    )
    parser.add_argument("--cases-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--variant", action="append", help="Variant to refresh. Defaults to all variant directories.")
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--mode", action="append", choices=DEFAULT_MODES, default=None)
    parser.add_argument("--algorithm-name", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit seeds per variant after filtering.")
    parser.add_argument("--summary-file", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true", help="Do not print one line per refreshed result file.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    for _, _, _, _, input_file, result_file in iter_result_files(
        args.cases_root,
        variants=args.variant,
        seeds=args.seed,
        modes=args.mode,
        algorithms=args.algorithm_name,
        limit=args.limit,
    ):
        row = refresh_result_metrics(input_file, result_file, dry_run=args.dry_run)
        rows.append(row)
        if not args.quiet:
            action = "checked" if args.dry_run else "refreshed"
            print(f"{action}: {result_file}")

    if args.summary_file is not None:
        write_summary(args.summary_file, rows)

    changed = sum(1 for row in rows if row["grf_changed"])
    legacy_differs = sum(1 for row in rows if row["legacy_differs_from_exact"])
    print(f"Results scanned: {len(rows)}")
    print(f"Corrected GRF changes: {changed}")
    print(f"Legacy differs from corrected exact GRF: {legacy_differs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
