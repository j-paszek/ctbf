#!/usr/bin/env python
import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES_ROOT = PROJECT_ROOT / "test" / "data" / "algorithm_cases"
LEGACY_TEST_VARIANTS = [
    "r2bss025",
    "r2bss05",
    "r2bss075",
    "r4bss05",
    "r4bss075",
    "r4bss05high",
    "r4bss05highdm",
]
ADAPTIVE_RADIUS_TEST_VARIANTS = [
    "rAbss025",
    "rAbss05",
    "rAbss075",
    "rAbss05high",
    "rAbss05highdm",
]
TEST_VARIANTS = LEGACY_TEST_VARIANTS
ALL_TEST_VARIANTS = LEGACY_TEST_VARIANTS + ADAPTIVE_RADIUS_TEST_VARIANTS


@dataclass
class MomentTotals:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0

    def add(self, value):
        value = float(value)
        self.count += 1
        self.total += value
        self.total_sq += value * value

    def add_many(self, values):
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return
        self.count += int(values.size)
        self.total += float(values.sum())
        self.total_sq += float(np.square(values).sum())

    @property
    def mean(self):
        return self.total / self.count if self.count else 0.0

    @property
    def variance(self):
        if not self.count:
            return 0.0
        return max(self.total_sq / self.count - self.mean * self.mean, 0.0)


@dataclass
class CaseStats:
    variant: str
    seed: int
    r_dist: float
    biopsy_size_scalable: float | None
    profile: str | None
    input_pair_distances: MomentTotals = field(default_factory=MomentTotals)
    radius_neighbors: MomentTotals = field(default_factory=MomentTotals)
    plausible_neighbors: MomentTotals = field(default_factory=MomentTotals)
    tied_candidates: MomentTotals = field(default_factory=MomentTotals)
    selected_parent_group_sizes: MomentTotals = field(default_factory=MomentTotals)
    parent_opportunities: int = 0
    actual_tie_breaks: int = 0
    raw_closest_ties: int = 0
    same_id_shortcuts: int = 0
    missing_parent_copies: int = 0
    unique_plausible_parent_choices: int = 0
    closest_parent_choices: int = 0
    multi_child_parent_groups: int = 0
    children_in_multi_child_groups: int = 0

    def to_row(self):
        mean_d = self.input_pair_distances.mean
        return {
            "variant": self.variant,
            "seed": self.seed,
            "r_dist": self.r_dist,
            "biopsy_size_scalable": self.biopsy_size_scalable,
            "profile": self.profile,
            "parent_opportunities": self.parent_opportunities,
            "r_neighborhood_mean": self.radius_neighbors.mean,
            "r_neighborhood_variance": self.radius_neighbors.variance,
            "plausible_r_neighborhood_mean": self.plausible_neighbors.mean,
            "plausible_r_neighborhood_variance": self.plausible_neighbors.variance,
            "actual_tie_breaks": self.actual_tie_breaks,
            "actual_tie_break_rate": safe_ratio(self.actual_tie_breaks, self.parent_opportunities),
            "raw_closest_ties": self.raw_closest_ties,
            "same_id_shortcuts": self.same_id_shortcuts,
            "same_id_shortcut_rate": safe_ratio(self.same_id_shortcuts, self.parent_opportunities),
            "missing_parent_copies": self.missing_parent_copies,
            "missing_parent_copy_rate": safe_ratio(self.missing_parent_copies, self.parent_opportunities),
            "unique_plausible_parent_choices": self.unique_plausible_parent_choices,
            "closest_parent_choices": self.closest_parent_choices,
            "tied_candidates_mean": self.tied_candidates.mean,
            "selected_parent_group_size_mean": self.selected_parent_group_sizes.mean,
            "selected_parent_group_size_variance": self.selected_parent_group_sizes.variance,
            "selected_parent_group_count": self.selected_parent_group_sizes.count,
            "multi_child_parent_groups": self.multi_child_parent_groups,
            "multi_child_parent_group_rate": safe_ratio(
                self.multi_child_parent_groups,
                self.selected_parent_group_sizes.count,
            ),
            "children_in_multi_child_groups": self.children_in_multi_child_groups,
            "children_in_multi_child_group_rate": safe_ratio(
                self.children_in_multi_child_groups,
                self.parent_opportunities,
            ),
            "input_pair_distance_count": self.input_pair_distances.count,
            "input_pair_distance_mean": mean_d,
            "input_pair_distance_variance": self.input_pair_distances.variance,
            "r_over_d": safe_ratio(self.r_dist, mean_d),
            "r_minus_d": self.r_dist - mean_d,
        }


@dataclass
class VariantStats:
    variant: str
    cases: int = 0
    r_values: set = field(default_factory=set)
    r_distances: MomentTotals = field(default_factory=MomentTotals)
    input_pair_distances: MomentTotals = field(default_factory=MomentTotals)
    radius_neighbors: MomentTotals = field(default_factory=MomentTotals)
    plausible_neighbors: MomentTotals = field(default_factory=MomentTotals)
    tied_candidates: MomentTotals = field(default_factory=MomentTotals)
    selected_parent_group_sizes: MomentTotals = field(default_factory=MomentTotals)
    parent_opportunities: int = 0
    actual_tie_breaks: int = 0
    raw_closest_ties: int = 0
    same_id_shortcuts: int = 0
    missing_parent_copies: int = 0
    unique_plausible_parent_choices: int = 0
    closest_parent_choices: int = 0
    multi_child_parent_groups: int = 0
    children_in_multi_child_groups: int = 0

    def add_case(self, stats):
        self.cases += 1
        self.r_values.add(stats.r_dist)
        self.r_distances.add(stats.r_dist)
        self.input_pair_distances.count += stats.input_pair_distances.count
        self.input_pair_distances.total += stats.input_pair_distances.total
        self.input_pair_distances.total_sq += stats.input_pair_distances.total_sq
        self.radius_neighbors.count += stats.radius_neighbors.count
        self.radius_neighbors.total += stats.radius_neighbors.total
        self.radius_neighbors.total_sq += stats.radius_neighbors.total_sq
        self.plausible_neighbors.count += stats.plausible_neighbors.count
        self.plausible_neighbors.total += stats.plausible_neighbors.total
        self.plausible_neighbors.total_sq += stats.plausible_neighbors.total_sq
        self.tied_candidates.count += stats.tied_candidates.count
        self.tied_candidates.total += stats.tied_candidates.total
        self.tied_candidates.total_sq += stats.tied_candidates.total_sq
        self.selected_parent_group_sizes.count += stats.selected_parent_group_sizes.count
        self.selected_parent_group_sizes.total += stats.selected_parent_group_sizes.total
        self.selected_parent_group_sizes.total_sq += stats.selected_parent_group_sizes.total_sq
        self.parent_opportunities += stats.parent_opportunities
        self.actual_tie_breaks += stats.actual_tie_breaks
        self.raw_closest_ties += stats.raw_closest_ties
        self.same_id_shortcuts += stats.same_id_shortcuts
        self.missing_parent_copies += stats.missing_parent_copies
        self.unique_plausible_parent_choices += stats.unique_plausible_parent_choices
        self.closest_parent_choices += stats.closest_parent_choices
        self.multi_child_parent_groups += stats.multi_child_parent_groups
        self.children_in_multi_child_groups += stats.children_in_multi_child_groups

    def to_row(self):
        r_dist = sorted(self.r_values)[0] if len(self.r_values) == 1 else self.r_distances.mean
        mean_d = self.input_pair_distances.mean
        return {
            "variant": self.variant,
            "cases": self.cases,
            "r_dist": r_dist,
            "r_dist_variance": self.r_distances.variance,
            "parent_opportunities": self.parent_opportunities,
            "r_neighborhood_mean": self.radius_neighbors.mean,
            "r_neighborhood_variance": self.radius_neighbors.variance,
            "plausible_r_neighborhood_mean": self.plausible_neighbors.mean,
            "plausible_r_neighborhood_variance": self.plausible_neighbors.variance,
            "actual_tie_breaks": self.actual_tie_breaks,
            "actual_tie_break_rate": safe_ratio(self.actual_tie_breaks, self.parent_opportunities),
            "raw_closest_ties": self.raw_closest_ties,
            "same_id_shortcuts": self.same_id_shortcuts,
            "same_id_shortcut_rate": safe_ratio(self.same_id_shortcuts, self.parent_opportunities),
            "missing_parent_copies": self.missing_parent_copies,
            "missing_parent_copy_rate": safe_ratio(self.missing_parent_copies, self.parent_opportunities),
            "unique_plausible_parent_choices": self.unique_plausible_parent_choices,
            "closest_parent_choices": self.closest_parent_choices,
            "tied_candidates_mean": self.tied_candidates.mean,
            "selected_parent_group_size_mean": self.selected_parent_group_sizes.mean,
            "selected_parent_group_size_variance": self.selected_parent_group_sizes.variance,
            "selected_parent_group_count": self.selected_parent_group_sizes.count,
            "multi_child_parent_groups": self.multi_child_parent_groups,
            "multi_child_parent_group_rate": safe_ratio(
                self.multi_child_parent_groups,
                self.selected_parent_group_sizes.count,
            ),
            "children_in_multi_child_groups": self.children_in_multi_child_groups,
            "children_in_multi_child_group_rate": safe_ratio(
                self.children_in_multi_child_groups,
                self.parent_opportunities,
            ),
            "input_pair_distance_count": self.input_pair_distances.count,
            "input_pair_distance_mean": mean_d,
            "input_pair_distance_variance": self.input_pair_distances.variance,
            "r_over_d": safe_ratio(r_dist, mean_d),
            "r_minus_d": r_dist - mean_d,
        }


def safe_ratio(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def clone_cell(cell):
    return {
        "node_id": cell.get("node_id"),
        "cell_id": cell.get("cell_id"),
        "generation": cell.get("generation"),
        "genome": list(cell.get("genome", [])),
    }


def cell_lists_from_input(input_case):
    return [
        [clone_cell(cell) for cell in biopsy["cells"]]
        for biopsy in input_case["biopsies"]
        if biopsy["cells"]
    ]


def extend_biopsy_levels(cell_lists):
    cell_levels = defaultdict(list)
    for level, cell_list in enumerate(cell_lists):
        for cell in cell_list:
            cell_levels[cell["cell_id"]].append(level)

    for cell_id, levels in cell_levels.items():
        if len(levels) <= 1:
            continue

        min_level, max_level = min(levels), max(levels)
        for level in range(min_level, max_level + 1):
            if any(cell["cell_id"] == cell_id for cell in cell_lists[level]):
                continue

            nearest_level = min(levels, key=lambda existing_level: abs(existing_level - level))
            original = next(cell for cell in cell_lists[nearest_level] if cell["cell_id"] == cell_id)
            cell_lists[level].append(clone_cell(original))

    return cell_lists


def is_biologically_plausible_ancestor(parent, child):
    parent_genome = np.asarray(parent["genome"])
    child_genome = np.asarray(child["genome"])
    return not bool(np.any((parent_genome == 0) & (child_genome > 0)))


def pairwise_distances(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape[0] < 2:
        return np.array([], dtype=float)
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def distance(full_dist_matrix, id_to_index, a, b):
    return full_dist_matrix[id_to_index[a["cell_id"]], id_to_index[b["cell_id"]]]


def analyze_case(input_case, *, extend_levels=True):
    matrix_data = input_case["distance_matrices"]["cnp2cnp"]
    ids = matrix_data["ids"]
    full_dist_matrix = np.asarray(matrix_data["matrix"], dtype=float)
    id_to_index = {cell_id: index for index, cell_id in enumerate(ids)}
    cell_lists = cell_lists_from_input(input_case)
    if extend_levels:
        cell_lists = extend_biopsy_levels(cell_lists)

    stats = CaseStats(
        variant=input_case["variant"],
        seed=int(input_case["seed"]),
        r_dist=float(input_case["r_dist"]),
        biopsy_size_scalable=input_case.get("biopsy_size_scalable"),
        profile=input_case.get("profile"),
    )
    stats.input_pair_distances.add_many(pairwise_distances(full_dist_matrix))

    for level_index in reversed(range(1, len(cell_lists))):
        upper_cells = cell_lists[level_index - 1]
        bottom_cells = cell_lists[level_index]
        children_by_parent = defaultdict(int)
        for child in bottom_cells:
            stats.parent_opportunities += 1
            selected_parent = None
            child_idx = id_to_index[child["cell_id"]]
            candidates = [
                parent
                for parent in upper_cells
                if full_dist_matrix[child_idx, id_to_index[parent["cell_id"]]] <= stats.r_dist
            ]
            stats.radius_neighbors.add(len(candidates))

            if len(candidates) > 1:
                min_raw_distance = min(
                    full_dist_matrix[child_idx, id_to_index[parent["cell_id"]]]
                    for parent in candidates
                )
                raw_tied = [
                    parent
                    for parent in candidates
                    if full_dist_matrix[child_idx, id_to_index[parent["cell_id"]]] == min_raw_distance
                ]
                if len(raw_tied) > 1:
                    stats.raw_closest_ties += 1

            same_id_matches = [parent for parent in candidates if parent["cell_id"] == child["cell_id"]]
            if same_id_matches:
                stats.same_id_shortcuts += 1
                stats.plausible_neighbors.add(0)
                selected_parent = same_id_matches[0]
                children_by_parent[id(selected_parent)] += 1
                continue

            plausible_candidates = [
                parent
                for parent in candidates
                if is_biologically_plausible_ancestor(parent, child)
            ]
            stats.plausible_neighbors.add(len(plausible_candidates))

            if len(plausible_candidates) == 0:
                stats.missing_parent_copies += 1
                upper_cells.append(clone_cell(child))
                continue
            if len(plausible_candidates) == 1:
                stats.unique_plausible_parent_choices += 1
                selected_parent = plausible_candidates[0]
                children_by_parent[id(selected_parent)] += 1
                continue

            min_distance = min(
                distance(full_dist_matrix, id_to_index, child, parent)
                for parent in plausible_candidates
            )
            tied_candidates = [
                parent
                for parent in plausible_candidates
                if distance(full_dist_matrix, id_to_index, child, parent) == min_distance
            ]
            if len(tied_candidates) > 1:
                stats.actual_tie_breaks += 1
                stats.tied_candidates.add(len(tied_candidates))
            else:
                stats.closest_parent_choices += 1
            selected_parent = tied_candidates[0]
            children_by_parent[id(selected_parent)] += 1

        for group_size in children_by_parent.values():
            stats.selected_parent_group_sizes.add(group_size)
            if group_size > 1:
                stats.multi_child_parent_groups += 1
                stats.children_in_multi_child_groups += group_size

    return stats


def selected_variants(variants):
    if not variants:
        return TEST_VARIANTS
    unknown = sorted(set(variants) - set(ALL_TEST_VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Available: {ALL_TEST_VARIANTS}")
    return variants


def case_input_paths(cases_root, variants, seeds=None, limit=None):
    seed_filter = set(seeds or [])
    for variant in variants:
        variant_dir = Path(cases_root) / variant
        if not variant_dir.exists():
            continue
        paths = [
            seed_dir / "input.json"
            for seed_dir in sorted(
                [path for path in variant_dir.iterdir() if path.is_dir() and path.name.isdigit()],
                key=lambda path: int(path.name),
            )
            if (seed_dir / "input.json").exists()
        ]
        if seed_filter:
            paths = [path for path in paths if int(path.parent.name) in seed_filter]
        if limit is not None:
            paths = paths[:limit]
        for path in paths:
            yield path


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_float(value):
    if value == "":
        return ""
    return f"{float(value):.6f}"


def summary_markdown(variant_rows, *, cases_root, extend_levels):
    lines = [
        "# Biopsy Neighborhood Stats",
        "",
        f"- Cases root: `{cases_root}`",
        f"- Extended repeated biopsy levels: `{extend_levels}`",
        "- `actual_tie_breaks` counts exact closest-distance ties after same-cell shortcuts and plausibility filtering, matching the place where `candidate_tie_breaker` is used.",
        "- `r_neighborhood_mean` counts all upper-level cells within radius `r` for each lower-level cell before same-cell and plausibility rules.",
        "- `d` is the mean pairwise distance over all unique sampled cell ids in the frozen input distance matrix.",
        "",
        "| variant | cases | r | d mean | r/d | r-neighborhood mean | r-neighborhood variance | actual tie-breaks | tie-break rate | missing-parent rate | same-id rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in variant_rows:
        lines.append(
            "| {variant} | {cases} | {r} | {d} | {r_over_d} | {rn_mean} | {rn_var} | {ties} | {tie_rate} | {missing_rate} | {same_rate} |".format(
                variant=row["variant"],
                cases=row["cases"],
                r=format_float(row["r_dist"]),
                d=format_float(row["input_pair_distance_mean"]),
                r_over_d=format_float(row["r_over_d"]),
                rn_mean=format_float(row["r_neighborhood_mean"]),
                rn_var=format_float(row["r_neighborhood_variance"]),
                ties=row["actual_tie_breaks"],
                tie_rate=format_float(row["actual_tie_break_rate"]),
                missing_rate=format_float(row["missing_parent_copy_rate"]),
                same_rate=format_float(row["same_id_shortcut_rate"]),
            )
        )
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze biopsy parent-selection neighborhoods from frozen JSON input cases."
    )
    parser.add_argument("--cases-root", type=Path, default=DEFAULT_CASES_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--variant", action="append", choices=ALL_TEST_VARIANTS)
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit seeds per selected variant.")
    parser.add_argument(
        "--no-extend-levels",
        action="store_true",
        help="Do not mimic extend_biopsy_levels before collecting parent-selection stats.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    variants = selected_variants(args.variant)
    extend_levels = not args.no_extend_levels
    case_stats = []
    variant_stats = {variant: VariantStats(variant) for variant in variants}

    for input_path in case_input_paths(args.cases_root, variants, seeds=args.seed, limit=args.limit):
        stats = analyze_case(load_json(input_path), extend_levels=extend_levels)
        case_stats.append(stats)
        variant_stats[stats.variant].add_case(stats)

    case_rows = [stats.to_row() for stats in case_stats]
    variant_rows = [
        variant_stats[variant].to_row()
        for variant in variants
        if variant_stats[variant].cases
    ]
    markdown = summary_markdown(
        variant_rows,
        cases_root=args.cases_root,
        extend_levels=extend_levels,
    )

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(args.output_dir / "biopsy_neighborhood_case_stats.csv", case_rows)
        write_csv(args.output_dir / "biopsy_neighborhood_variant_stats.csv", variant_rows)
        (args.output_dir / "biopsy_neighborhood_stats.md").write_text(markdown)
        print(f"Wrote stats to {args.output_dir}")

    print(markdown)


if __name__ == "__main__":
    main()
