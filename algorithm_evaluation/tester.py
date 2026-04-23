import argparse
import json
from pathlib import Path
import sys

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TEST_DATA_DIR = PROJECT_ROOT / "test" / "data"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ctbs import run_single_test
from ctbs_utils import get_biopsy_nodes_ids
from evaluator_full import evaluate_4
from evaluator import grf_tree
from reconstructor import build_evolution_tree, visualize_tree_plotly
from reconstructor_registry import (
    LEGACY_ALGORITHM_NAMES,
    get_algorithms_to_test,
    get_experimental_algorithms_to_test,
    get_legacy_algorithms_to_test,
)

CONFIG_BY_PROFILE = {
    "base": TEST_DATA_DIR / "config_for_pic.json",
    "high": TEST_DATA_DIR / "config_high.json",
    "highdm": TEST_DATA_DIR / "config_high_dm.json",
}
DEFAULT_BIOPSY_GENERATIONS = [4, 6, 8]
DEFAULT_SEEDS_FILE = TEST_DATA_DIR / "seeds.json"


def get_root_id(tr):
    roots = [n for n, indeg in tr.in_degree() if indeg == 0]
    if len(roots) != 1:
        raise ValueError(f"Tree must have exactly one root (found {len(roots)})")
    return roots[0]


def format_bss_token(bss):
    return str(bss).replace(".", "")


def build_variant_name(r_dist, biopsy_size_scalable, profile):
    variant = f"r{r_dist}bss{format_bss_token(biopsy_size_scalable)}"
    if profile == "high":
        variant += "high"
    elif profile == "highdm":
        variant += "highdm"
    return variant


def _normalize_seed_values(values):
    seeds = []
    seen = set()
    for value in values:
        seed = int(value)
        if seed not in seen:
            seen.add(seed)
            seeds.append(seed)
    return seeds


def load_seeds(seed_file):
    path = Path(seed_file)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
        if "seeds" in df.columns:
            return _normalize_seed_values(df["seeds"].dropna().tolist())
        if "seed" in df.columns:
            return _normalize_seed_values(df["seed"].dropna().tolist())
        raise ValueError(f"CSV seed file {path} must contain a 'seeds' or 'seed' column.")

    if suffix == ".json":
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if "seeds" in data:
                return _normalize_seed_values(data["seeds"])
            if "seed" in data:
                return _normalize_seed_values(data["seed"])
            raise ValueError(f"JSON seed file {path} must contain a 'seeds' or 'seed' field.")
        if isinstance(data, list):
            return _normalize_seed_values(data)
        raise ValueError(f"Unsupported JSON seed file structure in {path}.")

    raise ValueError(f"Unsupported seed file format for {path}. Use .csv or .json.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CTBF appendix benchmark for one test variant."
    )
    parser.add_argument("--r", type=int, required=True, dest="r_dist",
                        help="Reconstruction radius, e.g. 2 or 4.")
    parser.add_argument("--bss", type=float, required=True, dest="biopsy_size_scalable",
                        help="Biopsy-size scaling coefficient, e.g. 0.25, 0.5, 0.75.")
    parser.add_argument("--profile", choices=sorted(CONFIG_BY_PROFILE), default="base",
                        help="Simulation profile: base, high, or highdm.")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional config override. If omitted, derived from --profile.")
    parser.add_argument("--bedfile", type=str, default=None,
                        help="Optional simulator CSV file. Default: none.")
    parser.add_argument("--seeds-file", type=str, default=DEFAULT_SEEDS_FILE,
                        help="Path to a CSV or JSON file describing benchmark seeds.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Optional output directory. Default: results/<derived-variant>.")
    parser.add_argument("--seed", type=int, action="append", default=None,
                        help="Limit the run to selected seed(s). Can be passed multiple times.")
    parser.add_argument("--algorithm-index", type=int, action="append", default=None,
                        help="Limit the run to selected algorithm index/indices.")
    parser.add_argument("--algorithm-name", action="append", default=None,
                        help="Limit the run to selected algorithm name(s). Can be passed multiple times.")
    parser.add_argument("--list-algorithms", action="store_true",
                        help="Print available algorithm indexes and names, then exit.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the resolved benchmark plan without running simulations.")
    parser.add_argument("--write-newick", action="store_true",
                        help="Print Newick trees during runs.")
    return parser.parse_args()


def select_algorithm_indices(algorithms, algorithm_indexes=None, algorithm_names=None):
    selected = []
    algorithm_indexes = algorithm_indexes or []
    algorithm_names = algorithm_names or []

    for index in algorithm_indexes:
        if index < 0 or index >= len(algorithms):
            raise ValueError(f"Algorithm index {index} is out of range 0..{len(algorithms) - 1}.")
        selected.append(index)

    if algorithm_names:
        names_to_indices = {}
        duplicate_names = set()
        for index, algorithm in enumerate(algorithms):
            name = getattr(algorithm, "__name__", str(algorithm))
            if name in names_to_indices:
                duplicate_names.add(name)
            names_to_indices[name] = index
        if duplicate_names:
            duplicates = ", ".join(sorted(duplicate_names))
            raise ValueError(f"Algorithm names must be unique to select by name. Duplicates: {duplicates}")

        for name in algorithm_names:
            if name not in names_to_indices:
                available = ", ".join(names_to_indices)
                raise ValueError(f"Unknown algorithm name '{name}'. Available algorithms: {available}")
            selected.append(names_to_indices[name])

    if not selected:
        return list(range(len(algorithms)))
    return list(dict.fromkeys(selected))


def format_algorithm_listing(algorithms, legacy_count=None):
    lines = []
    for index, algorithm in enumerate(algorithms):
        name = getattr(algorithm, "__name__", str(algorithm))
        if legacy_count is None:
            category = "algorithm"
        elif index < legacy_count:
            category = "legacy"
        else:
            category = "experimental"
        lines.append(f"{index}: {name} [{category}]")
    return "\n".join(lines)


def format_run_plan(variant_name, config_path, seeds_source, output_dir, seeds, algorithms, selected_indices):
    lines = [
        f"Variant: {variant_name}",
        f"Config: {config_path}",
        f"Seeds source: {seeds_source}",
        f"Output directory: {output_dir}",
        f"Seeds: {len(seeds)}",
        "Seed values: " + ", ".join(str(seed) for seed in seeds),
        f"Algorithms: {selected_indices}",
    ]
    for index in selected_indices:
        algorithm_name = getattr(algorithms[index], "__name__", str(algorithms[index]))
        lines.append(f"  {index}: {algorithm_name}")
    return "\n".join(lines)


# === Summary function ===
def provide_summary(rec, nj, mode, seed, failures, rec_report, nj_report):
    p1 = rec[mode]["precision"]
    p2 = nj[mode]["precision"]
    f1 = rec[mode]["F1"]
    f2 = nj[mode]["F1"]

    # Append once per seed — ensure complete record
    if mode == "ancestors_multiset":
        rec_report["1-precision"].append(p1)
        rec_report["1-f1"].append(f1)
        nj_report["1-precision"].append(p2)
        nj_report["1-f1"].append(f2)
    elif mode == "ancestors_unique":
        rec_report["2-precision"].append(p1)
        rec_report["2-f1"].append(f1)
        nj_report["2-precision"].append(p2)
        nj_report["2-f1"].append(f2)
    elif mode == "ancestors_unique_restricted":
        rec_report["3-precision"].append(p1)
        rec_report["3-f1"].append(f1)
        nj_report["3-precision"].append(p2)
        nj_report["3-f1"].append(f2)


    print(f"Mode: {mode}")
    print(f"Precision: {p1:.4f} vs {p2:.4f}")
    print(f"F1: {f1:.4f} vs {f2:.4f}")

    if p1 < p2:
        failures[f"{mode}_precision_failures"].append(seed)
    if f1 < f2:
        failures[f"{mode}_f1_failures"].append(seed)


def check_one_alg(algo, counter, *, seeds, config_path, bedfile, biopsy_size_scalable,
                  biopsy_generations, r_dist, output_dir, write_newick):
    results_store = {
        "ancestors_multiset_precision_failures": [],
        "ancestors_multiset_f1_failures": [],
        "ancestors_unique_precision_failures": [],
        "ancestors_unique_f1_failures": [],
        "ancestors_unique_restricted_precision_failures": [],
        "ancestors_unique_restricted_f1_failures": []
    }
    rec_output = {"seed": [], "1-precision": [], "1-f1": [],
                  "2-precision": [], "2-f1": [], "3-precision": [], "3-f1": [], "grf": [],}
    nj_output = {"seed": [], "1-precision": [], "1-f1": [],
                 "2-precision": [], "2-f1": [], "3-precision": [], "3-f1": [], "grf": [],}
    algo_name = getattr(algo, "__name__", str(algo))
    for seed in seeds:
        print(f"\nTesting seed: {seed}")
        try:
            a, b, c = run_single_test(
                seed=seed,
                config=config_path,
                bedfile=bedfile,
                biopsy_size_scalable=biopsy_size_scalable,
                biopsy_generations=biopsy_generations,
                r_dist=r_dist,
                write_newick=write_newick,
                reconstruction_algorithm=algo,
            )

            biopsy_cell_ids = get_biopsy_nodes_ids(b, c)
            rec = evaluate_4(a, b, restrict_labels=biopsy_cell_ids)
            nj = evaluate_4(a, c, restrict_labels=biopsy_cell_ids)
            # --- GRF: true vs reconstructed, true vs NJ ---
            grf_rec = grf_tree(a, get_root_id(a), b, get_root_id(b))
            grf_nj = grf_tree(a, get_root_id(a), c, get_root_id(c))

            # Make sure seed is added once
            rec_output["seed"].append(seed)
            nj_output["seed"].append(seed)
            rec_output["grf"].append(grf_rec)
            nj_output["grf"].append(grf_nj)

            for mode in ["ancestors_multiset", "ancestors_unique", "ancestors_unique_restricted"]:
                provide_summary(rec, nj, mode, seed, results_store, rec_output, nj_output)

        except Exception as e:
            print(f"Error for seed {seed} using {algo_name}: {e}")

    print("\n\n=== Summary ===")
    print(results_store)

    # Save failure summary
    all_fail_seeds = sorted({s for v in results_store.values() for s in v})
    df_fail = pd.DataFrame({"seed": all_fail_seeds})
    for key, seeds in results_store.items():
        df_fail[key] = df_fail["seed"].apply(lambda x: 1 if x in seeds else 0)
    df_fail["total_failures"] = df_fail.drop(columns=["seed"]).sum(axis=1)
    output_dir.mkdir(parents=True, exist_ok=True)
    name1 = output_dir / f"{counter}out.csv"
    df_fail.to_csv(name1, index=False)

    # Save REC and NJ metric reports
    df_rec = pd.DataFrame(rec_output)
    name2 = output_dir / f"{counter}rec.csv"
    df_rec.to_csv(name2, index=False)

    df_nj = pd.DataFrame(nj_output)
    name3 = output_dir / f"{counter}nj.csv"
    df_nj.to_csv(name3, index=False)

    print("Results saved: ", name1, name2, name3)


if __name__ == "__main__":
    args = parse_args()
    config_path = Path(args.config) if args.config else CONFIG_BY_PROFILE[args.profile]
    variant_name = build_variant_name(args.r_dist, args.biopsy_size_scalable, args.profile)
    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "results" / variant_name
    algorithms = get_algorithms_to_test()
    if args.list_algorithms:
        print(format_algorithm_listing(algorithms, legacy_count=len(get_legacy_algorithms_to_test())))
        sys.exit(0)
    selected_seeds = args.seed if args.seed else load_seeds(args.seeds_file)

    selected_indices = select_algorithm_indices(
        algorithms,
        algorithm_indexes=args.algorithm_index,
        algorithm_names=args.algorithm_name,
    )
    seeds_source = "CLI --seed" if args.seed else args.seeds_file

    if args.dry_run:
        print(format_run_plan(
            variant_name,
            config_path,
            seeds_source,
            output_dir,
            selected_seeds,
            algorithms,
            selected_indices,
        ))
        sys.exit(0)

    print(f"Variant: {variant_name}")
    print(f"Config: {config_path}")
    print(f"Seeds source: {seeds_source}")
    print(f"Output directory: {output_dir}")
    print(f"Seeds: {len(selected_seeds)}")
    print(f"Algorithms: {selected_indices}")

    for counter in selected_indices:
        algo = algorithms[counter]
        algo_name = getattr(algo, "__name__", str(algo))
        print(f"\n--- Running tests with {algo_name} (index {counter}) ---")
        check_one_alg(
            algo,
            counter,
            seeds=selected_seeds,
            config_path=config_path,
            bedfile=args.bedfile,
            biopsy_size_scalable=args.biopsy_size_scalable,
            biopsy_generations=DEFAULT_BIOPSY_GENERATIONS,
            r_dist=args.r_dist,
            output_dir=output_dir,
            write_newick=args.write_newick,
        )
