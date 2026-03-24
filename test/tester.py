import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ctbs import run_single_test
from ctbs_utils import get_biopsy_nodes_ids
from evaluator_full import evaluate_4
from evaluator import grf_tree
from reconstructor import build_evolution_tree, visualize_tree_plotly, \
    neighbor_joining_adaptive_centrality, \
    neighbor_joining_adaptive_centrality_nonlinear, neighbor_joining_adaptive_centrality_reversed, \
    neighbor_joining_hybrid_opt, neighbor_joining_hybrid_opt_adaptive, neighbor_joining_hybrid_opt_v2, \
    neighbor_joining_hybrid_opt_refined, neighbor_joining_hybrid_anticentral_opt, \
    neighbor_joining_hybrid_anticentral_adaptive_v3, \
    neighbor_joining_baseline, make_nj_full_variant, make_nj_full_cps_variant, make_nj_hybrid_variant, \
    make_nj_hybrid_inv_cent_variant, neighbor_joining_hybrid_anticentral_adaptive_v3_plausible, \
    neighbor_joining_hybrid_anticentral_adaptive_v3_skip_unplausible, \
    neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony

CONFIG_BY_PROFILE = {
    "base": "data/config_for_pic.json",
    "high": "data/config_high.json",
    "highdm": "data/config_high_dm.json",
}
DEFAULT_BIOPSY_GENERATIONS = [4, 6, 8]
DEFAULT_SEEDS_FILE = "data/seeds.json"


def get_root_id(tr):
    roots = [n for n, indeg in tr.in_degree() if indeg == 0]
    if len(roots) != 1:
        raise ValueError(f"Tree must have exactly one root (found {len(roots)})")
    return roots[0]


def get_algorithms_to_test():
    neighbor_joining_full_full = make_nj_full_variant(True)
    neighbor_joining_full_partial = make_nj_full_variant(False)
    neighbor_joining_full_cps_full = make_nj_full_cps_variant(True)
    neighbor_joining_full_cps_partial = make_nj_full_cps_variant(False)
    neighbor_joining_hybrid_full = make_nj_hybrid_variant(True)
    neighbor_joining_hybrid_partial = make_nj_hybrid_variant(False)
    neighbor_joining_hybrid_inverse_centrality_full = make_nj_hybrid_inv_cent_variant(True)
    neighbor_joining_hybrid_inverse_centrality_partial = make_nj_hybrid_inv_cent_variant(False)
    return [
            neighbor_joining_baseline,
            neighbor_joining_full_full,
            neighbor_joining_full_partial,
            neighbor_joining_full_cps_full,
            neighbor_joining_full_cps_partial,
            neighbor_joining_hybrid_full,
            neighbor_joining_hybrid_partial,
            neighbor_joining_hybrid_inverse_centrality_full,
            neighbor_joining_hybrid_inverse_centrality_partial,
            neighbor_joining_adaptive_centrality,
            neighbor_joining_adaptive_centrality_nonlinear,
            neighbor_joining_adaptive_centrality_reversed,
            neighbor_joining_hybrid_opt,
            neighbor_joining_hybrid_opt_adaptive,
            neighbor_joining_hybrid_opt_v2,
            neighbor_joining_hybrid_opt_refined,
            neighbor_joining_hybrid_anticentral_opt,
            neighbor_joining_hybrid_anticentral_adaptive_v3,
            neighbor_joining_hybrid_anticentral_adaptive_v3_plausible,
            neighbor_joining_hybrid_anticentral_adaptive_v3_skip_unplausible,
            neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony
            ]


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
    parser.add_argument("--write-newick", action="store_true",
                        help="Print Newick trees during runs.")
    return parser.parse_args()


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
    config_path = args.config or CONFIG_BY_PROFILE[args.profile]
    variant_name = build_variant_name(args.r_dist, args.biopsy_size_scalable, args.profile)
    output_dir = Path(args.output_dir) if args.output_dir else Path("results") / variant_name
    selected_seeds = args.seed if args.seed else load_seeds(args.seeds_file)

    algorithms = get_algorithms_to_test()
    selected_indices = args.algorithm_index if args.algorithm_index is not None else list(range(len(algorithms)))

    print(f"Variant: {variant_name}")
    print(f"Config: {config_path}")
    print(f"Seeds source: {'CLI --seed' if args.seed else args.seeds_file}")
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
