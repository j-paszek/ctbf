import pandas as pd

from ctbs import run_single_test
from ctbs_utils import get_biopsy_nodes_ids
from evaluator_full import evaluate_4
from evaluator import grf_tree
from reconstructor import build_evolution_tree, visualize_tree_plotly, \
    neighbor_joining_adaptive_centrality, \
    neighbor_joining_adaptive_centrality_nonlinear, neighbor_joining_adaptive_centrality_reversed, \
    neighbor_joining_hybrid_opt, neighbor_joining_hybrid_opt_adaptive, neighbor_joining_hybrid_opt_v2, \
    neighbor_joining_hybrid_opt_refined, neighbor_joining_hybrid_anticentral_opt, \
    neighbor_joining_hybrid_anticentral_adaptive_v2, neighbor_joining_hybrid_anticentral_adaptive_v3, \
    neighbor_joining_baseline, make_nj_full_variant, make_nj_full_cps_variant, make_nj_hybrid_variant, \
    make_nj_hybrid_inv_cent_variant, neighbor_joining_hybrid_anticentral_adaptive_v3_plausible

df = pd.read_csv("data/f1results.csv", delimiter="\t")
all_seeds = df["seed"].unique().tolist()


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
            neighbor_joining_hybrid_anticentral_adaptive_v3_plausible
            ]


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


def check_one_alg(algo, counter):
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
    for seed in all_seeds:
        print(f"\nTesting seed: {seed}")
        try:
            a, b, c = run_single_test(
                seed=seed,
                config="data/config_for_pic.json",
                bedfile="data/pic.csv",
                biopsy_size_scalable=0.5,
                biopsy_generations=[4, 6, 8],
                r_dist=4,
                write_newick=True,
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
    name1 = "results/" + str(counter) + "out.csv"
    df_fail.to_csv(name1, index=False)

    # Save REC and NJ metric reports
    df_rec = pd.DataFrame(rec_output)
    name2 = "results/" + str(counter) + "rec.csv"
    df_rec.to_csv(name2, index=False)

    df_nj = pd.DataFrame(nj_output)
    name3 = "results/" + str(counter) + "nj.csv"
    df_nj.to_csv(name3, index=False)

    print("Results saved: ", name1, name2, name3)

if __name__ == "__main__":
    # === Run tests ===
    counter = -1
    for algo in get_algorithms_to_test():
        counter += 1 # 0-21/31, 1-23/34, 2-24/27, 3-25
        if counter != 18:
            continue

        algo_name = getattr(algo, "__name__", str(algo))
        print(f"\n--- Running tests with {algo_name} ---")
        check_one_alg(algo, counter)
