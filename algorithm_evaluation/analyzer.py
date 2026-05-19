import os.path
import math
from pathlib import Path
import pandas as pd
from tester import get_algorithms_to_test, load_seeds, DEFAULT_SEEDS_FILE
from scipy.stats import wilcoxon
from itertools import combinations
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
LEGACY_TEST_VARIANTS = ["r2bss05", "r2bss025", "r2bss075", "r4bss05", "r4bss05high", "r4bss075", "r4bss05highdm"]
ADAPTIVE_RADIUS_TEST_VARIANTS = ["rAbss025", "rAbss05", "rAbss075", "rAbss05high", "rAbss05highdm"]


# --------------------------------------------------------------
#  Compare two algorithms based on Wilcoxon test results
# --------------------------------------------------------------
def _decide_winner(res_for_metric, alg1, alg2, alpha=0.05):
    """
    res_for_metric is the dict:
        {"p_value": ..., "rec_mean": ..., "nj_mean": ...}
    """
    p = float(res_for_metric["p_value"])
    m1 = float(res_for_metric["rec_mean"])
    m2 = float(res_for_metric["nj_mean"])

    # No significant difference -> tie
    if p > alpha:
        return "tie"

    # Significant difference -> higher score wins
    if m1 > m2:
        return "alg1"
    elif m2 > m1:
        return "alg2"
    else:
        return "tie"


# --------------------------------------------------------------
#  Build win/loss/tie matrices
# --------------------------------------------------------------
def build_pairwise_ranking(all_algorithms, paired_results, metric):
    """
    paired_results[(alg1_name, alg2_name)] = {
        '3-precision': {...}, '3-f1': {...}, 'grf': {...}
    }

    metric ∈ {"3-precision", "3-f1", "grf"}
    """

    table = pd.DataFrame(
        0,
        index=all_algorithms,
        columns=["wins", "losses", "ties", "score"],
        dtype=int
    )

    for (alg1, alg2), metrics_dict in paired_results.items():
        if metric not in metrics_dict:
            continue

        res_for_metric = metrics_dict[metric]
        winner = _decide_winner(res_for_metric, alg1, alg2)

        if winner == "alg1":
            table.loc[alg1, "wins"] += 1
            table.loc[alg2, "losses"] += 1
        elif winner == "alg2":
            table.loc[alg2, "wins"] += 1
            table.loc[alg1, "losses"] += 1
        else:
            table.loc[alg1, "ties"] += 1
            table.loc[alg2, "ties"] += 1

    table["score"] = table["wins"] - table["losses"]
    return table


# --------------------------------------------------------------
#  Pretty print ranking table
# --------------------------------------------------------------
def print_ranking_table(table, title="Algorithm ranking"):
    print("\n" + "="*80)
    print(title)
    print("="*80)
    print(table)
    print("="*80 + "\n")


# --------------------------------------------------------------
#  Example of usage inside analyzer
# --------------------------------------------------------------
def rank_algorithms_from_results(all_algorithms, paired_results, title="RANKING", tv="VariantName"):
    # """
    # all_algorithms: list of algorithm identifiers (functions or strings)
    # paired_results: dict keyed by (alg1_name, alg2_name) -> metrics dict
    # """

    # --- Normalize algorithm identifiers to names (strings) ---
    alg_names = [
        alg if isinstance(alg, str) else alg.__name__
        for alg in all_algorithms
    ]

    mode = "nj" if "NJ" in title.upper() else "rec"
    metrics = ["3-precision", "3-f1", "grf"]

    print(f"\n=============== {title} ===============\n")

    for metric in metrics:
        print(f"------ Metric: {metric} ------")
        ranking = build_pairwise_ranking(alg_names, paired_results, metric)
        print("=" * 80)
        print(ranking)
        print("=" * 80)
        print()
        out_file = os.path.join(
            RESULTS_DIR,
            f"ranking_{tv}_{metric}_{mode}.csv"
        )
        ranking.to_csv(out_file, index=True)
        print(f"Saved ranking table → {out_file}")



def compare_two_results(file_local, file_full):
    A = pd.read_csv(file_local)
    B = pd.read_csv(file_full)

    # Match by seed
    merged = A.merge(B, on="seed", suffixes=("_rec", "_nj"))

    metrics = ["3-precision", "3-f1", "grf"]  # adapt to your column names

    results = {}
    for m in metrics:
        x = merged[f"{m}_rec"]
        y = merged[f"{m}_nj"]
        stat, p = wilcoxon(x, y)
        results[m] = {"p_value": p, "rec_mean": x.mean(), "nj_mean": y.mean()}

    return results


def summarize(df):
    numeric_cols = df.columns.drop("seed")

    # Compute summary statistics
    summary = pd.DataFrame({
        "sum": df[numeric_cols].sum(),
        "avg": df[numeric_cols].mean(),
        "min": df[numeric_cols].min(),
        "max": df[numeric_cols].max(),
        "std": df[numeric_cols].std()
    })

    print(summary)
    # summary.to_csv("summary_stats.csv")


def analize(i, alg_name, how_many, a, b, c):
    print(f"Analyzing {alg_name}")

    df = pd.read_csv(a)
    df1 = pd.read_csv(b)
    df2 = pd.read_csv(c)
    m = (
            (df["ancestors_multiset_precision_failures"] > 0) |
            (df["ancestors_multiset_f1_failures"] > 0)
    )
    x = df.loc[m, "seed"].tolist()
    u = len(x)
    m = (
            (df["ancestors_unique_precision_failures"] > 0) |
            (df["ancestors_unique_f1_failures"] > 0)
    )
    x = df.loc[m, "seed"].tolist()
    v = len(x)
    mask2 = (
            (
                    (df["ancestors_multiset_precision_failures"] == 0) &
                    (df["ancestors_multiset_f1_failures"] == 0)
            ) &
            (
                    (df["ancestors_unique_precision_failures"] > 0) |
                    (df["ancestors_unique_f1_failures"] > 0)
            )
    )
    w = df.loc[mask2, "seed"].tolist()

    m = (
            (df["ancestors_unique_restricted_precision_failures"] > 0) |
            (df["ancestors_unique_restricted_f1_failures"] > 0)
    )
    f = len(df.loc[m, "seed"].tolist())
    m = (df["ancestors_unique_restricted_precision_failures"] > 0)
    g = len(df.loc[m, "seed"].tolist())
    m = (df["ancestors_unique_restricted_f1_failures"] > 0)
    h = len(df.loc[m, "seed"].tolist())

    print("No. Rec < NJ; All: ", u, " Unique: ", v, " Unique but not all: ", w)
    print("Restricted failures; All: ", f, " precision: ", g, " F1: ", h)

    print("***** Rekonstruowane *****")
    summarize(df1)
    print("***** NJ *****")
    summarize(df2)
    if "grf" in df1.columns and "grf" in df2.columns:
        print(f"GRF avg: REC={df1['grf'].mean():.3f}, NJ={df2['grf'].mean():.3f}")
        wins = (df1["grf"] > df2["grf"]).sum()
        ties = (df1["grf"] == df2["grf"]).sum()
        losses = (df1["grf"] < df2["grf"]).sum()
        print(f"GRF comparison: REC wins={wins}, NJ wins={losses}, ties={ties}")
    # print("***** Correctness *****")
    if len(df1) == len(df2) == how_many:
        print("")
    else:
        print("Correctness failure: ", len(df1), len(df2), " instead of ", how_many)


def compare_two(i, j, rec=False):
    """
    Compare algorithms i and j.
    rec=False → NJ-like comparison (i out.csv)
    rec=True  → reconstructed comparison (i rec.csv)

    Stores results in paired_results_nj / paired_results_rec
    and prints Wilcoxon results exactly like before.
    """

    # ---------- select correct CSV file ----------
    if rec:
        fname_i = os.path.join(RESULTS_DIR, f"{i}rec.csv")
        fname_j = os.path.join(RESULTS_DIR, f"{j}rec.csv")
        store_dict = paired_results_rec
    else:
        fname_i = os.path.join(RESULTS_DIR, f"{i}nj.csv")
        fname_j = os.path.join(RESULTS_DIR, f"{j}nj.csv")
        store_dict = paired_results_nj

    # ---------- compute statistics ----------
    res = compare_two_results(fname_i, fname_j)

    # ---------- print results exactly as before ----------
    for metric, r in res.items():
        print(f"{metric}: p={r['p_value']:.4g}, alg1={r['rec_mean']:.3f}, alg2={r['nj_mean']:.3f}")

    # ---------- store in ranking dict (IMPORTANT) ----------
    # (alg_i_name, alg_j_name) → { metric: (p, alg1_mean, alg2_mean) }
    key = (alg_names[i].__name__, alg_names[j].__name__)
    store_dict[key] = {}

    for metric, r in res.items():
        store_dict[key][metric] = (
            r["p_value"],
            r["rec_mean"],   # mean for algorithm i
            r["nj_mean"],    # mean for algorithm j
        )


if __name__ == "__main__":
    test_variants = LEGACY_TEST_VARIANTS

    alg_names = get_algorithms_to_test()
    all_seeds = load_seeds(DEFAULT_SEEDS_FILE)

    for test_variant in test_variants:
        for i in range(len(alg_names)):
            a = os.path.join(RESULTS_DIR, test_variant, str(i) + "out.csv")
            b = os.path.join(RESULTS_DIR, test_variant, str(i) + "rec.csv")
            c = os.path.join(RESULTS_DIR, test_variant, str(i) + "nj.csv")
            analize(i, alg_names[i], len(all_seeds), a, b, c)

            res = compare_two_results(b, c)
            print("\nWilcoxon paired comparison:")
            for m, r in res.items():
                print(f"{m}: p={r['p_value']:.4g}, rec={r['rec_mean']:.3f}, nj={r['nj_mean']:.3f}")

            print()

        paired_results_nj = {}
        paired_results_rec = {}

        for i, j in combinations(range(len(alg_names)), 2):

            alg1_name = alg_names[i].__name__
            alg2_name = alg_names[j].__name__

            # --- NJ-like comparison ---
            nj_file_1 = os.path.join(RESULTS_DIR, test_variant, f"{i}nj.csv")
            nj_file_2 = os.path.join(RESULTS_DIR, test_variant, f"{j}nj.csv")
            res_nj = compare_two_results(nj_file_1, nj_file_2)
            paired_results_nj[(alg1_name, alg2_name)] = res_nj

            # --- REC comparison ---
            rec_file_1 = os.path.join(RESULTS_DIR, test_variant, f"{i}rec.csv")
            rec_file_2 = os.path.join(RESULTS_DIR, test_variant, f"{j}rec.csv")
            res_rec = compare_two_results(rec_file_1, rec_file_2)
            paired_results_rec[(alg1_name, alg2_name)] = res_rec

            print("\nWilcoxon paired comparison:", alg1_name, " and ", alg2_name)
            print("\nNJ like reconstructions")
            for m, r in res_nj.items():
                print(f"{m}: p={r['p_value']:.4g}, alg1={r['rec_mean']:.3f}, alg2={r['nj_mean']:.3f}")

            print("\nStandard leveled reconstructions")
            for m, r in res_rec.items():
                print(f"{m}: p={r['p_value']:.4g}, alg1={r['rec_mean']:.3f}, alg2={r['nj_mean']:.3f}")


        # Build algorithm IDs (strings) that match the keys used in paired_results_*
        algo_ids = [f.__name__ for f in alg_names]

        rank_algorithms_from_results(alg_names, paired_results_nj, title="NJ-like RANKING", tv=test_variant)
        rank_algorithms_from_results(alg_names, paired_results_rec, title="Standard-level RANKING", tv=test_variant)
