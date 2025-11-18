import os.path

import pandas as pd
from tester import get_algorithms_to_test
from scipy.stats import wilcoxon
from itertools import combinations


def compare_two_results(file_local, file_full):
    A = pd.read_csv(file_local)
    B = pd.read_csv(file_full)

    # Match by seed
    merged = A.merge(B, on="seed", suffixes=("_rec", "_nj"))

    metrics = ["3-precision", "3-f1"]  # adapt to your column names

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
    # print("***** Correctness *****")
    if len(df1) == len(df2) == how_many:
        print("")
    else:
        print("Correctness failure: ", len(df1), len(df2), " instead of ", how_many)


def compare_nj_algos(df):
    algos = sorted({col.split("_")[0] for col in df.columns})

    for m in metrics:
        for a, b in combinations(algos, 2):
            X = df[f"{a}_{m}"]
            Y = df[f"{b}_{m}"]
            p = wilcoxon(X, Y).pvalue
            print(f"{a} vs {b} [{m}]: p={p:.4g}, mean({a})={X.mean():.3f}, mean({b})={Y.mean():.3f}")


def compare_two(i, j, alg_names):
    print("\nWilcoxon paired comparison:", alg_names[i], " and ", alg_names[j])
    x = os.path.join("results", str(i) + "nj.csv")
    y = os.path.join("results", str(j) + "nj.csv")
    res = compare_two_results(x, y)
    for m, r in res.items():
        print(f"{m}: p={r['p_value']:.4g}, alg1={r['rec_mean']:.3f}, alg2={r['nj_mean']:.3f}")


if __name__ == "__main__":
    alg_names = get_algorithms_to_test()
    df = pd.read_csv("data/f1results.csv", delimiter="\t")
    all_seeds = df["seed"].unique().tolist()

    for i in range(len(alg_names)):
        a = os.path.join("results", str(i) + "out.csv")
        b = os.path.join("results", str(i) + "rec.csv")
        c = os.path.join("results", str(i) + "nj.csv")
        analize(i, alg_names[i], len(all_seeds), a, b, c)

        res = compare_two_results(b, c)
        print("\nWilcoxon paired comparison:")
        for m, r in res.items():
            print(f"{m}: p={r['p_value']:.4g}, rec={r['rec_mean']:.3f}, nj={r['nj_mean']:.3f}")

        print()

    for i, j in combinations(range(3), 2):
        compare_two(i, j, alg_names)

