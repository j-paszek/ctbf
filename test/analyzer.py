import pandas as pd
from tester import get_algorithms_to_test


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


def analize(i, alg_name):
    print(f"Analyzing {alg_name}")
    a = "results/" + str(i) + "out.csv"
    b = "results/" + str(i) + "rec.csv"
    c = "results/" + str(i) + "nj.csv"
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
    print("No. Rec < NJ; All: ", u, " Unique: ", v, " Unique but not all: ", w)

    print("***** Rekonstruowane *****")
    summarize(df1)
    print("***** NJ *****")
    summarize(df2)


if __name__ == "__main__":
    alg_names = get_algorithms_to_test()

    for i in range(len(alg_names)):
        analize(i, alg_names[i])


