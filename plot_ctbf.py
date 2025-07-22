import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
import json



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

import sys

infile = sys.argv[1] if len(sys.argv)>1 else "results_rpercentile"
res = []
with open(infile, "r") as f:
    data = f.readlines()
    
for d in data:
    print(d.rstrip().replace("'", '"'))
    res.append(json.loads(d.rstrip().replace("'", '"').replace("(","[").replace(")","]")))

df = pd.DataFrame(res)

# --- Binning GenTreeSize into categories ---
df['GenTreeSizeBin'] = pd.cut(
    df['GenTreeSize'],
    bins=[0, 1000, 1500, 2000, 1000000],
    labels=["<1000", "1000–1500","1500–2000", ">2000"]
)


# --- Create labels for axes and grouping ---
df['Tree-Biopsies'] = df.apply(lambda row: f"{row['TreeSize']}L-{row['NumBiopsies']}B", axis=1)
ordered_labels = ["6L-2B", "8L-2B", "8L-3B", "10L-2B", "10L-3B", "14L-2B"]
df['Tree-Biopsies'] = pd.Categorical(df['Tree-Biopsies'], categories=ordered_labels, ordered=True)

df['Biopsy-P'] = df.apply(lambda row: f"{row['biopsy_size']:.1f} | {row['RPercentile']}", axis=1)
label_tuples = [
    (label, float(label.split(' | ')[0]), float(label.split('|')[1]))
    for label in df['Biopsy-P'].unique()
]

sorted_labels = [label for label, _, _ in sorted(label_tuples, key=lambda x: (x[1], x[2]))]

# convert to ordered categorical
df['Biopsy-P'] = pd.Categorical(df['Biopsy-P'], categories=sorted_labels, ordered=True)

palette = sns.color_palette("rocket", 100)
custom_rocket_palette = reversed(palette[10::20])

g = sns.catplot(
    data=df,
    x='Biopsy-P', y='Result',
    hue='GenTreeSizeBin',
    kind='box',
    col='Tree-Biopsies',
    col_wrap=2,
    height=4, aspect=1.2,
    palette=custom_rocket_palette
)

g.set_titles("")

for ax, title in zip(g.axes.flat, df['Tree-Biopsies'].cat.categories):
    ax.text(
        0.02, 1, re.sub("Tree ([0-9]*)L-([0-9]*)B", r"Trees: \1 levels | \2 biopsies" , f"Tree {title}"),
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize=17
    )


g._legend.set_bbox_to_anchor((0.89, 0.29))
g.set_axis_labels("Biopsy Size | Percentile", "Similarity", fontsize=16)
g._legend.set_title("Simulated Tree Size")

g.fig.set_size_inches(10, 6.1)
for ax in g.axes.flat:
    plt.setp(ax.get_xticklabels(), visible=True, rotation=45, ha='center', fontsize=1)
    plt.setp(ax.get_xticklabels(),fontsize=16)
    plt.setp(ax.get_yticklabels(),fontsize=16)
g.fig.savefig("ctbf_results.pdf", dpi=300, bbox_inches='tight')

plt.show()

