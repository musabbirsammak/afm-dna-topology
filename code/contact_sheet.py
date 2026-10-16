"""
Makes one picture per class with 25 random molecules.

Run:
    python contact_sheet.py --crops crops
Output: 
    results/figures/qc/<class>.png
"""
import argparse, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--crops", default="crops")
ap.add_argument("--n", type=int, default=25)
a = ap.parse_args()

md = pd.read_csv(os.path.join(a.crops, "metadata.csv"))
md = md[md.keep]

OUT = os.path.join(REPO, "results", "figures", "qc")
os.makedirs(OUT, exist_ok=True)

for label, g in md.groupby("label"):
    g = g.sample(min(a.n, len(g)), random_state=0)
    k = int(np.ceil(np.sqrt(len(g))))
    fig, axes = plt.subplots(k, k, figsize=(2 * k, 2 * k))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, (_, r) in zip(np.ravel(axes), g.iterrows()):
        ax.imshow(np.load(os.path.join(a.crops, "npy", r.id + ".npy")), cmap="afmhot", vmin=0, vmax=1)
        ax.set_title(f"{r['sample'][:10]} {r.session}", fontsize=6)
    fig.suptitle(f"class {label}  (n shown = {len(g)})")
    fig.tight_layout()
    name = label.replace("^", "-")
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=120)
    plt.close(fig)
    print("saved", os.path.join(OUT, f"{name}.png"))