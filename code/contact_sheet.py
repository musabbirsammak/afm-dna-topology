"""
Makes one picture per knot type with 25 random molecules, for every crops folder.
Only molecules that passed the length filter (keep = True) are shown.

Run from the project folder:
    python code/contact_sheet.py                  # every folder named crops* that has a metadata.csv
    python code/contact_sheet.py --crops crops_1nm   # only this folder
    python code/contact_sheet.py --n 16           # a different number of molecules per picture
Output:
    results/figures/qc/<crops folder>/<knot type>.png
"""
import argparse, glob, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sheets(crops, n):
    name = os.path.basename(os.path.normpath(crops))
    md = pd.read_csv(os.path.join(crops, "metadata.csv"))
    md = md[md.keep]
    out = os.path.join(REPO, "results", "figures", "qc", name)
    os.makedirs(out, exist_ok=True)

    for label, g in md.groupby("label"):
        g = g.sample(min(n, len(g)), random_state=0)
        k = int(np.ceil(np.sqrt(len(g))))
        fig, axes = plt.subplots(k, k, figsize=(2 * k, 2 * k))
        for ax in np.ravel(axes):
            ax.axis("off")
        for ax, (_, r) in zip(np.ravel(axes), g.iterrows()):
            ax.imshow(np.load(os.path.join(crops, "npy", r.id + ".npy")), cmap="afmhot", vmin=0, vmax=1)
            ax.set_title(f"{r['sample'][:10]} {r.session}", fontsize=6)
        fig.suptitle(f"{name}: class {label}  (n shown = {len(g)})")
        fig.tight_layout()
        path = os.path.join(out, f"{label.replace('^', '-')}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print("saved", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default=None, help="one crops folder; default = every folder named crops* in the project")
    ap.add_argument("--n", type=int, default=25)
    a = ap.parse_args()

    if a.crops is None:
        folders = sorted(d for d in glob.glob(os.path.join(REPO, "crops*"))
                         if os.path.isfile(os.path.join(d, "metadata.csv")))
    else:
        folders = [a.crops if os.path.isdir(a.crops) else os.path.join(REPO, a.crops)]
    if not folders:
        print("no crops folder with a metadata.csv found - run extract_crops.py first")
    for d in folders:
        sheets(d, a.n)