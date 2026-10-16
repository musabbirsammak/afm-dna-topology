"""
Turn the saved predictions into the numbers for the results tables.

Input: one or more *_oof.csv files written by train.py
Run:
  python evaluate.py --oof results/classify/*_oof.csv

Prints, for each file:
  - model, all classes : accuracy and balanced accuracy, each with a 95% bootstrap
                         confidence interval (2,000 resamples), and macro-F1
  - model, knots only  : the same, using only molecules whose true type is 0_1, 3_1, 5_1 or 5_2.
                         These all have about 758 nm of DNA, so size cannot give the answer away.
                         Balanced accuracy here is the mean recall over these 4 types, so guessing
                         gives 25%. The model may still predict 4^2_1 for them, which counts as wrong.
  - TopoStats          : the same two lines for TopoStats' own knot-type calls on exactly the same
                         molecules (an empty call counts as wrong)
  - recall per class   : share of each true type that was predicted correctly
  - accuracy per imaging condition (Ni, MgNi, other)

Saves, next to each CSV:
  <name>_confusion.png : confusion matrix (rows = true type, columns = predicted type)
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import warnings
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

# TopoStats sometimes predicts classes not in the true labels (e.g. other knot types); sklearn warns on every
# bootstrap sample, which floods the log. The metrics are unaffected.
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

KNOTS = ["0_1", "3_1", "5_1", "5_2"]


def ci(y, p, fn, n=2000, seed=0):
    rng = np.random.default_rng(seed); y, p = np.asarray(y), np.asarray(p)
    vals = [fn(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return fn(y, p), np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def line(name, y, p):
    a = ci(y, p, accuracy_score)
    b = ci(y, p, balanced_accuracy_score)
    f = ci(y, p, lambda t, q: f1_score(t, q, average="macro"))
    print(f"  {name:28s} n={len(y):4d}  acc {a[0]:.1%} [{a[1]:.1%}-{a[2]:.1%}]  "
          f"bal.acc {b[0]:.1%} [{b[1]:.1%}-{b[2]:.1%}]  macro-F1 {f[0]:.2f}")


ap =argparse.ArgumentParser()
ap.add_argument("--oof", nargs="+", required=True)
args = ap.parse_args()

for path in args.oof:
    r = pd.read_csv(path)
    print(f"\n=== {path}")
    line("model, all classes", r.label, r.pred)
    k = r[r.label.isin(KNOTS)]
    line("model, knots only", k.label, k.pred)
    ts = r.ts_topology.fillna("").astype(str)
    line("TopoStats, all classes", r.label, ts)
    line("TopoStats, knots only", k.label, ts[k.index])
    print("  recall per class:", {c: f"{(g.pred == c).mean():.0%} (n={len(g)})" for c, g in r.groupby("label")})
    print("  accuracy per imaging condition:", {c: f"{(g.pred == g.label).mean():.0%} (n={len(g)})" for c, g in r.groupby("condition")})
    classes = sorted(r.label.unique())
    cm = confusion_matrix(r.label, r.pred, labels=classes)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45); ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(path.split("/")[-1], fontsize=8)
    fig.tight_layout(); fig.savefig(path.replace(".csv", "_confusion.png"), dpi=150); plt.close(fig)
    print("  saved", path.replace(".csv", "_confusion.png"))
