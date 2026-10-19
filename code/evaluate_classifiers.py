"""
Collect every whole-molecule classifier result into tables and figures for the paper.

Reads every results/classify*/<model>_<split>[_flip]_oof.csv written by train_crops.py
(one row per molecule, each predicted once by a model that never saw it).

Each run is described by:
  crops    the crops it was trained on, taken from the folder name
           (classify_1nm -> 1nm, classify_05nm -> 05nm, classify_1nm_norm -> 1nm_norm, ...)
  model    resnet18, convnext_tiny or dinov2
  split    session (whole imaging days held out) or image (only whole images held out)
  flip     True for the flip control run
  epochs   30, or 15 / 60 when the folder name ends in _ep15 / _ep60 (empty for DINOv2)

Two scores are used throughout:
  all    every molecule, all 5 types
  knots  only molecules whose true type is 0_1, 3_1, 5_1 or 5_2. These all have the same
         DNA length, so size cannot give the answer away. Balanced accuracy = the accuracy
         on each of these 4 types separately, averaged, so each type counts equally.
         Random guessing among 4 types scores 25%. The model still chooses from all 5 types,
         so calling a knot a catenane counts as wrong.
95% confidence intervals: 2,000 bootstrap resamples of the molecules (seed 0).

Output: results/classifier_performance/
  summary.csv            one row per run: accuracy, balanced accuracy and macro-F1 with 95% CIs
                         (all and knots), macro AUC, plus TopoStats on the same molecules
  per_class_recall.csv   share of each true type predicted correctly, per run (and TopoStats)
  per_condition.csv      accuracy per imaging buffer, per run
  per_session.csv        accuracy per imaging day, per run
  split_gap.csv          image split minus session split, same crops and model, paired over the
                         same molecules (knots balanced accuracy, with 95% CI)
  comparisons.csv        exact McNemar tests between related runs on the same molecules:
                         flip vs no flip, 15/60 vs 30 epochs, 0.5 vs 1 nm, norm/binary vs height,
                         model vs model, every run vs TopoStats. Both all and knots.
  confusion/<run>.csv    confusion matrix counts (rows = true type, columns = predicted type)
  confusion/<run>.png    the same as a picture, with counts and row percentages
  knots_balanced_accuracy.png   knots balanced accuracy of every run with 95% CIs

Run from the project folder:
  python code/evaluate_classifiers.py
"""
import glob, itertools, os, re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results")
OUT = os.path.join(RES, "classifier_performance")
KNOTS = ["0_1", "3_1", "5_1", "5_2"]
N_BOOT, SEED = 2000, 0
FILE_RE = re.compile(r"^(resnet18|convnext_tiny|dinov2)_(session|image)(_flip)?_oof\.csv$")
SESSION_BLUE, IMAGE_ORANGE = "#2a78d6", "#eb6834"


# ----------------------------------------------------------------------------- loading
def find_runs():
    runs = []
    for path in sorted(glob.glob(os.path.join(RES, "classify*", "*_oof.csv"))):
        m = FILE_RE.match(os.path.basename(path))
        if not m:
            continue
        crops = os.path.basename(os.path.dirname(path)).replace("classify", "", 1).lstrip("_") or "default"
        ep = re.search(r"_ep(\d+)$", crops)
        base = crops[: ep.start()] if ep else crops                     # 1nm_ep15 -> 1nm
        model, split, flip = m.group(1), m.group(2), bool(m.group(3))
        runs.append(dict(
            run=f"{crops}__{model}_{split}" + ("_flip" if flip else ""),
            path=path, crops=crops, base_crops=base, model=model, split=split, flip=flip,
            epochs=None if model == "dinov2" else (int(ep.group(1)) if ep else 30),
            input="norm" if base.endswith("_norm") else "binary" if base.endswith("_binary") else "height",
            resolution=base.split("_")[0]))
    return runs


def load(path):
    r = pd.read_csv(path, dtype={"session": str})
    r["ts_topology"] = r.ts_topology.fillna("").astype(str)
    return r


# ----------------------------------------------------------------------------- metrics
# Labels are turned into numbers 0..K-1; any prediction that is not one of the K classes
# (e.g. TopoStats giving no call) becomes K, which is always wrong.
def encode(y, p, classes):
    idx = {c: i for i, c in enumerate(classes)}
    return (np.array([idx[c] for c in y]), np.array([idx.get(c, len(classes)) for c in p]))


def metrics(y, p, k):
    """accuracy, balanced accuracy, macro-F1 (over the true types present)."""
    cm = np.bincount(y * (k + 1) + p, minlength=(k + 1) ** 2).reshape(k + 1, k + 1)
    rows, cols, tp = cm.sum(1), cm.sum(0), np.diag(cm)
    present = rows > 0
    acc = tp.sum() / len(y)
    bal = np.mean(tp[present] / rows[present])
    f1 = np.mean(2 * tp[present] / (rows[present] + cols[present]))
    return np.array([acc, bal, f1])


def boot_metrics(y, p, k):
    rng = np.random.default_rng(SEED)
    vals = np.array([metrics(y[i], p[i], k) for i in (rng.integers(0, len(y), len(y)) for _ in range(N_BOOT))])
    return metrics(y, p, k), np.percentile(vals, 2.5, 0), np.percentile(vals, 97.5, 0)


def scores(labels, preds, classes, prefix):
    out = {f"{prefix}_n": len(labels)}
    if len(labels) == 0:
        return out
    y, p = encode(labels, preds, classes)
    est, lo, hi = boot_metrics(y, p, len(classes))
    for j, name in enumerate(["acc", "balacc", "macroF1"]):
        out[f"{prefix}_{name}"], out[f"{prefix}_{name}_lo"], out[f"{prefix}_{name}_hi"] = est[j], lo[j], hi[j]
    return out


def macro_auc(r, classes):
    """One-vs-rest AUC for each type from the predicted probabilities, averaged over types."""
    present = [c for c in classes if (r.label == c).any()]
    if len(present) < 2:
        return np.nan
    p = r[[f"p_{c}" for c in present]].to_numpy()
    p = p / np.clip(p.sum(1, keepdims=True), 1e-12, None)
    return roc_auc_score(r.label, p, multi_class="ovr", average="macro", labels=present)


def mcnemar(ca, cb):
    a_only, b_only = int((ca & ~cb).sum()), int((~ca & cb).sum())
    return dict(both_right=int((ca & cb).sum()), only_A_right=a_only, only_B_right=b_only,
                both_wrong=int((~ca & ~cb).sum()),
                p_mcnemar=binomtest(a_only, a_only + b_only).pvalue if a_only + b_only else 1.0)


# ----------------------------------------------------------------------------- pictures
def confusion(r, classes, run):
    y, p = encode(r.label, r.pred, classes)
    k = len(classes)
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    pd.DataFrame(cm, index=[f"true_{c}" for c in classes], columns=[f"pred_{c}" for c in classes]) \
        .to_csv(os.path.join(OUT, "confusion", f"{run}.csv"))
    pct = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(k), classes, rotation=45)
    ax.set_yticks(range(k), classes)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{cm[i, j]}\n{pct[i, j]:.0%}", ha="center", va="center", fontsize=7,
                    color="white" if pct[i, j] > 0.6 else "#222222")
    ax.set_xlabel("predicted type"); ax.set_ylabel("true type")
    ax.set_title(run.replace("__", "  "), fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "confusion", f"{run}.png"), dpi=200); plt.close(fig)


def figure(summary):
    s = summary[summary.model != "TopoStats"].copy()
    s["row"] = s.crops + "  " + s.model + np.where(s.flip, " +flip", "")
    s = s.sort_values(["crops", "model", "flip"])
    rows = list(dict.fromkeys(s.row))
    y_of = {t: i for i, t in enumerate(rows)}
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(rows) + 1.6))
    for split, color, marker, dy, name in (("session", SESSION_BLUE, "o", -0.13, "session split (new imaging days)"),
                                           ("image", IMAGE_ORANGE, "s", 0.13, "image split (same imaging days)")):
        d = s[s.split == split]
        if d.empty:
            continue
        ax.errorbar(d.knots_balacc * 100, d.row.map(y_of) + dy,
                    xerr=[(d.knots_balacc - d.knots_balacc_lo) * 100, (d.knots_balacc_hi - d.knots_balacc) * 100],
                    fmt=marker, color=color, ms=6, elinewidth=2, capsize=0, label=name)
    ax.axvline(25, color="#888888", lw=1, ls="--")
    ax.text(25.5, -0.8, "chance 25%", fontsize=8, color="#555555", va="center")
    ts = summary[summary.model == "TopoStats"]
    if len(ts):                    # TopoStats on the same molecules
        x = ts.knots_balacc.iloc[0] * 100
        ax.axvline(x, color="#555555", lw=1, ls=":")
        ax.text(x + 0.5, -0.8, "TopoStats", fontsize=8, color="#555555", va="center")
    ax.set_yticks(range(len(rows)), rows, fontsize=8)
    ax.set_ylim(len(rows) - 0.4, -1.2)
    ax.set_xlim(0, 100)
    ax.set_xlabel("knots balanced accuracy (%), types 0_1, 3_1, 5_1, 5_2")
    ax.grid(axis="x", color="#e5e5e5", lw=0.8); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "knots_balanced_accuracy.png"), dpi=200); plt.close(fig)


# ----------------------------------------------------------------------------- which runs to compare
def reason(a, b):
    """Why runs a and b should be compared, or None. Both must use the same split."""
    if a["split"] != b["split"]:
        return None
    same_model = a["model"] == b["model"]
    if same_model and a["crops"] == b["crops"] and a["flip"] != b["flip"]:
        return "flip vs no flip"
    if a["flip"] or b["flip"]:
        return None
    if same_model and a["base_crops"] == b["base_crops"] and a["epochs"] != b["epochs"]:
        return "epochs"
    if same_model and a["epochs"] == b["epochs"] and a["input"] == b["input"] == "height" \
            and a["resolution"] != b["resolution"]:
        return "resolution (0.5 vs 1 nm/px)"
    if same_model and a["epochs"] == b["epochs"] and a["resolution"] == b["resolution"] and a["input"] != b["input"]:
        return "input (height / norm / binary)"
    if not same_model and a["crops"] == b["crops"]:
        return "model vs model"
    return None


# ----------------------------------------------------------------------------- main
def main():
    runs = find_runs()
    if not runs:
        print(f"no *_oof.csv files found in {RES}/classify*/ - run train_crops.py first")
        return
    os.makedirs(os.path.join(OUT, "confusion"), exist_ok=True)
    print(f"{len(runs)} runs found\n")

    data, summary, recall, cond, sess = {}, [], [], [], []
    for meta in runs:
        r = load(meta["path"]); data[meta["run"]] = r
        classes = sorted(r.label.unique())
        k = r[r.label.isin(KNOTS)]
        row = {key: meta[key] for key in ("run", "crops", "model", "split", "flip", "epochs", "input", "resolution")}
        row.update(scores(r.label, r.pred, classes, "all"))
        row.update(scores(k.label, k.pred, classes, "knots"))
        row["all_macroAUC"] = macro_auc(r, classes)
        row["knots_macroAUC"] = macro_auc(k, classes)
        summary.append(row)
        for c, g in r.groupby("label"):
            recall.append(dict(run=meta["run"], label=c, recall=(g.pred == c).mean()))
        for c, g in r.groupby("condition"):
            kk = g.label.isin(KNOTS)
            cond.append(dict(run=meta["run"], condition=c, n=len(g), accuracy=(g.pred == g.label).mean(),
                             knots_n=int(kk.sum()),
                             knots_accuracy=(g.pred == g.label)[kk].mean() if kk.any() else np.nan))
        for c, g in r.groupby("session"):
            sess.append(dict(run=meta["run"], session=c, n=len(g), types=" ".join(sorted(g.label.unique())),
                             accuracy=(g.pred == g.label).mean()))
        confusion(r, classes, meta["run"])
        print(f"  {meta['run']:42s} knots balanced acc {row['knots_balacc']:6.1%} "
              f"[{row['knots_balacc_lo']:.1%} - {row['knots_balacc_hi']:.1%}]   all acc {row['all_acc']:.1%}")

    # TopoStats on the same molecules. Its calls do not depend on the model, split or crop style,
    # so it is scored once per distinct set of molecules (1 nm and 0.5 nm crops may differ slightly).
    seen = {}
    for m in sorted(runs, key=lambda m: m["base_crops"] != "1nm"):      # name it after 1nm if possible
        seen.setdefault(frozenset(data[m["run"]].id), m["base_crops"])
    for base in seen.values():
        meta = next(m for m in runs if m["base_crops"] == base)
        r = data[meta["run"]]; k = r[r.label.isin(KNOTS)]
        classes = sorted(r.label.unique())
        row = dict(run=f"{base}__TopoStats", crops=base, model="TopoStats", split="", flip=False,
                   epochs=None, input=meta["input"], resolution=meta["resolution"])
        row.update(scores(r.label, r.ts_topology, classes, "all"))
        row.update(scores(k.label, k.ts_topology, classes, "knots"))
        summary.append(row)
        for c, g in r.groupby("label"):
            recall.append(dict(run=row["run"], label=c, recall=(g.ts_topology == c).mean()))
        print(f"  {row['run']:42s} knots balanced acc {row['knots_balacc']:6.1%} "
              f"[{row['knots_balacc_lo']:.1%} - {row['knots_balacc_hi']:.1%}]   all acc {row['all_acc']:.1%}")

    summary = pd.DataFrame(summary)
    summary.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    pd.DataFrame(recall).pivot(index="run", columns="label", values="recall") \
        .to_csv(os.path.join(OUT, "per_class_recall.csv"))
    pd.DataFrame(cond).to_csv(os.path.join(OUT, "per_condition.csv"), index=False)
    pd.DataFrame(sess).to_csv(os.path.join(OUT, "per_session.csv"), index=False)

    # image split minus session split, paired over the same molecules (knots only)
    gaps = []
    for m in runs:
        twin = f"{m['crops']}__{m['model']}_session"
        if m["split"] != "image" or m["flip"] or twin not in data:
            continue
        a, b = data[m["run"]].set_index("id"), data[twin].set_index("id")
        ids = a.index.intersection(b.index)
        ids = ids[a.loc[ids, "label"].isin(KNOTS).to_numpy()]
        classes = sorted(a.label.unique())
        y, pa = encode(a.loc[ids, "label"], a.loc[ids, "pred"], classes)
        _, pb = encode(a.loc[ids, "label"], b.loc[ids, "pred"], classes)
        K = len(classes)
        rng = np.random.default_rng(SEED)
        diffs = [metrics(y[i], pa[i], K)[1] - metrics(y[i], pb[i], K)[1]
                 for i in (rng.integers(0, len(y), len(y)) for _ in range(N_BOOT))]
        ia, ib = metrics(y, pa, K)[1], metrics(y, pb, K)[1]
        gaps.append(dict(crops=m["crops"], model=m["model"], knots_n=len(ids),
                         image_knots_balacc=ia, session_knots_balacc=ib, gap=ia - ib,
                         gap_lo=np.percentile(diffs, 2.5), gap_hi=np.percentile(diffs, 97.5)))
    pd.DataFrame(gaps).to_csv(os.path.join(OUT, "split_gap.csv"), index=False)

    # McNemar tests between related runs, and every run vs TopoStats
    comp = []
    for a, b in itertools.combinations(runs, 2):
        why = reason(a, b)
        if not why:
            continue
        A, B = data[a["run"]].set_index("id"), data[b["run"]].set_index("id")
        ids = A.index.intersection(B.index)
        for scope in ("all", "knots"):
            idx = ids if scope == "all" else ids[A.loc[ids, "label"].isin(KNOTS).to_numpy()]
            ca = (A.loc[idx, "pred"] == A.loc[idx, "label"]).to_numpy()
            cb = (B.loc[idx, "pred"] == B.loc[idx, "label"]).to_numpy()
            comp.append(dict(comparison=why, scope=scope, run_A=a["run"], run_B=b["run"], n=len(idx),
                             acc_A=ca.mean(), acc_B=cb.mean(), **mcnemar(ca, cb)))
    for m in runs:
        r = data[m["run"]]
        for scope in ("all", "knots"):
            g = r if scope == "all" else r[r.label.isin(KNOTS)]
            ca, cb = (g.pred == g.label).to_numpy(), (g.ts_topology == g.label).to_numpy()
            comp.append(dict(comparison="model vs TopoStats", scope=scope, run_A=m["run"], run_B="TopoStats",
                             n=len(g), acc_A=ca.mean(), acc_B=cb.mean(), **mcnemar(ca, cb)))
    pd.DataFrame(comp).to_csv(os.path.join(OUT, "comparisons.csv"), index=False)

    figure(summary)
    print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    main()