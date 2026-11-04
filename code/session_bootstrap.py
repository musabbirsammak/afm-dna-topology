"""
Confidence intervals that resample imaging sessions instead of single molecules.

Why: molecules imaged in the same session are not independent, because they share the tip,
the surface and the buffer of that session. The intervals in summary.csv resample molecules,
which treats them as independent and makes the intervals too narrow. Here whole sessions are
resampled instead (a cluster bootstrap).

How: for each knot type, the sessions that contain that type are drawn with replacement, as many
as there are, and all molecules of that type from the drawn sessions are used. Resampling within
each type keeps every type in every resample, so balanced accuracy always averages over the same
types. A session with two types (for example 2022-11-16, with 5_1 and 5_2) is drawn separately
for each type. 2,000 resamples, seed 0, the same as evaluate_classifiers.py.

Reads the same files as evaluate_classifiers.py: results/classify*/<model>_<split>[_flip]_oof.csv
and uses its functions, so runs, metrics and TopoStats scoring are identical.

Run from the project folder, after evaluate_classifiers.py:
  python code/session_bootstrap.py

Output: results/classifier_performance_session_boot/
  summary.csv     one row per run and for TopoStats. Accuracy and balanced accuracy, all classes and
                  knots only, each with the molecule-level 95% CI (as in the paper, *_mol_lo/_hi)
                  and the session-level 95% CI (*_sess_lo/_hi)
  split_gap.csv   image split minus session split (knots-only balanced accuracy) on the same
                  molecules, with the session-level 95% CI
  fig2_knots_balanced_accuracy.pdf/.png   the knots-only figure redrawn with the session-level CIs
"""
import os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import evaluate_classifiers as ec
except ImportError:
    sys.exit("evaluate_classifiers.py must be in the same folder as this script")

OUT = os.path.join(ec.RES, "classifier_performance_session_boot")


def session_resamples(d, n=ec.N_BOOT, seed=ec.SEED):
    """Row positions of n session-level resamples of frame d (columns label, session)."""
    by_class = {}
    for (c, _), idx in d.groupby(["label", "session"]).indices.items():   # keys come sorted
        by_class.setdefault(c, []).append(idx)
    rng = np.random.default_rng(seed)
    for _ in range(n):
        parts = []
        for c in sorted(by_class):
            blocks = by_class[c]
            parts.extend(blocks[j] for j in rng.integers(0, len(blocks), len(blocks)))
        yield np.concatenate(parts)


def score(d, pred_col, classes, prefix):
    """Accuracy and balanced accuracy with molecule-level and session-level 95% CIs."""
    d = d.reset_index(drop=True)
    y, p = ec.encode(d.label, d[pred_col], classes)
    K = len(classes)
    est, mol_lo, mol_hi = ec.boot_metrics(y, p, K)
    sess = np.array([ec.metrics(y[i], p[i], K) for i in session_resamples(d)])
    s_lo, s_hi = np.percentile(sess, 2.5, 0), np.percentile(sess, 97.5, 0)
    out = {f"{prefix}_n": len(d), f"{prefix}_sessions": d.session.nunique()}
    for j, name in ((0, "acc"), (1, "balacc")):
        out[f"{prefix}_{name}"] = est[j]
        out[f"{prefix}_{name}_mol_lo"], out[f"{prefix}_{name}_mol_hi"] = mol_lo[j], mol_hi[j]
        out[f"{prefix}_{name}_sess_lo"], out[f"{prefix}_{name}_sess_hi"] = s_lo[j], s_hi[j]
    return out


def main():
    runs = ec.find_runs()
    if not runs:
        sys.exit(f"no *_oof.csv files found in {ec.RES}/classify*/ - run train_crops.py first")
    os.makedirs(OUT, exist_ok=True)
    print(f"{len(runs)} runs found\n")

    data, rows = {}, []
    for meta in runs:
        r = ec.load(meta["path"]); data[meta["run"]] = r
        classes = sorted(r.label.unique())
        row = {key: meta[key] for key in ("run", "crops", "model", "split", "flip", "epochs", "input", "resolution")}
        row.update(score(r, "pred", classes, "all"))
        row.update(score(r[r.label.isin(ec.KNOTS)], "pred", classes, "knots"))
        rows.append(row)
        print(f"  {meta['run']:42s} knots bal.acc {row['knots_balacc']:6.1%}  "
              f"molecules [{row['knots_balacc_mol_lo']:.1%} - {row['knots_balacc_mol_hi']:.1%}]  "
              f"sessions [{row['knots_balacc_sess_lo']:.1%} - {row['knots_balacc_sess_hi']:.1%}]")

    # TopoStats, once per distinct set of molecules, named as in evaluate_classifiers.py
    seen = {}
    for m in sorted(runs, key=lambda m: m["base_crops"] != "1nm"):
        seen.setdefault(frozenset(data[m["run"]].id), m["base_crops"])
    for base in seen.values():
        meta = next(m for m in runs if m["base_crops"] == base)
        r = data[meta["run"]]
        classes = sorted(r.label.unique())
        row = dict(run=f"{base}__TopoStats", crops=base, model="TopoStats", split="", flip=False,
                   epochs=None, input=meta["input"], resolution=meta["resolution"])
        row.update(score(r, "ts_topology", classes, "all"))
        row.update(score(r[r.label.isin(ec.KNOTS)], "ts_topology", classes, "knots"))
        rows.append(row)
        print(f"  {row['run']:42s} knots bal.acc {row['knots_balacc']:6.1%}  "
              f"molecules [{row['knots_balacc_mol_lo']:.1%} - {row['knots_balacc_mol_hi']:.1%}]  "
              f"sessions [{row['knots_balacc_sess_lo']:.1%} - {row['knots_balacc_sess_hi']:.1%}]")
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUT, "summary.csv"), index=False)

    # image split minus session split on the same knot molecules, same resamples for both runs
    gaps = []
    for m in runs:
        twin = f"{m['crops']}__{m['model']}_session"
        if m["split"] != "image" or m["flip"] or twin not in data:
            continue
        a, b = data[m["run"]].set_index("id"), data[twin].set_index("id")
        ids = a.index.intersection(b.index)
        ids = ids[a.loc[ids, "label"].isin(ec.KNOTS).to_numpy()]
        d = a.loc[ids, ["label", "session"]].reset_index(drop=True)
        classes = sorted(a.label.unique()); K = len(classes)
        y, pa = ec.encode(d.label, a.loc[ids, "pred"], classes)
        _, pb = ec.encode(d.label, b.loc[ids, "pred"], classes)
        diffs = [ec.metrics(y[i], pa[i], K)[1] - ec.metrics(y[i], pb[i], K)[1] for i in session_resamples(d)]
        ia, ib = ec.metrics(y, pa, K)[1], ec.metrics(y, pb, K)[1]
        gaps.append(dict(crops=m["crops"], model=m["model"], knots_n=len(ids),
                         image_knots_balacc=ia, session_knots_balacc=ib, gap=ia - ib,
                         gap_sess_lo=np.percentile(diffs, 2.5), gap_sess_hi=np.percentile(diffs, 97.5)))
    pd.DataFrame(gaps).to_csv(os.path.join(OUT, "split_gap.csv"), index=False)

    # the knots-only figure, with the session-level intervals
    fig = summary.rename(columns={"knots_balacc_sess_lo": "knots_balacc_lo", "knots_balacc_sess_hi": "knots_balacc_hi"})
    ec.OUT = OUT
    ec.figure(fig)
    print(f"\nsaved summary.csv, split_gap.csv and fig2_knots_balanced_accuracy.pdf/.png to {OUT}")


if __name__ == "__main__":
    main()
