"""
Collect every crossing-classifier result into tables for the paper.

Reads every results/crossings_<resolution>/crossing_oof_<mode>.csv written by train_crossings.py
(one row per labelled crossing, each predicted once by a model that never saw its imaging day).

Each run is described by:
  resolution  taken from the folder name (crossings_1nm -> 1nm, crossings_05nm -> 05nm, ...)
  mode        true_labels, teacher or teacher_finetune

TopoStats (the FWHM rule) is scored on the same crossings.
95% confidence intervals: 2,000 bootstrap resamples of the crossings (seed 0).

Comparisons, each an exact McNemar test on the same crossings:
  CNN vs TopoStats    every run
  resolution          same mode, different resolution (e.g. 0.5 vs 1 nm per pixel)
  training regime     same resolution, different mode

p values are Holm-adjusted over all comparisons in comparisons.csv (the Stage 1 family).
Both unadjusted and adjusted values are saved.

Output: results/crossing_performance/
  summary.csv       one row per run, plus TopoStats: accuracy with 95% CI
  per_class.csv     accuracy per knot type, per run and for TopoStats
  per_session.csv   accuracy per imaging day, per run and for TopoStats
  comparisons.csv   the McNemar tests, with Holm-adjusted p values

Run from the project folder, after train_crossings.py:
  python code/evaluate_crossings.py
"""
import glob, itertools, os, re

import numpy as np
import pandas as pd
from scipy.stats import binomtest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results")
OUT = os.path.join(RES, "crossing_performance")
MODES = ["true_labels", "teacher", "teacher_finetune"]
FILE_RE = re.compile(r"^crossing_oof_(true_labels|teacher|teacher_finetune)\.csv$")
N_BOOT, SEED = 2000, 0


def as_bool(s):
    """CSV columns of True/False can come back as text."""
    return s.map(lambda v: v if isinstance(v, bool) or pd.isna(v) else str(v).strip().lower() == "true")


def find_runs():
    runs = []
    for path in glob.glob(os.path.join(RES, "crossings_*", "crossing_oof_*.csv")):
        m = FILE_RE.match(os.path.basename(path))
        if not m:
            continue
        res = os.path.basename(os.path.dirname(path)).replace("crossings_", "", 1)
        runs.append(dict(run=f"{res}__{m.group(1)}", path=path, resolution=res, mode=m.group(1)))
    # 1 nm first, then the other resolutions; modes in the order true_labels, teacher, teacher_finetune
    return sorted(runs, key=lambda r: (r["resolution"] != "1nm", r["resolution"], MODES.index(r["mode"])))


def load(path):
    r = pd.read_csv(path, dtype={"session": str})
    for c in ("model_correct", "ts_correct", "true_a_on_top"):
        r[c] = as_bool(r[c]).astype(bool)
    return r.set_index("id")


# ----------------------------------------------------------------------------- statistics
def acc_ci(correct):
    """Accuracy and 95% percentile bootstrap CI."""
    v = np.asarray(correct, float)
    rng = np.random.default_rng(SEED)
    boots = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(N_BOOT)]
    return v.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def mcnemar(ca, cb):
    """Exact McNemar test: binomial test on the crossings only one of the two got right."""
    a_only, b_only = int((ca & ~cb).sum()), int((~ca & cb).sum())
    return dict(both_right=int((ca & cb).sum()), only_A_right=a_only, only_B_right=b_only,
                both_wrong=int((~ca & ~cb).sum()),
                p_mcnemar=binomtest(a_only, a_only + b_only).pvalue if a_only + b_only else 1.0)


def holm(p):
    """Holm-adjusted p values, returned in the original order."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.minimum(np.maximum.accumulate((m - np.arange(m)) * p[order]), 1.0)
    out = np.empty(m)
    out[order] = adj
    return out


def reason(a, b):
    """Why runs a and b should be compared, or None."""
    if a["mode"] == b["mode"] and a["resolution"] != b["resolution"]:
        return "resolution"
    if a["resolution"] == b["resolution"] and a["mode"] != b["mode"]:
        return "training regime"
    return None


def main():
    runs = find_runs()
    if not runs:
        print(f"no crossing_oof_*.csv files found in {RES}/crossings_*/ - run train_crossings.py first")
        return
    os.makedirs(OUT, exist_ok=True)
    data = {m["run"]: load(m["path"]) for m in runs}
    print(f"{len(runs)} runs found\n")

    # every run must give the same correct answer for the crossings it shares with the others
    ref = data[runs[0]["run"]]
    for m in runs[1:]:
        r = data[m["run"]]
        ids = ref.index.intersection(r.index)
        bad = int((ref.loc[ids, "true_a_on_top"] != r.loc[ids, "true_a_on_top"]).sum())
        if bad:
            print(f"WARNING: {m['run']} and {runs[0]['run']} disagree on the correct answer of {bad} crossings")

    summary, per_class, per_session = [], [], []

    def add(run, resolution, mode, r, col):
        acc, lo, hi = acc_ci(r[col])
        summary.append(dict(run=run, resolution=resolution, mode=mode, n=len(r),
                            accuracy=round(acc, 4), accuracy_lo=round(lo, 4), accuracy_hi=round(hi, 4)))
        for c, g in r.groupby("label"):
            per_class.append(dict(run=run, label=c, n=len(g), accuracy=round(g[col].mean(), 4)))
        for c, g in r.groupby("session"):
            per_session.append(dict(run=run, session=c, n=len(g), types=" ".join(sorted(g.label.unique())),
                                    accuracy=round(g[col].mean(), 4)))
        print(f"  {run:30s} n={len(r):3d}  accuracy {acc:6.1%}  [{lo:.1%} - {hi:.1%}]")

    for m in runs:
        add(m["run"], m["resolution"], m["mode"], data[m["run"]], "model_correct")

    # TopoStats does not depend on the run, so it is scored once per distinct set of crossings
    seen = {}
    for m in runs:
        seen.setdefault(frozenset(data[m["run"]].index), m)
    for m in seen.values():
        add(f"{m['resolution']}__TopoStats", m["resolution"], "TopoStats", data[m["run"]], "ts_correct")

    pd.DataFrame(summary).to_csv(os.path.join(OUT, "summary.csv"), index=False)
    pd.DataFrame(per_class).pivot(index="run", columns="label", values="accuracy") \
        .to_csv(os.path.join(OUT, "per_class.csv"))
    pd.DataFrame(per_session).to_csv(os.path.join(OUT, "per_session.csv"), index=False)

    # McNemar tests: every run vs TopoStats, then related runs vs each other
    comp = []
    for m in runs:
        r = data[m["run"]]
        ca, cb = r.model_correct.to_numpy(), r.ts_correct.to_numpy()
        comp.append(dict(comparison="CNN vs TopoStats", run_A=m["run"], run_B=f"{m['resolution']}__TopoStats",
                         n=len(r), acc_A=round(ca.mean(), 4), acc_B=round(cb.mean(), 4), **mcnemar(ca, cb)))
    for a, b in itertools.combinations(runs, 2):
        why = reason(a, b)
        if not why:
            continue
        A, B = data[a["run"]], data[b["run"]]
        ids = A.index.intersection(B.index)
        ca, cb = A.loc[ids, "model_correct"].to_numpy(), B.loc[ids, "model_correct"].to_numpy()
        comp.append(dict(comparison=why, run_A=a["run"], run_B=b["run"], n=len(ids),
                         acc_A=round(ca.mean(), 4), acc_B=round(cb.mean(), 4), **mcnemar(ca, cb)))
    comp = pd.DataFrame(comp)
    comp["p_holm"] = holm(comp.p_mcnemar)
    comp["p_mcnemar"] = comp.p_mcnemar.round(4)
    comp["p_holm"] = comp.p_holm.round(4)
    comp.to_csv(os.path.join(OUT, "comparisons.csv"), index=False)

    print(f"\n{len(comp)} comparisons (Holm-adjusted together):")
    for x in comp.itertuples():
        print(f"  {x.comparison:17s} {x.run_A:28s} vs {x.run_B:28s} {x.acc_A:6.1%} vs {x.acc_B:6.1%} | "
              f"only A {x.only_A_right:2d}, only B {x.only_B_right:2d} | p = {x.p_mcnemar:.3f}, Holm p = {x.p_holm:.3f}")
    print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    main()