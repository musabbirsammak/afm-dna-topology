"""
Train a small network to answer "is strand A (red) on top of strand B (blue)?"
and compare it with TopoStats' rule (the wider strand is on top) on exactly the same crossings.

Uses:
  crossings/npy/<id>.npy     3 x 64 x 64 squares (height, strand A, strand B)  - from extract_crossings.py
  crossings/labels.csv       the correct answers (167 crossings)              - from make_labels.py
  crossings/metadata.csv     TopoStats' answers for all crossings (only for --teacher and --predict_all)
Molecules whose crossings ALL disagree with the expected sign ("all opposite") are left out.

Testing: crossings are grouped by imaging day. Each day is tested by a model that never saw
that day in training, so every labelled crossing is predicted exactly once.

Training modes:
  (default)            learn only from the correct answers of the training days
  --teacher            learn from TopoStats' own answers for all crossings of the training days
                       (about 2,000, about 80% correct; no correct answers used)
  --teacher --finetune the teacher model, then trained further on the correct answers of the training days

Making the most of little data (each square -> 16 versions, all with a known answer):
  - 4 rotations x mirror or not (a mirror image does NOT change which strand is on top)
  - red and blue swapped, with the answer flipped
  - heights scaled by 0.85-1.15 plus a little noise, so a session's overall brightness helps less
At test time the prediction is averaged over all 16 versions.

Output (default: results/crossings/):
  crossing_oof_<mode>.csv        one row per labelled crossing: correct answer, the network's
                                 answer and TopoStats' answer (mode = true_labels, teacher, teacher_finetune)
  all_crossings_predicted_<mode>.csv   only with --predict_all (see below)

Run (a laptop CPU is fine):
  python code/train_crossings.py                         # every crossings_* folder, all three modes
  python code/train_crossings.py --nm_per_px 0.5         # only crossings_05nm, all three modes
  python code/train_crossings.py --mode teacher          # every folder, only the teacher mode

  Optional: --predict_all also trains one final model on ALL labelled days, using the same mode, 
  and predicts every crossing in metadata.csv (2,457). Saved as all_crossings_predicted_<mode>.csv.
"""
import argparse, glob, os, time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from scipy.stats import binomtest

from extract_crossings import tag

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class SmallCNN(nn.Module):
    """~100k parameters: enough for a local 64 x 64 decision, small enough for ~170 examples."""
    def __init__(self):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.f = nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 64))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.f(x)).squeeze(1)


def dihedral(x, k, flip):
    x = torch.rot90(x, k, (-2, -1))
    return torch.flip(x, (-1,)) if flip else x


def swap(x):
    return x[:, [0, 2, 1]]


def augment(x, y):
    """x: B x 3 x 64 x 64, y: B. Random dihedral, random A/B swap, height jitter."""
    k = np.random.randint(4)
    x = dihedral(x, k, np.random.rand() < 0.5)
    s = torch.rand(x.shape[0], device=x.device) < 0.5
    x = torch.where(s[:, None, None, None], swap(x), x)
    y = torch.where(s, 1 - y, y)
    h = x[:, :1] * (0.85 + 0.3 * torch.rand(x.shape[0], 1, 1, 1, device=x.device))
    h = h + 0.02 * torch.randn_like(h) * (h > 0)
    return torch.cat([h.clamp(0, 1), x[:, 1:]], 1), y


@torch.no_grad()
def predict(model, x):
    """P(A on top), averaged over 8 dihedral versions x 2 (A/B swapped -> 1 - p)."""
    model.eval()
    ps = []
    for k in range(4):
        for fl in (False, True):
            xx = dihedral(x, k, fl)
            ps.append(torch.sigmoid(model(xx)))
            ps.append(1 - torch.sigmoid(model(swap(xx))))
    return torch.stack(ps).mean(0)


def fit(x, y, epochs, seed, dev, model=None, lr=1e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model if model is not None else SmallCNN().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = nn.BCEWithLogitsLoss()
    n = len(x)
    for ep in range(epochs):
        model.train()
        for idx in torch.randperm(n).split(32):
            xb, yb = augment(x[idx], y[idx])
            loss = lossf(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return model


def as_bool(s):
    """CSV columns of True/False can come back as text; empty cells stay missing (NaN)."""
    return s.map(lambda v: v if isinstance(v, bool) or pd.isna(v) else str(v).strip().lower() == "true")


MODES = {"true_labels": (False, False), "teacher": (True, False), "teacher_finetune": (True, True)}  # mode: (teacher, finetune)


def train_one(a, teacher, finetune, xtr, ytr, xt, yt, seed, dev):
    """One model in the chosen mode. xt, yt = TopoStats' answers (teacher data), may be None."""
    if teacher:
        m = fit(xt, yt, a.teacher_epochs, seed=seed, dev=dev)
        if finetune:
            m = fit(xtr, ytr, max(10, a.epochs // 3), seed=seed + 50, dev=dev, model=m, lr=2e-4)
        return m
    return fit(xtr, ytr, a.epochs, seed=seed, dev=dev)


def boot_ci(v, n=2000, seed=0):
    rng = np.random.default_rng(seed); v = np.asarray(v, float)
    m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return v.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def compare(g):
    """CNN vs TopoStats on the crossings in g: accuracy, 95% bootstrap CI, discordant counts, exact McNemar."""
    m, mlo, mhi = boot_ci(g.model_correct)
    t, tlo, thi = boot_ci(g.ts_correct)
    mo = int((g.model_correct & ~g.ts_correct).sum())
    to = int((~g.model_correct & g.ts_correct).sum())
    return dict(n=len(g),
                cnn_accuracy=round(m, 4), cnn_ci_lo=round(mlo, 4), cnn_ci_hi=round(mhi, 4),
                topostats_accuracy=round(t, 4), topostats_ci_lo=round(tlo, 4), topostats_ci_hi=round(thi, 4),
                both_right=int((g.model_correct & g.ts_correct).sum()), only_cnn_right=mo,
                only_topostats_right=to, both_wrong=int((~g.model_correct & ~g.ts_correct).sum()),
                p_mcnemar=round(binomtest(mo, mo + to).pvalue, 4) if mo + to > 0 else 1.0)


def report(r, mode):
    """Print the comparison for all crossings, per knot class and per TopoStats confidence; return it as rows."""
    lo = r.ts_confidence < 0.263
    groups = [("all", r)] + [(f"class {lab}", g) for lab, g in r.groupby("label")] \
             + [("confidence < 0.263", r[lo]), ("confidence >= 0.263", r[~lo])]
    rows = [dict(mode=mode, group=name, **compare(g)) for name, g in groups if len(g)]
    print("\n=== crossings never seen in training (grouped by imaging day) ===")
    for x in rows:
        print(f"  {x['group']:20s} n={x['n']:3d} | "
              f"CNN {x['cnn_accuracy']:.1%} [{x['cnn_ci_lo']:.1%}-{x['cnn_ci_hi']:.1%}] | "
              f"TopoStats {x['topostats_accuracy']:.1%} [{x['topostats_ci_lo']:.1%}-{x['topostats_ci_hi']:.1%}] | "
              f"only CNN right {x['only_cnn_right']}, only TopoStats right {x['only_topostats_right']} | "
              f"McNemar p = {x['p_mcnemar']:.3f}")
    return rows


def load(crossings, teacher):
    """Labelled crossings of one folder and, for the teacher modes, TopoStats' calls on all its crossings."""
    lab = pd.read_csv(os.path.join(crossings, "labels.csv"), dtype={"session": str})
    lab = lab[~lab.pattern.str.startswith("all opposite")].reset_index(drop=True)
    lab["true_a_on_top"] = as_bool(lab.true_a_on_top)
    lab["ts_correct"] = as_bool(lab.ts_correct)
    lab = lab[lab.true_a_on_top.notna()].reset_index(drop=True)   # a crossing without an answer cannot be used
    x = torch.from_numpy(np.stack([np.load(os.path.join(crossings, "npy", i + ".npy")) for i in lab.id])).float()
    y = torch.tensor(lab.true_a_on_top.astype(bool).values, dtype=torch.float)
    md = xt_all = yt_all = None
    if teacher:
        md = pd.read_csv(os.path.join(crossings, "metadata.csv"), dtype={"session": str})
        md["ts_a_on_top"] = as_bool(md.ts_a_on_top)
        md = md[md.ts_a_on_top.notna()].reset_index(drop=True)
        xt_all = torch.from_numpy(np.stack([np.load(os.path.join(crossings, "npy", i + ".npy")) for i in md.id])).float()
        yt_all = torch.tensor(md.ts_a_on_top.astype(bool).values, dtype=torch.float)
    return lab, x, y, md, xt_all, yt_all


def run_mode(a, crossings, data, mode, out, dev):
    lab, x, y, md, xt_all, yt_all = data
    teacher, finetune = MODES[mode]
    k = min(a.folds, lab.session.nunique())
    print(f"\n##### {os.path.basename(os.path.normpath(crossings))} | mode: {mode} | device {dev} | "
          f"{len(lab)} labelled crossings | {lab.session.nunique()} imaging days | {k} folds")
    if teacher:
        print(f"teacher data: {len(md)} crossings with TopoStats calls")

    probs = np.zeros(len(lab)); fold_of = np.zeros(len(lab), int); t0 = time.time()
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=k).split(lab, groups=lab.session)):
        test_days = sorted(lab.session.iloc[te].unique())
        xtr, ytr = x[tr].to(dev), y[tr].to(dev)
        p = np.zeros(len(te)); train_acc = []
        if teacher:
            keep = ~md.session.isin(test_days).values          # teacher data never includes the test days
            xt, yt = xt_all[keep].to(dev), yt_all[keep].to(dev)
        else:
            xt = yt = None
        for s in range(a.seeds):
            m = train_one(a, teacher, finetune, xtr, ytr, xt, yt, seed=100 * fold + s, dev=dev)
            p += predict(m, x[te].to(dev)).cpu().numpy() / a.seeds
            train_acc.append(((predict(m, xtr).cpu().numpy() > 0.5) == ytr.cpu().numpy().astype(bool)).mean())
        probs[te] = p; fold_of[te] = fold
        print(f"  fold {fold}: test days {test_days}, {len(te)} crossings, "
              f"CNN {((p > 0.5) == y[te].numpy().astype(bool)).mean():.0%} | "
              f"on its own training crossings {np.mean(train_acc):.0%} ({time.time() - t0:.0f}s)")

    r = lab.copy()
    r["fold"] = fold_of; r["p_a_on_top"] = probs; r["pred_a_on_top"] = probs > 0.5
    r["model_correct"] = r.pred_a_on_top == r.true_a_on_top.astype(bool)
    r["ts_correct"] = r.ts_correct.astype(bool)
    r.to_csv(os.path.join(out, f"crossing_oof_{mode}.csv"), index=False)
    rows = report(r, mode)
    print(f"\nsaved {out}/crossing_oof_{mode}.csv")

    if a.predict_all:
        # one final model per seed, trained on ALL labelled days in the SAME mode as above
        allc = pd.read_csv(os.path.join(crossings, "metadata.csv"), dtype={"session": str})
        allc["ts_a_on_top"] = as_bool(allc.ts_a_on_top)
        xa = torch.from_numpy(np.stack([np.load(os.path.join(crossings, "npy", i + ".npy")) for i in allc.id])).float()
        xt, yt = (xt_all.to(dev), yt_all.to(dev)) if teacher else (None, None)
        p = np.zeros(len(allc))
        for s in range(a.seeds):
            m = train_one(a, teacher, finetune, x.to(dev), y.to(dev), xt, yt, seed=999 + s, dev=dev)
            p += np.concatenate([predict(m, xb.to(dev)).cpu().numpy() for xb in xa.split(256)]) / a.seeds
        allc["cnn_p_a_on_top"] = p
        allc["cnn_a_on_top"] = p > 0.5
        allc.to_csv(os.path.join(out, f"all_crossings_predicted_{mode}.csv"), index=False)
        has = allc.ts_a_on_top.notna()
        agree = (allc.cnn_a_on_top[has] == allc.ts_a_on_top[has].astype(bool)).mean()
        print(f"saved {out}/all_crossings_predicted_{mode}.csv ({len(allc)} crossings; "
              f"CNN agrees with TopoStats on {agree:.0%} of the {has.sum()} crossings TopoStats answered)")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nm_per_px", type=float, default=None, help="only this resolution (default: every crossings_* folder)")
    ap.add_argument("--crossings", default=None, help="only this crossings folder (default: every crossings_* folder)")
    ap.add_argument("--mode", nargs="+", choices=list(MODES), default=list(MODES), help="default: all three")
    ap.add_argument("--out", default=None, help="only for a single folder; default: results/<name of the crossings folder>")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=3, help="models per fold, predictions averaged")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--predict_all", action="store_true")
    ap.add_argument("--teacher_epochs", type=int, default=20)
    a = ap.parse_args()

    if a.crossings:
        folders = [a.crossings]
    elif a.nm_per_px is not None:
        folders = [os.path.join(REPO, f"crossings_{tag(a.nm_per_px)}")]
    else:
        folders = sorted(f for f in glob.glob(os.path.join(REPO, "crossings_*"))
                         if os.path.exists(os.path.join(f, "labels.csv")))
    if not folders:
        raise SystemExit("no crossings_* folder with labels.csv: run extract_crossings.py and make_labels.py first")
    print("crossings folders:", [os.path.basename(os.path.normpath(f)) for f in folders], "| modes:", a.mode)

    dev = device()
    for folder in folders:
        name = os.path.basename(os.path.normpath(folder))
        out = a.out if (a.out and len(folders) == 1) else os.path.join(REPO, "results", name)
        os.makedirs(out, exist_ok=True)
        data = load(folder, teacher=any(MODES[m][0] for m in a.mode))
        rows = []
        for mode in a.mode:
            rows += run_mode(a, folder, data, mode, out, dev)

        # all printed numbers of this folder in one table; modes not run this time keep their earlier rows
        path = os.path.join(out, "crossing_results.csv")
        table = pd.DataFrame(rows)
        if os.path.exists(path):
            old = pd.read_csv(path)
            table = pd.concat([old[~old["mode"].isin(a.mode)], table], ignore_index=True)
        table["mode"] = pd.Categorical(table["mode"], list(MODES), ordered=True)
        table.sort_values("mode", kind="stable").to_csv(path, index=False)
        print(f"\nsaved {path}")