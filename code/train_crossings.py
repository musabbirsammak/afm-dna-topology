"""
WEEK 2, STEP 3 - Train a small network to say "is strand A (red) on top of strand B (blue)?"
and compare it with TopoStats (FWHM rule) on exactly the same crossings.

Uses:
  ../crossings/npy/<id>.npy      3 x 64 x 64 squares (height, strand A, strand B)  - from extract_crossings.py
  ../crossings/labels.csv        automatic correct answers                          - from make_labels.py

Test is done on IMAGING DAYS THE MODEL NEVER SAW (crossings grouped by session), every crossing
predicted exactly once. Two "molecule-level" cheats are impossible here: the square is 64 nm wide
and only one crossing is asked about.

Tricks for the small dataset (each square -> 16 versions, all with a correct answer):
  - 8 rotations/mirror images (a mirror image does NOT change which strand is on top)
  - swap red and blue and flip the answer
  - small random height scaling + noise, so absolute brightness of a session is less useful
At test time the prediction is averaged over all 16 versions.

Run (Mac is fine):
  python train_crossings.py --crossings ../crossings --out ../results_crossings                       # learn from the ~170 correct answers only
  python train_crossings.py --crossings ../crossings --out ../results_crossings --teacher             # learn from TopoStats' calls on ~2,000 crossings
  python train_crossings.py --crossings ../crossings --out ../results_crossings --teacher --finetune  # ...then polish with the correct answers
Test crossings are always the correct answers of imaging days left out of ALL training.
Optional, after you are happy with the numbers: predict EVERY crossing (2,457) with a model trained on all labels
  python train_crossings.py --crossings ../crossings --out ../results_crossings --predict_all
"""
import argparse, os, time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

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


def boot_ci(v, n=2000, seed=0):
    rng = np.random.default_rng(seed); v = np.asarray(v, float)
    m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return v.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def report(r):
    print("\n=== crossings never seen in training (grouped by imaging day) ===")
    for name, col in (("small CNN", "model_correct"), ("TopoStats FWHM", "ts_correct")):
        m, lo, hi = boot_ci(r[col])
        print(f"  {name:15s} accuracy {m:.1%}  [95% CI {lo:.1%}-{hi:.1%}]  n={len(r)}")
    both = (r.model_correct & r.ts_correct).sum(); mo = (r.model_correct & ~r.ts_correct).sum()
    to = (~r.model_correct & r.ts_correct).sum(); none = (~r.model_correct & ~r.ts_correct).sum()
    print(f"  both right {both} | only CNN right {mo} | only TopoStats right {to} | both wrong {none}")
    if mo + to > 0:
        from scipy.stats import binomtest
        print(f"  McNemar exact test (is the difference real?): p = {binomtest(mo, mo + to).pvalue:.3f}")
    print("\n  by knot class:")
    for lab, g in r.groupby("label"):
        print(f"    {lab}: CNN {g.model_correct.mean():.0%} | TopoStats {g.ts_correct.mean():.0%} | n={len(g)}")
    lo = r.ts_confidence < 0.263
    print("\n  by TopoStats confidence:")
    print(f"    unreliable (<0.263): CNN {r[lo].model_correct.mean():.0%} | TopoStats {r[lo].ts_correct.mean():.0%} | n={lo.sum()}")
    print(f"    reliable  (>=0.263): CNN {r[~lo].model_correct.mean():.0%} | TopoStats {r[~lo].ts_correct.mean():.0%} | n={(~lo).sum()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--crossings", default=os.path.join(REPO, "crossings"))
    ap.add_argument("--out", default=os.path.join(REPO, "results", "crossings"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=3, help="models per fold, predictions averaged")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--predict_all", action="store_true")
    ap.add_argument("--teacher", action="store_true",
                    help="train on TopoStats' own calls for ALL crossings of the training days (~2,000, ~80%% correct)")
    ap.add_argument("--finetune", action="store_true",
                    help="with --teacher: afterwards fine-tune on the correct answers of the training days")
    ap.add_argument("--teacher_epochs", type=int, default=20)
    a = ap.parse_args()
    dev = device()
    os.makedirs(a.out, exist_ok=True)

    lab = pd.read_csv(os.path.join(a.crossings, "labels.csv"), dtype={"session": str})
    lab = lab[~lab.pattern.str.startswith("all opposite")].reset_index(drop=True)
    x = torch.from_numpy(np.stack([np.load(os.path.join(a.crossings, "npy", i + ".npy")) for i in lab.id])).float()
    y = torch.tensor(lab.true_a_on_top.astype(bool).values, dtype=torch.float)
    k = min(a.folds, lab.session.nunique())
    print(f"device {dev} | {len(lab)} labelled crossings | {lab.session.nunique()} imaging days | {k} folds")

    if a.teacher:
        md = pd.read_csv(os.path.join(a.crossings, "metadata.csv"), dtype={"session": str})
        md = md[md.ts_a_on_top.notna()].reset_index(drop=True)
        xt_all = torch.from_numpy(np.stack([np.load(os.path.join(a.crossings, "npy", i + ".npy")) for i in md.id])).float()
        yt_all = torch.tensor(md.ts_a_on_top.astype(str).str.lower().eq("true").values, dtype=torch.float)
        print(f"teacher mode: {len(md)} crossings with TopoStats calls available" + (" + fine-tuning" if a.finetune else ""))

    probs = np.zeros(len(lab)); fold_of = np.zeros(len(lab), int); t0 = time.time()
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=k).split(lab, groups=lab.session)):
        test_days = sorted(lab.session.iloc[te].unique())
        xtr, ytr = x[tr].to(dev), y[tr].to(dev)
        p = np.zeros(len(te)); train_acc = []
        for s in range(a.seeds):
            if a.teacher:
                keep = ~md.session.isin(test_days).values
                m = fit(xt_all[keep].to(dev), yt_all[keep].to(dev), a.teacher_epochs, seed=100 * fold + s, dev=dev)
                if a.finetune:
                    m = fit(xtr, ytr, max(10, a.epochs // 3), seed=100 * fold + s + 50, dev=dev, model=m, lr=2e-4)
            else:
                m = fit(xtr, ytr, a.epochs, seed=100 * fold + s, dev=dev)
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
    tag = "teacher_finetune" if (a.teacher and a.finetune) else ("teacher" if a.teacher else "true_labels")
    r.to_csv(os.path.join(a.out, f"crossing_oof_{tag}.csv"), index=False)
    print(f"\nmode: {tag}")
    report(r)
    print(f"\nsaved {a.out}/crossing_oof_{tag}.csv")

    if a.predict_all:
        md = pd.read_csv(os.path.join(a.crossings, "metadata.csv"))
        xa = torch.from_numpy(np.stack([np.load(os.path.join(a.crossings, "npy", i + ".npy")) for i in md.id])).float()
        p = np.zeros(len(md))
        for s in range(a.seeds):
            m = fit(x.to(dev), y.to(dev), a.epochs, seed=999 + s, dev=dev)
            p += np.concatenate([predict(m, xb.to(dev)).cpu().numpy() for xb in xa.split(256)]) / a.seeds
        md["cnn_p_a_on_top"] = p
        md.to_csv(os.path.join(a.out, "all_crossings_predicted.csv"), index=False)
        agree = ((p > 0.5) == md.ts_a_on_top.astype(bool)).mean()
        print(f"saved {a.out}/all_crossings_predicted.csv ({len(md)} crossings; CNN agrees with TopoStats on {agree:.0%})")