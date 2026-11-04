"""
Can the imaging session be recognized from the images of molecules of one knot type?

Why: the paper says the whole-molecule classifiers recognize the imaging session rather than
the knot. This tests that directly. Within one DNA preparation (column "sample", for example the
nicked catenanes) every molecule has the same topology and the same supercoiling, so the only
thing that can tell the sessions apart is how the images were acquired. Probing within a
preparation, not within a knot type, matters because supercoiled and nicked molecules of the same
type look different and were often imaged on different days. If a simple
classifier predicts the session well above chance, sessions leave a visible signature that a
knot classifier can exploit.

How:
  features  frozen DINOv2 ViT-S/14 (vit_small_patch14_dinov2.lvd142m), computed exactly as in the
            DINOv2 classifier of train_crops.py: crop copied to 3 channels, resized to 224 px,
            ImageNet normalization, features averaged over rotations of 0, 90, 180 and 270 deg.
  probe     logistic regression (C = 1, class_weight = "balanced"), the same as in Stage 0,
            trained to predict the session.
  folds     5-fold StratifiedGroupKFold with the AFM image as the group, so molecules from one
            image are never in both training and test.
  score     balanced accuracy over sessions. Chance = 1 / number of sessions.
  p value   permutation test. The sessions are shuffled between whole images (each image keeps
            all its molecules together and the number of images per session stays the same),
            and the whole cross-validation is repeated --perm times.

Probes are run for each preparation, and again for each preparation within one imaging buffer
(column "condition"), so that the 2022 Ni and 2023 Mg/Ni buffers alone cannot explain the result.
Sessions with fewer than --min_images images are left out, because cross-validation cannot both
train and test on them. A probe needs at least 2 sessions.

Run from the project folder:
  python code/session_probe.py                          # crops_1nm, 200 permutations
  python code/session_probe.py --crops crops_1nm --perm 1000

Output: results/session_probe/<crops folder>/
  dinov2_features.npz   cached features (delete it to recompute)
  summary.csv           one row per probe: molecules, images, sessions, balanced accuracy,
                        chance level, ratio to chance, permutation p value
  per_session.csv       recall of each session in each probe
  confusion/<probe>.csv confusion matrix (rows = true session, columns = predicted session)
"""
import argparse, os, time

# torch must be imported before numpy/scikit-learn: on macOS with conda the other order can
# load two copies of OpenMP and crash with "segmentation fault". train_crops.py uses this order too.
import torch
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
import warnings

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# small sessions trigger "least populated class has fewer members than n_splits"; expected here
warnings.filterwarnings("ignore", message="The least populated class")


# ----------------------------------------------------------------------------- features
class Crops(Dataset):
    """The crops exactly as the DINOv2 run of train_crops.py sees them (3 channels, 224 px, ImageNet norm)."""
    def __init__(self, ids, crops):
        self.ids, self.crops = list(ids), crops
        self.tf = T.Compose([T.Resize(224, antialias=True), T.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        x = torch.from_numpy(np.load(os.path.join(self.crops, "npy", self.ids[i] + ".npy"))).float()[None].repeat(3, 1, 1)
        return self.tf(x)


def pick_device(name):
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def dinov2_features(df, crops, cache, workers, device):
    if os.path.isfile(cache):
        z = np.load(cache, allow_pickle=True)
        if list(z["ids"]) == list(df.id):
            print(f"features loaded from {cache}")
            return z["X"]
        print("cached features are for other molecules - recomputing")

    import timm

    dev = pick_device(device)
    print(f"computing DINOv2 features for {len(df)} molecules on {dev} (done once, then cached)")
    torch.manual_seed(0)
    m = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224).to(dev).eval()
    feats, t0 = [], time.time()
    with torch.no_grad():
        for x in DataLoader(Crops(df.id, crops), batch_size=64, num_workers=workers):
            x = x.to(dev)
            feats.append((sum(m(torch.rot90(x, k, (2, 3))) for k in range(4)) / 4).cpu().numpy())
    X = np.concatenate(feats)
    np.savez(cache, X=X, ids=np.array(df.id, dtype=object))
    print(f"features computed on {dev} in {time.time() - t0:.0f}s, saved to {cache}")
    return X


# ----------------------------------------------------------------------------- probe
def cross_val_predict(X, y, groups, k, seed=0):
    pred = np.empty(len(y), dtype=object)
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    for tr, te in cv.split(X, y, groups=groups):
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)
        pred[te] = clf.fit(X[tr], y[tr]).predict(X[te])
    return pred


def run_probe(X, d, k, n_perm, seed=0):
    """d: molecules of the probe (columns session, image). Returns observed score, predictions, p value."""
    y, g = d.session.to_numpy(), d.image.to_numpy()
    k = min(k, d.image.nunique())
    pred = cross_val_predict(X, y, g, k)
    obs = balanced_accuracy_score(y, pred)

    # shuffle sessions between whole images
    img_sess = d.groupby("image").session.first()
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        perm = pd.Series(rng.permutation(img_sess.to_numpy()), index=img_sess.index)
        yp = d.image.map(perm).to_numpy()
        null.append(balanced_accuracy_score(yp, cross_val_predict(X, yp, g, k)))
    p = (1 + np.sum(np.array(null) >= obs)) / (1 + n_perm) if n_perm else np.nan
    return obs, pred, p, (np.mean(null) if null else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="crops_1nm")
    ap.add_argument("--k", type=int, default=5, help="number of folds")
    ap.add_argument("--min_images", type=int, default=5, help="leave out sessions with fewer images")
    ap.add_argument("--perm", type=int, default=200, help="permutations for the p value (0 = none)")
    ap.add_argument("--workers", type=int, default=0, help="data-loading processes (0 is safest on macOS)")
    ap.add_argument("--device", default="auto", help="auto, cpu, mps or cuda")
    a = ap.parse_args()

    crops = a.crops if os.path.isdir(a.crops) else os.path.join(REPO, a.crops)
    split_file = os.path.join(crops, "splits.csv")
    if not os.path.isfile(split_file):
        raise SystemExit(f"{split_file} not found - run make_splits.py first")
    name = os.path.basename(os.path.normpath(crops))
    out = os.path.join(REPO, "results", "session_probe", name)
    os.makedirs(os.path.join(out, "confusion"), exist_ok=True)

    df = pd.read_csv(split_file, dtype={"session": str, "image": str}).reset_index(drop=True)
    X = dinov2_features(df, crops, os.path.join(out, "dinov2_features.npz"), a.workers, a.device)
    print(f"{len(df)} molecules, {df.session.nunique()} sessions, {df.image.nunique()} images\n")

    preps = sorted(df["sample"].astype(str).unique())
    groups = [(sp, "all", df["sample"].astype(str) == sp) for sp in preps]
    if "condition" in df.columns:
        groups += [(sp, v, (df["sample"].astype(str) == sp) & (df.condition == v))
                   for sp in preps for v in sorted(df.condition.dropna().unique())]

    summary, per_session, done = [], [], set()
    for prep, cond, mask in groups:
        d = df[mask]
        n_img = d.groupby("session").image.nunique()
        keep = n_img[n_img >= a.min_images].index
        d = d[d.session.isin(keep)]
        ids = frozenset(d.id)
        if d.session.nunique() < 2 or ids in done:      # too few sessions, or same molecules as a probe already run
            continue
        done.add(ids)
        probe = f"{prep}_{cond}"
        t0 = time.time()
        obs, pred, p, null_mean = run_probe(X[d.index], d, a.k, a.perm)
        chance = 1 / d.session.nunique()
        summary.append(dict(probe=probe, sample=prep, label=" ".join(sorted(d.label.unique())),
                            condition=cond, molecules=len(d),
                            images=d.image.nunique(), sessions=d.session.nunique(),
                            session_list=" ".join(sorted(d.session.unique())),
                            balanced_accuracy=obs, chance=chance, ratio_to_chance=obs / chance,
                            permutation_mean=null_mean, p_permutation=p, permutations=a.perm))
        sessions = sorted(d.session.unique())
        cm = confusion_matrix(d.session, pred.astype(str), labels=sessions)
        pd.DataFrame(cm, index=[f"true_{s}" for s in sessions], columns=[f"pred_{s}" for s in sessions]) \
            .to_csv(os.path.join(out, "confusion", f"{probe}.csv"))
        for s, row in zip(sessions, cm):
            per_session.append(dict(probe=probe, session=s, molecules=int(row.sum()),
                                    images=int(n_img[s]), recall=row[sessions.index(s)] / max(row.sum(), 1)))
        print(f"  {probe:28s} {len(d):4d} molecules  {d.session.nunique()} sessions  "
              f"balanced acc {obs:.1%} (chance {chance:.1%})  p = {p:.3g}  ({time.time() - t0:.0f}s)")

    pd.DataFrame(summary).to_csv(os.path.join(out, "summary.csv"), index=False)
    pd.DataFrame(per_session).to_csv(os.path.join(out, "per_session.csv"), index=False)
    print(f"\nsaved summary.csv, per_session.csv and confusion/ to {out}")


if __name__ == "__main__":
    main()
