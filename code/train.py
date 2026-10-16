"""
Train and test the whole-molecule classifiers with 5-fold cross-validation.

Three models:
  resnet18      : ImageNet-pretrained ResNet-18, fine-tuned on the crops
  convnext_tiny : ImageNet-pretrained ConvNeXt-Tiny, fine-tuned on the crops
                  (a newer CNN, trained with exactly the same settings as resnet18)
  dinov2        : frozen DINOv2 ViT-S/14 features + logistic regression (no fine-tuning)

The two fine-tuned models use the same recipe: 30 epochs, AdamW (lr 3e-4, weight decay 1e-4),
cosine schedule, batch 32, class-weighted loss, random rotation and small shifts during training,
and predictions averaged over 4 rotations (0, 90, 180, 270 deg) at test time.

Every molecule is predicted exactly once, by the model trained on the folds it is NOT in.
The folds come from splits.csv (made by make_splits.py):
  --split session : whole imaging days are held out (the honest test)
  --split image   : only whole images are held out (the optimistic test)

Output: results/classify/<model>_<split>_oof.csv
  one row per molecule: true label, predicted label, probability of each class.
  The folder follows the crops folder: crops -> results/classify, crops_norm -> results/classify_norm,
  crops_binary -> results/classify_binary. Use --out to choose another folder.

Run (examples):
  python train.py --crops crops --model dinov2        --split session
  python train.py --crops crops --model dinov2        --split image
  python train.py --crops crops --model resnet18      --split session
  python train.py --crops crops --model resnet18      --split image
  python train.py --crops crops --model convnext_tiny --split session
  python train.py --crops crops --model convnext_tiny --split image
  python train.py --crops crops --model resnet18      --split session --flip   # control run
Then:
  python evaluate.py --oof results/classify/*_oof.csv

Flips: no horizontal/vertical flips by default. A flip turns a knot into its mirror image. Our
labels do not tell mirror images apart (5_1 and its mirror are both "5_1"), so a flip does not
change the correct answer. --flip is a control run that checks the results do not depend on this.
"""
import argparse, os, time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader, Dataset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class Crops(Dataset):
    def __init__(self, df, crops, classes, train, flip):
        self.df, self.crops, self.cls = df.reset_index(drop=True), crops, {c: i for i, c in enumerate(classes)}
        aug = [T.RandomRotation(180, fill=0), T.RandomAffine(0, translate=(0.05, 0.05), fill=0)] if train else []
        if train and flip:
            aug += [T.RandomHorizontalFlip(), T.RandomVerticalFlip()]
        self.tf = T.Compose(aug + [T.Resize(224, antialias=True), T.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        x = torch.from_numpy(np.load(os.path.join(self.crops, "npy", r.id + ".npy"))).float()[None].repeat(3, 1, 1)
        return self.tf(x), self.cls[r.label]


def resnet18(n_classes, pretrained):
    import torchvision
    w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = torchvision.models.resnet18(weights=w)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


def convnext_tiny(n_classes, pretrained):
    import torchvision
    w = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    m = torchvision.models.convnext_tiny(weights=w)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, n_classes)   # last layer -> our classes
    return m


FINETUNE = {"resnet18": resnet18, "convnext_tiny": convnext_tiny}   # models trained with run_resnet


def run_resnet(df, crops, classes, fold_col, args, dev):
    """Fine-tune one of the FINETUNE models (args.model) on each fold. Same recipe for all of them."""
    out = []
    for fold in sorted(df[fold_col].unique()):
        tr, te = df[df[fold_col] != fold], df[df[fold_col] == fold]
        counts = tr.label.value_counts().reindex(classes).fillna(0).values
        w = torch.tensor(counts.sum() / np.maximum(counts, 1) / len(classes), dtype=torch.float, device=dev)
        model = FINETUNE[args.model](len(classes), not args.no_pretrained).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
        dl = DataLoader(Crops(tr, crops, classes, True, args.flip), batch_size=32, shuffle=True,
                        num_workers=args.workers, drop_last=len(tr) > 32)
        lossf = nn.CrossEntropyLoss(weight=w)
        t0 = time.time()
        for ep in range(args.epochs):
            model.train(); tot = 0
            for x, y in dl:
                x, y = x.to(dev), y.to(dev)
                loss = lossf(model(x), y)
                opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(y)
            sched.step()
            if ep % 5 == 4 or ep == args.epochs - 1:
                print(f"  fold {fold} epoch {ep + 1}/{args.epochs} loss {tot / len(tr):.3f} ({time.time() - t0:.0f}s)")
        # test-time: average over 4 rotations (0, 90, 180, 270 deg)
        model.eval(); probs = []
        with torch.no_grad():
            for x, _ in DataLoader(Crops(te, crops, classes, False, False), batch_size=64, num_workers=args.workers):
                x = x.to(dev)
                p = sum(torch.softmax(model(torch.rot90(x, k, (2, 3))), 1) for k in range(4)) / 4
                probs.append(p.cpu().numpy())
        out.append(save_fold(te, np.concatenate(probs), classes, fold))
    return pd.concat(out)


def run_dinov2(df, crops, classes, fold_col, args, dev):
    import timm
    from sklearn.linear_model import LogisticRegression
    m = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=not args.no_pretrained,
                          num_classes=0, img_size=224).to(dev).eval()
    feats = []
    with torch.no_grad():
        for x, _ in DataLoader(Crops(df, crops, classes, False, False), batch_size=64, num_workers=args.workers):
            x = x.to(dev)
            f = sum(m(torch.rot90(x, k, (2, 3))) for k in range(4)) / 4      # rotation-averaged features
            feats.append(f.cpu().numpy())
    X = np.concatenate(feats); y = df.label.values
    out = []
    for fold in sorted(df[fold_col].unique()):
        tr, te = (df[fold_col] != fold).values, (df[fold_col] == fold).values
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(X[tr], y[tr])
        p = np.zeros((te.sum(), len(classes)))
        p[:, [classes.index(c) for c in clf.classes_]] = clf.predict_proba(X[te])
        out.append(save_fold(df[te], p, classes, fold))
        print(f"  fold {fold}: {te.sum()} test molecules")
    return pd.concat(out)


def save_fold(te, p, classes, fold):
    r = te[["id", "label", "sample", "session", "condition", "ts_topology"]].copy()
    r["fold"] = fold
    r["pred"] = [classes[i] for i in p.argmax(1)]
    for i, c in enumerate(classes):
        r[f"p_{c}"] = p[:, i]
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default=os.path.join(REPO, "crops"))
    ap.add_argument("--model", choices=["resnet18", "convnext_tiny", "dinov2"], default="dinov2")
    ap.add_argument("--split", choices=["session", "image"], default="session")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--flip", action="store_true", help="add flip augmentation (control experiment)")
    ap.add_argument("--no_pretrained", action="store_true", help="only for testing without internet")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not os.path.isdir(args.crops):
        args.crops = os.path.join(REPO, args.crops)
    if args.out is None:   # crops -> classify, crops_norm -> classify_norm, crops_binary -> classify_binary
        name = os.path.basename(os.path.normpath(args.crops))
        args.out = os.path.join(REPO, "results", name.replace("crops", "classify"))

    torch.manual_seed(0); np.random.seed(0)
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    df = pd.read_csv(os.path.join(args.crops, "splits.csv"))
    classes = sorted(df.label.unique())
    fold_col = f"fold_{args.split}"
    print(f"device {dev} | {len(df)} molecules | classes {classes} | split {fold_col}")
    res = (run_resnet if args.model in FINETUNE else run_dinov2)(df, args.crops, classes, fold_col, args, dev)
    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.model}_{args.split}" + ("_flip" if args.flip else "")
    path = os.path.join(args.out, f"{tag}_oof.csv")
    res.to_csv(path, index=False)
    acc = (res.label == res.pred).mean()
    print(f"\nsaved {path} | overall accuracy {acc:.1%}  -> now run evaluate.py")