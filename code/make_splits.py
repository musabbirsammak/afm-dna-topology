"""
Decide which molecules are used for training and which for testing.
Only molecules that passed the length filter in extract_crops.py 
(keep = True) are used.

Two kinds of split are written to splits.csv:

  fold_session : 5 folds. All molecules from one imaging SESSION (one day of imaging)
                 are always in the same fold. the model never sees the test day during training.
  fold_image   : 5 folds. All molecules from one IMAGE are kept together, but images
                 from the same day can end up in both training and test. used only to show how 
                 much the session style leaks.

Each fold also tries to contain every knot type in similar numbers.
Example: if 20% of all molecules are torus knots, each fold aims for about 20% torus knots.
This does not always work in the session split. Almost all trefoils were imaged on one day,
so almost all trefoils end up in the same fold.

Run:
  python make_splits.py --crops crops
  python make_splits.py --crops crops --k 10
"""
import argparse, os

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--crops", default=os.path.join(REPO, "crops"))
ap.add_argument("--k", type=int, default=5)
a = ap.parse_args()
if not os.path.isdir(a.crops):
    a.crops = os.path.join(REPO, a.crops)

md = pd.read_csv(os.path.join(a.crops, "metadata.csv"))
md = md[md.keep].reset_index(drop=True)

for col, group in (("fold_session", "session"), ("fold_image", "image")):
    md[col] = -1
    k = min(a.k, md[group].nunique())
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=0)
    for fold, (_, te) in enumerate(cv.split(md, md.label, groups=md[group])):
        md.loc[te, col] = fold

md.to_csv(os.path.join(a.crops, "splits.csv"), index=False)
print(f"saved {a.crops}/splits.csv with {len(md)} molecules\n")
print("molecules per class:\n", md.label.value_counts().to_string(), "\n")
print("sessions per class:\n", md.groupby("label").session.nunique().to_string(), "\n")
print("SESSION split - test molecules per fold and class:")
print(pd.crosstab(md.label, md.fold_session).to_string(), "\n")
print("IMAGE split - test molecules per fold and class:")
print(pd.crosstab(md.label, md.fold_image).to_string())
