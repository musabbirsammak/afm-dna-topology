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

Output, for each crops folder:
  <crops folder>/splits.csv                       molecules with their fold numbers (read by train.py)
  results/splits/<crops folder>/class_counts.csv  molecules and sessions per knot type
  results/splits/<crops folder>/folds_session.csv test molecules per fold and knot type, session split
  results/splits/<crops folder>/folds_image.csv   the same for the image split

Run from the project folder:
  python code/make_splits.py                  # every folder named crops* that has a metadata.csv
  python code/make_splits.py --crops crops    # only this folder
  python code/make_splits.py --k 10           # a different number of folds
"""
import argparse, glob, os

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_splits(crops, k_max):
    name = os.path.basename(os.path.normpath(crops))
    md = pd.read_csv(os.path.join(crops, "metadata.csv"))
    md = md[md.keep].reset_index(drop=True)

    for col, group in (("fold_session", "session"), ("fold_image", "image")):
        md[col] = -1
        k = min(k_max, md[group].nunique())
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=0)
        for fold, (_, te) in enumerate(cv.split(md, md.label, groups=md[group])):
            md.loc[te, col] = fold

    md.to_csv(os.path.join(crops, "splits.csv"), index=False)

    res = os.path.join(REPO, "results", "splits", name)
    os.makedirs(res, exist_ok=True)
    counts = pd.DataFrame({"molecules": md.label.value_counts(),
                           "sessions": md.groupby("label").session.nunique()})
    f_session = pd.crosstab(md.label, md.fold_session)
    f_image = pd.crosstab(md.label, md.fold_image)
    counts.to_csv(os.path.join(res, "class_counts.csv"))
    f_session.to_csv(os.path.join(res, "folds_session.csv"))
    f_image.to_csv(os.path.join(res, "folds_image.csv"))

    print(f"\n===== {name}: {len(md)} molecules")
    print(f"saved {crops}/splits.csv and the tables in {res}/\n")
    print("molecules and sessions per class:\n", counts.to_string(), "\n")
    print("SESSION split - test molecules per fold and class:\n", f_session.to_string(), "\n")
    print("IMAGE split - test molecules per fold and class:\n", f_image.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default=None, help="one crops folder; default = every folder named crops* in the project")
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()

    if a.crops is None:
        folders = sorted(d for d in glob.glob(os.path.join(REPO, "crops*"))
                         if os.path.isfile(os.path.join(d, "metadata.csv")))
    else:
        folders = [a.crops if os.path.isdir(a.crops) else os.path.join(REPO, a.crops)]
    if not folders:
        print("no crops folder with a metadata.csv found - run extract_crops.py first")
    for d in folders:
        make_splits(d, a.k)