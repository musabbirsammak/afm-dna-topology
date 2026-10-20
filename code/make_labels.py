"""
Find the correct answer to "which strand is on top?" for some crossings, automatically,
without labelling anything by hand.

How it works:
  - Every crossing has a sign, + or -. It depends on the direction of the two strands and on
    which strand is on top. Swap which strand is on top and the sign flips.
  - TopoStats saves a sign for every crossing (called "writhe"), worked out from its own
    choice of top strand.
  - In a CLEAN knot, every crossing has the same sign, and for our samples this sign is known:
        twist knot 5_2 (pDIR) -> all crossings negative   (Holmes et al., 2025)
        torus knot 5_1 (pINV) -> all crossings positive   (Holmes et al., 2025)
        trefoil 3_1 (pINV)    -> all crossings positive   (Holmes et al., 2025: pINV gives torus knots
                                 3_1 and 5_1 of one chirality, and the pINV 5_1 is positive)
        
  - So, for each crossing in a clean knot:
        TopoStats' sign = expected sign -> TopoStats chose the right top strand
        TopoStats' sign = opposite sign -> TopoStats chose the wrong one, so the other strand is on top

A molecule is CLEAN when all of these hold:
  - it is a trefoil, torus knot or twist knot. unknots and catenanes are not used.
  - TopoStats found exactly the minimum number of crossings for that knot (3 for 3_1, 5 for 5_1 and 5_2)
  - every crossing has exactly 2 strands
  - it was not removed by the length filter in extract_crops.py

Only crossings with a sign of + or - are used.

Each labelled molecule also gets a pattern:
  "all as expected"       every TopoStats sign matches the expected sign
  "some flipped"          some match and some don't (normal: TopoStats made some mistakes)
  "all opposite (check!)" none match. Either a mirror-image knot or a systematic error.
                          train_crossings.py leaves these molecules out.

Input:  crossings/metadata.csv  (from extract_crossings.py)

Output (the same table, saved twice):
  crossings/labels.csv
  results/crossings/crossing_labels.csv
  one row per labelled crossing:
    id, mol            which sample and which molecule
    label, session     knot type and imaging day
    expected_sign      the sign this crossing should have
    writhe             the sign TopoStats gave it
    ts_a_on_top        TopoStats' answer: is strand A on top?
    ts_correct         was TopoStats right?
    true_a_on_top      the correct answer
    ts_confidence      TopoStats' reliability score for this crossing
    pattern            the molecule pattern above

Prints:
  - how many clean molecules and labelled crossings were found, per knot type
  - how often TopoStats was right, per knot type and overall
  - the same split by TopoStats' reliability score (below / above 0.263, the threshold
    suggested by Holmes et al., 2025)

Run from the code folder:
  python make_labels.py
  python make_labels.py --trefoil_sign -    # sensitivity check only: assume the opposite trefoil chirality
"""
import argparse, os

import pandas as pd
from extract_crossings import tag

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_CROSSINGS = {"3_1": 3, "5_1": 5, "5_2": 5}

#   crossing sign of each knot in a minimal diagram (Holmes et al., 2025, Fig. 4):
#   pINV gives torus knots 3_1 and 5_1 of one chirality; the pINV 5_1 has positive crossings,
#   so the pINV trefoil (also a torus knot of the same handedness) has positive crossings too.
#   pDIR gives twist knots; the pDIR 5_2 has negative crossings.
KNOWN_SIGN = {"3_1": "+", "5_1": "+", "5_2": "-"}

ap = argparse.ArgumentParser()
ap.add_argument("--nm_per_px", type=float, default=1.0, help="resolution used in extract_crossings.py")
ap.add_argument("--crossings", default=None, help="default: crossings_<resolution>")
ap.add_argument("--trefoil_sign", choices=["+", "-"], default="+",
                help="expected trefoil crossing sign (default +, from Holmes et al. 2025); '-' only as a sensitivity check")
a = ap.parse_args()

if a.crossings is None:
    a.crossings = os.path.join(REPO, f"crossings_{tag(a.nm_per_px)}")

md = pd.read_csv(os.path.join(a.crossings, "metadata.csv"))
md["mol"] = md["sample"] + "|" + md["image"] + "|" + md["grain"]

# one row per crossing (2-strand crossings have exactly one sample each)
mol = md.groupby("mol").agg(label=("label", "first"), n_nodes=("node", "nunique"),
                            all_simple=("n_strands", lambda s: (s == 2).all()),
                            kept=("molecule_kept", "first")).reset_index()
mol["clean"] = mol.apply(lambda r: r.label in MIN_CROSSINGS and r.n_nodes == MIN_CROSSINGS[r.label]
                         and r.all_simple and r.kept is not False, axis=1)
clean = md[md.mol.isin(mol[mol.clean].mol) & md.writhe.isin(["+", "-"])].copy()
print(f"clean knot molecules: {mol.clean.sum()}  ->  {len(clean)} crossings\n")
print("clean molecules per class:", mol[mol.clean].label.value_counts().to_dict())

signs = dict(KNOWN_SIGN)
signs["3_1"] = a.trefoil_sign
tref = clean[clean.label == "3_1"].writhe.value_counts()
print(f"\ntrefoil crossing signs as called by TopoStats: {tref.to_dict()} | expected sign used: '{signs['3_1']}'")

clean["expected_sign"] = clean.label.map(signs)
clean["ts_correct"] = clean.writhe == clean.expected_sign
clean["true_a_on_top"] = clean.apply(lambda r: r.ts_a_on_top if r.ts_correct else (not r.ts_a_on_top), axis=1)

# molecule sign pattern: all TopoStats signs as expected / some wrong / all opposite (suspicious)
pat = clean.groupby("mol").ts_correct.agg(["sum", "size"])
pat["pattern"] = pat.apply(lambda r: "all as expected" if r["sum"] == r["size"]
                           else ("all opposite (check!)" if r["sum"] == 0 else "some flipped"), axis=1)
clean = clean.merge(pat[["pattern"]], left_on="mol", right_index=True)

mols = clean.drop_duplicates("mol")
patterns = pd.crosstab(mols.label, mols.pattern, margins=True, margins_name="all")
print("\nmolecule sign patterns per class:")
print(patterns.to_string())

print("\nTopoStats (FWHM) accuracy on the auto-labelled crossings:")
for lab, g in clean.groupby("label"):
    print(f"  {lab}: {g.ts_correct.mean():.0%} of {len(g)} crossings")
print(f"  all: {clean.ts_correct.mean():.0%} of {len(clean)} crossings")
lo = clean.ts_confidence < 0.263
print(f"  confidence < 0.263 (paper's 'unreliable'): {clean[lo].ts_correct.mean():.0%} of {lo.sum()}"
      f" | >= 0.263: {clean[~lo].ts_correct.mean():.0%} of {(~lo).sum()}")
print("  (an 'all opposite' molecule is either a mirror-image knot or wrong labels -> look at it before trusting)")

out = clean[["id", "mol", "label", "session", "expected_sign", "writhe", "ts_a_on_top", "ts_correct",
             "true_a_on_top", "ts_confidence", "pattern"]]
out.to_csv(os.path.join(a.crossings, "labels.csv"), index=False)
print(f"\nsaved {a.crossings}/labels.csv  ({len(out)} labelled crossings)")

# the labels and the printed numbers as tables, in results/<crossings folder>/
res = os.path.join(REPO, "results", os.path.basename(os.path.normpath(a.crossings)))
os.makedirs(res, exist_ok=True)
out.to_csv(os.path.join(res, "crossing_labels.csv"), index=False)

per_class = []
for lab, g in clean.groupby("label"):
    per_class.append(dict(label=lab, expected_sign=signs[lab],
                          clean_molecules=int((mol.clean & (mol.label == lab)).sum()),
                          labelled_crossings=len(g), topostats_correct=int(g.ts_correct.sum()),
                          topostats_accuracy=round(g.ts_correct.mean(), 4)))
per_class.append(dict(label="all", expected_sign="", clean_molecules=int(mol.clean.sum()),
                      labelled_crossings=len(clean), topostats_correct=int(clean.ts_correct.sum()),
                      topostats_accuracy=round(clean.ts_correct.mean(), 4)))
pd.DataFrame(per_class).to_csv(os.path.join(res, "labels_per_class.csv"), index=False)

patterns.to_csv(os.path.join(res, "sign_patterns.csv"))

conf = []
for name, m in (("< 0.263 (unreliable)", lo), (">= 0.263 (reliable)", ~lo), ("all", lo | ~lo)):
    g = clean[m]
    conf.append(dict(topostats_confidence=name, crossings=int(m.sum()), topostats_correct=int(g.ts_correct.sum()),
                     topostats_accuracy=round(g.ts_correct.mean(), 4) if len(g) else None))
pd.DataFrame(conf).to_csv(os.path.join(res, "topostats_by_confidence.csv"), index=False)

pd.DataFrame([dict(topostats_plus_calls=int(tref.get("+", 0)), topostats_minus_calls=int(tref.get("-", 0)),
                   sign_used=signs["3_1"],
                   sign_from="Holmes et al. 2025 (pINV torus knot)" if a.trefoil_sign == "+" else "set by hand (sensitivity check)")]
             ).to_csv(os.path.join(res, "trefoil_sign.csv"), index=False)

print(f"saved to {res}/: crossing_labels.csv, labels_per_class.csv, sign_patterns.csv, "
      f"topostats_by_confidence.csv, trefoil_sign.csv")