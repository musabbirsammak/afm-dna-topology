"""
Cut one image per DNA molecule out of the .topostats files. 
Each molecule is a single classification example with one knot type, 
so one crop covers the whole molecule; more crops of the same molecule 
would only repeat it.

The crop holds the whole molecule, crossings included. Each .topostats 
file stores the image’s pixel size, `pixel_to_nm_scaling` 
(0.29 to 0.98 nm per pixel). `ndimage.zoom` rescales each crop by its pixel 
size to 1 nm per pixel (bilinear). If the result is wider than 256 pixels, 
a second zoom shrinks it to fit. But resampling to 1 nm per pixel, and 
shrinking molecules larger than 256 nm to fit the canvas, can blur closely 
spaced crossings into one.

What it does, for every molecule (TopoStats "grain") in every .topostats file:
  1. takes the height image of that molecule and its mask
  2. removes everything that is not the molecule (other DNA, dirt)
  3. resamples it to the SAME physical scale for every image (default 1 nm per pixel)
  4. pastes it in the centre of a fixed square canvas (default 256 x 256 px = 256 x 256 nm)
  5. saves it as .npy (for training) and .png (for you to look at)
and writes one row per molecule to metadata.csv.

Run:
    python extract_crops.py --root "PATH/TO/knots_and_cats" --out crops
Check one file first (prints what is inside, saves nothing):
    python extract_crops.py --root "PATH/TO/knots_and_cats" --check
"""
import argparse, glob, os, re

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Knot Types of Dataset
LABELS = {
    "unknot_plasmid_nic": "0_1", "unknot_plasmid_sup": "0_1",
    "trefoil": "3_1", "torus": "5_1", "twist": "5_2",
    "cat_nicked": "4^2_1", "cat_supercoiled": "4^2_1",
}

# total DNA length of each molecule type in nm (used only to throw away broken pieces/clumps,
# the same rule is applied to every class)
ALLOWED_LENGTHS = [758.0, 554.0]
LENGTH_TOL = 0.10           # keep molecules within +-10 % of one of the allowed lengths
H_MIN, H_MAX = -1.0, 5.0    # height window in nm (DNA is ~2 nm high, crossings ~4 nm)


def px_nm(f):
    v = float(f["pixel_to_nm_scaling"][()])
    return v * 1e9 if v < 1e-6 else v


def session_of(name):
    m = re.search(r"(202[0-9][01]\d[0-3]\d)", name)
    return m.group(1) if m else "unknown"


def condition_of(name):
    n = name.lower()
    if "mgni" in n:
        return "MgNi"
    if "nickel" in n or "nicl" in n:
        return "Ni"
    return "other"


def grain_ids(f):
    g = f.get("disordered_traces/above")
    return sorted(g.keys(), key=lambda s: int(s.split("_")[1])) if g else []


def contour_nm(f, g):
    sp = f.get(f"splining/above/{g}")
    if sp is None:
        return np.nan, 0
    tot = 0.0
    for m in sp.keys():
        L = float(sp[m]["tracing_stats/contour_length"][()])
        tot += L * 1e9 if L < 1e-3 else L
    return tot, len(sp.keys())


def ts_topology(f, g):
    ot = f.get(f"ordered_traces/above/{g}")
    if ot is None:
        return ""
    for m in ot.keys():
        t = ot[m]["mol_stats/topology"][()]
        return t.decode() if isinstance(t, bytes) else str(t)
    return ""


def make_crop(img, mask, px, nm_per_px, size, mode="height"):
    """Mask, resample to nm_per_px, centre on a size x size canvas. Returns (canvas, scaled_flag)."""
    mask = ndimage.binary_dilation(mask > 0, iterations=max(1, int(round(2 / px))))   # ~2 nm margin
    img = np.where(mask, img, 0.0)
    z = px / nm_per_px
    img = ndimage.zoom(img, z, order=1)
    mask = ndimage.zoom(mask.astype(np.uint8), z, order=0) > 0
    img = np.where(mask, img, 0.0)
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None, False
    img = img[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    scaled = False
    if max(img.shape) > size:                     # molecule bigger than the canvas: shrink to fit
        s = size / max(img.shape)
        img = ndimage.zoom(img, s, order=1)
        scaled = True
    on = img != 0
    if mode == "binary":                 # shape only: 1 on DNA, 0 elsewhere (no height, no imaging style)
        img = on.astype(np.float32)
    elif mode == "norm":                 # heights divided by this molecule's own backbone height
        base = np.median(img[on]) if on.any() else 1.0
        img = np.clip(img / max(base, 1e-3), 0, 2.5) / 2.5   # backbone -> 0.4, crossings (~2x) -> 0.8
        img[~on] = 0
    else:                                # absolute height in nm (default)
        img = np.clip((img - H_MIN) / (H_MAX - H_MIN), 0, 1)
        img[img <= (0 - H_MIN) / (H_MAX - H_MIN) + 1e-6] = 0
    canvas = np.zeros((size, size), np.float32)
    r0 = (size - img.shape[0]) // 2
    c0 = (size - img.shape[1]) // 2
    canvas[r0:r0 + img.shape[0], c0:c0 + img.shape[1]] = img
    return canvas, scaled


def check(root):
    files = sorted(glob.glob(os.path.join(root, "*", "processed", "*.topostats")))
    print(f"found {len(files)} .topostats files")
    for s in LABELS:
        n = len(glob.glob(os.path.join(root, s, "processed", "*.topostats")))
        print(f"  {s:20s} {n:4d} files")
    if not files:
        print("Nothing found: --root must be the folder that CONTAINS trefoil/, torus/, twist/ ...")
        return
    with h5py.File(files[0], "r") as f:
        g = grain_ids(f)[0]
        d = f[f"disordered_traces/above/{g}"]
        im = d["original_image"][()]
        print(f"\nfirst file: {files[0]}\npixel size: {px_nm(f):.3f} nm | grains: {len(grain_ids(f))}")
        print(f"{g}: crop {im.shape}, height min/max {im.min():.2f}/{im.max():.2f} nm, bbox {d['bbox'][()]}")
        print(f"contour length (TopoStats): {contour_nm(f, g)[0]:.0f} nm, topology: {ts_topology(f, g)}")


def main(root, out, nm_per_px, size, mode="height"):
    files = sorted(glob.glob(os.path.join(root, "*", "processed", "*.topostats")))
    os.makedirs(os.path.join(out, "npy"), exist_ok=True)
    rows = []
    for i, path in enumerate(files):
        sample = os.path.basename(os.path.dirname(os.path.dirname(path)))
        if sample not in LABELS:
            continue
        image = os.path.basename(path)[: -len(".topostats")]
        with h5py.File(path, "r") as f:
            px = px_nm(f)
            full_shape = f["image"].shape
            for g in grain_ids(f):
                d = f[f"disordered_traces/above/{g}"]
                img = d["original_image"][()].astype(np.float32)
                mask = d["original_grain"][()]
                bbox = d["bbox"][()]
                L, n_loops = contour_nm(f, g)
                if np.isnan(L):                               # fallback: skeleton length
                    L = float((d["pruned_skeleton"][()] > 0).sum()) * px * 1.12
                on_edge = bool(min(bbox[0], bbox[1]) <= 2 or bbox[2] >= full_shape[0] - 2 or bbox[3] >= full_shape[1] - 2)
                ok_len = any(abs(L - a) / a <= LENGTH_TOL for a in ALLOWED_LENGTHS)
                crop, scaled = make_crop(img, mask, px, nm_per_px, size, mode)
                if crop is None:
                    continue
                mid = f"{sample}__{image}__{g}"
                np.save(os.path.join(out, "npy", mid + ".npy"), crop)
                pdir = os.path.join(out, "png", LABELS[sample])
                os.makedirs(pdir, exist_ok=True)
                Image.fromarray((crop * 255).astype(np.uint8)).save(os.path.join(pdir, mid + ".png"))
                ys, xs = np.where(crop > 0)
                rows.append(dict(id=mid, sample=sample, label=LABELS[sample], image=image, grain=g,
                                 session=session_of(image), condition=condition_of(image),
                                 px_nm=round(px, 4), contour_nm=round(L, 1), ts_loops=n_loops,
                                 ts_topology=ts_topology(f, g),
                                 extent_nm=int(max(np.ptp(ys) + 1, np.ptp(xs) + 1) * nm_per_px) if len(ys) else 0,
                                 scaled_to_fit=scaled, on_edge=on_edge, length_ok=ok_len,
                                 keep=bool(ok_len and not on_edge)))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(files)} files")
    md = pd.DataFrame(rows)
    md.to_csv(os.path.join(out, "metadata.csv"), index=False)
    print(f"\nsaved {len(md)} molecules to {out}/  ({md.keep.sum()} pass the filters)\n")
    print(md.groupby(["label", "sample"]).agg(molecules=("id", "size"), kept=("keep", "sum"),
                                              sessions=("session", "nunique"),
                                              shrunk=("scaled_to_fit", "sum")).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(REPO, "data", "knots_and_cats"), help="folder that contains trefoil/, torus/, twist/, ...")
    ap.add_argument("--out", default=None)
    ap.add_argument("--nm_per_px", type=float, default=1.0)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--mode", choices=["height", "norm", "binary"], default="height",
                    help="height = absolute nm; norm = divided by the molecule's own DNA height; binary = shape only")
    a = ap.parse_args()
    if a.out is None:
        a.out = os.path.join(REPO, {"height": "crops", "norm": "crops_norm", "binary": "crops_binary"}[a.mode])
    check(a.root) if a.check else main(a.root, a.out, a.nm_per_px, a.size, a.mode)
