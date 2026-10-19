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

Run from the project folder (every option is optional; defaults in brackets):
python code/extract_crops.py
    --root PATH          folder that contains trefoil/, torus/, twist/, ...  [data/knots_and_cats]
    --mode MODE          height = absolute height in nm
                         norm   = height divided by the molecule's own DNA height
                         binary = shape only                                  [height]
    --out PATH           output folder  [crops, crops_norm or crops_binary, following --mode]
    --nm_per_px X        pixel size of the crops in nm                        [1.0]
    --size N             canvas size in pixels                                [256]
    --check              print what is inside the first file and save nothing
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
# the same rule is applied to every class). In all_mol_statistics.csv, the middle
# value of the traced lengths is 756–765 nm for knots and unknots and 553–555 nm 
# for catenanes. The length calculated from the DNA sequence is a little longer, 
# about 768 nm for the 2,260 base-pair plasmid (2,260 × 0.34 nm per base pair). 
# TopoStats’ traces come out slightly short, so the lengths measured in the 
# images are the right reference.
ALLOWED_LENGTHS = [758.0, 554.0]
# keep molecules within +-10 % of one of the allowed lengths (from my judgement)
LENGTH_TOL = 0.10
 # height window in nm. DNA is ~2 nm high, crossings ~4 nm
H_MIN, H_MAX = -1.0, 5.0


# Reads the pixel size from the file and returns it in nanometres, 
# multiplying by 10^9 if the file stored it in metres.
def px_nm(f):
    v = float(f["pixel_to_nm_scaling"][()])
    return v * 1e9 if v < 1e-6 else v


# Finds the imaging date in the file name (8 digits like 20231011, 
# year 2020–2029) and returns it as the session ID, or "unknown" if 
# the name has no date.
def session_of(name):
    m = re.search(r"(202[0-9][01]\d[0-3]\d)", name)
    return m.group(1) if m else "unknown"


# Reads the imaging buffer from the file name: “MgNi” if the name contains
# “mgni”, “Ni” if it contains “nickel” or “nicl”, otherwise “other”.
def condition_of(name):
    n = name.lower()
    if "mgni" in n:
        return "MgNi"
    if "nickel" in n or "nicl" in n:
        return "Ni"
    return "other"

## FUNCTION INPUTS
## `f` is one opened .topostats file, which holds one image. 
## `g` is the name of one molecule inside it, such as "grain_3".

# lists the molecules (“grains”) in a .topostats file and sorts them by number 
# (grain_0, grain_1, grain_2 … grain_10). If the file has no traced molecules, 
# it returns an empty list. A grain is TopoStats’ name for one separate object 
# it finds in an image, usually one DNA molecule. We use grains so that each 
# molecule can be cut out and labelled on its own.
def grain_ids(f):
    # TopoStats looks for two kinds of objects: those higher than the surface 
    # (“above”) and those lower than it (“below”, such as holes or pits). DNA 
    # lies on top of the surface, so it always shows up under “above”.
    g = f.get("disordered_traces/above")
    return sorted(g.keys(), key=lambda s: int(s.split("_")[1])) if g else []


# Adds up the traced DNA length of one molecule in nanometres.
def contour_nm(f, g):
    sp = f.get(f"splining/above/{g}")
    if sp is None:
        return np.nan, 0
    tot = 0.0
    for m in sp.keys():
        L = float(sp[m]["tracing_stats/contour_length"][()])
        tot += L * 1e9 if L < 1e-3 else L
    return tot, len(sp.keys())


# reads the knot type that TopoStats assigned to molecule g in file f 
# (for example “3_1”) and returns it as text. If TopoStats gave no answer, 
# it returns an empty string.
def ts_topology(f, g):
    ot = f.get(f"ordered_traces/above/{g}")
    if ot is None:
        return ""
    for m in ot.keys():
        t = ot[m]["mol_stats/topology"][()]
        return t.decode() if isinstance(t, bytes) else str(t)
    return ""


# Turns one molecule into a 256 × 256 image
def make_crop(img, mask, px, nm_per_px, size, mode="height"):
    """Mask, resample to nm_per_px, centre on a size x size canvas. Returns (canvas, scaled_flag)."""
    # mask marks which pixels belong to the molecule. 
    # Dilation grows it outward by about 2 nm, so the edges of the DNA 
    # aren’t cut off. 2 / px turns 2 nm into a number of pixels.
    mask = ndimage.binary_dilation(mask > 0, iterations=max(1, int(round(2 / px))))   # ~2 nm margin
    # Pixels outside the molecule are set to 0, so the model sees only this molecule 
    # and not its neighbours.
    img = np.where(mask, img, 0.0)

    # The zoom halves the number of pixels, not the size of a pixel.The 
    # molecule’s real width stays the same, so when it is covered by half as 
    # many pixels, each pixel covers twice as much, eg., 0.5 nm × 2 = 1 nm.
    # order=1 smoothes the pixels.
    z = px / nm_per_px
    img = ndimage.zoom(img, z, order=1)
    # This rescales the mask by the same factor as the image, 
    # so the two still line up. the mask is True/False, and zoom needs numbers.
    # So, mask.astype(np.uint8). order=0 means each new pixel copies the value 
    # of the nearest old pixel, without averaging. The result stays only 
    # 0s and 1s, with no in-between values at the edges. > 0 turns the 1s and 0s 
    # back into True/False.
    mask = ndimage.zoom(mask.astype(np.uint8), z, order=0) > 0
    # keeps the image value wherever the mask is True (on the molecule) 
    # and sets every other pixel to 0
    img = np.where(mask, img, 0.0)

    # finds the smallest box that contains the molecule and crops to it. 
    # If the mask is empty, the molecule is skipped.
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None, False
    img = img[rows.min():rows.max() + 1, cols.min():cols.max() + 1]

    # If the molecule is wider than 256 nm, the whole crop is shrunk to fit. 
    # scaled = True records this, because the crop is then no longer 1 nm 
    # per pixel.
    scaled = False
    if max(img.shape) > size:
        s = size / max(img.shape)
        img = ndimage.zoom(img, s, order=1)
        scaled = True
    
    on = img != 0
    # shape only: 1 on DNA, 0 elsewhere (no height, no imaging style)
    if mode == "binary":
        img = on.astype(np.float32)
    # each pixel’s height is divided by the molecule’s typical (median) height. 
    # Backbone becomes about 0.4 and crossings about 0.8. 
    # This removes differences in overall brightness between sessions.
    elif mode == "norm":
        base = np.median(img[on]) if on.any() else 1.0
        img = np.clip(img / max(base, 1e-3), 0, 2.5) / 2.5
        img[~on] = 0
    # absolute height in nm (default). clipped to the range −1 to 5 nm 
    # and scaled to 0–1. Pixels at or below 0 nm are set to 0, 
    # so they look like background.
    else:
        img = np.clip((img - H_MIN) / (H_MAX - H_MIN), 0, 1)
        img[img <= (0 - H_MIN) / (H_MAX - H_MIN) + 1e-6] = 0
    
    # The crop is pasted into the middle of an empty 256 × 256 image, 
    # so all inputs have the same size.
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
    summary = md.groupby(["label", "sample"]).agg(molecules=("id", "size"), kept=("keep", "sum"),
                                                  sessions=("session", "nunique"),
                                                  shrunk=("scaled_to_fit", "sum"))
    print(summary.to_string())
    summary.to_csv(os.path.join(out, "summary.csv"))
    print(f"\nsaved {out}/summary.csv")


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
