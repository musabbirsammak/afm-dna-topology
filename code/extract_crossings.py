"""
Cut a small square image around every DNA crossing, for the crossing classifier.

A crossing is a point where one DNA strand passes over another. TopoStats has already found
every crossing (it calls them nodes) and saved the following in each .topostats file:
  - where the crossing is                  nodestats/above/stats/grain_i/node_j/node_coords
  - the path of each strand through it     .../node_j/branch_stats/k/ordered_coords
  - TopoStats' own decision on which strand is on top, from the width of each strand's
    height profile (FWHM, full width at half maximum; the wider strand is called the top one)
                                           .../node_j/branch_stats/k/fwhm/fwhm
Nothing has to be detected again. This script reads these positions and cuts around them.

Which crossings are used:
  - crossings that TopoStats marked as an error are skipped. although in the dataset there is no such error
  - crossings with fewer than 2 strands are skipped (nothing to compare)
  - at a crossing with more than 2 strands, every PAIR of strands becomes its own sample
    (2 strands -> 1 pair, 3 strands -> 3 pairs, 4 strands -> 6 pairs). we simply use itertools
    to select all possible pairs.

Each sample is a 3-channel array of 3 x 64 x 64 pixels at 1 nm per pixel (a 64 x 64 nm square
centred on the crossing):
    channel 0 : height image, scaled like the molecule crops (-1 to 5 nm -> 0 to 1).
                Other molecules nearby are blanked out.
    channel 1 : strand A drawn as a line about 3 nm wide (1 on the strand, 0 elsewhere)
    channel 2 : strand B drawn the same way

The question a model answers later: is strand A on top of strand B?

Note: the original images have pixels of 0.29 to 0.98 nm, so cutting at 1 nm per pixel loses
some detail. To change this, edit NM_PER_PX and SIZE below (e.g. 0.5 and 128 keeps a 64 nm square).

Output (default: crossings/ in the project folder):
    crossings/npy/<id>.npy     the 3 x 64 x 64 arrays (for training)
    crossings/png/<id>.png     colour preview: grey = height, red = strand A, blue = strand B
    results/figures/qc/crossings_sheet.png
                               40 random previews. OPEN IT: each square should show
                               a crossing in the middle
    crossings/metadata.csv     one row per sample: sample, knot type, imaging session, buffer,
                               strand widths, TopoStats' decision (ts_a_on_top), its reliability
                               score (ts_confidence), the crossing sign (writhe), and whether the
                               molecule passed the length filter in extract_crops.py (molecule_kept)

Before running: run extract_crops.py first, so that crops/metadata.csv exists
(it is only used to fill in molecule_kept; without it that column stays empty).

Run from the code folder:
    python code/extract_crossings.py --nm_per_px 0.5 --out crossings_05
    python code/make_labels.py --crossings crossings_05
    python code/train_crossings.py --crossings crossings_05 --out results/crossings_05 --teacher --finetune
"""
import argparse, glob, itertools, os

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from extract_crops import LABELS, H_MAX, H_MIN, condition_of, grain_ids, px_nm, session_of

LINE_NM = 3.0   # width of the drawn strand lines in nm
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tag(nm_per_px):
    """Name for a resolution, as in the Stage 0 runs: 1 -> '1nm', 0.5 -> '05nm', 0.3 -> '03nm'."""
    return f"{nm_per_px:g}".replace(".", "") + "nm"


def as_str(v):
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray) and v.dtype.kind in "SO":
        return " ".join(x.decode() if isinstance(x, bytes) else str(x) for x in v.ravel())
    return str(v)


def read_nodes(f, g):
    """List of dicts, one per crossing of grain g."""
    st = f.get(f"nodestats/above/stats/{g}")
    out = []
    if st is None:
        return out
    for nk in sorted(st.keys(), key=lambda s: int(s.split("_")[1])):
        n = st[nk]
        if "error" in n and bool(n["error"][()]):
            continue
        strands = []
        bs = n.get("branch_stats")
        if bs is not None:
            for k in sorted(bs.keys(), key=int):
                b = bs[k]
                fw = float(b["fwhm/fwhm"][()]) if "fwhm/fwhm" in b else np.nan
                strands.append(dict(k=int(k), path=np.asarray(b["ordered_coords"][()], float), fwhm=fw))
        out.append(dict(name=nk, coords=np.asarray(n["node_coords"][()], float), strands=strands,
                        confidence=float(n["confidence"][()]) if "confidence" in n else np.nan,
                        writhe=as_str(n["writhe"][()]) if "writhe" in n else ""))
    return out


def skeleton_of(f, g):
    p = f"nodestats/above/images/{g}/grain/grain_skeleton"
    if p in f:
        return f[p][()] > 0
    return f[f"disordered_traces/above/{g}/pruned_skeleton"][()] > 0


def pick_orientation(skel, nodes):
    """node_coords/ordered_coords may be stored (row, col) or (col, row): keep the one that lies on the skeleton."""
    pts = np.vstack([n["coords"] for n in nodes]).astype(int) if nodes else np.zeros((0, 2), int)
    def hit(p):
        ok = (p[:, 0] >= 0) & (p[:, 0] < skel.shape[0]) & (p[:, 1] >= 0) & (p[:, 1] < skel.shape[1])
        return skel[p[ok, 0], p[ok, 1]].mean() if ok.any() else 0.0
    a, b = hit(pts), hit(pts[:, ::-1])
    return (False, a) if a >= b else (True, b)


def cut(img, centre, px):
    """64 x 64 patch at 1 nm/px around centre (row, col) in crop pixels; bilinear, zeros outside."""
    step = NM_PER_PX / px                                 # crop pixels per patch pixel
    off = (np.arange(SIZE) - (SIZE - 1) / 2) * step
    rr, cc = np.meshgrid(centre[0] + off, centre[1] + off, indexing="ij")
    return ndimage.map_coordinates(img, [rr, cc], order=1, cval=0.0)


def draw(path, centre, px):
    """Rasterise a strand path (crop pixels) into the patch frame as a ~3 nm wide line."""
    m = np.zeros((SIZE, SIZE), bool)
    if len(path) == 0:
        return m
    q = (path - centre) * px / NM_PER_PX + (SIZE - 1) / 2  # -> patch pixels
    dense = [q[:1]]
    for a, b in zip(q[:-1], q[1:]):
        n = int(np.ceil(np.abs(b - a).max())) + 1
        dense.append(np.linspace(a, b, n))
    d = np.round(np.vstack(dense)).astype(int)
    ok = (d[:, 0] >= 0) & (d[:, 0] < SIZE) & (d[:, 1] >= 0) & (d[:, 1] < SIZE)
    m[d[ok, 0], d[ok, 1]] = True
    r = max(0, int(round((LINE_NM / NM_PER_PX - 1) / 2)))
    return ndimage.binary_dilation(m, iterations=r) if r else m


def dist_to_centre_nm(path, centre, px):
    """Closest distance (nm) between a strand path and the crossing centre."""
    if len(path) == 0:
        return np.inf
    return float(np.min(np.linalg.norm(path - centre, axis=1)) * px)


def preview(x):
    h = (x[0] * 255).astype(np.uint8)
    rgb = np.stack([h, h, h], -1).astype(np.float32)
    rgb[x[1] > 0] = 0.5 * rgb[x[1] > 0] + 0.5 * np.array([230, 40, 40])
    rgb[x[2] > 0] = 0.5 * rgb[x[2] > 0] + 0.5 * np.array([40, 90, 230])
    return Image.fromarray(rgb.clip(0, 255).astype(np.uint8)).resize((128, 128), Image.NEAREST)


def files_of(root):
    return sorted(glob.glob(os.path.join(root, "*", "processed", "*.topostats")))


def check(root):
    files = files_of(root)
    print(f"found {len(files)} .topostats files")
    for path in files[:200]:
        with h5py.File(path, "r") as f:
            for g in grain_ids(f):
                nodes = read_nodes(f, g)
                if not nodes:
                    continue
                skel = skeleton_of(f, g)
                swap, frac = pick_orientation(skel, nodes)
                n0 = nodes[0]
                print(f"\nfile {path}\n{g}: {len(nodes)} crossings | coords swapped: {swap} | "
                      f"crossing pixels on skeleton: {frac:.0%}")
                print(f"first crossing: {len(n0['coords'])} pixels, {len(n0['strands'])} strands, "
                      f"FWHM {[round(s['fwhm'], 2) for s in n0['strands']]}, confidence {n0['confidence']:.2f}, "
                      f"writhe '{n0['writhe']}'")
                return
    print("no crossings found in the first 200 files")


def main(root, out, crops_meta):
    os.makedirs(os.path.join(out, "npy"), exist_ok=True)
    os.makedirs(os.path.join(out, "png"), exist_ok=True)
    keep = None
    if crops_meta and os.path.exists(crops_meta):
        cm = pd.read_csv(crops_meta)
        keep = {(r.sample, r.image, r.grain): bool(r.keep) for r in cm.itertuples()}
    rows, skipped, on_skel = [], {"<2 strands": 0}, []
    files = files_of(root)
    for i, path in enumerate(files):
        sample = os.path.basename(os.path.dirname(os.path.dirname(path)))
        if sample not in LABELS:
            continue
        image = os.path.basename(path)[: -len(".topostats")]
        with h5py.File(path, "r") as f:
            px = px_nm(f)
            for g in grain_ids(f):
                nodes = read_nodes(f, g)
                if not nodes:
                    continue
                d = f[f"disordered_traces/above/{g}"]
                img = d["original_image"][()].astype(np.float32)
                mask = ndimage.binary_dilation(d["original_grain"][()] > 0, iterations=max(1, int(round(2 / px))))
                img = np.where(mask, img, 0.0)
                skel = skeleton_of(f, g)
                swap, frac = pick_orientation(skel, nodes)
                on_skel.append(frac)
                for n in nodes:
                    if len(n["strands"]) < 2:
                        skipped["<2 strands"] += 1
                        continue
                    coords = n["coords"][:, ::-1] if swap else n["coords"]
                    centre = coords.mean(0)
                    height = cut(img, centre, px)
                    height = np.clip((height - H_MIN) / (H_MAX - H_MIN), 0, 1)
                    height[height <= (0 - H_MIN) / (H_MAX - H_MIN) + 1e-6] = 0
                    for a, b in itertools.combinations(n["strands"], 2):
                        pa = a["path"][:, ::-1] if swap else a["path"]
                        pb = b["path"][:, ::-1] if swap else b["path"]

                        # a strand that never reaches the crossing is discarded
                        da, db = dist_to_centre_nm(pa, centre, px), dist_to_centre_nm(pb, centre, px)
                        if max(da, db) > 5.0:          
                            skipped["strand far from crossing"] = skipped.get("strand far from crossing", 0) + 1
                            continue
                        
                        x = np.stack([height, draw(pa, centre, px), draw(pb, centre, px)]).astype(np.float32)
                        sid = f"{sample}__{image}__{g}__{n['name']}__s{a['k']}{b['k']}"
                        np.save(os.path.join(out, "npy", sid + ".npy"), x)
                        preview(x).save(os.path.join(out, "png", sid + ".png"))
                        rows.append(dict(
                            id=sid, sample=sample, label=LABELS[sample], image=image, grain=g, node=n["name"],
                            session=session_of(image), condition=condition_of(image), px_nm=round(px, 4),
                            n_strands=len(n["strands"]), strand_a=a["k"], strand_b=b["k"],
                            node_pixels=len(n["coords"]), centre_row=round(centre[0], 1), centre_col=round(centre[1], 1),
                            fwhm_a=a["fwhm"], fwhm_b=b["fwhm"],
                            ts_a_on_top=bool(a["fwhm"] > b["fwhm"]) if np.isfinite(a["fwhm"]) and np.isfinite(b["fwhm"]) else None,
                            ts_confidence=n["confidence"], writhe=n["writhe"],
                            coords_swapped=swap,
                            molecule_kept=keep.get((sample, image, g)) if keep is not None else None))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(files)} files, {len(rows)} samples so far")
    
    md = pd.DataFrame(rows)
    md.to_csv(os.path.join(out, "metadata.csv"), index=False)
    n_cross = md[["image", "grain", "node"]].drop_duplicates().shape[0]
    print(f"\nsaved {len(md)} samples ({n_cross} crossings) to {out}/")
    print(f"skipped: {skipped}")
    print(f"crossing pixels lying on the skeleton (should be ~100%): median {np.median(on_skel):.0%}")
    per_class = pd.crosstab(md.label, md.n_strands, margins=True, margins_name="all")
    print("\nsamples per class and strands per crossing:")
    print(per_class.to_string())

    # the same numbers as tables, in results/<output folder>/
    res = os.path.join(REPO, "results", os.path.basename(os.path.normpath(out)))
    os.makedirs(res, exist_ok=True)
    pd.DataFrame([dict(
        nm_per_px=NM_PER_PX, window_nm=WINDOW_NM, patch_px=SIZE,
        topostats_files=len(files),
        molecules_with_crossings=md[["sample", "image", "grain"]].drop_duplicates().shape[0],
        crossings=n_cross, samples=len(md),
        skipped_less_than_2_strands=skipped.get("<2 strands", 0),
        skipped_strand_far_from_crossing=skipped.get("strand far from crossing", 0),
        median_crossing_pixels_on_skeleton=round(float(np.median(on_skel)), 3),
    )]).to_csv(os.path.join(res, "extraction_summary.csv"), index=False)
    per_class.to_csv(os.path.join(res, "samples_per_class.csv"))
    print(f"\nsaved {res}/extraction_summary.csv and {res}/samples_per_class.csv")

    # contact sheet for the centring check
    pick = md.sample(min(40, len(md)), random_state=0)
    sheet = Image.new("RGB", (8 * 132, 5 * 132), "white")
    for j, r in enumerate(pick.itertuples()):
        sheet.paste(Image.open(os.path.join(out, "png", r.id + ".png")), ((j % 8) * 132 + 2, (j // 8) * 132 + 2))
    qc = os.path.join(REPO, "results", "figures", "qc")
    os.makedirs(qc, exist_ok=True)
    sheet_path = os.path.join(qc, f"{os.path.basename(os.path.normpath(out))}_sheet.png")
    sheet.save(sheet_path)
    print(f"\nsaved {sheet_path} -> open it: every square should show a crossing in the MIDDLE,"
          f"\nwith the red and blue lines running along the two DNA strands.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(REPO, "data", "knots_and_cats"), help="folder that contains trefoil/, torus/, twist/, ...")
    ap.add_argument("--out", default=None, help="default: crossings_<resolution>, e.g. crossings_05nm")
    ap.add_argument("--crops_meta", default=os.path.join(REPO, "crops", "metadata.csv"))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--nm_per_px", type=float, default=1.0, help="patch resolution in nm per pixel (default 1.0)")
    ap.add_argument("--window_nm", type=float, default=64.0, help="width of the square around each crossing in nm (default 64)")
    a = ap.parse_args()
    if a.out is None:
        a.out = os.path.join(REPO, f"crossings_{tag(a.nm_per_px)}")
    NM_PER_PX = a.nm_per_px
    WINDOW_NM = a.window_nm
    SIZE = int(round(a.window_nm / a.nm_per_px))
    print(f"patches: {a.window_nm:g} x {a.window_nm:g} nm at {NM_PER_PX:g} nm per pixel = {SIZE} x {SIZE} pixels")
    check(a.root) if a.check else main(a.root, a.out, a.crops_meta)