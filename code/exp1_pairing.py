"""
Experiment 1 - physics-only strand pairing for AFM DNA molecules.

For every molecule (grain) in every .topostats file:
  1. build the skeleton graph: crossing regions (nodes) + DNA segments between them
  2. enumerate every way of pairing the branches at each node
  3. score each pairing with   E = E_len + lambda_bend * E_bend
        E_len  : traced loops must match the expected loop lengths of the sample
        E_bend : DNA continues smoothly through a crossing (worm-like chain)
  4. keep the lowest-energy pairing and compare its loops with TopoStats' loops

No learning, no crossing order yet (that is Experiment 2).

The solver is not told which sample a molecule comes from (it accepts any species of the dataset),
unless --given is used. Results are evaluated on the molecules of Stage 0, those that pass the length
and edge filter of extract_crops.py (keep column of crops_1nm/metadata.csv). A molecule the solver
cannot handle counts as wrong.

Output (results/pairing/):
  exp1_agnostic.csv            one row per molecule (exp1_given.csv with --given)
  exp1_agnostic_summary.csv    loops correct per knot class, knots and catenanes, and all molecules:
                               Experiment 1 and TopoStats on the same molecules, 95% bootstrap CIs,
                               exact McNemar test, Holm-adjusted p values
  exp1_agnostic_status.csv     solver status per knot class

Run from the project folder:
  python code/exp1_pairing.py                                   # all molecules, sample not given (fair)
  python code/exp1_pairing.py --probe path/to/one.topostats     # check one file, prints only

Requires: numpy, scipy, h5py, pandas
"""
import argparse, glob, itertools, math, os, sys
from collections import deque, defaultdict

import h5py
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import binomtest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Expected loop lengths (nm) per sample folder, measured from all_mol_statistics.csv
# (design values: 2260 bp plasmid = 768 nm; p4CAT catenane = 426 + 135 nm).
TARGETS = {
    "trefoil": [758.0], "torus": [758.0], "twist": [758.0],
    "unknot_plasmid_sup": [758.0], "unknot_plasmid_nic": [758.0],
    "cat_supercoiled": [418.0, 136.0], "cat_nicked": [418.0, 136.0],
}
EIGHT = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
BIG = 1e3            # penalty per missing/extra loop
SIGMA = 0.05         # allowed relative length error (1 sigma)
ARM_PX = 15          # pixels used to estimate a branch direction
MAX_CONFIGS = 200_000


# ----------------------------------------------------------------------------- loading
def px_nm(f):
    v = float(f["pixel_to_nm_scaling"][()])
    return v * 1e9 if v < 1e-6 else v           # stored in m in some versions


def grains(f):
    g = f.get("disordered_traces/above")
    return sorted(g.keys(), key=lambda s: int(s.split("_")[1])) if g else []


def load_grain(f, g):
    """Return skeleton (bool crop) and list of node pixel arrays (row, col)."""
    base = f"nodestats/above/images/{g}/grain/grain_skeleton"
    if base in f:
        skel = f[base][()] > 0
    else:
        skel = f[f"disordered_traces/above/{g}/pruned_skeleton"][()] > 0
    nodes = []
    st = f.get(f"nodestats/above/stats/{g}")
    if st is not None:
        for k in sorted(st.keys(), key=lambda s: int(s.split("_")[1])):
            nodes.append(np.asarray(st[k]["node_coords"][()], dtype=int))
    return skel, nodes


def topostats_loops(f, g, px):
    """Loops TopoStats traced for this grain: list of contour lengths (nm)."""
    out = []
    sp = f.get(f"splining/above/{g}")
    if sp is None:
        return out
    for m in sp.keys():
        L = float(sp[m]["tracing_stats/contour_length"][()])
        out.append(L * 1e9 if L < 1e-3 else L)
    return out


def topostats_topology(f, g):
    ot = f.get(f"ordered_traces/above/{g}")
    if ot is None:
        return ""
    for m in ot.keys():
        t = ot[m]["mol_stats/topology"][()]
        return t.decode() if isinstance(t, bytes) else str(t)
    return ""


# ----------------------------------------------------------------------------- graph
def orient_nodes(skel, nodes):
    """node_coords may be (row,col) or (col,row); pick the one lying on the skeleton."""
    if not nodes:
        return nodes, 1.0
    def hit(arrs):
        tot = ok = 0
        for a in arrs:
            for r, c in a:
                tot += 1
                if 0 <= r < skel.shape[0] and 0 <= c < skel.shape[1] and skel[r, c]:
                    ok += 1
        return ok / max(tot, 1)
    a, b = hit(nodes), hit([n[:, ::-1] for n in nodes])
    return (nodes, a) if a >= b else ([n[:, ::-1] for n in nodes], b)


def build_graph(skel, nodes, px):
    H, W = skel.shape
    node_id = -np.ones(skel.shape, int)
    for i, arr in enumerate(nodes):
        for r, c in arr:
            if 0 <= r < H and 0 <= c < W:
                node_id[r, c] = i
    # absorb tiny stubs (<5 px) that touch a node into that node, then re-split
    for _ in range(3):
        rest = skel & (node_id < 0)
        lab, n = ndimage.label(rest, structure=np.ones((3, 3)))
        changed = False
        for k in range(1, n + 1):
            pix = np.argwhere(lab == k)
            if len(pix) >= 5:
                continue
            touch = {node_id[r + dr, c + dc] for r, c in pix for dr, dc in EIGHT
                     if 0 <= r + dr < H and 0 <= c + dc < W and node_id[r + dr, c + dc] >= 0}
            if len(touch) == 1:
                node_id[tuple(pix.T)] = touch.pop(); changed = True
        if not changed:
            break
    rest = skel & (node_id < 0)
    lab, n = ndimage.label(rest, structure=np.ones((3, 3)))

    segs = []                       # each: dict(length, ends=[(node, pix, dir)])
    for k in range(1, n + 1):
        pix = [tuple(p) for p in np.argwhere(lab == k)]
        S = set(pix)
        nb = {p: [(p[0] + dr, p[1] + dc) for dr, dc in EIGHT if (p[0] + dr, p[1] + dc) in S] for p in pix}
        touching = {}               # pixel -> node it touches
        for p in pix:
            for dr, dc in EIGHT:
                q = (p[0] + dr, p[1] + dc)
                if 0 <= q[0] < H and 0 <= q[1] < W and node_id[q] >= 0:
                    touching.setdefault(p, node_id[q])
        ends = [p for p in pix if len(nb[p]) <= 1]
        T = [p for p in ends if p in touching] or list(touching)
        if not T:
            if not ends:            # closed loop with no crossing
                segs.append(dict(length=len(pix) * px * 1.1, ends=[], free=True, pix=pix))
            else:                   # fragment not connected to any node
                segs.append(dict(length=len(pix) * px, ends=[], free=False, dangling=True))
            continue
        a = T[0]
        dist = bfs_dist(a, nb)
        cand = [p for p in T if p != a]
        if not cand:                # only one attachment point -> loose end
            segs.append(dict(length=len(pix) * px, ends=[], free=False, dangling=True))
            continue
        b = max(cand, key=lambda p: dist.get(p, -1))
        path = bfs_path(a, b, nb)
        length = path_length(path) * px
        e = []
        for endpix, walk in ((a, path), (b, path[::-1])):
            q = walk[min(ARM_PX, len(walk) - 1)]
            v = np.array([q[0] - endpix[0], q[1] - endpix[1]], float)
            nv = np.linalg.norm(v)
            e.append((int(touching[endpix]), endpix, v / nv if nv else v))
        loose = len([p for p in ends if p not in touching]) > 0 and len(pix) > 3 * ARM_PX
        segs.append(dict(length=length, ends=e, free=False, dangling=False, spur=loose, path=path))
    return segs


def bfs_far(start, nb):
    prev = {start: None}; dq = deque([start]); last = start
    while dq:
        p = dq.popleft(); last = p
        for q in nb[p]:
            if q not in prev:
                prev[q] = p; dq.append(q)
    path = [last]
    while prev[path[-1]] is not None:
        path.append(prev[path[-1]])
    return path[::-1]


def path_length(path, w=7):
    """Length of a pixel path after a moving-average smoothing (removes staircase bias)."""
    P = np.asarray(path, float)
    if len(P) > w:
        k = np.ones(w) / w
        P = np.c_[np.convolve(P[:, 0], k, "valid"), np.convolve(P[:, 1], k, "valid")]
        head = np.linalg.norm(np.asarray(path[w // 2], float) - path[0])
        tail = np.linalg.norm(np.asarray(path[-1], float) - path[-1 - w // 2])
        return float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)) + head + tail)
    return float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))) if len(P) > 1 else 0.0


def bfs_dist(a, nb):
    d = {a: 0}; dq = deque([a])
    while dq:
        p = dq.popleft()
        for q in nb[p]:
            if q not in d:
                d[q] = d[p] + 1; dq.append(q)
    return d


def bfs_path(a, b, nb):
    prev = {a: None}; dq = deque([a])
    while dq:
        p = dq.popleft()
        if p == b:
            break
        for q in nb[p]:
            if q not in prev:
                prev[q] = p; dq.append(q)
    path = [b]
    while prev.get(path[-1]) is not None:
        path.append(prev[path[-1]])
    return path[::-1]


# ----------------------------------------------------------------------------- enumeration
def matchings(items):
    if not items:
        yield []
        return
    a = items[0]
    for i in range(1, len(items)):
        rest = items[1:i] + items[i + 1:]
        for m in matchings(rest):
            yield [(a, items[i])] + m


def bend(d1, d2):
    """Angle (rad) by which the DNA deviates from straight when it enters along d1 and leaves along d2."""
    c = float(np.clip(-np.dot(d1, d2), -1, 1))
    return math.acos(c)


AGNOSTIC = [[758.0], [418.0, 136.0]]   # every species present in the dataset


def len_energy(loops, target):
    if target == "agnostic":               # molecule type is NOT given: best over all species
        return min(len_energy(loops, t) for t in AGNOSTIC)
    L, T = sorted(loops, reverse=True), sorted(target, reverse=True)
    e = BIG * abs(len(L) - len(T))
    for l, t in zip(L, T):
        e += ((l - t) / (SIGMA * t)) ** 2
    return e


def solve(segs, target, lam):
    ends_at = defaultdict(list)          # node -> [(seg, endidx)]
    for s, sg in enumerate(segs):
        for j, (nd, _, _) in enumerate(sg["ends"]):
            ends_at[nd].append((s, j))
    if any(sg.get("dangling") for sg in segs):
        return dict(status="open_or_fragment")
    if any(len(v) % 2 for v in ends_at.values()):
        return dict(status="odd_degree")
    nodes = sorted(ends_at)
    per_node = []
    for nd in nodes:
        opts = []
        for m in matchings(ends_at[nd]):
            eb = 0.0; part = {}
            for (s1, j1), (s2, j2) in m:
                _, p1, d1 = segs[s1]["ends"][j1]; _, p2, d2 = segs[s2]["ends"][j2]
                eb += bend(d1, d2) ** 2
                part[(s1, j1)] = (s2, j2); part[(s2, j2)] = (s1, j1)
            opts.append((eb, part))
        per_node.append(opts)
    n_conf = int(np.prod([len(o) for o in per_node])) if per_node else 1
    if n_conf > MAX_CONFIGS:                       # beam: keep 5 smoothest per node
        per_node = [sorted(o, key=lambda x: x[0])[:5] for o in per_node]
    free = [sg["length"] for sg in segs if sg.get("free")]
    best = []
    for combo in (itertools.product(*per_node) if per_node else [()]):
        part = {}; eb = 0.0
        for o in combo:
            eb += o[0]; part.update(o[1])
        loops = list(free); seen = set(); bad = False
        for s0, sg in enumerate(segs):
            if sg.get("free") or s0 in seen:
                continue
            L = 0.0; cur, j = s0, 0
            for _ in range(4 * len(segs) + 4):
                seen.add(cur); L += segs[cur]["length"]
                nxt = part.get((cur, 1 - j))
                if nxt is None:
                    bad = True; break
                _, p1, _ = segs[cur]["ends"][1 - j]; _, p2, _ = segs[nxt[0]]["ends"][nxt[1]]
                L += math.hypot(p1[0] - p2[0], p1[1] - p2[1]) * PX[0]   # path through the node
                cur, j = nxt
                if (cur, j) == (s0, 0):
                    break
            else:
                bad = True
            if bad:
                break
            loops.append(L)
        if bad:
            continue
        El = len_energy(loops, target)
        best.append((El + lam * eb, El, eb, sorted(loops, reverse=True), part))
    if not best:
        return dict(status="no_valid_config")
    best.sort(key=lambda x: x[0])
    E, El, Eb, loops, part = best[0]
    margin = best[1][0] - E if len(best) > 1 else float("inf")
    return dict(status="ok", n_nodes=len(nodes), degrees=[len(ends_at[n]) for n in nodes],
                n_configs=n_conf, E=E, E_len=El, E_bend=Eb, loops=loops, margin=margin, part=part)


PX = [1.0]   # current pixel size (nm), used inside solve()


# ----------------------------------------------------------------------------- evaluation (also used by exp2_topology.py)
N_BOOT, SEED = 2000, 0


def kept_molecules(crops_meta):
    """The molecules of Stage 0: those passing the length and edge filter (keep column of the crops metadata)."""
    if not os.path.exists(crops_meta):
        raise SystemExit(f"{crops_meta} not found: run extract_crops.py first, or pass --crops_meta <path>")
    cm = pd.read_csv(crops_meta)
    return cm.loc[cm.keep.astype(str).str.lower() == "true", ["sample", "image", "grain", "label"]]


def acc_ci(correct):
    """Accuracy and 95% percentile bootstrap CI."""
    v = np.asarray(correct, float)
    rng = np.random.default_rng(SEED)
    boots = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(N_BOOT)]
    return v.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def holm(p):
    """Holm-adjusted p values, returned in the original order."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.minimum(np.maximum.accumulate((m - np.arange(m)) * p[order]), 1.0)
    out = np.empty(m)
    out[order] = adj
    return out


def groups_of(d, extra=()):
    """One group per knot class, any extra groups, and all molecules."""
    out = [(f"class {c}", g) for c, g in d.groupby("label")]
    out += [(name, d[mask]) for name, mask in extra]
    out.append(("all", d))
    return [(n, g) for n, g in out if len(g)]


def compare(groups, col_a, col_b, name_a, name_b):
    """Two methods on the same molecules, per group: accuracy with 95% CI, exact McNemar, Holm over the groups."""
    rows = []
    for gname, g in groups:
        a, b = g[col_a].astype(bool).to_numpy(), g[col_b].astype(bool).to_numpy()
        ma, la, ha = acc_ci(a)
        mb, lb, hb = acc_ci(b)
        a_only, b_only = int((a & ~b).sum()), int((~a & b).sum())
        rows.append({"group": gname, "n": len(g),
                     f"{name_a}_accuracy": round(ma, 4), f"{name_a}_ci_lo": round(la, 4), f"{name_a}_ci_hi": round(ha, 4),
                     f"{name_b}_accuracy": round(mb, 4), f"{name_b}_ci_lo": round(lb, 4), f"{name_b}_ci_hi": round(hb, 4),
                     "both_right": int((a & b).sum()), f"only_{name_a}_right": a_only,
                     f"only_{name_b}_right": b_only, "both_wrong": int((~a & ~b).sum()),
                     "p_mcnemar": binomtest(a_only, a_only + b_only).pvalue if a_only + b_only else 1.0})
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p_mcnemar).round(4)
    t["p_mcnemar"] = t.p_mcnemar.round(4)
    return t


def loops_ok(loops, sample):
    if sample.startswith("cat"):
        return len(loops) == 2 and 110 <= min(loops) <= 165 and abs(max(loops) - 418) / 418 < 0.10
    return len(loops) == 1 and abs(loops[0] - 758) / 758 < 0.10


# ----------------------------------------------------------------------------- main
def probe(path, lam, agnostic=False):
    with h5py.File(path, "r") as f:
        px = px_nm(f); PX[0] = px
        print(f"file: {path}\npixel size: {px:.3f} nm | grains: {grains(f)}")
        sample = os.path.basename(os.path.dirname(os.path.dirname(path)))
        target = "agnostic" if agnostic else TARGETS.get(sample, [758.0])
        for g in grains(f):
            skel, nodes = load_grain(f, g)
            nodes, frac = orient_nodes(skel, nodes)
            segs = build_graph(skel, nodes, px)
            res = solve(segs, target, lam)
            print(f"\n{g}: crop {skel.shape}, skeleton px {skel.sum()}, nodes {len(nodes)} "
                  f"(node pixels on skeleton: {frac:.0%})")
            print(f"  segments: {len(segs)}  lengths nm: {[round(s['length']) for s in segs]}")
            print(f"  ends per node: {dict(sorted(((n, sum(e[0]==n for s in segs for e in s['ends'])) for n in range(len(nodes)))))}")
            print(f"  TopoStats loops nm: {[round(x) for x in topostats_loops(f, g, px)]}  topology: {topostats_topology(f, g)}")
            print(f"  Experiment 1: { {k: v for k, v in res.items() if k != 'part'} }")


def batch(root, out, lam, agnostic, kept):
    files = sorted(glob.glob(os.path.join(root, "*", "processed", "*.topostats")))
    print(f"{len(files)} .topostats files")
    rows = []
    for i, path in enumerate(files):
        sample = os.path.basename(os.path.dirname(os.path.dirname(path)))
        if sample not in TARGETS:
            continue
        target = "agnostic" if agnostic else TARGETS[sample]
        try:
            with h5py.File(path, "r") as f:
                px = px_nm(f); PX[0] = px
                for g in grains(f):
                    skel, nodes = load_grain(f, g)
                    nodes, frac = orient_nodes(skel, nodes)
                    segs = build_graph(skel, nodes, px)
                    res = solve(segs, target, lam)
                    ts = topostats_loops(f, g, px)
                    tot = sum(s["length"] for s in segs)
                    rows.append(dict(sample=sample, image=os.path.basename(path)[:-10], grain=g,
                                     skel_len_nm=round(tot, 1), n_nodes=len(nodes), node_on_skel=round(frac, 2),
                                     status=res["status"], degrees=res.get("degrees"),
                                     n_configs=res.get("n_configs"), E=res.get("E"),
                                     E_len=res.get("E_len"), E_bend=res.get("E_bend"),
                                     margin=res.get("margin"),
                                     exp1_loops=[round(x) for x in res.get("loops", [])],
                                     exp1_ok=loops_ok(res.get("loops", []), sample) if res["status"] == "ok" else False,
                                     ts_loops=[round(x) for x in ts], ts_ok=loops_ok(ts, sample),
                                     ts_topology=topostats_topology(f, g)))
        except Exception as ex:                      # keep going, report at the end
            rows.append(dict(sample=sample, image=os.path.basename(path), grain="", status=f"error: {ex}"))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(files)}")
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")

    # evaluation on the molecules of Stage 0; a molecule without a result counts as wrong
    d = kept.merge(df, on=["sample", "image", "grain"], how="left")
    d["exp1_ok"] = d.exp1_ok.fillna(False).astype(bool)
    d["ts_ok"] = d.ts_ok.fillna(False).astype(bool)
    d["status"] = d.status.fillna("missing")
    res_dir = os.path.dirname(os.path.abspath(out))
    stem = os.path.splitext(os.path.basename(out))[0]
    summary = compare(groups_of(d, [("knots and catenanes", d.label != "0_1")]), "exp1_ok", "ts_ok", "exp1", "topostats")
    summary.to_csv(os.path.join(res_dir, f"{stem}_summary.csv"), index=False)
    status = pd.crosstab(d.label, d.status, margins=True, margins_name="all")
    status.to_csv(os.path.join(res_dir, f"{stem}_status.csv"))
    print(f"\nloops correct on the {len(d)} molecules of Stage 0 (failures count as wrong):")
    print(summary[["group", "n", "exp1_accuracy", "topostats_accuracy", "only_exp1_right",
                   "only_topostats_right", "p_mcnemar", "p_holm"]].to_string(index=False))
    print("\nsolver status per class:")
    print(status.to_string())
    print(f"\nsaved {stem}_summary.csv and {stem}_status.csv in {res_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe")
    ap.add_argument("--root", default=os.path.join(REPO, "data", "knots_and_cats"))
    ap.add_argument("--out", default=None, help="default: results/pairing/exp1_agnostic.csv (exp1_given.csv with --given)")
    ap.add_argument("--crops_meta", default=os.path.join(REPO, "crops_1nm", "metadata.csv"),
                    help="crops metadata of Stage 0; its keep column selects the molecules that are evaluated")
    ap.add_argument("--lam", type=float, default=1.0, help="weight of the bending term")
    ap.add_argument("--given", action="store_true",
                    help="tell the solver which sample it is (optimistic). Default: agnostic (fair, no label leakage)")
    a = ap.parse_args()
    if a.out is None:
        a.out = os.path.join(REPO, "results", "pairing", "exp1_given.csv" if a.given else "exp1_agnostic.csv")
    if a.probe:
        probe(a.probe, a.lam, not a.given)
    else:
        batch(a.root, a.out, a.lam, not a.given, kept_molecules(a.crops_meta))
