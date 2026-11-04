"""
Experiment 2 - from the physics-based tracing (Experiment 1) to the KNOT TYPE of every molecule.

For every molecule in every .topostats file:
  1. Experiment 1: build the skeleton graph and choose the strand connections at every crossing region
     with the physics rule (right total DNA length, smooth bends) - exp1_pairing.py
  2. Decide which strand is on top at every crossing, with the same idea TopoStats uses:
     the strand whose height profile through the crossing is WIDER is on top (FWHM rule, Holmes et al. 2025),
     here measured along OUR strand connections. (--rule height: the higher strand is on top instead.)
  3. Build a 3D curve: DNA lies at z = 0, a strand passing over a crossing is lifted above the other.
  4. Compute the knot / link type with Topoly (HOMFLY-PT polynomial).
  5. Compare with the sample's known topology, and with what TopoStats reported for the same molecule.

The solver is not told which sample a molecule comes from, unless --given is used. Results are evaluated
on the molecules of Stage 0, those that pass the length and edge filter of extract_crops.py (keep column
of crops_1nm/metadata.csv). A molecule without a knot type (solver failure, Topoly timeout) counts as wrong.

Output (results/pairing/):
  exp2_topology.csv            one row per molecule (exp2_topology_height.csv with --rule height)
  exp2_topology_summary.csv    knot type correct per knot class, knots and catenanes, knots only and
                               all molecules: Experiment 2 and TopoStats on the same molecules,
                               95% bootstrap CIs, exact McNemar test, Holm-adjusted p values
  exp2_topology_answers.csv    how often each knot type (or failure) was returned, per knot class

Run from the project folder (needs: pip install topoly):
  python code/exp2_topology.py                                                       # all molecules, fair
  python code/exp2_topology.py --probe data/knots_and_cats/twist/processed/<image>.topostats   # one file
"""
import argparse, glob, io, math, os, contextlib
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
from scipy import ndimage

import exp1_pairing as E1

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXPECTED = {"trefoil": "3_1", "torus": "5_1", "twist": "5_2", "unknot_plasmid_sup": "0_1",
            "unknot_plasmid_nic": "0_1", "cat_supercoiled": "4^2_1", "cat_nicked": "4^2_1"}
LINK_NAMES = {"L2a1": "2^2_1", "L4a1": "4^2_1", "L5a1": "5^2_1", "L6a1": "6^2_1", "L6a2": "6^2_2",
              "L6a3": "6^2_3", "L7a1": "7^2_1", "L4a1#L2a1": "4^2_1#2^2_1"}
ARM = 12          # pixels of strand taken on each side of a crossing for the height profile
LIFT = 4.0        # z-step (in pixels) between stacked strands in the 3D curve


# ----------------------------------------------------------------------------- helpers
def image_of(f, g):
    return f[f"disordered_traces/above/{g}/original_image"][()].astype(float)


def walk_closed(pix):
    """Order the pixels of a closed skeleton loop (no crossings)."""
    S = set(pix)
    nb = {p: [(p[0] + dr, p[1] + dc) for dr, dc in E1.EIGHT if (p[0] + dr, p[1] + dc) in S] for p in pix}
    start = pix[0]; out = [start]; prev = None; cur = start
    for _ in range(len(pix)):
        nxt = [q for q in nb[cur] if q != prev and q not in out[-3:]]
        if not nxt:
            break
        prev, cur = cur, nxt[0]
        if cur == start:
            break
        out.append(cur)
    if len(out) < 0.8 * len(pix):                   # fall back: sort around the centre
        P = np.asarray(pix, float); c = P.mean(0)
        out = [pix[i] for i in np.argsort(np.arctan2(P[:, 0] - c[0], P[:, 1] - c[1]))]
    return out


def traverse(segs, part):
    """Loops as lists of steps: ('seg', s, j_entry) and ('chord', node, end_out, end_in)."""
    loops, seen = [], set()
    for s0, sg in enumerate(segs):
        if sg.get("free"):
            loops.append([("free", s0)]); continue
        if s0 in seen:
            continue
        steps, cur, j = [], s0, 0
        for _ in range(4 * len(segs) + 4):
            seen.add(cur)
            steps.append(("seg", cur, j))
            nxt = part[(cur, 1 - j)]
            steps.append(("chord", segs[cur]["ends"][1 - j][0], (cur, 1 - j), nxt))
            cur, j = nxt
            if (cur, j) == (s0, 0):
                break
        loops.append(steps)
    return loops


def oriented(seg, j):
    return seg["path"] if j == 0 else seg["path"][::-1]


def chord_profile(img, segs, e_in, e_out):
    """Height profile along: last ARM px of the incoming strand -> straight line through the node -> first ARM px out."""
    s1, j1 = e_in; s2, j2 = e_out
    a = oriented(segs[s1], 1 - j1)[-ARM:]           # path of s1 ending at its end j1 (at the node)
    b = oriented(segs[s2], j2)[:ARM]                # path starting at the node
    p1, p2 = np.asarray(a[-1], float), np.asarray(b[0], float)
    n = max(2, int(np.ceil(np.linalg.norm(p2 - p1) * 2)))
    mid = np.linspace(p1, p2, n)[1:-1]
    P = np.vstack([np.asarray(a, float), mid, np.asarray(b, float)])
    h = ndimage.map_coordinates(img, [P[:, 0], P[:, 1]], order=1)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    lo, hi = len(a) - 1, len(a) + len(mid)          # indices of the part inside the node
    return h, d, lo, hi


def fwhm(h, d, lo, hi):
    if len(h) < 5:
        return 0.0, 0.0
    i = lo + int(np.argmax(h[lo:hi + 1])) if hi >= lo else int(np.argmax(h))
    base = min(np.median(h[:3]), np.median(h[-3:]))
    half = (h[i] + base) / 2
    l = i
    while l > 0 and h[l - 1] >= half:
        l -= 1
    r = i
    while r < len(h) - 1 and h[r + 1] >= half:
        r += 1
    return float(d[r] - d[l]), float(h[i])


def segments_cross(p1, p2, q1, q2):
    def o(a, b, c):
        return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    return o(p1, p2, q1) * o(p1, p2, q2) < 0 and o(q1, q2, p1) * o(q1, q2, p2) < 0


def stacking(img, segs, part, rule):
    """For every node: order its strand connections ('chords') from top to bottom."""
    chords = defaultdict(dict)                      # node -> {chord_key: (score, p1, p2)}
    for (s, j), (s2, j2) in part.items():
        key = frozenset([(s, j), (s2, j2)])
        nd = segs[s]["ends"][j][0]
        if key in chords[nd]:
            continue
        h, d, lo, hi = chord_profile(img, segs, (s, j), (s2, j2))  # arrive through end j of s, leave through end j2 of s2
        w, peak = fwhm(h, d, lo, hi)
        score = w if rule == "fwhm" else peak
        chords[nd][key] = (score, segs[s]["ends"][j][1], segs[s2]["ends"][j2][1])
    level, n_cross = {}, 0
    for nd, ch in chords.items():
        keys = sorted(ch, key=lambda k: -ch[k][0])
        for rank, k in enumerate(keys):
            level[k] = len(keys) - rank             # top strand = highest level
        ks = list(ch)
        for i in range(len(ks)):
            for jj in range(i + 1, len(ks)):
                if segments_cross(ch[ks[i]][1], ch[ks[i]][2], ch[ks[jj]][1], ch[ks[jj]][2]):
                    n_cross += 1
    return level, n_cross


def curves_3d(segs, loops, level, flip=False):
    chains = []
    for steps in loops:
        pts = []
        for st in steps:
            if st[0] == "free":
                for r, c in walk_closed(segs[st[1]]["pix"])[::3]:
                    pts.append([c, r, 0.0])
            elif st[0] == "seg":
                for r, c in oriented(segs[st[1]], st[2])[::3]:
                    pts.append([c, r, 0.0])
            else:
                _, nd, eo, ei = st
                p1 = segs[eo[0]]["ends"][eo[1]][1]; p2 = segs[ei[0]]["ends"][ei[1]][1]
                z = LIFT * level.get(frozenset([eo, ei]), 0) * (-1 if flip else 1)
                n = max(2, int(math.hypot(p2[0] - p1[0], p2[1] - p1[1])))
                pts.append([p1[1], p1[0], 0.0]); pts.append([p1[1], p1[0], z])
                for t in np.linspace(0, 1, n)[1:-1]:
                    pts.append([p1[1] + t * (p2[1] - p1[1]), p1[0] + t * (p2[0] - p1[0]), z])
                pts.append([p2[1], p2[0], z]); pts.append([p2[1], p2[0], 0.0])
        clean = [pts[0]] if pts else []
        for p in pts[1:]:
            if p != clean[-1]:
                clean.append(p)
        if len(clean) > 1 and clean[-1] == clean[0]:
            clean.pop()
        jitter = np.random.default_rng(len(chains)).normal(0, 1e-3, (len(clean), 3))
        chains.append((np.asarray(clean, float) + jitter).round(4).tolist())
    return chains


class TopolyRunner:
    """Runs Topoly in a separate helper process (this same script with --worker), so a molecule that makes
    Topoly hang is stopped after `timeout` seconds instead of freezing the whole run."""
    MARK = "@@TOPOLOGY@@"

    def __init__(self, timeout):
        self.timeout = timeout
        self.proc = None

    def _start(self):
        import subprocess, sys
        self.proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--worker"],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(os.devnull if not os.environ.get("EXP2_DEBUG") else "worker_stderr.txt", "a"),
                                     text=True, bufsize=1)

    def _kill(self):
        if self.proc is not None:
            try:
                self.proc.kill(); self.proc.wait(timeout=5)
            except Exception:
                pass
        self.proc = None

    def __call__(self, chains, n_cross):
        import json, select, time
        if n_cross > MAX_CROSS:
            return "too_complex"
        if self.proc is None or self.proc.poll() is not None:
            self._start()
        try:
            self.proc.stdin.write(json.dumps(chains) + "\n"); self.proc.stdin.flush()
        except Exception:
            self._kill(); return "worker_error"
        t_end = time.time() + self.timeout
        while True:
            left = t_end - time.time()
            if left <= 0 or not select.select([self.proc.stdout], [], [], left)[0]:
                self._kill(); return "timeout"
            line = self.proc.stdout.readline()
            if not line:
                self._kill(); return "worker_error"
            if line.startswith(self.MARK):
                return line[len(self.MARK):].strip()

    def close(self):
        self._kill()


def worker():
    """Helper-process loop: one JSON list of chains per input line -> one marked result line.
    Uses private copies of stdin/stdout, because Topoly closes/redirects the standard ones."""
    import json
    fin = os.fdopen(os.dup(0), "r")
    fout = os.fdopen(os.dup(1), "w")
    for line in fin:
        try:
            res = topology(json.loads(line))
        except Exception as ex:
            res = f"error:{type(ex).__name__}"
        fout.write(TopolyRunner.MARK + res + "\n"); fout.flush()


MAX_CROSS = 14     # skip topology for tangles with more crossings than this (counted inside crossing regions)
RUNNER = None


def topology(chains):
    import topoly
    from topoly import Closure
    if not chains or any(len(c) < 4 for c in chains):
        return "invalid"
    inp = chains[0] if len(chains) == 1 else chains
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            t = topoly.homfly(inp, closure=Closure.CLOSED, hide_trivial=False, chiral=False,
                              run_parallel=False, cuda=False)
    except Exception as ex:
        return f"error:{type(ex).__name__}"
    if isinstance(t, dict):
        t = max(t, key=t.get) if t else "unknown"
    return normalise(str(t), len(chains))


def normalise(t, n_loops=1):
    t = t.strip().lstrip("+-")
    parts = t.split("U")
    if all(p.strip("+-") == "0_1" for p in parts):
        return "0_1" if n_loops == 1 else "U".join(["0_1"] * n_loops)
    return LINK_NAMES.get(t, t)


def ts_name(t):
    """TopoStats' call in the same notation as ours. The number of loops is kept, so a molecule that
    TopoStats traced as two unlinked circles ('0_1U0_1') is not counted as an unknot (as in Stage 0)."""
    if not t:
        return ""
    t = str(t)
    return normalise(t, len(t.split("U")))


# ----------------------------------------------------------------------------- runs
def analyse(f, g, px, target, lam, rule):
    skel, nodes = E1.load_grain(f, g)
    nodes, _ = E1.orient_nodes(skel, nodes)
    E1.PX[0] = px
    segs = E1.build_graph(skel, nodes, px)
    res = E1.solve(segs, target, lam)
    out = dict(status=res["status"], skel_len_nm=round(sum(s["length"] for s in segs), 1))
    if res["status"] != "ok":
        return out
    img = image_of(f, g)
    loops = traverse(segs, res["part"])
    level, n_cross = stacking(img, segs, res["part"], rule)
    chains = curves_3d(segs, loops, level)
    out.update(loops=[round(x) for x in res["loops"]], n_loops=len(loops), n_crossings=n_cross,
               topology=RUNNER(chains, n_cross) if RUNNER is not None else topology(chains))
    return out


def probe(path, lam, rule, agnostic):
    with h5py.File(path, "r") as f:
        px = E1.px_nm(f)
        sample = os.path.basename(os.path.dirname(os.path.dirname(path)))
        target = "agnostic" if agnostic else E1.TARGETS.get(sample, [758.0])
        print(f"{path}\nsample {sample} (expected {EXPECTED.get(sample)}), pixel {px:.3f} nm")
        for g in E1.grains(f):
            r = analyse(f, g, px, target, lam, rule)
            print(f"  {g}: TopoStats -> {ts_name(E1.topostats_topology(f, g))!r:12s} | Exp2 -> {r}")


def batch(root, out, lam, rule, agnostic, kept):
    files = sorted(glob.glob(os.path.join(root, "*", "processed", "*.topostats")))
    print(f"{len(files)} .topostats files")
    rows = []
    for i, path in enumerate(files):
        sample = os.path.basename(os.path.dirname(os.path.dirname(path)))
        if sample not in EXPECTED:
            continue
        target = "agnostic" if agnostic else E1.TARGETS[sample]
        try:
            with h5py.File(path, "r") as f:
                px = E1.px_nm(f)
                for g in E1.grains(f):
                    try:
                        r = analyse(f, g, px, target, lam, rule)
                    except Exception as ex:
                        r = dict(status=f"error: {type(ex).__name__}: {ex}")
                    r.update(sample=sample, image=os.path.basename(path)[:-10], grain=g,
                             expected=EXPECTED[sample], ts_topology=ts_name(E1.topostats_topology(f, g)))
                    rows.append(r)
        except Exception as ex:
            rows.append(dict(sample=sample, image=os.path.basename(path), status=f"error: {ex}"))
        print(f"  {i + 1}/{len(files)}  {sample}/{os.path.basename(path)[:-10]}", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")

    # evaluation on the molecules of Stage 0; a molecule without a result counts as wrong
    d = kept.merge(df, on=["sample", "image", "grain"], how="left")
    if "topology" not in d:
        d["topology"] = np.nan
    d["exp2_ok"] = d.topology.fillna("") == d.label
    d["ts_ok"] = d.ts_topology.fillna("") == d.label
    res_dir = os.path.dirname(os.path.abspath(out))
    stem = os.path.splitext(os.path.basename(out))[0]
    extra = [("knots and catenanes", d.label != "0_1"), ("knots (3_1, 5_1, 5_2)", d.label.isin(["3_1", "5_1", "5_2"]))]
    summary = E1.compare(E1.groups_of(d, extra), "exp2_ok", "ts_ok", "exp2", "topostats")
    summary.to_csv(os.path.join(res_dir, f"{stem}_summary.csv"), index=False)
    answers = (d.assign(answer=d.topology.fillna(d.status).fillna("missing"))
                .groupby("label").answer.value_counts().rename("molecules").reset_index())
    answers.to_csv(os.path.join(res_dir, f"{stem}_answers.csv"), index=False)
    print(f"\nknot type correct on the {len(d)} molecules of Stage 0 (failures count as wrong):")
    print(summary[["group", "n", "exp2_accuracy", "topostats_accuracy", "only_exp2_right",
                   "only_topostats_right", "p_mcnemar", "p_holm"]].to_string(index=False))
    print("\nmost common answers per class:")
    for c, g in answers.groupby("label"):
        print(f"  {c:6s} {dict(zip(g.answer.head(5), g.molecules.head(5)))}")
    print(f"\nsaved {stem}_summary.csv and {stem}_answers.csv in {res_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(REPO, "data", "knots_and_cats"))
    ap.add_argument("--probe")
    ap.add_argument("--out", default=None, help="default: results/pairing/exp2_topology.csv")
    ap.add_argument("--crops_meta", default=os.path.join(REPO, "crops_1nm", "metadata.csv"),
                    help="crops metadata of Stage 0; its keep column selects the molecules that are evaluated")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--rule", choices=["fwhm", "height"], default="fwhm",
                    help="how to decide the top strand: wider height profile (TopoStats' rule) or higher peak")
    ap.add_argument("--given", action="store_true",
                    help="tell the solver which sample it is (optimistic). Default: agnostic (fair)")
    ap.add_argument("--timeout", type=float, default=20, help="seconds allowed per molecule for Topoly")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.out is None:
        a.out = os.path.join(REPO, "results", "pairing",
                             "exp2_topology.csv" if a.rule == "fwhm" else "exp2_topology_height.csv")
    if a.worker:
        worker(); raise SystemExit
    RUNNER = TopolyRunner(a.timeout)
    if a.probe:
        probe(a.probe, a.lam, a.rule, not a.given)
    elif a.root:
        batch(a.root, a.out, a.lam, a.rule, not a.given, E1.kept_molecules(a.crops_meta))
    else:
        ap.print_help()
    RUNNER.close()
