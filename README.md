# Session-aware testing, strand pairing and crossing limits for DNA knots in AFM images

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Code, labels and results for the paper

> Sammak MH, Biplob KBMB, Toma TR, Mahmud I. *A session-aware test protocol, energy-based strand
> pairing and a measured crossing-resolution limit for automated DNA knot identification in atomic
> force microscopy images.*

The paper presents three methods for identifying the knot type of single DNA molecules in atomic force
microscopy (AFM) images. Each method is evaluated against TopoStats on the same 721 molecules of the
public dataset of Holmes et al. (2025).

| Method | What it does | Scripts |
|---|---|---|
| Session-aware test protocol | Holds out whole imaging sessions, scores the four knot types that cannot be separated by size, compares with TopoStats on the same molecules, and tests whether images identify their session | `extract_crops.py`, `make_splits.py`, `train_crops.py`, `evaluate_classifiers.py`, `session_bootstrap.py`, `session_probe.py` |
| Energy-based strand pairing | Connects strands through crossing regions by DNA length and stiffness | `exp1_pairing.py` |
| Crossing-resolution limit | Labels the upper strand at crossings from the known handedness of the knots, tests the width rule and a crossing classifier on these labels, and counts the crossings lost when crossing regions merge | `extract_crossings.py`, `make_labels.py`, `train_crossings.py`, `evaluate_crossings.py`, `exp2_topology.py` |

The script names use the stage numbers of Fig 2 of the paper. Stage 0 is the whole-molecule classifier,
Stage 1 the upper strand at each crossing, Stage 2 the strand pairing and Stage 3 the knot type.

## Reproduce everything

```
bash run.sh
```

This runs every step in order and writes all tables and figures to `results/`. If it is interrupted,
run it again and finished steps are skipped. To start from scratch, run `FORCE=1 bash run.sh`.
At the end it prints where each table and figure of the paper is.

### 1. Get the data

Download the dataset of Gamill et al. (2025), <https://doi.org/10.15131/shef.data.27143238>, as described
in [`data/README.md`](data/README.md). The TopoStats files must end up at

```
data/knots_and_cats/<sample>/processed/<image>.topostats
```

`run.sh` checks this and stops with a message if the files are not found.

### 2. Install the packages

The analyses were run with Python 3.14.8 and the package versions in `requirements.txt`.

```
pip install -r requirements.txt
```

### 3. Hardware

Fine-tuning ResNet-18 and ConvNeXt-Tiny needs a CUDA GPU. On a CPU it takes days. All other steps run
on a CPU.

On macOS with conda, import `torch` before `scikit-learn` in any new script. The other order can crash
Python with a segmentation fault. The scripts here already do this.

## Where each result of the paper is

All paths are relative to `results/`.

| Paper item | File |
|---|---|
| Table 1, molecules and sessions per class | `splits/crops_1nm/class_counts.csv`, and `../crops_1nm/splits.csv` for images and sessions per molecule |
| Table 2, crossing labels from knot handedness | `crossings_1nm/labels_per_class.csv`, `crossings_1nm/sign_patterns.csv`, `crossings_1nm/trefoil_sign.csv` |
| Table 3, whole-molecule classification | `classifier_performance/summary.csv`; session-level CIs in `classifier_performance_session_boot/summary.csv` |
| Fig 4, confusion matrices of ResNet-18 | `classifier_performance/fig3_confusion.pdf` |
| Table 4, session probe | `session_probe/crops_1nm/summary.csv` |
| Table 5, traced loops | `pairing/exp1_agnostic_summary.csv` |
| Table 6, crossing classifiers | `crossing_performance/summary.csv`, `crossing_performance/comparisons.csv` |
| Table 7, knot type | `pairing/exp2_topology_summary.csv` |
| Table 8, crossings found per molecule | column `n_crossings` of `pairing/exp2_topology.csv` |
| S1 Appendix, control analyses | `classifier_performance/summary.csv`, `classifier_performance/comparisons.csv`, `classifier_performance_session_boot/split_gap.csv` |
| S2 Appendix, crossing classifier details | `crossings_*/extraction_summary.csv`, `crossings_*/samples_per_class.csv`, `crossings_*/crossing_results.csv` |
| S1 Fig, knots-only balanced accuracy of all runs | `classifier_performance_session_boot/fig2_knots_balanced_accuracy.png` |
| S2 Fig, crossing patches | `figures/qc/crossings_1nm_sheet.png` |
| S1 Table, recall per class | `classifier_performance/per_class_recall.csv` |
| S2 Table, accuracy per session | `classifier_performance/per_session.csv` |
| S3 Table, session folds | `splits/crops_1nm/folds_session.csv` |
| S4 Table, types assigned | `pairing/exp2_topology_answers.csv` |

The Holm-adjusted p values are in the column `p_holm` of each `comparisons.csv` and `*_summary.csv`.

## Released data

Besides the code, this repository contains

- the session and image splits of every molecule: `crops_*/splits.csv`
- the crossing labels from knot handedness: `crossings_*/labels.csv`
- the out-of-fold prediction for every molecule and every labeled crossing: `results/classify_*/` and
  `results/crossings_*/`
- the logs of the classifier training and evaluation runs: `results/logs/`

The image arrays (`npy/`, `png/`) are not included because of their size. `run.sh` rebuilds them from
the dataset.

## What each script does

All scripts are in `code/` and are run from the project folder, for example `python code/make_splits.py`.

**Session-aware test protocol (Stage 0)**

- `extract_crops.py` cuts each molecule out of its image, resamples it to a fixed pixel size and keeps
  molecules whose DNA length matches a known molecule (721 of 759). Output: `crops_<name>/`.
- `make_splits.py` assigns molecules to five folds, once grouped by imaging session and once grouped by
  image. Output: `crops_<name>/splits.csv` and `results/splits/`.
- `contact_sheet.py` draws example crops for a visual check. Output: `results/figures/qc/`.
- `train_crops.py` trains ResNet-18, ConvNeXt-Tiny or DINOv2 with logistic regression, and predicts every
  molecule once with a model that did not see it. Output: `results/classify_<name>/<model>_<split>_oof.csv`.
- `evaluate_classifiers.py` computes all metrics, confidence intervals, McNemar tests and figures.
  Output: `results/classifier_performance/`.
- `evaluate.py` prints a quick summary of one or more `*_oof.csv` files and saves a confusion matrix next
  to each.
- `session_bootstrap.py` recomputes the confidence intervals by resampling imaging sessions instead of
  molecules. Output: `results/classifier_performance_session_boot/`.
- `session_probe.py` tests whether the imaging session can be predicted from images of molecules of one
  preparation. Output: `results/session_probe/`.

**Crossing labels and crossing classifier (Stage 1)**

- `extract_crossings.py` cuts a 64 x 64 nm patch around every crossing region found by TopoStats.
  Output: `crossings_<resolution>/`.
- `make_labels.py` labels the upper strand at crossings of clean knots from the known handedness of each
  knot, without manual annotation. Output: `crossings_<resolution>/labels.csv` and
  `results/crossings_<resolution>/`.
- `train_crossings.py` trains a small CNN in three ways (known crossings only, TopoStats decisions,
  TopoStats decisions then known crossings) and compares it with TopoStats on the same crossings.
  Output: `results/crossings_<resolution>/`.
- `evaluate_crossings.py` collects all crossing results into tables. Output: `results/crossing_performance/`.

**Strand pairing and knot type (Stages 2 and 3)**

- `exp1_pairing.py` connects strands through crossing regions by minimizing an energy based on DNA length
  and bending, without being told the sample. Output: `results/pairing/exp1_agnostic*.csv`.
- `exp2_topology.py` orders the crossings along these connections with the width rule, computes the knot
  type with Topoly and counts the crossings found. Output: `results/pairing/exp2_topology*.csv`.

## Folder layout

```
code/          all scripts
run.sh         runs the whole pipeline
data/          the downloaded dataset (not included; see data/README.md)
crops_*/       molecule metadata and splits (made by extract_crops.py and make_splits.py)
crossings_*/   crossing metadata and labels (made by extract_crossings.py and make_labels.py)
results/       all tables, figures, predictions and logs
```

## How to cite

Cite the paper above and this repository. Citation details for the repository are in
[`CITATION.cff`](CITATION.cff). Cite the dataset as

Gamill M, Holmes E, Catley T, Wiggins L, Pyne A, Whittle S, et al. DNA knots, catenanes, and replication
intermediates via high-resolution AFM [dataset]. The University of Sheffield; 2025.
doi:10.15131/shef.data.27143238

## License

MIT, see [`LICENSE`](LICENSE).