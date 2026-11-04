# Classifiers learn the imaging session, not the knot

Code for the paper *Classifiers learn the imaging session, not the knot: a step-by-step evaluation of
machine learning for DNA topology in AFM images*.

The paper asks whether machine learning can identify the knot type of single DNA molecules in atomic
force microscopy (AFM) images, and which step of the existing pipeline (TopoStats) limits it. It
evaluates four stages on the same 721 molecules:

| Stage | Question | Scripts |
|---|---|---|
| 0 | Can an image classifier read the knot type directly? | `extract_crops.py`, `make_splits.py`, `train_crops.py`, `evaluate_classifiers.py`, `session_bootstrap.py`, `session_probe.py` |
| 1 | Which strand lies on top at a crossing? | `extract_crossings.py`, `make_labels.py`, `train_crossings.py`, `evaluate_crossings.py` |
| 2 | How should strands be connected through crossings? | `exp1_pairing.py` |
| 3 | Is the final knot type correct? | `exp2_topology.py` |

## Reproduce everything

```
bash run.sh
```

This runs every step in order and writes all tables and figures to `results/`. If it is interrupted,
run it again and finished steps are skipped. To start from scratch, run `FORCE=1 bash run.sh`.
At the end it prints where each table of the paper is.

### 1. Get the data

Download the public dataset of Gamill et al. (2025) from the University of Sheffield repository,
<https://doi.org/10.15131/shef.data.27143238>, and place it so that the TopoStats files are at

```
data/knots_and_cats/<sample>/processed/<image>.topostats
```

`run.sh` checks this and stops with a message if the files are not found.

### 2. Install the packages

```
pip install numpy scipy pandas scikit-learn matplotlib h5py pillow torch torchvision timm topoly
```

### 3. Hardware

Fine-tuning ResNet-18 and ConvNeXt-Tiny (Stage 0) needs a CUDA GPU. On a CPU it takes days.
All other steps run on a CPU.

On macOS with conda, import `torch` before `scikit-learn` in any new script. The other order can crash
Python with a segmentation fault. The scripts here already do this.

## What each script does

All scripts are in `code/` and are run from the project folder, for example `python code/make_splits.py`.

**Stage 0: whole-molecule classifiers**

- `extract_crops.py` cuts each molecule out of its image, resamples it to a fixed pixel size and keeps
  molecules whose DNA length matches a known molecule (721 of 759). Output: `crops_<name>/`.
- `make_splits.py` assigns molecules to five folds, once grouped by imaging session and once grouped by
  image. Output: `crops_<name>/splits.csv` and `results/splits/`.
- `contact_sheet.py` draws example crops for a visual check.
- `train_crops.py` trains ResNet-18, ConvNeXt-Tiny or DINOv2 with logistic regression, and predicts every
  molecule once with a model that did not see it. Output: `results/classify_<name>/<model>_<split>_oof.csv`.
- `evaluate_classifiers.py` computes all metrics, confidence intervals, McNemar tests and figures.
  Output: `results/classifier_performance/`.
- `session_bootstrap.py` recomputes the confidence intervals by resampling imaging sessions instead of
  molecules. Output: `results/classifier_performance_session_boot/`.
- `session_probe.py` tests whether the imaging session can be predicted from images of molecules of one
  preparation. Output: `results/session_probe/`.

**Stage 1: crossing labels and crossing classifier**

- `extract_crossings.py` cuts a 64 x 64 nm patch around every crossing found by TopoStats.
  Output: `crossings_<resolution>/`.
- `make_labels.py` labels the upper strand at crossings of clean knots from the known handedness of each
  knot, without manual annotation. Output: `crossings_<resolution>/labels.csv`.
- `train_crossings.py` trains a small CNN in three regimes (labels only, TopoStats as teacher, teacher
  followed by fine-tuning) and compares it with TopoStats on the same crossings.
  Output: `results/crossings_<resolution>/`.
- `evaluate_crossings.py` collects all crossing results into tables. Output: `results/crossing_performance/`.

**Stages 2 and 3: strand pairing and topology**

- `exp1_pairing.py` connects strands through crossing regions by minimizing an energy based on DNA length
  and bending, without being told the sample. Output: `results/pairing/exp1_agnostic*.csv`.
- `exp2_topology.py` orders the crossings along these connections, computes the knot type with Topoly and
  counts the crossings found. Output: `results/pairing/exp2_topology*.csv`.

## Released data

Besides the code, this repository contains

- the session and image splits: `crops_*/splits.csv`
- the crossing labels: `crossings_*/labels.csv`
- the prediction for every molecule and every labeled crossing: `results/classify_*/` and `results/crossings_*/`

The image arrays (`npy/`, `png/`) are not included because of their size. `run.sh` rebuilds them from
the dataset.

## Folder layout

```
code/          all scripts
run.sh         runs the whole pipeline
data/          the downloaded dataset (not included)
crops_*/       molecule crops, metadata and splits (made by extract_crops.py and make_splits.py)
crossings_*/   crossing patches, metadata and labels (made by extract_crossings.py and make_labels.py)
results/       all tables, figures and predictions
```
