# Results

Every file here is written by a script in `code/` (run all of them with `bash run.sh`). The table
"Where each result of the paper is" in the main [README](../README.md) lists the file behind each table
and figure of the paper.

| Folder | Contents | Written by |
|---|---|---|
| `splits/<crops>/` | molecules and sessions per class, and the test molecules of each fold | `make_splits.py` |
| `classify_<crops>/` | out-of-fold predictions of each classifier run (`*_oof.csv`) and a confusion matrix per run (`*_oof_confusion.png`) | `train_crops.py`, `evaluate.py` |
| `classifier_performance/` | metrics with molecule-level 95% CIs, recall per class, accuracy per condition and per session, McNemar tests with Holm-adjusted p values, confusion matrices and figures | `evaluate_classifiers.py` |
| `classifier_performance_session_boot/` | the same metrics with session-level 95% CIs, and the knots-only figure with these CIs | `session_bootstrap.py` |
| `session_probe/crops_1nm/` | session probe per preparation and buffer, recall per session, confusion matrices | `session_probe.py` |
| `crossings_<resolution>/` | extraction counts, crossing labels, TopoStats accuracy on the labels, out-of-fold predictions of the crossing classifier in each training regime | `extract_crossings.py`, `make_labels.py`, `train_crossings.py` |
| `crossing_performance/` | crossing classifier and TopoStats accuracy with 95% CIs, per class and per session, McNemar tests with Holm-adjusted p values | `evaluate_crossings.py` |
| `pairing/` | energy-based pairing (`exp1_agnostic*`) and knot type (`exp2_topology*`), one row per molecule plus summaries | `exp1_pairing.py`, `exp2_topology.py` |
| `figures/qc/` | example molecule crops per class and example crossing patches | `contact_sheet.py`, `extract_crossings.py` |
| `logs/` | printed output of the classifier training and evaluation runs | `run.sh` |