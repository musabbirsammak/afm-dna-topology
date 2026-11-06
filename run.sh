#!/usr/bin/env bash
# Reproduce every result in the paper with one command.
#
#   bash run.sh             run everything; steps whose output already exists are skipped
#   FORCE=1 bash run.sh     run everything again from the start
#
# Before running:
#   1. Download the public dataset (Gamill et al., 2025, https://doi.org/10.15131/shef.data.27143238)
#      and place it so that the .topostats files are at
#          data/knots_and_cats/<sample>/processed/<image>.topostats
#   2. Install the Python packages: pip install -r requirements.txt
#
# Hardware: a CUDA GPU is strongly recommended. Fine-tuning ResNet-18 and ConvNeXt-Tiny on a CPU
# takes days. Everything else runs on a CPU.
#
# Output: everything goes to results/. Each step also writes what it prints to results/logs/<step>.txt.
# Where each result of the paper ends up is listed at the end of this script.

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results/logs

# step <file that marks the step as done, or ""> <name> <python arguments...>
step () {
  local done_file=$1 name=$2; shift 2
  if [ -n "$done_file" ] && [ -e "$done_file" ] && [ "${FORCE:-0}" != 1 ]; then
    echo ">> $name (already done, skipped)"; return
  fi
  echo ">> $name"
  python "$@" 2>&1 | tee "results/logs/$name.txt"
}

# ------------------------------------------------------------------ 0. checks
n=$(find data/knots_and_cats -name "*.topostats" 2>/dev/null | wc -l | tr -d ' ')
if [ "$n" -eq 0 ]; then
  echo "No .topostats files found under data/knots_and_cats/."
  echo "Download the dataset from https://doi.org/10.15131/shef.data.27143238 and place it as"
  echo "data/knots_and_cats/<sample>/processed/<image>.topostats, then run this script again."
  exit 1
fi
echo "found $n .topostats files"
python - <<'EOF'
import importlib.util, sys
need = ["numpy", "scipy", "pandas", "sklearn", "matplotlib", "h5py", "PIL", "torch", "torchvision", "timm", "topoly"]
missing = [m for m in need if importlib.util.find_spec(m) is None]
if missing:
    sys.exit("missing Python packages: " + ", ".join(missing))
import torch
print("GPU:", "yes" if torch.cuda.is_available() else "NO - Stage 0 fine-tuning will be very slow")
EOF

# ------------------------------------------------------------------ 1. molecule crops and splits
step crops_1nm/metadata.csv        crops_1nm        code/extract_crops.py --nm_per_px 1.0 --size 256 --out crops_1nm
step crops_05nm/metadata.csv       crops_05nm       code/extract_crops.py --nm_per_px 0.5 --size 512 --out crops_05nm
step crops_1nm_norm/metadata.csv   crops_1nm_norm   code/extract_crops.py --nm_per_px 1.0 --size 256 --mode norm   --out crops_1nm_norm
step crops_1nm_binary/metadata.csv crops_1nm_binary code/extract_crops.py --nm_per_px 1.0 --size 256 --mode binary --out crops_1nm_binary
step "" make_splits    code/make_splits.py
step "" contact_sheets code/contact_sheet.py

# ------------------------------------------------------------------ 2. Stage 0: whole-molecule classifiers
# main runs: 1 nm height crops, all models, both splits
for m in dinov2 resnet18 convnext_tiny; do
  for s in session image; do
    step "results/classify_1nm/${m}_${s}_oof.csv" "train_1nm_${m}_${s}" \
         code/train_crops.py --crops crops_1nm --model "$m" --split "$s"
  done
done
# control: flip augmentation
step results/classify_1nm/resnet18_session_flip_oof.csv train_1nm_resnet18_session_flip \
     code/train_crops.py --crops crops_1nm --model resnet18 --split session --flip
# control: training length (15 and 60 epochs; 30 is the main run)
for e in 15 60; do
  for m in resnet18 convnext_tiny; do
    step "results/classify_1nm_ep${e}/${m}_session_oof.csv" "train_1nm_${m}_session_ep${e}" \
         code/train_crops.py --crops crops_1nm --model "$m" --split session --epochs "$e" --out "results/classify_1nm_ep${e}"
  done
done
# control: sampling density. 0.5 nm crops at their full size of 512 x 512 pixels (not resized to 224)
for m in resnet18 convnext_tiny; do
  step "results/classify_05nm/${m}_session_oof.csv" "train_05nm_${m}_session" \
       code/train_crops.py --crops crops_05nm --model "$m" --split session --no_resize --out results/classify_05nm
done
# control: input encoding (DINOv2 on normalized and binary crops)
step results/classify_1nm_norm/dinov2_session_oof.csv   train_1nm_norm_dinov2_session \
     code/train_crops.py --crops crops_1nm_norm   --model dinov2 --split session
step results/classify_1nm_binary/dinov2_session_oof.csv train_1nm_binary_dinov2_session \
     code/train_crops.py --crops crops_1nm_binary --model dinov2 --split session

# tables, figures, session-level intervals and the session probe
step "" evaluate_classifiers code/evaluate_classifiers.py
step "" session_bootstrap    code/session_bootstrap.py
step results/session_probe/crops_1nm/summary.csv session_probe code/session_probe.py --crops crops_1nm

# ------------------------------------------------------------------ 3. Stage 1: crossing labels and crossing classifier
for px in 1 0.5; do
  t=$([ "$px" = 1 ] && echo 1nm || echo 05nm)
  step "crossings_${t}/metadata.csv" "extract_crossings_${t}" code/extract_crossings.py --nm_per_px "$px"
  step "crossings_${t}/labels.csv"   "make_labels_${t}"       code/make_labels.py --nm_per_px "$px"
done
# trains all three regimes (labels only, teacher, teacher + fine-tuning) for every crossings_* folder
if [ -e results/crossings_1nm/crossing_oof_teacher_finetune.csv ] && \
   [ -e results/crossings_05nm/crossing_oof_teacher_finetune.csv ] && [ "${FORCE:-0}" != 1 ]; then
  echo ">> train_crossings (already done, skipped)"
else
  step "" train_crossings code/train_crossings.py
fi
step "" evaluate_crossings code/evaluate_crossings.py

# ------------------------------------------------------------------ 4. Stages 2 and 3: strand pairing and topology
step results/pairing/exp1_agnostic.csv exp1_pairing  code/exp1_pairing.py
step results/pairing/exp2_topology.csv exp2_topology code/exp2_topology.py

cat <<'EOF'

  ALL DONE. Where the results of the paper are:
    Table 1, S3 Table                   results/splits/crops_1nm/
    Table 2                             results/crossings_1nm/labels_per_class.csv
    Table 3, Fig 4, S1-S2 Tables        results/classifier_performance/
    Table 3 CIs, S1 Fig                 results/classifier_performance_session_boot/
    Table 4                             results/session_probe/crops_1nm/summary.csv
    Table 5                             results/pairing/exp1_agnostic_summary.csv
    Table 6                             results/crossing_performance/
    Table 7, S4 Table                   results/pairing/exp2_topology_summary.csv, exp2_topology_answers.csv
    Table 8                             results/pairing/exp2_topology.csv (column n_crossings)
    S2 Fig                              results/figures/qc/crossings_1nm_sheet.png
EOF