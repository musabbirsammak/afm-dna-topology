#!/usr/bin/env bash
# Train and evaluate the whole-molecule classifiers on every crops folder made by prepare_crops.sh.
# ResNet-18 and ConvNeXt-Tiny need a GPU (RunPod). Run:  bash train_crops.sh
# Results: results/classify_<name>/<model>_<split>_oof.csv   Logs: results/logs/<step>.txt
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results/logs
r () { name=$1; shift; echo ">> $name"; python "$@" 2>&1 | tee "results/logs/$name.txt"; }

# main crops (1 nm, height): all models, both splits, plus the flip control
for m in dinov2 resnet18 convnext_tiny; do
  for s in session image; do
    r "train_1nm_${m}_${s}" code/train_crops.py --crops crops_1nm --model "$m" --split "$s"
  done
done
r train_1nm_resnet18_session_flip code/train_crops.py --crops crops_1nm --model resnet18 --split session --flip

# 0.5 nm crops: the fine-tuned models on the honest split
for m in resnet18 convnext_tiny; do
  r "train_05nm_${m}_session" code/train_crops.py --crops crops_05nm --model "$m" --split session
done

# brightness controls (1 nm): DINOv2 on the honest split
r train_1nm_norm_dinov2_session   code/train_crops.py --crops crops_1nm_norm   --model dinov2 --split session
r train_1nm_binary_dinov2_session code/train_crops.py --crops crops_1nm_binary --model dinov2 --split session

# numbers for the tables
r evaluate code/evaluate.py --oof results/classify_1nm/*_oof.csv results/classify_05nm/*_oof.csv \
                                  results/classify_1nm_norm/*_oof.csv results/classify_1nm_binary/*_oof.csv

echo "ALL DONE"