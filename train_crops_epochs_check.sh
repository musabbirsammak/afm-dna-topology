#!/usr/bin/env bash
# Epoch sensitivity check: does training for fewer or more epochs change the result?
# ResNet-18 and ConvNeXt-Tiny on crops_1nm, session split, at 15 and 60 epochs (30 is the main run).
# Needs a GPU (RunPod). Run:  bash train_epochs_check.sh
# Results: results/classify_1nm_ep15/ and results/classify_1nm_ep60/   Logs: results/logs/<step>.txt
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results/logs
r () { name=$1; shift; echo ">> $name"; python "$@" 2>&1 | tee "results/logs/$name.txt"; }

for e in 15 60; do
  for m in resnet18 convnext_tiny; do
    r "train_1nm_${m}_session_ep${e}" code/train_crops.py --crops crops_1nm --model "$m" --split session \
                                      --epochs "$e" --out "results/classify_1nm_ep${e}"
  done
done

r evaluate_epochs code/evaluate.py --oof results/classify_1nm_ep15/*_oof.csv results/classify_1nm_ep60/*_oof.csv

echo "ALL DONE"