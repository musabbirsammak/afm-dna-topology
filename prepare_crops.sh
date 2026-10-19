#!/usr/bin/env bash
# Cut the molecule crops, make the train/test splits and the contact sheets.
# Run from anywhere:  bash prepare_crops.sh
# Logs: results/logs/<step>.txt
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results/logs
r () { name=$1; shift; echo ">> $name"; python "$@" 2>&1 | tee "results/logs/$name.txt"; }

r crops_1nm         code/extract_crops.py --nm_per_px 1.0 --size 256 --out crops_1nm
r crops_05nm        code/extract_crops.py --nm_per_px 0.5 --size 512 --out crops_05nm
r crops_1nm_norm    code/extract_crops.py --nm_per_px 1.0 --size 256 --mode norm   --out crops_1nm_norm
r crops_1nm_binary  code/extract_crops.py --nm_per_px 1.0 --size 256 --mode binary --out crops_1nm_binary

r make_splits       code/make_splits.py
r contact_sheets    code/contact_sheet.py

echo "ALL DONE"