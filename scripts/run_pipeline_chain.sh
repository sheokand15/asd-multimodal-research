#!/usr/bin/env bash
# Sequential execution (memory-safe on a 14 GB machine). Each step logs to outputs/logs/.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
step() { echo "[$(date +%T)] START $1"; shift; "$@"; echo "[$(date +%T)] EXIT $?"; }
step build_mmasd   $PY scripts/build_mmasd.py --workers 4 > outputs/logs/build_mmasd.log 2>&1
step move4as_part2 $PY scripts/run_move4as.py --only motion_mae late_fusion_deep motion_feat_lr_all eeg_feat_lr_all > outputs/logs/run_move4as_stdout_part2.log 2>&1
step sensitivity   $PY scripts/run_move4as_sensitivity.py > outputs/logs/run_move4as_sensitivity_stdout.log 2>&1
step audit         $PY scripts/audit_datasets.py > outputs/logs/audit.log 2>&1
