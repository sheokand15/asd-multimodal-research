#!/usr/bin/env bash
# FULL_TRAINING: all experiments sequentially (memory-safe on a 14 GB machine).
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
CFG=configs/full_training.yaml
L=outputs/full/logs
mkdir -p "$L"
step() { echo "[$(date +%T)] START $1"; shift; "$@"; echo "[$(date +%T)] EXIT $?"; }
step questionnaire $PY scripts/run_questionnaire.py --config $CFG > $L/run_questionnaire_stdout.log 2>&1
step move4as       $PY scripts/run_move4as.py --config $CFG > $L/run_move4as_stdout.log 2>&1
step sensitivity   $PY scripts/run_move4as_sensitivity.py $CFG > $L/run_move4as_sensitivity_stdout.log 2>&1
step mmasd         $PY scripts/run_mmasd.py --config $CFG > $L/run_mmasd_stdout.log 2>&1
step report        $PY scripts/make_report.py $CFG > $L/make_report.log 2>&1
