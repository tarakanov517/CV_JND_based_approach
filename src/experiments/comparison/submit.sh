#!/bin/bash
# Run from the directory containing main.py, with the Python environment activated.
set -euo pipefail
python3 -m compileall -q main_fixed.py main_adaptive.py main_conditional.py evaluate_comparison.py experiments/comparison
python3 -m unittest discover -s experiments/comparison -p 'test_*.py'
mkdir -p logs
ROOT="${1:-results/comparison_v1}"
WEIGHTS="${2:-weights.pth}"
read -r TRAIN_COUNT EVAL_COUNT < <(python3 -c 'import sys; sys.path.insert(0,"experiments/comparison"); from jobs import training_jobs,evaluation_jobs; print(len(training_jobs()),len(evaluation_jobs()))')
BASE_JOB=$(sbatch --parsable --array=0-2%3 experiments/comparison/run.sh baseline --root "$ROOT" --weights "$WEIGHTS")
BASE_JOB="${BASE_JOB%%;*}"
TRAIN_JOB=$(sbatch --parsable --dependency="afterok:$BASE_JOB" --array="0-$((TRAIN_COUNT-1))%3" experiments/comparison/run.sh train --root "$ROOT")
TRAIN_JOB="${TRAIN_JOB%%;*}"
EVAL_JOB=$(sbatch --parsable --dependency="afterok:$TRAIN_JOB" --array="0-$((EVAL_COUNT-1))%3" experiments/comparison/run.sh evaluate --root "$ROOT")
echo "baseline=$BASE_JOB train=$TRAIN_JOB evaluate=$EVAL_JOB"
