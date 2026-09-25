#!/bin/bash
# Runs the analysis notebooks headless; executed copies (with figures) go to $RESULTS.
# Usage: sbatch -A <project> slurm/03_notebooks.sh [general|ewt|pud|all]
#SBATCH --job-name=notebooks
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/config.sh"

WHAT=${1:-all}
run() {  # run <notebook> <data dir> <output name>
  ATTN_DATA_DIR=$2 jupyter nbconvert --to notebook --execute $1 \
      --ExecutePreprocessor.timeout=-1 --output-dir $RESULTS --output $3
}
mkdir -p $RESULTS
if [ $WHAT = general ] || [ $WHAT = all ]; then
  run Norm_General_Analysis.ipynb $DATA/wiki general_wiki.ipynb
fi
if [ $WHAT = ewt ] || [ $WHAT = all ]; then
  run Norm_Syntax_Analysis.ipynb $DATA/ewt syntax_ewt.ipynb
fi
if [ $WHAT = pud ] || [ $WHAT = all ]; then
  run Norm_Syntax_Analysis.ipynb $DATA/pud syntax_pud.ipynb
fi
