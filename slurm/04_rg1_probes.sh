#!/bin/bash
# RG1: parent probes over BERT attention (controls, linear vs. MLP, weights,
# ablations, single-layer and relation-conditioned probes). Needs the outputs of
# 02_extract_syntax.sh (including train_random_norms.pkl).
#SBATCH --job-name=rg1_probes
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/config.sh"

mkdir -p $RESULTS
ATTN_DATA_DIR=$DATA jupyter nbconvert --to notebook --execute RG1_Probe_Analysis.ipynb \
    --ExecutePreprocessor.timeout=-1 --output-dir $RESULTS --output rg1_probes.ipynb
