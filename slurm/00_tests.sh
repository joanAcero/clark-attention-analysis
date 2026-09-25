#!/bin/bash
#SBATCH --job-name=norm_tests
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/config.sh"

python -m pytest tests -v -rs
