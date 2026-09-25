#!/bin/bash
# Experiments B and C: word-level attention + norms on UD EWT (train/dev) and PUD,
# plus the random-initialization control. Files that already exist are skipped,
# so rerunning only extracts what is missing (delete a file to redo it).
#SBATCH --job-name=extract_syntax
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/config.sh"

extract() {  # extract <input json> <output pkl> [extra args]
  local in=$1 out=$2; shift 2
  if [ -f $out ]; then echo "exists, skipping: $out"; return; fi
  python extract_norms.py --preprocessed-data-file $in --bert-dir $BERT --word_level \
      --outfile $out "$@"
}

for split in train dev; do
  extract $DATA/ewt/$split.json $DATA/ewt/${split}_norms.pkl
  # control: same architecture with random weights
  extract $DATA/ewt/$split.json $DATA/ewt/${split}_random_norms.pkl --random_init --seed 0
done

# PUD (test only): evaluated as "dev", probes trained on EWT train
extract $DATA/pud/dev.json $DATA/pud/dev_norms.pkl
extract $DATA/pud/dev.json $DATA/pud/dev_random_norms.pkl --random_init --seed 0
ln -sfn $DATA/ewt/train_norms.pkl $DATA/pud/train_norms.pkl
ln -sfn $DATA/ewt/train_random_norms.pkl $DATA/pud/train_random_norms.pkl
