#!/bin/bash
# Experiments B and C: word-level attention + norms on UD EWT (train/dev) and PUD,
# plus the random-initialization control.
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

for split in train dev; do
  python extract_norms.py --preprocessed-data-file $DATA/ewt/$split.json --bert-dir $BERT --word_level
done
python extract_norms.py --preprocessed-data-file $DATA/ewt/dev.json --bert-dir $BERT --word_level \
    --random_init --seed 0 --outfile $DATA/ewt/dev_random_norms.pkl

# PUD (test only): evaluated as "dev", probes trained on EWT train
python extract_norms.py --preprocessed-data-file $DATA/pud/dev.json --bert-dir $BERT --word_level
python extract_norms.py --preprocessed-data-file $DATA/pud/dev.json --bert-dir $BERT --word_level \
    --random_init --seed 0 --outfile $DATA/pud/dev_random_norms.pkl
ln -sfn $DATA/ewt/train_norms.pkl $DATA/pud/train_norms.pkl
