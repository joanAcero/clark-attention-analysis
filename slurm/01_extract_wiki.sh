#!/bin/bash
# Experiment A: attention + norms on the paper's 1000 Wikipedia segments,
# reproduction check against the released maps, and head distances.
#SBATCH --job-name=extract_wiki
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/config.sh"

for f in $DATA/clark/unlabeled_attn.pkl $DATA/wiki/unlabeled.json; do
  if [ ! -f $f ]; then
    echo "ERROR: $f not found. Download Clark et al.'s data into $DATA/clark/ and rerun" \
         "'bash slurm/prepare_data.sh' (see slurm/README.md)." >&2
    exit 1
  fi
done

SEGMENT_IDS=${SEGMENT_IDS:-zeros}   # override with: sbatch --export=ALL,SEGMENT_IDS=pair ...
W=$DATA/wiki
python extract_norms.py --preprocessed-data-file $W/unlabeled.json --bert-dir $BERT \
    --outputs attns,norms,fx_norms --segment_ids $SEGMENT_IDS --outfile $W/unlabeled_norms.pkl

echo "=== released (TensorFlow) vs. re-extracted attention, segment_ids=$SEGMENT_IDS ==="
python released_data.py compare --released $DATA/clark/unlabeled_attn.pkl --extracted $W/unlabeled_norms.pkl

python head_distances.py --attn-data-file $W/unlabeled_norms.pkl --key attns --outfile $W/head_distances_attns.pkl
python head_distances.py --attn-data-file $W/unlabeled_norms.pkl --key norms --outfile $W/head_distances_norms.pkl
