#!/bin/bash
# Downloads and preprocessing. Light work (downloads + JSON conversion): run it on
# the login node with `bash slurm/prepare_data.sh`, not with sbatch, since compute
# nodes may not have internet access.
set -euo pipefail
source "$(dirname "$0")/config.sh"

mkdir -p $DATA/ud $DATA/ewt $DATA/pud $DATA/wiki $DATA/models $RESULTS

# BERT (same weights as the paper's uncased_L-12_H-768_A-12 checkpoint)
# (google-bert/bert-base-uncased is the current name of bert-base-uncased; plain HTTP
# download because the Xet backend can fail on the legacy name / behind proxies)
if [ ! -f $BERT/model.safetensors ]; then
  HF_HUB_DISABLE_XET=1 python -c "from huggingface_hub import snapshot_download; snapshot_download('google-bert/bert-base-uncased', local_dir='$BERT', allow_patterns=['config.json', 'vocab.txt', 'model.safetensors'])"
fi
ls -la $BERT

# Universal Dependencies
for f in UD_English-EWT/master/en_ewt-ud-train.conllu UD_English-EWT/master/en_ewt-ud-dev.conllu \
         UD_English-PUD/master/en_pud-ud-test.conllu; do
  [ -f $DATA/ud/$(basename $f) ] || wget -q -P $DATA/ud https://raw.githubusercontent.com/UniversalDependencies/$f
done
python preprocess_conllu.py --conllu $DATA/ud/en_ewt-ud-train.conllu --outfile $DATA/ewt/train.json
python preprocess_conllu.py --conllu $DATA/ud/en_ewt-ud-dev.conllu   --outfile $DATA/ewt/dev.json
python preprocess_conllu.py --conllu $DATA/ud/en_pud-ud-test.conllu  --outfile $DATA/pud/dev.json

# Clark et al.'s released data: download it by hand (see README) into $DATA/clark/
if [ -f $DATA/clark/unlabeled_attn.pkl ]; then
  python released_data.py tokens --released $DATA/clark/unlabeled_attn.pkl --outfile $DATA/wiki/unlabeled.json
else
  echo "WARNING: $DATA/clark/unlabeled_attn.pkl not found; experiment A (Wikipedia) needs it"
fi
if [ -d $DATA/clark/glove ]; then
  ln -sfn $DATA/clark/glove $DATA/ewt/glove
  ln -sfn $DATA/clark/glove $DATA/pud/glove
fi
echo "done"
