# Reproducing Clark et al. (2019) with attention weights and with norms

This guide runs the experiments of
[What Does BERT Look At?](https://arxiv.org/abs/1906.04341) (Clark et al., 2019) twice:
once on the attention weights α (to check the reproduction), and once on the
norm-based maps ‖α f(x)‖ of
[Attention is Not Only a Weight](https://www.aclweb.org/anthology/2020.emnlp-main.574/) (Kobayashi et al., 2020).
All commands are run from the repository root. `$DATA` is a directory of your choice.

| Experiment | Paper section | Data | Output |
| --- | --- | --- | --- |
| A. General behaviour of heads | Clark §3, §6; Kobayashi §4 | Clark's 1000 Wikipedia segments | `Norm_General_Analysis.ipynb` |
| B. Heads vs. dependency syntax | Clark §4.2 (Table 1) | PTB-SD dev (paper) and/or UD EWT / PUD | `Norm_Syntax_Analysis.ipynb` |
| C. Attention probes | Clark §5 | PTB-SD train/dev (paper) and/or UD EWT train/dev | `Norm_Syntax_Analysis.ipynb` |

## 0. Setup

```bash
conda env create -f environment.yml
conda activate bert-norm-analysis
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"  # should print True
```
If CUDA is not available, reinstall torch for your driver from
https://pytorch.org/get-started/locally/ inside the environment.

Run the tests. The last one compares our norms with Kobayashi et al.'s own
implementation, which is their modified transformers 3.0:
```bash
git clone https://github.com/gorokoba560/norm-analysis-of-transformer $HOME/norm-analysis-of-transformer
pip install sentencepiece sacremoses          # needed only to import their fork
KOBAYASHI_SRC=$HOME/norm-analysis-of-transformer/emnlp2020/transformers/src python -m pytest tests -v
```
All tests should pass, with none skipped.

Model: `--bert-dir bert-base-uncased` downloads the model from the HuggingFace hub. This is a
conversion of the checkpoint the paper used (`uncased_L-12_H-768_A-12`). On a cluster
without internet, download it first (`huggingface-cli download bert-base-uncased --local-dir $DATA/bert-base-uncased`)
and pass that directory instead.

## 1. Data

1. **Clark et al.'s data** (Google Drive link in the [README](README.md#attention-analysis)).
   Put `unlabeled_attn.pkl` in `$DATA/`, and the `glove/` folder (`embeddings.pkl`, `vocab.pkl`) in `$DATA/glove/`.
   The released `depparse` files have dummy labels ("ROOT" everywhere), so they cannot be used for B and C.
2. **Universal Dependencies** (free):
   ```bash
   mkdir -p $DATA/ud
   for f in UD_English-EWT/master/en_ewt-ud-train.conllu UD_English-EWT/master/en_ewt-ud-dev.conllu \
            UD_English-PUD/master/en_pud-ud-test.conllu; do
     wget -P $DATA/ud https://raw.githubusercontent.com/UniversalDependencies/$f
   done
   ```
3. **Penn Treebank with Stanford Dependencies** (optional; needs an LDC licence, which is the only way
   to compare numbers directly with Clark's Table 1). Convert the WSJ constituency trees with Stanford CoreNLP:
   ```bash
   # one file per split, e.g. sections 02-21 -> train, 22 -> dev
   java -cp "stanford-corenlp/*" edu.stanford.nlp.trees.EnglishGrammaticalStructure \
       -treeFile wsj_dev.mrg -basic -conllx > $DATA/ptb/dev.conllx
   # keep punctuation in the output (older CoreNLP versions drop it unless given -keepPunct)
   ```
   The CoNLL-X output has the same columns as CoNLL-U, so `preprocess_conllu.py` reads it.
   Check the split and the converter version against the paper: both change the numbers.

## 2. Experiment A: general analysis (Wikipedia)

```bash
# exact inputs (wordpieces, [CLS]/[SEP]) of the released attention maps
python released_data.py tokens --released $DATA/unlabeled_attn.pkl --outfile $DATA/unlabeled.json

# attention + norms from the same inputs (about a minute on a GPU)
python extract_norms.py --preprocessed-data-file $DATA/unlabeled.json --bert-dir bert-base-uncased \
    --outputs attns,norms,fx_norms --segment_ids zeros
#   -> $DATA/unlabeled_norms.pkl

# reproduction check: our attention vs. the released (TensorFlow) attention
python released_data.py compare --released $DATA/unlabeled_attn.pkl --extracted $DATA/unlabeled_norms.pkl
```
**What to expect from `compare`:** the maps are stored in float16, so differences of about 1e-3 and
row-argmax agreement close to 1 mean the reproduction is exact. Much larger differences most likely come from
segment ids. `extract_attention.py` sets them all to 0 even for two-segment inputs, while standard BERT and
Kobayashi et al. use 1 for the second segment. Re-extract with `--segment_ids pair`, compare again, and use
whichever setting matches.

```bash
# JS divergences between heads (Section 6); uses the GPU
python head_distances.py --attn-data-file $DATA/unlabeled_norms.pkl --key attns --outfile $DATA/head_distances_attns.pkl
python head_distances.py --attn-data-file $DATA/unlabeled_norms.pkl --key norms --outfile $DATA/head_distances_norms.pkl

ATTN_DATA_DIR=$DATA jupyter lab Norm_General_Analysis.ipynb
```
**Reference results:**
* α (Clark, Section 3): over half of the attention in layers 6–10 goes to [SEP].
* Norms (Kobayashi, Section 4): the contribution ‖αf‖ of [SEP] is small, because ‖f([SEP])‖ is small exactly in the heads that attend to it.
  Compare the "norm (raw)" column with their figures, and the "norm (normalized)" column with Clark's thresholds.
* Clark §3.3: entropies are high in the lower layers, and [CLS] attends broadly in the last layer.

Memory: `unlabeled_norms.pkl` with `attns,norms,fx_norms` is about 8 GB on disk and in RAM.
To save memory, drop `fx_norms` or extract `attns` and `norms` to two files (`--outputs`, `--outfile`).

## 3. Experiments B and C: syntax (heads and probes)

The syntax notebook expects `dev_norms.pkl`, `train_norms.pkl` and optionally `dev_random_norms.pkl` in
`$ATTN_DATA_DIR`. Make one directory per dataset:

```bash
# --- UD English EWT (train/dev) ---
D=$DATA/ewt; mkdir -p $D
python preprocess_conllu.py --conllu $DATA/ud/en_ewt-ud-train.conllu --outfile $D/train.json
python preprocess_conllu.py --conllu $DATA/ud/en_ewt-ud-dev.conllu   --outfile $D/dev.json
for split in train dev; do
  python extract_norms.py --preprocessed-data-file $D/$split.json --bert-dir bert-base-uncased --word_level
done
# control: same architecture with random weights
python extract_norms.py --preprocessed-data-file $D/dev.json --bert-dir bert-base-uncased --word_level \
    --random_init --seed 0 --outfile $D/dev_random_norms.pkl
ATTN_DATA_DIR=$D jupyter lab Norm_Syntax_Analysis.ipynb
```
* **PTB-SD:** the same commands with `--conllu $DATA/ptb/train.conllx` / `dev.conllx` and `D=$DATA/ptb`.
* **PUD** (for comparison with Htut et al. and Limisiewicz et al.): PUD has only a test split, so use it as `dev.json` and EWT's `train_norms.pkl` for the probes.
* `--strip_subtypes` maps `nsubj:pass` to `nsubj` and similar, which matches how most of the UD literature reports results.
* Sentences longer than 128 wordpieces are dropped, as in the original. The script prints how many.

**Reference results, PTB-SD only.** Best head vs. best offset baseline (Clark Table 1, dev):

| relation | best head | baseline |
| --- | --- | --- |
| nsubj | 58.5 | 45.5 |
| dobj | 86.8 | 40.0 |
| amod | 75.6 | 68.3 |
| advmod | 48.8 | 40.2 |
| det | 94.3 | 51.7 |

Probes (Clark Section 5, UAS): attention-only 61, attention-and-words 77, words-and-distances baseline 58.
Right-branching baseline 26.3, and the best single head used as a parser 34.5.
UD labels differ from SD labels (e.g. `obj` vs `dobj`, `case` instead of `prep`/`pobj`), so UD numbers are not directly comparable.

**What the notebook adds for the norm analysis:**
* The best-head table for α and ‖αf‖ (and for the random-init control) side by side.
* A held-out variant: heads are selected on one half of the data and scored on the other. The paper selects and scores on the same data, which is optimistic.
* Per-head accuracy differences, ‖αf‖ minus α, for the frequent relations.
* The attention-only probe on α, on raw ‖αf‖ and on row-normalized ‖αf‖, over 3 seeds, with the learned weight per layer and head.
* The GloVe probes, if `$D/glove/` exists. Copy or symlink Clark's `glove/` there.

## 4. Choices that change the numbers

| Choice | Default | Alternative | Why it matters |
| --- | --- | --- | --- |
| `--segment_ids` | `zeros` (Clark's code) | `pair` (standard BERT, Kobayashi) | the inputs differ for two-segment data |
| `--word_norm_mode` | `vector`: ‖merge(αf)‖ | `sum`: merge(‖αf‖) | `sum` is an upper bound that ignores cancellation between the pieces of a word |
| Row normalization of ‖αf‖ | raw and normalized both reported | | argmax is unaffected; averages, entropies, JS and probes are affected |
| `fix_root_alignment` (GloVe probes) | `False` (reproduces the original code) | `True` | in the original code the word-pair embedding for candidate head j is that of word j+1 (see `analysis_utils.Probe`) |
| Probe seed | 0, 1, 2 | | the original reports one run |
| `head_distances.py --device` | GPU/CPU torch (float32) | `numpy` (original, float16 for the paper's data) | small numerical differences |

## 5. Why a HuggingFace model instead of Kobayashi's modified transformers

Kobayashi et al. patched transformers 3.0 (`BertNormOutput`) to return ‖f(x)‖, ‖αf(x)‖ and ‖Σαf(x)‖.
Those quantities depend only on the attention weights α, the value vectors x W<sub>V</sub> + b<sub>V</sub>, and the output
projection W<sub>O</sub>. An unmodified model exposes all three: α through `output_attentions`, the value vectors through a
forward hook on the value layer, and W<sub>O</sub> as a weight. `norm_utils.py` recomputes the norms from these.

This approach avoids installing a 2020 fork of transformers alongside a modern PyTorch and CUDA.
It also avoids the memory cost of the fork, which builds a `[batch, heads, n, n, 768]` tensor; `norm_utils.py`
computes the same norms exactly in the 64-dimensional value space.
`tests/test_norms.py` checks that the result matches their code number for number.
The price is a dependence on transformers' internal module names (`encoder.layer[i].attention.self.value`).
It is tested with transformers 4.46.3 and 5.17.0.
