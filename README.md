# BERT Attention Analysis

This repository contains code for [What Does BERT Look At? An Analysis of BERT's Attention](https://arxiv.org/abs/1906.04341).
It includes code for getting attention maps from BERT and writing them to disk, analyzing BERT's attention in general (sections 3 and 6 of the paper), and comparing its attention to dependency syntax (sections 4.2 and 5).
We will add the code for the coreference resolution analysis (section 4.3 of the paper) soon!

## Requirements
For extracting attention maps from text:
* [Tensorflow](https://www.tensorflow.org/)
* [NumPy](http://www.numpy.org/)

Additional requirements for the attention analysis:
* [Jupyter](https://jupyter.org/https://jupyter.org/)
* [MatplotLib](https://matplotlib.org/)
* [seaborn](https://seaborn.pydata.org/index.html)
* [scikit-learn](https://scikit-learn.org/)

## Attention Analysis
`Syntax_Analysis.ipynb` and `General_Analysis.ipynb`
contain code for analyzing BERT's attention, including reproducing the figures and tables in the paper.

You can download the data needed to run the notebooks (including BERT attention maps on Wikipedia
and the Penn Treebank) from [here](https://drive.google.com/open?id=1DEIBQIl0Q0az5ZuLoy4_lYabIfLSKBg-). However, note that the Penn Treebank annotations are not
freely available, so the Penn Treebank data only includes dummy labels.
If you want to run the analysis on your own data, you can use the scripts described below to extract BERT attention maps.

## Extracting BERT Attention Maps
We provide a script for running BERT over text and writing the resulting
attention maps to disk.
The input data should be a [JSON](https://www.json.org/) file containing a
list of dicts, each one corresponding to a single example to be passed in
to BERT. Each dict must contain exactly one of the following fields:
* `"text"`: A string.
* `"words"`: A list of strings. Needed if you want word-level rather than
token-level attention.
* `"tokens"`: A list of strings corresponding to BERT wordpiece tokenization.

If the present field is "tokens," the script expects [CLS]/[SEP] tokens
to be already added; otherwise it adds these tokens to the
beginning/end of the text automatically.
Note that if an example is longer than `max_sequence_length` tokens
after BERT wordpiece tokenization, attention maps will not be extracted for it.
Attention extraction adds two additional fields to each dict:
* `"attns"`: A numpy array of size [num_layers, heads_per_layer, sequence_length,
sequence_length] containing attention weights.
* `"tokens"`: If `"tokens"` was not already provided for the example, the
BERT-wordpiece-tokenized text (list of strings).

Other fields already in the feature dicts will be preserved. For example
if each dict has a `tags` key containing POS tags, they will stay
in the data after attention extraction so they can be used when
analyzing the data.

Attention extraction is run with
```
python extract_attention.py --preprocessed_data_file <path-to-your-data> --bert_dir <directory-containing-BERT-model>
```
The following optional arguments can also be added:
* `--max_sequence_length`: Maximum input sequence length after tokenization (default is 128).
* `--batch_size`: Batch size when running BERT over examples (default is 16).
* `--debug`: Use a tiny BERT model for fast debugging.
* `--cased`: Do not lowercase the input text.
* `--word_level`: Compute word-level instead of token-level attention (see Section 4.1 of the paper).

The feature dicts with added attention maps (numpy arrays with shape [n_layers, n_heads_per_layer, n_tokens, n_tokens]) are written to `<path-to-your-data>_attn.pkl`


## Pre-processing Scripts
We include two pre-processing scripts for going from a raw data file to
JSON that can be supplied to ``attention_extractor.py``.

`preprocess_unlabeled.py` does BERT-pre-training-style preprocessing for unlabeled text
(i.e, taking two consecutive text spans, truncating them so they are at most
`max_sequence_length` tokens, and adding [CLS]/[SEP] tokens).
Each line of the input data file
should be one sentence. Documents should be separated by empty lines.
Example usage:
```
python preprocess_unlabeled.py --data-file $ATTN_DATA_DIR/unlabeled.txt --bert-dir $ATTN_DATA_DIR/uncased_L-12_H-768_A-12
```
will create the file `$ATTN_DATA_DIR/unlabeled.json` containing pre-processed data.
After pre-processing, you can run `extract_attention.py` to get attention maps, e.g.,
```
python extract_attention.py --preprocessed-data-file $ATTN_DATA_DIR/unlabeled.json --bert-dir $ATTN_DATA_DIR/uncased_L-12_H-768_A-12
```


`preprocess_depparse.py` pre-processes dependency parsing data.
Dependency parsing data should consist of two files `train.txt` and `dev.txt` under a common directory.
Each line in the files should contain a word followed by a space followed by <index_of_head>-<dependency_label>
(e.g., 0-root). Examples should be separated by empty lines. Example usage:
```
python preprocess_depparse.py --data-dir $ATTN_DATA_DIR/depparse
```

After pre-processing, you can run `extract_attention.py` to get attention maps, e.g.,
```
python extract_attention.py --preprocessed-data-file $ATTN_DATA_DIR/depparse/dev.json --bert-dir $ATTN_DATA_DIR/uncased_L-12_H-768_A-12 --word_level
```
## Computing Distances Between Attention Heads
`head_distances.py` computes the average Jenson-Shannon divergence between the attention weights of all pairs of attention heads and writes the results to disk as a numpy array of shape [n_heads, n_heads]. These distances can be used to cluster BERT's attention heads (see Section 6 and Figure 6 of the paper; code for doing this clustering is in `General_Analysis.ipynb`). Example usage (requires that attention maps have already been extracted):
```
python head_distances.py --attn-data-file $ATTN_DATA_DIR/unlabeled_attn.pkl --outfile $ATTN_DATA_DIR/head_distances.pkl
```

## Norm-Based Analysis (Kobayashi et al., 2020)

**Step-by-step instructions to reproduce the paper and run the norm-based experiments: [REPRODUCE.md](REPRODUCE.md).**
**Results and conclusions of the norm-based analysis: [RESULTS_RG2.md](RESULTS_RG2.md).** RG1 (analysis of the attention probe): `RG1_Probe_Analysis.ipynb`, `probes.py`.
Environment: `conda env create -f environment.yml` (or `pip install -r requirements.txt`).

This fork adds a PyTorch pipeline that reproduces the analyses above with the
norm-based maps of [Kobayashi et al. (2020), Attention is Not Only a Weight: Analyzing Transformers with Vector Norms](https://www.aclweb.org/anthology/2020.emnlp-main.574/).
Head *h*'s output at position *i* is a sum of vectors α<sub>ij</sub> f(x<sub>j</sub>), with
f(x) = (x W<sub>V</sub> + b<sub>V</sub>) W<sub>O</sub>; the maps ‖α<sub>ij</sub> f(x<sub>j</sub>)‖
are analysed in place of α<sub>ij</sub>.

Requirements: see `requirements.txt` (Python 3, [PyTorch](https://pytorch.org/),
[transformers](https://github.com/huggingface/transformers) 4.46–5.x, NumPy, Jupyter, Matplotlib, seaborn, scikit-learn).
TensorFlow is no longer needed by `utils.py` and `bert/tokenization.py`.

| File | Purpose |
| --- | --- |
| `extract_norms.py` | PyTorch counterpart of `extract_attention.py`: writes `attns`, `norms` = ‖αf(x)‖, and optionally `summed_norms` = ‖Σ<sub>h</sub> αf(x)‖ and `fx_norms` = ‖f(x)‖, at token or word level |
| `norm_utils.py` | the norm computation (same quantities as `BertNormOutput` in [Kobayashi et al.'s code](https://github.com/gorokoba560/norm-analysis-of-transformer/tree/master/emnlp2020)) and word-level conversion |
| `preprocess_conllu.py` | Universal Dependencies (CoNLL-U) → JSON for the syntax analysis |
| `analysis_utils.py` | Python 3 / PyTorch port of the notebooks' analysis code (token statistics, entropies, per-head syntax accuracy, baselines, probes) for any map |
| `Norm_General_Analysis.ipynb`, `Norm_Syntax_Analysis.ipynb` | the paper's analyses run on α and on ‖αf(x)‖ side by side (generated by `make_notebooks.py`) |
| `head_distances.py --key norms` | JS divergences between heads for norm-based maps (rows normalized first); now vectorized in PyTorch (`--device numpy` for the original loop) |
| `released_data.py` | recovers the inputs of the paper's released attention maps and compares them with re-extracted maps |
| `tests/` | `python -m pytest tests`; set `KOBAYASHI_SRC` to also compare against Kobayashi et al.'s implementation (see `tests/test_norms.py`) |

Example (`--bert-dir` is a HuggingFace model name or directory):
```
# General analysis (token level)
python preprocess_unlabeled.py --data-file $DATA/unlabeled.txt --bert-dir bert-base-uncased
python extract_norms.py --preprocessed-data-file $DATA/unlabeled.json --bert-dir bert-base-uncased --outputs attns,norms,fx_norms
python head_distances.py --attn-data-file $DATA/unlabeled_norms.pkl --key attns --outfile $DATA/head_distances_attns.pkl
python head_distances.py --attn-data-file $DATA/unlabeled_norms.pkl --key norms --outfile $DATA/head_distances_norms.pkl

# Syntax (word level), e.g. on UD English EWT
python preprocess_conllu.py --conllu en_ewt-ud-train.conllu --outfile $DATA/train.json
python preprocess_conllu.py --conllu en_ewt-ud-dev.conllu --outfile $DATA/dev.json
python extract_norms.py --preprocessed-data-file $DATA/train.json --bert-dir bert-base-uncased --word_level
python extract_norms.py --preprocessed-data-file $DATA/dev.json --bert-dir bert-base-uncased --word_level
python extract_norms.py --preprocessed-data-file $DATA/dev.json --bert-dir bert-base-uncased --word_level \
    --random_init --outfile $DATA/dev_random_norms.pkl   # control
ATTN_DATA_DIR=$DATA jupyter notebook Norm_Syntax_Analysis.ipynb
```

Things to be aware of:
* **Word-level norms.** Clark et al. merge the pieces of a word by summing columns and averaging rows.
  `--word_norm_mode vector` (default) applies this merge to the vectors α<sub>ij</sub> f(x<sub>j</sub>) and then
  takes the norm; `--word_norm_mode sum` applies it to the token-level norms, which is an upper bound (triangle inequality).
* **Scale.** Rows of ‖αf(x)‖ do not sum to one. Argmax-based per-head evaluation is unaffected, but averages,
  entropies, JS divergences and probes are; the notebooks report raw and row-normalized variants.
* **Segment ids.** `extract_attention.py` feeds all-zero segment ids even for the two-segment Wikipedia inputs, whereas
  Kobayashi et al. use standard sentence-pair ids. `extract_norms.py` defaults to the former (`--segment_ids zeros`) to
  match the paper's data; use `--segment_ids pair` for the latter.
* The original checkpoint (`uncased_L-12_H-768_A-12`) corresponds to `bert-base-uncased` on the HuggingFace hub.

## Citation
If you find the code or data helpful, please cite the original paper:

```
@inproceedings{clark2019what,
  title = {What Does BERT Look At? An Analysis of BERT's Attention},
  author = {Kevin Clark and Urvashi Khandelwal and Omer Levy and Christopher D. Manning},
  booktitle = {BlackBoxNLP@ACL},
  year = {2019}
}
```

## Contact
[Kevin Clark](https://cs.stanford.edu/~kevclark/) (@clarkkev).
