"""Generates Norm_General_Analysis.ipynb and Norm_Syntax_Analysis.ipynb
(Python 3 versions of the paper's notebooks that compare attention weights
with the norm-based maps of Kobayashi et al., 2020).

  python make_notebooks.py
"""

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

KERNEL = {"display_name": "Python 3", "language": "python", "name": "python3"}


def notebook(cells):
  nb = new_notebook(cells=[
      new_markdown_cell(c[3:]) if c.startswith("md:") else new_code_cell(c)
      for c in cells])
  nb.metadata["kernelspec"] = KERNEL
  return nb


GENERAL = [
    """md:# General Analysis: Attention Weights vs. Norm-Based Maps

Python 3 version of `General_Analysis.ipynb` (Sections 3 and 6 of
[Clark et al., 2019](https://arxiv.org/abs/1906.04341)) that runs every analysis
on the attention weights $\\alpha$ **and** on the norm-based maps
$\\|\\alpha f(x)\\|$ of [Kobayashi et al., 2020](https://www.aclweb.org/anthology/2020.emnlp-main.574/).

Data: token-level maps written by `extract_norms.py`, e.g.
```
python preprocess_unlabeled.py --data-file $DATA/unlabeled.txt --bert-dir bert-base-uncased
python extract_norms.py --preprocessed-data-file $DATA/unlabeled.json --bert-dir bert-base-uncased \\
    --outputs attns,norms,fx_norms
python head_distances.py --attn-data-file $DATA/unlabeled_norms.pkl --key attns --outfile $DATA/head_distances_attns.pkl
python head_distances.py --attn-data-file $DATA/unlabeled_norms.pkl --key norms --outfile $DATA/head_distances_norms.pkl
```

**Scale.** Rows of $\\alpha$ sum to one; rows of $\\|\\alpha f(x)\\|$ do not. Each
statistic is therefore reported for the norms both *raw* (the quantity plotted by
Kobayashi et al.) and *row-normalized* (the share of a token's summed norm going to
each position, on the same scale as $\\alpha$ and usable with Clark et al.'s
thresholds). Entropies and JS divergences are only defined for normalized rows.""",
    """import os
import sys

import numpy as np
import seaborn as sns
import sklearn.manifold
from matplotlib import cm
from matplotlib import pyplot as plt

import analysis_utils as au
import utils

sns.set_style("darkgrid")

DATA_DIR = os.environ.get("ATTN_DATA_DIR", "./data")
DATA_FILE = os.path.join(DATA_DIR, "unlabeled_norms.pkl")
HEAD_DISTANCES = {k: os.path.join(DATA_DIR, "head_distances_%s.pkl" % k)
                  for k in ["attns", "norms"]}""",
    """# list of dicts with "tokens", "attns" [layers, heads, n, n],
# "norms" [layers, heads, n, n] and optionally "fx_norms" [layers, heads, n]
data = utils.load_pickle(DATA_FILE)
n_layers, n_heads = data[0]["attns"].shape[:2]
print(len(data), "examples;", n_layers, "layers x", n_heads, "heads")""",
    """md:### Average attention to particular tokens/positions (Sections 3.1 and 3.2)""",
    """# (map name, key, normalize)
MAPS = [("attention", "attns", False),
        ("norm (raw)", "norms", False),
        ("norm (normalized)", "norms", True)]
avg_stats = {name: au.token_stats(data, key, normalize)
             for name, key, normalize in MAPS}""",
    """BLACK, GREEN, SEA, BLUE = "k", "#59d98e", "#159d82", "#3498db"
PURPLE, GREY, RED, ORANGE = "#9b59b6", "#95a5a6", "#e74c3c", "#f39c12"


def get_data_points(head_data):
  xs, ys, avgs = [], [], []
  for layer in range(head_data.shape[0]):
    for head in range(head_data.shape[1]):
      ys.append(head_data[layer, head])
      xs.append(1 + layer)
    avgs.append(head_data[layer].mean())
  return xs, ys, avgs


def add_line(stats, key, ax, color, label, plot_avgs=True):
  xs, ys, avgs = get_data_points(stats[key])
  ax.scatter(xs, ys, s=12, label=label, color=color)
  if plot_avgs:
    ax.plot(1 + np.arange(len(avgs)), avgs, color=color)
  ax.legend(loc="best")
  ax.set_xlabel("Layer")


# Figure 2 of Clark et al. (average attention per head), one column per map
fig, axes = plt.subplots(3, len(MAPS), figsize=(5 * len(MAPS), 10))
for col, (name, _, _) in enumerate(MAPS):
  stats = avg_stats[name]
  for key, color, label in [("cls", RED, "[CLS]"), ("sep", BLUE, "[SEP]"),
                            ("punct", PURPLE, ". or ,")]:
    add_line(stats, key, axes[0, col], color, label)
  for key, color, label in [("rest_sep", BLUE, "other -> [SEP]"),
                            ("sep_sep", GREEN, "[SEP] -> [SEP]")]:
    add_line(stats, key, axes[1, col], color, label)
  for key, color, label in [("left", RED, "next token"),
                            ("right", BLUE, "prev token"),
                            ("self", PURPLE, "current token")]:
    add_line(stats, key, axes[2, col], color, label, plot_avgs=False)
  axes[0, col].set_title(name)
  for row in range(3):
    axes[row, col].set_ylabel("Avg. " + name)
plt.tight_layout()
plt.show()""",
    """md:### $\\|f(x)\\|$ by token type (Kobayashi et al., Section 4)
Kobayashi et al.'s explanation of the [SEP] result: the heads that put large
$\\alpha$ on [SEP] have small $\\|f(\\text{[SEP]})\\|$. Requires `fx_norms`
(`--outputs attns,norms,fx_norms`).""",
    """if "fx_norms" in data[0]:
  groups = {"[CLS]": [], "[SEP]": [], ". or ,": [], "other": []}
  fx = {g: np.zeros((n_layers, n_heads)) for g in groups}
  counts = {g: 0 for g in groups}
  for doc in data:
    f = np.asarray(doc["fx_norms"], dtype=np.float64)
    for position, token in enumerate(doc["tokens"]):
      g = (token if token in ("[CLS]", "[SEP]") else
           ". or ," if token in (".", ",") else "other")
      fx[g] += f[:, :, position]
      counts[g] += 1
  plt.figure(figsize=(5, 4))
  for (g, total), color in zip(fx.items(), [RED, BLUE, PURPLE, GREY]):
    xs, ys, avgs = get_data_points(total / max(counts[g], 1))
    plt.scatter(xs, ys, s=8, color=color, label=g)
    plt.plot(1 + np.arange(n_layers), avgs, color=color)
  plt.xlabel("Layer")
  plt.ylabel("Avg. ||f(x)||")
  plt.legend()
  plt.show()
else:
  print("no fx_norms in the data")""",
    """md:### Entropies (Section 3.3)""",
    """entropy_stats = {key: au.entropy_stats(data, key)
                 for key in ["attns", "norms"]}

fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
for col, key in enumerate(["attns", "norms"]):
  uniform, entropies, entropies_cls = entropy_stats[key]
  for row, (values, label, color) in enumerate([
      (entropies, "heads", BLUE), (entropies_cls, "heads from [CLS]", RED)]):
    ax = axes[row, col]
    xs, es, avgs = get_data_points(values)
    ax.scatter(xs, es, c=color, s=5, label=label)
    ax.plot(1 + np.arange(n_layers), avgs, c=color)
    ax.plot([1, n_layers], [uniform, uniform], c="k", linestyle="--")
    ax.set_title(key + " (normalized rows)" if key != "attns" else key)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Avg. entropy (nats)")
    ax.legend(loc="lower right")
plt.tight_layout()
plt.show()""",
    """md:### Clustering heads (Section 6)
Uses the output of `head_distances.py` (JS divergences between heads; norms are
row-normalized first). Behaviour labels use Clark et al.'s thresholds, which were
set for attention; for norms they are applied to the *normalized* statistics.""",
    """ENTROPY_THRESHOLD = 3.8
POSITION_THRESHOLD = 0.5
SPECIAL_TOKEN_THRESHOLD = 0.6
# heads identified in Clark et al. (for bert-base-uncased)
LINGUISTIC_HEADS = {(4, 3): "Coreference", (7, 10): "Determiner",
                    (7, 9): "Direct object", (8, 5): "Object of prep.",
                    (3, 9): "Passive auxiliary", (6, 5): "Possesive"}


def plot_clusters(key, stats, entropies, ax):
  js = utils.load_pickle(HEAD_DISTANCES[key])
  mds = sklearn.manifold.MDS(metric=True, n_init=5, eps=1e-10, max_iter=1000,
                             dissimilarity="precomputed", random_state=0)
  pts = mds.fit_transform(js).reshape((n_layers, n_heads, 2))
  seen = set()
  for layer in range(n_layers):
    for head in range(n_heads):
      label, color, marker, size = "", GREY, "o", 4
      if stats["right"][layer, head] > POSITION_THRESHOLD:
        label, color, marker = "attend to next", RED, ">"
      if stats["left"][layer, head] > POSITION_THRESHOLD:
        label, color, marker = "attend to prev.", BLUE, "<"
      if entropies[layer, head] > ENTROPY_THRESHOLD:
        label, color, marker = "attend broadly", ORANGE, "^"
      if stats["cls"][layer, head] > SPECIAL_TOKEN_THRESHOLD:
        label, color, marker, size = "attend to [CLS]", PURPLE, "$C$", 5
      if stats["sep"][layer, head] > SPECIAL_TOKEN_THRESHOLD:
        label, color, marker, size = "attend to [SEP]", GREEN, "$S$", 5
      if stats["punct"][layer, head] > SPECIAL_TOKEN_THRESHOLD:
        label, color, marker, size = "attend to . and ,", SEA, "s", 3.2
      x, y = pts[layer, head]
      if (layer, head) in LINGUISTIC_HEADS:
        label, color, marker = "", BLACK, "x"
        ax.text(x, y, LINGUISTIC_HEADS[(layer, head)], color=color)
      if label in seen:
        label = ""
      seen.add(label)
      ax.plot([x], [y], marker=marker, markersize=size, color=color,
              label=label, linestyle="")
  ax.set_xticks([])
  ax.set_yticks([])
  ax.legend(loc="best")
  ax.set_title(key)


if all(os.path.exists(p) for p in HEAD_DISTANCES.values()):
  fig, axes = plt.subplots(1, 2, figsize=(12, 6))
  plot_clusters("attns", avg_stats["attention"], entropy_stats["attns"][1],
                axes[0])
  plot_clusters("norms", avg_stats["norm (normalized)"],
                entropy_stats["norms"][1], axes[1])
  plt.show()
else:
  print("run head_distances.py first")""",
]


SYNTAX = [
    """md:# Dependency Syntax: Attention Weights vs. Norm-Based Maps

Python 3 / PyTorch version of `Syntax_Analysis.ipynb` (Sections 4.2 and 5 of
[Clark et al., 2019](https://arxiv.org/abs/1906.04341)), run on the attention
weights $\\alpha$ (M1) and on the norm-based maps $\\|\\alpha f(x)\\|$ (M2) of
[Kobayashi et al., 2020](https://www.aclweb.org/anthology/2020.emnlp-main.574/).

Data: word-level maps written by `extract_norms.py --word_level`, e.g. for UD
(the paper used the WSJ Penn Treebank with Stanford Dependencies, which is not
freely available):
```
python preprocess_conllu.py --conllu en_ewt-ud-train.conllu --outfile $DATA/train.json
python preprocess_conllu.py --conllu en_ewt-ud-dev.conllu --outfile $DATA/dev.json
python extract_norms.py --preprocessed-data-file $DATA/dev.json --bert-dir bert-base-uncased --word_level
python extract_norms.py --preprocessed-data-file $DATA/train.json --bert-dir bert-base-uncased --word_level
# control: same model architecture, random weights
python extract_norms.py --preprocessed-data-file $DATA/dev.json --bert-dir bert-base-uncased --word_level \\
    --random_init --outfile $DATA/dev_random_norms.pkl
```
Note that examples longer than `--max_sequence_length` tokens are dropped, as in
the original code.""",
    """import collections
import os

import numpy as np
import torch
from matplotlib import pyplot as plt

import analysis_utils as au
import utils

DATA_DIR = os.environ.get("ATTN_DATA_DIR", "./data")
DEV_FILE = os.path.join(DATA_DIR, "dev_norms.pkl")
TRAIN_FILE = os.path.join(DATA_DIR, "train_norms.pkl")
RANDOM_DEV_FILE = os.path.join(DATA_DIR, "dev_random_norms.pkl")  # optional
GLOVE_DIR = os.path.join(DATA_DIR, "glove")  # optional, from the paper's data""",
    """# list of dicts with "words", "heads" (0 = ROOT, 1 = first word), "relns",
# "attns" and "norms" ([layers, heads, n_words + 2, n_words + 2], including
# [CLS] and [SEP])
dev_data = utils.load_pickle(DEV_FILE)
print(len(dev_data), "dev sentences")
print("words:", dev_data[0]["words"])
print("heads:", dev_data[0]["heads"])
print("relns:", dev_data[0]["relns"])
print("attns:", dev_data[0]["attns"].shape, "norms:", dev_data[0]["norms"].shape)

reln_counts = collections.Counter(r for e in dev_data for r in e["relns"])
print(reln_counts.most_common(10))""",
    """md:### Individual heads (Section 4.2, Table 1)
Each word is assigned the word it attends to most (or, for `h<-d`, the word
that attends to it most), ignoring the diagonal and [CLS]/[SEP]. The argmax is
invariant to rescaling a row, so no normalization is needed here; M2 differs
from M1 only by scaling column $j$ with $\\|f(x_j)\\|$.""",
    """head_scores = {"attns": au.get_head_scores(dev_data, "attns"),
               "norms": au.get_head_scores(dev_data, "norms")}
if os.path.exists(RANDOM_DEV_FILE):
  random_dev_data = utils.load_pickle(RANDOM_DEV_FILE)
  head_scores["attns (random)"] = au.get_head_scores(random_dev_data, "attns")
  head_scores["norms (random)"] = au.get_head_scores(random_dev_data, "norms")
baseline_scores = au.get_baseline_scores(dev_data)
table = au.best_head_table(dev_data, head_scores, baseline_scores)
au.print_best_head_table(table)""",
    """md:Best head per relation is selected on the same data it is scored on (as in
the paper), so these numbers are optimistic; for a fair comparison select heads
on one split and report on another:""",
    """def held_out_best_heads(select_data, eval_data, key):
  select = au.get_head_scores(select_data, key)
  evaluate = au.get_head_scores(eval_data, key)
  rows = {}
  for reln, count, _, _, best in au.best_head_table(
      eval_data, {key: select}, au.get_baseline_scores(eval_data)):
    _, layer, head, direction = best[key]
    rows[reln] = evaluate[direction][layer][head].get(reln, 0.0)
  return rows

half = len(dev_data) // 2
for key in ["attns", "norms"]:
  held_out = held_out_best_heads(dev_data[:half], dev_data[half:], key)
  print(key, {r: round(100 * a, 1) for r, a in held_out.items()})""",
    """md:### Per-head accuracy, M2 - M1
Difference in accuracy of every head between the norm-based and the attention
maps, for a few relations (both directions, best of the two).""",
    """RELNS = [r for r, _ in reln_counts.most_common() if r not in ("punct", "root")][:6]
fig, axes = plt.subplots(1, len(RELNS), figsize=(3.2 * len(RELNS), 3))
n_layers, n_heads = dev_data[0]["attns"].shape[:2]
for ax, reln in zip(axes, RELNS):
  diff = np.zeros((n_layers, n_heads))
  for l in range(n_layers):
    for h in range(n_heads):
      best = lambda key: max(head_scores[key][d][l][h].get(reln, 0.0)
                             for d in ["dep->head", "head<-dep"])
      diff[l, h] = best("norms") - best("attns")
  lim = max(np.abs(diff).max(), 1e-6)
  im = ax.imshow(100 * diff, cmap="RdBu", vmin=-100 * lim, vmax=100 * lim)
  ax.set_title(reln)
  ax.set_xlabel("head")
  ax.set_ylabel("layer")
  plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.show()""",
    """md:### Probing classifiers (Section 5)
PyTorch port of the paper's probes (batch size 1, Adam with lr 0.002, one epoch,
UAS excluding punctuation). The attention-only probe is a linear combination of
the 144 maps and their transposes. Unlike the argmax above, it is sensitive to
the scale of the features, so for M2 we report raw and row-normalized maps.
Results vary with the random seed; report several.""",
    """train_data = utils.load_pickle(TRAIN_FILE)
print(len(train_data), "train sentences")

SEEDS = [0, 1, 2]
probe_results = collections.defaultdict(list)
probes = {}
for key, normalize in [("attns", False), ("norms", False), ("norms", True)]:
  name = key + (" (normalized)" if normalize else "")
  for seed in SEEDS:
    torch.manual_seed(seed)
    probe = au.attn_linear_combo(
        n_maps=int(np.prod(dev_data[0][key].shape[:2])))
    uas = au.run_training(probe, train_data, dev_data, key=key,
                          normalize=normalize, log_every=0)
    probe_results[name].append(uas)
    probes[(name, seed)] = probe
for name, uas in probe_results.items():
  print("{:20s} UAS {:.1f} +- {:.1f}".format(
      name, 100 * np.mean(uas), 100 * np.std(uas)))""",
    """# learned weights of the attention-only probes (seed 0): [dependent->candidate
# maps | candidate->dependent maps], one weight per layer and head
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, name in zip(axes, probe_results):
  w = probes[(name, 0)].attn_map_weights.detach().numpy()
  w = w.reshape(2, n_layers, n_heads).transpose(1, 0, 2).reshape(n_layers, -1)
  lim = np.abs(w).max()
  im = ax.imshow(w, cmap="RdBu", vmin=-lim, vmax=lim)
  ax.axvline(n_heads - 0.5, color="k")
  ax.set_title(name)
  ax.set_xlabel("head (left: d->h, right: h<-d)")
  ax.set_ylabel("layer")
  plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.show()""",
    """md:Probes with GloVe word embeddings need the paper's `glove/embeddings.pkl`
and `glove/vocab.pkl`. See the `fix_root_alignment` note in
`analysis_utils.Probe` about how candidate heads and word embeddings are aligned in
the original code.""",
    """if os.path.exists(os.path.join(GLOVE_DIR, "embeddings.pkl")):
  embeddings = au.WordEmbeddings(os.path.join(GLOVE_DIR, "embeddings.pkl"),
                                 os.path.join(GLOVE_DIR, "vocab.pkl"))
  for key in ["attns", "norms"]:
    torch.manual_seed(0)
    print(key, "attn-and-words")
    au.run_training(au.attn_and_words(embeddings), train_data, dev_data, key=key,
                    log_every=0)
  torch.manual_seed(0)
  print("words-and-distances baseline")
  au.run_training(au.words_and_distances(embeddings), train_data, dev_data,
                  log_every=0)
else:
  print("no GloVe data found in", GLOVE_DIR)""",
]


def main():
  nbformat.write(notebook(GENERAL), "Norm_General_Analysis.ipynb")
  nbformat.write(notebook(SYNTAX), "Norm_Syntax_Analysis.ipynb")


if __name__ == "__main__":
  main()
