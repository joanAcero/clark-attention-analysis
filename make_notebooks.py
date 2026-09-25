"""Generates Norm_General_Analysis.ipynb, Norm_Syntax_Analysis.ipynb and
Norm_Syntax_Extended.ipynb
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
  # "right" = attention from token i to i+1 (np.eye(n, n, 1)). NOTE: the
  # original notebook labels "left" as "next token" and "right" as "prev
  # token", which contradicts its own clustering code and the selectors.
  for key, color, label in [("right", RED, "next token"),
                            ("left", BLUE, "prev token"),
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


EXTENDED = [
    """md:# Dependency Syntax: Extended Analyses (α vs. ‖αf(x)‖)

Follow-up to `Norm_Syntax_Analysis.ipynb`, on the same word-level data
(`$ATTN_DATA_DIR/dev_norms.pkl`, `train_norms.pkl`, optional
`dev_random_norms.pkl`, `train_random_norms.pkl`, `glove/`):

1. **Held-out head selection + paired significance.** The best head per
   relation is selected on one half of the sentences and scored on the other
   (2-fold cross-fitting, so every token is scored by a head not selected on
   it). α and ‖αf‖ are compared on the same tokens with an exact McNemar test,
   a sentence-level bootstrap CI of the difference, and Holm correction over
   relations.
2. **Sink ablation.** The per-head evaluation already ignores [CLS]/[SEP]; here
   punctuation is also removed from the candidate heads. If the gain of
   ‖αf‖ over α disappears once punctuation is removed from α, the norm-based
   gain is explained by discounting punctuation sinks, not by other information
   in ‖f(x)‖.
3. **Tree decoding from single heads** (R1 in the proposal): directed maximum
   spanning trees (Chu-Liu-Edmonds, gold root given, both directions) and
   undirected maximum spanning trees, on raw and row-renormalized maps (the
   MST objective, unlike the argmax, depends on row scale). Baselines:
   right-branching UAS and adjacent-chain UUAS.
4. **GloVe probe** with row-normalized norms and with both root alignments.
5. **Random-initialization control for the attention-only probe.**""",
    """import collections
import os

import numpy as np
import torch

import analysis_utils as au
import syntax_extended as se
import utils

DATA_DIR = os.environ.get("ATTN_DATA_DIR", "./data")
dev_data = utils.load_pickle(os.path.join(DATA_DIR, "dev_norms.pkl"))
RANDOM_DEV_FILE = os.path.join(DATA_DIR, "dev_random_norms.pkl")
TRAIN_FILE = os.path.join(DATA_DIR, "train_norms.pkl")
TRAIN_RANDOM_FILE = os.path.join(DATA_DIR, "train_random_norms.pkl")
GLOVE_DIR = os.path.join(DATA_DIR, "glove")

reln_counts = collections.Counter(r for e in dev_data for r in e["relns"])
RELNS = ["all"] + [r for r, c in reln_counts.most_common()
                   if c >= 100 and r not in ("punct", "root")]
print(len(dev_data), "sentences;", len(RELNS) - 1, "relations with >= 100 tokens")
print("punctuation words:", sum(se.is_punct(w) for e in dev_data for w in e["words"]))""",
    """md:## 1–2. Held-out best heads, significance, and sink ablation""",
    """VARIANTS = {"attns": ("attns", False), "attns -punct": ("attns", True),
            "norms": ("norms", False), "norms -punct": ("norms", True)}
cf = {name: se.cross_fit_correctness(dev_data, key, RELNS, exclude_punct=ep)
      for name, (key, ep) in VARIANTS.items()}
if os.path.exists(RANDOM_DEV_FILE):
  random_dev_data = utils.load_pickle(RANDOM_DEV_FILE)
  cf["attns (random)"] = se.cross_fit_correctness(random_dev_data, "attns", RELNS)
  cf["norms (random)"] = se.cross_fit_correctness(random_dev_data, "norms", RELNS)

print("held-out accuracy (%)")
print("{:12s}".format("reln") + "".join("{:>16s}".format(v) for v in cf))
for r in RELNS:
  if r in cf["attns"]:
    print("{:12s}".format(r[:12]) + "".join(
        "{:16.1f}".format(100 * cf[v][r][0].mean()) for v in cf if r in cf[v]))""",
    """md:**Does ‖αf‖ beat α?** (positive diff = norms better; `*` = Holm-adjusted p < 0.05)""",
    """_ = se.compare_maps(cf["attns"], cf["norms"], "attns", "norms")""",
    """md:**Sink ablation.** First: how much does removing punctuation candidates help α?
Second, the key comparison: ‖αf‖ vs α *without punctuation*. If this difference is
~0, the norm gain is the punctuation effect.""",
    """_ = se.compare_maps(cf["attns"], cf["attns -punct"], "attns", "a-punct")""",
    """_ = se.compare_maps(cf["attns -punct"], cf["norms"], "a-punct", "norms")""",
    """_ = se.compare_maps(cf["attns -punct"], cf["norms -punct"], "a-punct", "n-punct")""",
    """# heads selected in each fold (direction index: 0 = dep->head, 1 = head<-dep;
# layer and head 0-indexed)
for r in RELNS[:12]:
  print("{:12s} attns {}   norms {}".format(r[:12], cf["attns"][r][2], cf["norms"][r][2]))""",
    """md:## 3. Tree decoding from single heads
Every head is decoded on every sentence (≈1 s per head and variant per 1000
sentences). *In-sample* = best head selected and scored on all data (as in most
prior work); *held-out* = 2-fold cross-fitting. UAS/UUAS exclude punctuation;
the gold root is given for the directed trees.""",
    """print({k: round(100 * v, 1) for k, v in se.tree_baselines(dev_data).items()})

tree_variants = [("attns", False), ("attns", True), ("norms", False), ("norms", True)]
tree_results = {}
for key, renorm in tree_variants:
  name = key + (" renorm" if renorm else " raw")
  tree_results[name] = se.tree_table(se.tree_counts_all_heads(dev_data, key, renorm))
if os.path.exists(RANDOM_DEV_FILE):
  for key in ["attns", "norms"]:
    tree_results[key + " renorm (random)"] = se.tree_table(
        se.tree_counts_all_heads(random_dev_data, key, True))

print("{:24s}".format("") + "".join("{:>28s}".format(m) for m in se.TREE_METRICS))
print("{:24s}".format("") + "".join("{:>28s}".format("in-sample (head) | held-out")
                                    for m in se.TREE_METRICS))
for name, rows in tree_results.items():
  line = "{:24s}".format(name)
  for m in se.TREE_METRICS:
    score, (l, h), held = rows[m]
    line += "{:>28s}".format("{:.1f} ({}-{}) | {:.1f}".format(100 * score, l, h, 100 * held))
  print(line)""",
    """md:## 4. GloVe probes
`attn_and_words` with α and with row-normalized ‖αf‖, each with the original
root alignment (as in the paper's code) and with `fix_root_alignment=True`.""",
    """if os.path.exists(os.path.join(GLOVE_DIR, "embeddings.pkl")):
  train_data = utils.load_pickle(TRAIN_FILE)
  embeddings = au.WordEmbeddings(os.path.join(GLOVE_DIR, "embeddings.pkl"),
                                 os.path.join(GLOVE_DIR, "vocab.pkl"))
  for key, normalize in [("attns", False), ("norms", True)]:
    for fix in [False, True]:
      torch.manual_seed(0)
      print("{} normalize={} fix_root_alignment={}".format(key, normalize, fix))
      au.run_training(au.attn_and_words(embeddings, fix_root_alignment=fix),
                      train_data, dev_data, key=key, normalize=normalize,
                      log_every=0)
  torch.manual_seed(0)
  print("words-and-distances, fix_root_alignment=True")
  au.run_training(au.words_and_distances(embeddings, fix_root_alignment=True),
                  train_data, dev_data, log_every=0)
else:
  print("no GloVe data found in", GLOVE_DIR)""",
    """md:## 5. Random-initialization control for the attention-only probe
Needs `train_random_norms.pkl` and `dev_random_norms.pkl` (same architecture,
random weights; see `slurm/02_extract_syntax.sh`).""",
    """if os.path.exists(TRAIN_RANDOM_FILE) and os.path.exists(RANDOM_DEV_FILE):
  train_random = utils.load_pickle(TRAIN_RANDOM_FILE)
  for key, normalize in [("attns", False), ("norms", True)]:
    uas = []
    for seed in [0, 1, 2]:
      torch.manual_seed(seed)
      uas.append(au.run_training(au.attn_linear_combo(), train_random,
                                 random_dev_data, key=key, normalize=normalize,
                                 log_every=0))
    print("random-init {:6s} normalize={}: UAS {:.1f} +- {:.1f}".format(
        key, normalize, 100 * np.mean(uas), 100 * np.std(uas)))
else:
  print("random-init train/dev maps not found")""",
]


RG1 = [
    """md:# RG1: What does the attention-only parent probe use?

Clark et al.'s attention-only probe (61 UAS) combines the 144 heads linearly, but its
weights were never analysed, and it has not been compared with controls that
differ in only one respect. This notebook trains the **parent probe** of the
proposal on BERT-base attention (UD English EWT train; evaluated on EWT dev and PUD):

$p(j \\mid i) \\propto \\exp g(\\phi_{ij})$, $\\phi_{ij} = [\\alpha^k_{ij}, \\alpha^k_{ji}]_{k=1..144}$;
$p(\\text{ROOT} \\mid i) \\propto \\exp g_r(\\sigma_i)$, $\\sigma_i = [\\alpha^k_{i,\\text{CLS}}, \\alpha^k_{i,\\text{SEP}}]_k$,

with $g$ linear or a one-hidden-layer MLP, independent decisions per token
(no tree constraint), standardized features (weights = effect of one SD).

| Question | Analysis |
| --- | --- |
| Q1. How much of the probe's accuracy is due to BERT vs. the probe? | same probes on a randomly initialized BERT, and on positional features with the same number of parameters |
| Q2. Weighted vote over heads, or interactions? | linear vs. MLP, relative to the same gap on the controls |
| Q3. Which heads and layers drive the parent decision? | standardized weights (stability over seeds), head/layer ablation, single-layer probes |
| Q4. Does the combination change with the relation? | relation-conditioned linear probe (one weight vector per relation), similarity of the weight vectors |

`ATTN_DATA_DIR` must contain `ewt/{train,dev,train_random,dev_random}_norms.pkl`
and `pud/{dev,dev_random}_norms.pkl` (from `slurm/02_extract_syntax.sh`).""",
    """import collections
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy

import probes
import syntax_extended as se
import utils

DATA = os.environ.get("ATTN_DATA_DIR", "./data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [0, 1, 2, 3, 4]
REL_MIN_COUNT = int(os.environ.get("REL_MIN_COUNT", 200))  # relation-conditioned probe
L, H = 12, 12
# heads found by the per-head analysis (0-indexed layer, head)
KNOWN_HEADS = {(7, 10): "det/case", (7, 9): "obj", (6, 5): "poss", (3, 9): "aux:pass",
               (5, 3): "obl", (7, 1): "nsubj", (6, 0): "conj"}

load = lambda *p: utils.load_pickle(os.path.join(DATA, *p))
train_all = load("ewt", "train_norms.pkl")
n_val = len(train_all) // 10  # fixed validation split for early stopping
train, val = train_all[:-n_val], train_all[-n_val:]
dev, pud = load("ewt", "dev_norms.pkl"), load("pud", "dev_norms.pkl")
rtrain_all = load("ewt", "train_random_norms.pkl")
rtrain, rval = rtrain_all[:-n_val], rtrain_all[-n_val:]
rdev, rpud = load("ewt", "dev_random_norms.pkl"), load("pud", "dev_random_norms.pkl")
print(len(train), "train,", len(val), "val,", len(dev), "EWT dev,", len(pud), "PUD;", DEVICE)


def feature_name(f, root=False):
  d, k = divmod(f, L * H)
  l, h = divmod(k, H)
  kind = ("->[CLS]", "->[SEP]")[d] if root else ("d->h", "h<-d")[d]
  tag = KNOWN_HEADS.get((l, h), "")
  return "{:7s} {:2d}-{:2d} {}".format(kind, l, h, tag)""",
    """md:## Q1–Q2. Probes and controls""",
    """CONFIGS = {  # name: (train, val, dev, pud, feature_type)
    "BERT attention": (train, val, dev, pud, "attention"),
    "random-init attention": (rtrain, rval, rdev, rpud, "attention"),
    "position (matched)": (train, val, dev, pud, "position"),
}
results = collections.defaultdict(list)
linear_probes, mlp_probes = [], []
batchers = {}
for name, (tr, va, de, pu, ft) in CONFIGS.items():
  tb = probes.make_batcher(tr, feature_type=ft)
  vb = probes.make_batcher(va, feature_type=ft, standardizer_data=tr)
  db = probes.make_batcher(de, feature_type=ft, standardizer_data=tr)
  pb = probes.make_batcher(pu, feature_type=ft, standardizer_data=tr)
  batchers[name] = (tb, vb, db, pb)
  for kind in ["linear", "mlp"]:
    for seed in SEEDS:
      probe = probes.train_probe(tb, vb, kind, seed=seed, device=DEVICE, verbose=False)
      ewt_res = probes.evaluate(probe, db, DEVICE)
      pud_res = probes.evaluate(probe, pb, DEVICE)
      results[(name, kind)].append((ewt_res["UAS"], ewt_res["root"], pud_res["UAS"]))
      if name == "BERT attention":
        (linear_probes if kind == "linear" else mlp_probes).append(probe)
    r = np.array(results[(name, kind)]) * 100
    print("{:24s} {:6s}  EWT UAS {:.1f} ± {:.1f}  root {:.1f}  PUD UAS {:.1f} ± {:.1f}".format(
        name, kind, r[:, 0].mean(), r[:, 0].std(), r[:, 1].mean(), r[:, 2].mean(), r[:, 2].std()))""",
    """md:**Reading.** Q1: the margin of *BERT attention* over *random-init attention* and
*position* is what BERT's attention adds beyond architecture and word order.
Q2: if MLP − linear is large for BERT attention but small for the controls,
the parent decision needs interactions between heads; if it is similar, the
extra capacity is exploited independently of what BERT encodes.""",
    """gap = {name: np.mean([m[0] for m in results[(name, "mlp")]]) -
                np.mean([l[0] for l in results[(name, "linear")]]) for name in CONFIGS}
print({k: round(100 * v, 1) for k, v in gap.items()})""",
    """md:## Q3. Which heads drive the linear probe?
Standardized weights, averaged over seeds. Left: dependent attends to
candidate (d->h); middle: candidate attends to dependent (h<-d); right: ROOT
features (attention of the dependent to [CLS] / [SEP]).""",
    """pair_w = np.array([p.weights()[0] for p in linear_probes])
root_w = np.array([p.weights()[1] for p in linear_probes])
print("seed stability (mean pairwise corr, top-10 Jaccard): pair {} root {}".format(
    np.round(probes.seed_stability(pair_w), 2), np.round(probes.seed_stability(root_w), 2)))

mean_pair, mean_root = pair_w.mean(0), root_w.mean(0)
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
grids = [probes.weight_grid(mean_pair, L, H)[0], probes.weight_grid(mean_pair, L, H)[1],
         probes.weight_grid(mean_root, L, H)[0], probes.weight_grid(mean_root, L, H)[1]]
for ax, g, title in zip(axes, grids, ["d->h", "h<-d", "root: ->[CLS]", "root: ->[SEP]"]):
  lim = np.abs(g).max()
  im = ax.imshow(g, cmap="RdBu", vmin=-lim, vmax=lim)
  ax.set_title(title); ax.set_xlabel("head"); ax.set_ylabel("layer")
  plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout(); plt.show()

print("top pair weights (mean ± sd over seeds):")
for f in np.argsort(-np.abs(mean_pair))[:15]:
  print("  {:+.2f} ± {:.2f}  {}".format(mean_pair[f], pair_w[:, f].std(), feature_name(f)))
print("top root weights:")
for f in np.argsort(-np.abs(mean_root))[:8]:
  print("  {:+.2f} ± {:.2f}  {}".format(mean_root[f], root_w[:, f].std(), feature_name(f, True)))

plt.figure(figsize=(5, 3))
w = np.abs(probes.weight_grid(mean_pair, L, H)).sum(-1)
plt.plot(range(L), w[0], "o-", label="d->h"); plt.plot(range(L), w[1], "o-", label="h<-d")
plt.xlabel("layer"); plt.ylabel("sum |weight|"); plt.legend(); plt.show()""",
    """md:**Ablation.** Weights of correlated features are not importances, so we also
set each head's features (all four) to their training mean and measure the UAS
drop on EWT dev (seed-0 probes); and the same per layer.""",
    """_, db, _ = batchers["BERT attention"][1:]
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (name, probe) in zip(axes, [("linear", linear_probes[0]), ("mlp", mlp_probes[0])]):
  base, drops = probes.head_ablation(probe, db, L, H, DEVICE)
  im = ax.imshow(100 * drops, cmap="Reds")
  ax.set_title("{} (UAS {:.1f}): UAS drop per head".format(name, 100 * base))
  ax.set_xlabel("head"); ax.set_ylabel("layer"); plt.colorbar(im, ax=ax, fraction=0.046)
  top = np.argsort(-drops.ravel())[:8]
  print(name, "largest drops:", ["{}-{} {:.1f}{}".format(k // H, k % H, 100 * drops.ravel()[k],
        " (" + KNOWN_HEADS[(k // H, k % H)] + ")" if (k // H, k % H) in KNOWN_HEADS else "")
        for k in top])
  _, ldrops = probes.layer_ablation(probe, db, L, H, DEVICE)
  print(name, "layer ablation drops:", np.round(100 * ldrops, 1))
plt.tight_layout(); plt.show()""",
    """md:**Single-layer and cumulative probes** (the attention-space analogue of
Tenney et al.): a linear probe using only the heads of layer *l* (24 pair + 24
root features), and one using layers 0..*l*.""",
    """single, cumulative = np.zeros((L, 3)), np.zeros(L)
for l in range(L):
  for s, seed in enumerate([0, 1, 2]):
    tb = probes.make_batcher(train, layers=[l]); vb = probes.make_batcher(val, layers=[l], standardizer_data=train)
    db_l = probes.make_batcher(dev, layers=[l], standardizer_data=train)
    single[l, s] = probes.evaluate(probes.train_probe(tb, vb, "linear", seed=seed, device=DEVICE,
                                                      verbose=False), db_l, DEVICE)["UAS"]
  layers = list(range(l + 1))
  tb = probes.make_batcher(train, layers=layers); vb = probes.make_batcher(val, layers=layers, standardizer_data=train)
  db_l = probes.make_batcher(dev, layers=layers, standardizer_data=train)
  cumulative[l] = probes.evaluate(probes.train_probe(tb, vb, "linear", seed=0, device=DEVICE,
                                                     verbose=False), db_l, DEVICE)["UAS"]
  print("layer {:2d}: single {:.1f} ± {:.1f}   cumulative {:.1f}".format(
      l, 100 * single[l].mean(), 100 * single[l].std(), 100 * cumulative[l]))
plt.figure(figsize=(5, 3))
plt.errorbar(range(L), 100 * single.mean(1), 100 * single.std(1), fmt="o-", label="single layer")
plt.plot(range(L), 100 * cumulative, "s-", label="layers 0..l")
plt.xlabel("layer"); plt.ylabel("EWT dev UAS"); plt.legend(); plt.show()""",
    """md:## Q4. Relation-conditioned probe
One linear weight vector per relation of the dependent (relations with ≥ 200
training tokens; the rest, and `root`, share one vector). The **gold relation is
given**, so accuracies are an upper bound for "knowing which relation to look
for", not a parser; the relation also leaks information the unconditioned probe
does not have (e.g. the typical direction of the head), so compare per-relation
accuracies and weights, not the overall UAS. We compare per-relation accuracy with the unconditioned probe and the
similarity of the weight vectors across relations.""",
    """rels = probes.relation_vocab(train, min_count=REL_MIN_COUNT)
tb = probes.make_batcher(train, relations=rels)
vb = probes.make_batcher(val, relations=rels, standardizer_data=train)
db_r = probes.make_batcher(dev, relations=rels, standardizer_data=train)
rel_probes = [probes.train_probe(tb, vb, "relation", seed=s, device=DEVICE, verbose=False,
                                 n_relations=len(rels)) for s in [0, 1, 2]]
print("relation-conditioned UAS:", [round(100 * probes.evaluate(p, db_r, DEVICE)["UAS"], 1)
                                    for p in rel_probes])
acc_rel = probes.per_relation_accuracy(rel_probes[0], db_r, dev, DEVICE)
acc_lin = probes.per_relation_accuracy(linear_probes[0], batchers["BERT attention"][2], dev, DEVICE)
head_acc, _ = se.head_accuracies(dev, "attns")
counts = collections.Counter(r for e in dev for r in e["relns"])
print("{:12s} {:>6s} {:>8s} {:>10s} {:>10s}".format("reln", "n", "linear", "rel-cond", "best head"))
for r in [r for r in rels if r not in ("<other>", "root")]:
  if counts[r]:
    print("{:12s} {:6d} {:8.1f} {:10.1f} {:10.1f}".format(
        r[:12], counts[r], 100 * acc_lin.get(r, 0), 100 * acc_rel.get(r, 0), 100 * head_acc[r].max()))""",
    """W = np.mean([p.weights()[0] for p in rel_probes], 0)  # [R, 288]
names = [r for r in rels]
keep = [i for i, r in enumerate(names) if r not in ("<other>",)]
assert len(keep) >= 2, "need at least two relations with >= REL_MIN_COUNT tokens"
Wk = W[keep] / np.linalg.norm(W[keep], axis=1, keepdims=True)
sim = Wk @ Wk.T
order = hierarchy.leaves_list(hierarchy.linkage(Wk, "average", metric="cosine"))
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
im = axes[0].imshow(sim[np.ix_(order, order)], cmap="RdBu", vmin=-1, vmax=1)
lab = [names[keep[i]] for i in order]
axes[0].set_xticks(range(len(lab))); axes[0].set_xticklabels(lab, rotation=90)
axes[0].set_yticks(range(len(lab))); axes[0].set_yticklabels(lab)
axes[0].set_title("cosine similarity of relation weight vectors"); plt.colorbar(im, ax=axes[0])
hierarchy.dendrogram(hierarchy.linkage(Wk, "average", metric="cosine"),
                     labels=[names[i] for i in keep], ax=axes[1], orientation="right")
plt.tight_layout(); plt.show()
print("seed stability of relation weights (corr, top-10 Jaccard):",
      np.round(probes.seed_stability([p.weights()[0].ravel() for p in rel_probes]), 2))
for i in keep[:15]:
  top = np.argsort(-W[i])[:3]
  print("{:12s} top heads: {}".format(names[i][:12], " | ".join(feature_name(f) for f in top)))""",
]


def main():
  nbformat.write(notebook(GENERAL), "Norm_General_Analysis.ipynb")
  nbformat.write(notebook(SYNTAX), "Norm_Syntax_Analysis.ipynb")
  nbformat.write(notebook(EXTENDED), "Norm_Syntax_Extended.ipynb")
  nbformat.write(notebook(RG1), "RG1_Probe_Analysis.ipynb")


if __name__ == "__main__":
  main()
