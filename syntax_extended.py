"""Extended syntax analyses for comparing attention weights with norm-based
maps (see Norm_Syntax_Extended.ipynb):

  * vectorized per-head evaluation (same metric as analysis_utils, all heads at
    once), optionally excluding punctuation as candidate heads ("sink
    ablation"),
  * head selection on held-out data (2-fold cross-fitting) and paired
    significance tests (McNemar, sentence-level bootstrap, Holm correction),
  * tree decoding from single heads: directed maximum spanning trees
    (Chu-Liu-Edmonds, gold root) and undirected maximum spanning trees, with
    right-branching / adjacency baselines.

Maps are word-level [layers, heads, n + 2, n + 2] arrays with [CLS] at 0 and
[SEP] at n + 1, so word i (1-based, as in "heads") is at index i.
"""

import collections
import unicodedata

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import binomtest

DIRECTIONS = ["dep->head", "head<-dep"]


def is_punct(word):
  """True for words made only of punctuation/symbol characters (label-free,
  so it can be used on candidate heads without looking at the gold tree)."""
  return len(word) > 0 and all(unicodedata.category(c)[0] in "PS" for c in word)


###############################################################################
# Per-head predictions and evaluation
###############################################################################

def all_head_predictions(example, key, exclude_punct=False):
  """Predicted heads (1-based) of every word for every head and direction:
  [2, layers, heads, n]. Same rule as analysis_utils.attn_head_predictor:
  argmax over the other words, ignoring self and [CLS]/[SEP]; optionally also
  ignoring punctuation candidates."""
  maps = np.asarray(example[key], dtype=np.float64)[:, :, 1:-1, 1:-1]
  maps = np.stack([maps, maps.transpose(0, 1, 3, 2)])  # [2, L, H, n, n]
  n = maps.shape[-1]
  maps[..., np.arange(n), np.arange(n)] = -np.inf
  if exclude_punct:
    punct = np.array([is_punct(w) for w in example["words"]])
    if punct.any() and not punct.all():
      maps[..., punct] = -np.inf
  return maps.argmax(-1) + 1


def _correct(example, predictions):
  """Correctness of predictions [..., n] with the paper's special case for
  'poss' (Stanford Dependencies)."""
  words, heads = example["words"], np.asarray(example["heads"])
  correct = predictions == heads
  for i, r in enumerate(example["relns"]):
    if r == "poss" and i + 1 < len(words) and words[i + 1] in ("'s", "s'"):
      use_next = predictions[..., i] < len(words)
      correct[..., i] = np.where(use_next, predictions[..., i + 1] == heads[i],
                                 correct[..., i])
  return correct


def head_accuracies(data, key, exclude_punct=False):
  """acc[reln] = [2, layers, heads] accuracy of every head/direction, plus
  counts[reln]."""
  correct = collections.defaultdict(float)
  counts = collections.Counter()
  for example in data:
    c = _correct(example, all_head_predictions(example, key, exclude_punct))
    for i, r in enumerate(example["relns"]):
      for reln in (r, "all"):
        correct[reln] = correct[reln] + c[..., i]
        counts[reln] += 1
  return {r: correct[r] / counts[r] for r in counts}, counts


def best_heads(acc):
  """reln -> (direction index, layer, head) with the highest accuracy."""
  return {r: np.unravel_index(np.argmax(a), a.shape) for r, a in acc.items()}


def head_token_correctness(data, key, choice, reln, exclude_punct=False):
  """Per-token correctness (bool) of one head on the tokens with relation
  `reln` ("all" = every token), and the sentence index of each token."""
  d, l, h = choice
  correct, sents = [], []
  for s, example in enumerate(data):
    preds = all_head_predictions(example, key, exclude_punct)[d, l, h]
    c = _correct(example, preds)
    for i, r in enumerate(example["relns"]):
      if reln == "all" or r == reln:
        correct.append(c[i])
        sents.append(s)
  return np.array(correct, dtype=bool), np.array(sents)


def cross_fit_correctness(data, key, relns, exclude_punct=False, n_folds=2):
  """Held-out evaluation of the best head per relation: the head is selected
  on the other folds, and the per-token correctness of all folds is
  concatenated (so every token is evaluated with a head not selected on it).
  Returns {reln: (correct, sentence_ids, [selected heads per fold])}."""
  folds = [list(range(f, len(data), n_folds)) for f in range(n_folds)]
  out = {r: ([], [], []) for r in relns}
  for f, fold in enumerate(folds):
    select = [data[i] for g, other in enumerate(folds) if g != f for i in other]
    evaluate = [data[i] for i in fold]
    acc, _ = head_accuracies(select, key, exclude_punct)
    chosen = best_heads(acc)
    for r in relns:
      if r not in chosen:
        continue
      c, s = head_token_correctness(evaluate, key, chosen[r], r, exclude_punct)
      out[r][0].append(c)
      out[r][1].append(np.asarray(fold)[s] if len(s) else s)
      out[r][2].append(tuple(int(x) for x in chosen[r]))
  return {r: (np.concatenate(c), np.concatenate(s), heads)
          for r, (c, s, heads) in out.items() if c}


###############################################################################
# Paired statistics
###############################################################################

def mcnemar(a, b):
  """Exact McNemar test for paired binary outcomes (two-sided)."""
  n01 = int(np.sum(a & ~b))
  n10 = int(np.sum(~a & b))
  if n01 + n10 == 0:
    return 1.0
  return binomtest(n01, n01 + n10, 0.5).pvalue


def bootstrap_diff(a, b, sents, n_boot=2000, seed=0):
  """95% CI of accuracy(b) - accuracy(a), resampling sentences."""
  rng = np.random.RandomState(seed)
  ids, inv = np.unique(sents, return_inverse=True)
  sa = np.bincount(inv, weights=a.astype(float))
  sb = np.bincount(inv, weights=b.astype(float))
  n = np.bincount(inv)
  diffs = []
  for _ in range(n_boot):
    idx = rng.randint(0, len(ids), len(ids))
    diffs.append((sb[idx].sum() - sa[idx].sum()) / n[idx].sum())
  return np.percentile(diffs, [2.5, 97.5])


def holm(pvalues):
  """Holm-Bonferroni adjusted p-values (dict -> dict)."""
  items = sorted(pvalues.items(), key=lambda kv: kv[1])
  m, adjusted, running = len(items), {}, 0.0
  for rank, (k, p) in enumerate(items):
    running = max(running, min(1.0, (m - rank) * p))
    adjusted[k] = running
  return adjusted


def compare_maps(cf_a, cf_b, name_a, name_b, min_count=100):
  """Table comparing two held-out evaluations (outputs of
  cross_fit_correctness on the same data): accuracies, difference with
  bootstrap CI, McNemar p-value and Holm-adjusted p-value."""
  rows, pvals = [], {}
  for reln in cf_a:
    if reln not in cf_b:
      continue
    a, sents, _ = cf_a[reln]
    b, sents_b, _ = cf_b[reln]
    assert np.array_equal(sents, sents_b)
    if len(a) < min_count and reln != "all":
      continue
    lo, hi = bootstrap_diff(a, b, sents)
    p = mcnemar(a, b)
    pvals[reln] = p
    rows.append((reln, len(a), a.mean(), b.mean(), lo, hi, p))
  adjusted = holm(pvals)
  print("{:12s} {:>6s} {:>8s} {:>8s} {:>7s} {:>17s} {:>9s} {:>9s}".format(
      "reln", "n", name_a[:8], name_b[:8], "diff", "95% CI", "p", "p_holm"))
  for reln, n, acc_a, acc_b, lo, hi, p in rows:
    print("{:12s} {:6d} {:8.1f} {:8.1f} {:+7.1f} [{:+6.1f}, {:+6.1f}] "
          "{:9.2g} {:9.2g}{}".format(
              reln[:12], n, 100 * acc_a, 100 * acc_b, 100 * (acc_b - acc_a),
              100 * lo, 100 * hi, p, adjusted[reln],
              " *" if adjusted[reln] < 0.05 else ""))
  return rows


###############################################################################
# Tree decoding
###############################################################################

def chu_liu_edmonds(scores):
  """Maximum spanning arborescence rooted at node 0.

  scores: [n + 1, n + 1] array, scores[h, d] = score of the edge h -> d
  (-inf for forbidden edges). Returns heads [n + 1] (heads[0] = -1)."""
  scores = np.array(scores, dtype=np.float64)
  n = scores.shape[0]
  scores[:, 0] = -np.inf
  np.fill_diagonal(scores, -np.inf)
  heads = scores.argmax(0)
  heads[0] = -1
  cycle = _find_cycle(heads)
  if cycle is None:
    return heads
  # contract the cycle into a new node c
  cycle = np.array(sorted(cycle))
  in_cycle = np.zeros(n, dtype=bool)
  in_cycle[cycle] = True
  rest = np.where(~in_cycle)[0]  # includes the root
  cycle_score = scores[heads[cycle], cycle].sum()
  m = len(rest)
  new = np.full((m + 1, m + 1), -np.inf)
  new[:m, :m] = scores[np.ix_(rest, rest)]
  # edges out of the cycle: best source inside the cycle for each target
  out_scores = scores[np.ix_(cycle, rest)]
  out_src = cycle[out_scores.argmax(0)]
  new[m, :m] = out_scores.max(0)
  # edges into the cycle: gain of breaking the cycle at each node
  in_scores = (scores[np.ix_(rest, cycle)] - scores[heads[cycle], cycle] +
               cycle_score)
  in_dst = cycle[in_scores.argmax(1)]
  new[:m, m] = in_scores.max(1)
  sub = chu_liu_edmonds(new)
  result = heads.copy()
  for j in range(1, m):  # nodes outside the cycle (rest[0] is the root)
    h = sub[j]
    result[rest[j]] = rest[h] if h < m else out_src[j]
  entering = sub[m]  # index in `rest` of the head of the contracted node
  broken = in_dst[entering]
  result[broken] = rest[entering]
  result[0] = -1
  return result


def _find_cycle(heads):
  n = len(heads)
  visited = np.zeros(n, dtype=int)  # 0 new, 1 on current path, 2 done
  for start in range(1, n):
    path, node = [], start
    while node > 0 and visited[node] == 0:
      visited[node] = 1
      path.append(node)
      node = heads[node]
    if node > 0 and visited[node] == 1:
      return path[path.index(node):]
    for p in path:
      visited[p] = 2
  return None


def _word_map(example, key, layer, head, renormalize):
  m = np.asarray(example[key][layer][head], dtype=np.float64)[1:-1, 1:-1].copy()
  np.fill_diagonal(m, 0.0)
  if renormalize:
    m = m / np.maximum(m.sum(-1, keepdims=True), 1e-12)
  return m  # m[i, j]: word i+1 attends to word j+1


def decode_directed(example, key, layer, head, direction, renormalize=False):
  """Directed MST (gold root) from one head. Returns predicted heads."""
  m = _word_map(example, key, layer, head, renormalize)
  n = m.shape[0]
  # edge h -> d scored by how much d attends to h ("dep->head") or h to d
  edge = m.T if direction == "dep->head" else m
  scores = np.full((n + 1, n + 1), -np.inf)
  scores[1:, 1:] = edge
  root = example["heads"].index(0) + 1
  scores[0, root] = 0.0
  scores[1:, root] = -np.inf  # the gold root only attaches to ROOT
  return chu_liu_edmonds(scores)[1:]


def decode_undirected(example, key, layer, head, renormalize=False):
  """Undirected maximum spanning tree over the words. Returns a set of
  frozenset edges (1-based)."""
  m = _word_map(example, key, layer, head, renormalize)
  sym = m + m.T
  if sym.shape[0] < 2:
    return set()
  # maximum spanning tree = minimum spanning tree of (big - weights)
  weights = (sym.max() + 1.0) - sym
  np.fill_diagonal(weights, 0.0)
  tree = minimum_spanning_tree(weights).tocoo()
  return {frozenset((i + 1, j + 1)) for i, j in zip(tree.row, tree.col)}


def _eval_mask(example, ignore_punct=True):
  return np.array([r != "punct" for r in example["relns"]]) if ignore_punct \
      else np.ones(len(example["relns"]), dtype=bool)


def uas(example, predicted, ignore_punct=True):
  mask = _eval_mask(example, ignore_punct)
  return int(np.sum((np.asarray(predicted) == example["heads"]) & mask)), \
      int(mask.sum())


def uuas(example, edges, ignore_punct=True):
  mask = _eval_mask(example, ignore_punct)
  correct = total = 0
  for i, (h, keep) in enumerate(zip(example["heads"], mask)):
    if h == 0 or not keep:
      continue
    total += 1
    correct += frozenset((h, i + 1)) in edges
  return correct, total


def tree_scores(data, key, layer, head, renormalize=False):
  """UAS (directed MST, both directions) and UUAS (undirected MST)."""
  res = {}
  for direction in DIRECTIONS:
    c = t = 0
    for ex in data:
      ci, ti = uas(ex, decode_directed(ex, key, layer, head, direction,
                                       renormalize))
      c, t = c + ci, t + ti
    res["UAS " + direction] = c / max(t, 1)
  c = t = 0
  for ex in data:
    ci, ti = uuas(ex, decode_undirected(ex, key, layer, head, renormalize))
    c, t = c + ci, t + ti
  res["UUAS"] = c / max(t, 1)
  return res


def tree_baselines(data):
  """Right-branching (each word -> next word, last word -> ROOT) UAS, and
  UUAS of the chain of adjacent words."""
  c = t = cu = tu = 0
  for ex in data:
    n = len(ex["words"])
    ci, ti = uas(ex, [i + 2 if i + 1 < n else 0 for i in range(n)])
    c, t = c + ci, t + ti
    ci, ti = uuas(ex, {frozenset((i, i + 1)) for i in range(1, n)})
    cu, tu = cu + ci, tu + ti
  return {"right-branching UAS": c / max(t, 1), "adjacent-chain UUAS": cu / max(tu, 1)}


TREE_METRICS = ["UAS dep->head", "UAS head<-dep", "UUAS"]


def tree_counts_all_heads(data, key, renormalize=False):
  """Per-sentence (correct, total) for every head and tree metric:
  arrays [n_sentences, 3, layers, heads] (metrics as in TREE_METRICS)."""
  n_layers, n_heads = np.asarray(data[0][key]).shape[:2]
  correct = np.zeros((len(data), 3, n_layers, n_heads))
  total = np.zeros((len(data), 3, n_layers, n_heads))
  for s, ex in enumerate(data):
    for l in range(n_layers):
      for h in range(n_heads):
        for k, direction in enumerate(DIRECTIONS):
          correct[s, k, l, h], total[s, k, l, h] = uas(
              ex, decode_directed(ex, key, l, h, direction, renormalize))
        correct[s, 2, l, h], total[s, 2, l, h] = uuas(
            ex, decode_undirected(ex, key, l, h, renormalize))
  return correct, total


def tree_table(counts, n_folds=2):
  """Best head per tree metric: in-sample (selected and scored on all data, as
  in most prior work) and held-out (selected on the other folds)."""
  correct, total = counts
  n = correct.shape[0]
  in_sample = correct.sum(0) / np.maximum(total.sum(0), 1)
  rows = {}
  for k, metric in enumerate(TREE_METRICS):
    l, h = np.unravel_index(np.argmax(in_sample[k]), in_sample[k].shape)
    c_held = t_held = 0
    for f in range(n_folds):
      test = np.arange(f, n, n_folds)
      train = np.setdiff1d(np.arange(n), test)
      acc = correct[train, k].sum(0) / np.maximum(total[train, k].sum(0), 1)
      lf, hf = np.unravel_index(np.argmax(acc), acc.shape)
      c_held += correct[test, k, lf, hf].sum()
      t_held += total[test, k, lf, hf].sum()
    rows[metric] = (in_sample[k, l, h], (int(l), int(h)), c_held / max(t_held, 1))
  return rows
