"""Python 3 / PyTorch port of the analysis code in General_Analysis.ipynb and
Syntax_Analysis.ipynb, parameterized by which map to analyse.

Every function takes a `key` naming the field of the example dicts to use:
"attns" (attention weights, as in Clark et al.) or "norms"
(||alpha f(x)||, Kobayashi et al.), or any other [layers, heads, n, n] field.
The logic is otherwise kept identical to the original notebooks; deviations are
marked with "NOTE".
"""

import collections

import numpy as np
import torch

import norm_utils


def get_maps(example, key, normalize=False):
  maps = np.asarray(example[key], dtype=np.float64)
  return norm_utils.normalize_rows(maps) if normalize else maps


###############################################################################
# General analysis (Sections 3 and 6 of Clark et al.)
###############################################################################

def token_stats(data, key="attns", normalize=False):
  """Average attention (or norm) to particular tokens/positions (Section 3.1,
  3.2). With key="norms" and normalize=False this is the average summed
  ||alpha f(x)|| given to e.g. [SEP], as in Figure 4 of Kobayashi et al.; with
  normalize=True the rows are first rescaled to sum to one, giving the share of
  the norm going to each token type, which is on the same scale as attention.
  """
  n_docs = len(data)
  n_layers, n_heads = np.asarray(data[0][key]).shape[:2]
  avg = {k: np.zeros((n_layers, n_heads)) for k in [
      "self", "right", "left", "sep", "sep_sep", "rest_sep", "cls", "punct"]}
  for doc in data:
    maps = get_maps(doc, key, normalize)
    n_tokens = maps.shape[-1]
    tokens = doc["tokens"]
    if len(tokens) != n_tokens:  # word-level maps
      tokens = ["[CLS]"] + doc["words"] + ["[SEP]"]

    seps, clss, puncts = (np.zeros(n_tokens) for _ in range(3))
    for position, token in enumerate(tokens):
      if token == "[SEP]":
        seps[position] = 1
      if token == "[CLS]":
        clss[position] = 1
      if token == "." or token == ",":
        puncts[position] = 1

    sep_seps = np.ones((n_tokens, n_tokens)) * seps[np.newaxis] * \
        seps[:, np.newaxis]
    rest_seps = np.ones((n_tokens, n_tokens)) * \
        (np.ones(n_tokens) - seps)[:, np.newaxis] * seps[np.newaxis]
    selectors = {
        "self": np.eye(n_tokens, n_tokens),
        "right": np.eye(n_tokens, n_tokens, 1),
        "left": np.eye(n_tokens, n_tokens, -1),
        "sep": np.tile(seps[np.newaxis], [n_tokens, 1]),
        "sep_sep": sep_seps,
        "rest_sep": rest_seps,
        "cls": np.tile(clss[np.newaxis], [n_tokens, 1]),
        "punct": np.tile(puncts[np.newaxis], [n_tokens, 1]),
    }
    for k, selector in selectors.items():
      if k == "sep_sep":
        denom = 2
      elif k == "rest_sep":
        denom = n_tokens - 2
      else:
        denom = n_tokens
      avg[k] += ((maps * selector[np.newaxis, np.newaxis]).sum(-1).sum(-1) /
                 (n_docs * denom))
  return avg


def entropy_stats(data, key="attns"):
  """Attention entropies (Section 3.3). Maps are row-normalized first
  (NOTE: a no-op for attention; required for norms, whose rows do not sum to
  one), then smoothed as in the original notebook."""
  n_docs = len(data)
  n_layers, n_heads = np.asarray(data[0][key]).shape[:2]
  uniform = 0
  entropies = np.zeros((n_layers, n_heads))
  entropies_cls = np.zeros((n_layers, n_heads))
  for doc in data:
    maps = get_maps(doc, key, normalize=True)
    maps = 0.9999 * maps + (0.0001 / maps.shape[-1])
    uniform -= np.log(1.0 / maps.shape[-1])
    entropies -= (maps * np.log(maps)).sum(-1).mean(-1)
    entropies_cls -= (maps * np.log(maps))[:, :, 0].sum(-1)
  return uniform / n_docs, entropies / n_docs, entropies_cls / n_docs


###############################################################################
# Individual heads vs. dependency syntax (Section 4.2 of Clark et al.)
###############################################################################

def evaluate_predictor(data, prediction_fn):
  """Compute accuracies for each relation for the given predictor."""
  n_correct, n_incorrect = collections.Counter(), collections.Counter()
  for example in data:
    words = example["words"]
    predictions = prediction_fn(example)
    for i, (p, y, r) in enumerate(zip(predictions, example["heads"],
                                      example["relns"])):
      is_correct = (p == y)
      if r == "poss" and p < len(words):
        # Special case for poss (see discussion in Section 4.2).
        # NOTE: the original condition,
        #   i < len(words) and words[i + 1] == "'s" or words[i + 1] == "s'"
        # raises an IndexError for a sentence-final "poss" word; we use the
        # evidently intended condition. It only fires for Stanford
        # Dependencies labels (UD uses "nmod:poss").
        if i + 1 < len(words) and words[i + 1] in ("'s", "s'"):
          is_correct = (predictions[i + 1] == y)
      if is_correct:
        n_correct[r] += 1
        n_correct["all"] += 1
      else:
        n_incorrect[r] += 1
        n_incorrect["all"] += 1
  return {k: n_correct[k] / float(n_correct[k] + n_incorrect[k])
          for k in set(n_correct) | set(n_incorrect)}


def attn_head_predictor(layer, head, key="attns", mode="normal"):
  """Assign each word the most-attended-to other word as its head."""
  def predict(example):
    attn = np.array(example[key][layer][head], dtype=np.float64)
    if mode == "transpose":
      attn = attn.T
    elif mode == "both":
      attn += attn.T
    else:
      assert mode == "normal"
    # ignore attention to self and [CLS]/[SEP] tokens
    attn[range(attn.shape[0]), range(attn.shape[0])] = 0
    attn = attn[1:-1, 1:-1]
    return np.argmax(attn, axis=-1) + 1  # +1 because ROOT is at index 0
  return predict


def offset_predictor(offset):
  """Simple baseline: assign each word the word a fixed offset from
  it (e.g., the word to its right) as its head."""
  def predict(example):
    return [max(0, min(i + offset + 1, len(example["words"])))
            for i in range(len(example["words"]))]
  return predict


def get_scores(data, key="attns", mode="normal"):
  """Get the accuracies of every attention head."""
  n_layers, n_heads = np.asarray(data[0][key]).shape[:2]
  scores = collections.defaultdict(dict)
  for layer in range(n_layers):
    for head in range(n_heads):
      scores[layer][head] = evaluate_predictor(
          data, attn_head_predictor(layer, head, key, mode))
  return scores


def get_head_scores(data, key="attns"):
  # head_scores[direction][layer][head][dep_relation] = accuracy
  return {
      "dep->head": get_scores(data, key, "normal"),
      "head<-dep": get_scores(data, key, "transpose"),
  }


def get_baseline_scores(data):
  # baseline_scores[offset][dep_relation] = accuracy
  return {i: evaluate_predictor(data, offset_predictor(i))
          for i in range(-3, 3)}


def get_all_scores(head_scores, reln):
  """Get all attention head scores for a particular relation."""
  all_scores = []
  for direction, layer_head_scores in head_scores.items():
    for layer, per_head in layer_head_scores.items():
      for head, scores in per_head.items():
        all_scores.append((scores.get(reln, 0.0), layer, head, direction))
  return sorted(all_scores, reverse=True)


def best_head_table(data, head_scores_by_map, baseline_scores, min_count=100):
  """Best head per relation vs. the best fixed-offset baseline (Table 1 of
  Clark et al.), for one or several maps. Returns a list of rows
  (reln, count, baseline_offset, baseline_acc, {map: (acc, layer, head,
  direction)})."""
  reln_counts = collections.Counter(
      r for example in data for r in example["relns"])
  reln_counts["all"] = sum(len(example["relns"]) for example in data)
  rows = []
  for reln, _ in [("all", 0)] + reln_counts.most_common():
    if reln == "root" or reln == "punct" or (reln == "all" and rows):
      continue
    if reln_counts[reln] < min_count and reln != "all":
      break
    baseline_acc, baseline_offset = max(
        (scores.get(reln, 0.0), i) for i, scores in baseline_scores.items())
    best = {name: get_all_scores(scores, reln)[0]
            for name, scores in head_scores_by_map.items()}
    rows.append((reln, reln_counts[reln], baseline_offset, baseline_acc, best))
  return rows


def print_best_head_table(rows):
  names = list(rows[0][4].keys())
  header = "{:10s} | {:5s} | {:>9s}".format("reln", "count", "baseline")
  for name in names:
    header += " | {:>20s}".format(name)
  print(header)
  print("-" * len(header))
  for reln, count, offset, baseline_acc, best in rows:
    line = "{:10s} | {:5d} | {:+d}: {:5.1f}".format(
        reln[:10], count, offset, 100 * baseline_acc)
    for name in names:
      acc, layer, head, direction = best[name]
      line += " | {:5.1f} ({:2d}-{:2d} {:s})".format(
          100 * acc, layer, head, "d->h" if direction == "dep->head" else
          "h<-d")
    print(line)


###############################################################################
# Probing classifiers (Section 5 of Clark et al.), ported from TF1 to PyTorch
###############################################################################

N_DISTANCE_FEATURES = 8


def make_distance_features(seq_len):
  """Constructs distance features for a sentence."""
  # how much ahead/behind the other word is
  distances = np.zeros((seq_len, seq_len))
  for i in range(seq_len):
    for j in range(seq_len):
      if i < j:
        distances[i, j] = (j - i) / float(seq_len)
  feature_matrices = [distances, distances.T]

  # indicator features on if other word is up to 2 words ahead/behind
  for k in range(3):
    for direction in ([1] if k == 0 else [-1, 1]):
      feature_matrices.append(np.eye(seq_len, k=k*direction))
  features = np.stack(feature_matrices)

  # additional indicator feature for ROOT
  features = np.concatenate(
      [np.zeros([N_DISTANCE_FEATURES - 1, seq_len, 1]),
       features], -1)
  root = np.zeros((1, seq_len, seq_len + 1))
  root[:, :, 0] = 1

  return np.concatenate([features, root], 0)


class WordEmbeddings(object):
  """Pretrained GloVe embeddings in the format of the paper's data
  (embeddings.pkl: [vocab, dim] array; vocab.pkl: word -> row)."""

  def __init__(self, embeddings_file, vocab_file):
    import utils
    self.pretrained_embeddings = np.asarray(
        utils.load_pickle(embeddings_file), dtype=np.float32)
    self.vocab = utils.load_pickle(vocab_file)

  def tokid(self, w):
    return self.vocab.get(w.lower(), 0)


def _glorot_uniform_(tensor, fan_in, fan_out):
  limit = np.sqrt(6.0 / (fan_in + fan_out))
  with torch.no_grad():
    tensor.uniform_(-limit, limit)
  return tensor


class Probe(torch.nn.Module):
  """The probing classifier used in Section 5 of Clark et al.

  Initialization mirrors the TF1 defaults of the original (Glorot-uniform
  weights, zero biases). With the default arguments this is the
  "attention-only" probe: a linear combination of the 144 attention maps and
  their transposes (288 features) plus two scalars scoring ROOT from the
  attention to [CLS]/[SEP].

  NOTE on fix_root_alignment: in the original code the attention/distance
  features put ROOT in column 0 (candidate heads are [ROOT, w_1, ..., w_n]),
  but the word-pair representations append the dummy ROOT column at the end
  ([w_1, ..., w_n, ROOT]), so the embedding used for candidate head w_j is that
  of w_{j+1}. The default (False) keeps this behaviour to reproduce the paper;
  True puts the dummy ROOT column first. This only affects probes that use
  word embeddings.
  """

  def __init__(self, n_maps=144, use_distance_features=False, use_words=False,
               use_attns=True, include_transpose=True, hidden_layer=False,
               embeddings=None, fix_root_alignment=False):
    super().__init__()
    self.use_distance_features = use_distance_features
    self.use_words = use_words
    self.use_attns = use_attns
    self.include_transpose = include_transpose
    self.hidden_layer = hidden_layer
    self.fix_root_alignment = fix_root_alignment
    self.embeddings = embeddings

    n_features = 0
    if use_attns:
      n_features += 2 * n_maps if include_transpose else n_maps
      self.root_start = torch.nn.Parameter(_glorot_uniform_(
          torch.empty(()), 1, 1))
      self.root_end = torch.nn.Parameter(_glorot_uniform_(
          torch.empty(()), 1, 1))
    else:
      n_features += 1
    if use_distance_features:
      n_features += N_DISTANCE_FEATURES
    if use_words:
      emb = torch.tensor(embeddings.pretrained_embeddings)
      self.word_embedding = torch.nn.Embedding.from_pretrained(emb,
                                                               freeze=True)
      word_dim = 2 * emb.shape[1]
      if not use_attns:
        n_features += word_dim
    self.n_features = n_features

    def dense(n_in, n_out):
      layer = torch.nn.Linear(n_in, n_out)
      _glorot_uniform_(layer.weight, n_in, n_out)
      torch.nn.init.zeros_(layer.bias)
      return layer

    if use_words and use_attns:
      self.weights = dense(word_dim, n_features)
    elif hidden_layer:
      self.hidden = dense(n_features, 256)
      self.out = dense(256, 1)
    else:
      self.attn_map_weights = torch.nn.Parameter(_glorot_uniform_(
          torch.empty(n_features), n_features, n_features))

  def forward(self, attns, words):
    """attns: [layers, heads, n + 2, n + 2] maps (with [CLS]/[SEP]);
    words: list of n words. Returns [n, n + 1] logits over (ROOT, w_1..w_n)."""
    n_words = len(words)
    features = []
    if self.use_attns:
      attns = torch.as_tensor(np.asarray(attns, dtype=np.float32))
      seq_len = attns.shape[-1]
      maps = attns.reshape(-1, seq_len, seq_len)
      if self.include_transpose:
        maps = torch.cat([maps, maps.transpose(1, 2)], 0)
      root = (self.root_start * maps[:, 1:-1, 0] +
              self.root_end * maps[:, 1:-1, -1])
      features.append(torch.cat([root.unsqueeze(-1), maps[:, 1:-1, 1:-1]],
                                -1))
    else:
      features.append(torch.zeros(1, n_words, n_words + 1))
    if self.use_distance_features:
      features.append(torch.tensor(make_distance_features(n_words),
                                   dtype=torch.float32))
    features = torch.cat(features, 0)

    if self.use_words:
      ids = torch.tensor([self.embeddings.tokid(w) for w in words])
      emb = self.word_embedding(ids)
      tiled_vertical = emb.unsqueeze(0).expand(n_words, -1, -1)
      tiled_horizontal = emb.unsqueeze(1).expand(-1, n_words, -1)
      word_reprs = torch.cat([tiled_horizontal, tiled_vertical], -1)
      dummy = torch.zeros(n_words, 1, word_reprs.shape[-1])
      if self.fix_root_alignment:
        word_reprs = torch.cat([dummy, word_reprs], 1)
      else:
        word_reprs = torch.cat([word_reprs, dummy], 1)
      if not self.use_attns:
        features = torch.cat([features, word_reprs.permute(2, 0, 1)], 0)

    features = features.permute(1, 2, 0)  # [n, n + 1, n_features]
    if self.use_words and self.use_attns:
      return (self.weights(word_reprs) * features).sum(-1)
    if self.hidden_layer:
      return self.out(torch.tanh(self.hidden(features))).squeeze(-1)
    return (self.attn_map_weights * features).sum(-1)


def attn_linear_combo(**kwargs):
  return Probe(**kwargs)


def attn_and_words(embeddings, **kwargs):
  return Probe(use_words=True, embeddings=embeddings, **kwargs)


def words_and_distances(embeddings, **kwargs):
  return Probe(use_distance_features=True, use_attns=False, use_words=True,
               hidden_layer=True, embeddings=embeddings, **kwargs)


def evaluate_probe(probe, data, key="attns", normalize=False):
  """UAS excluding punctuation (standard for Stanford Dependencies)."""
  correct, total = 0, 0
  probe.eval()
  with torch.no_grad():
    for example in data:
      logits = probe(get_maps(example, key, normalize), example["words"])
      for head, prediction, reln in zip(
          example["heads"], logits.argmax(-1).tolist(), example["relns"]):
        if reln != "punct":
          correct += int(head == prediction)
          total += 1
  return correct / float(total)


def run_training(probe, train_data, dev_data, key="attns", normalize=False,
                 n_epochs=1, lr=0.002, seed=None, log_every=2000):
  """Trains (batch size 1, Adam, summed cross-entropy, one epoch, data in the
  given order, as in the original) and returns the dev UAS."""
  if seed is not None:
    torch.manual_seed(seed)
  opt = torch.optim.Adam(probe.parameters(), lr=lr)
  for epoch in range(n_epochs):
    probe.train()
    for i, example in enumerate(train_data):
      if log_every and i % log_every == 0:
        print("epoch {:} {:}/{:}".format(epoch + 1, i, len(train_data)))
      logits = probe(get_maps(example, key, normalize), example["words"])
      loss = torch.nn.functional.cross_entropy(
          logits, torch.tensor(example["heads"]), reduction="sum")
      opt.zero_grad()
      loss.backward()
      opt.step()
  uas = evaluate_probe(probe, dev_data, key, normalize)
  print("UAS: {:.1f}".format(100 * uas))
  return uas
