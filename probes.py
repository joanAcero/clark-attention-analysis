"""Parent probes over attention maps (RG1).

For each dependent i, a softmax over the candidate parents j != i and ROOT:

  p(j | i)    ∝ exp(g(phi_ij)),   phi_ij = [M^k_ij, M^k_ji]_k   (pair features)
  p(ROOT | i) ∝ exp(g_r(sigma_i)), sigma_i = [M^k_i,CLS, M^k_i,SEP]_k (root features)

where M^k is the word-level map of head k (attention weights, or any other
[layers, heads, n + 2, n + 2] map). With 144 heads there are 288 pair and 288
root features. g is linear ("linear") or a one-hidden-layer MLP ("mlp").
Decisions are independent per token (no tree constraint).

Controls:
  * the same probe on maps from a randomly initialized BERT (other data file);
  * feature_type="position": positional features with the same dimensionality
    (one-hot of the relative offset j - i for pairs, one-hot of the position of
    i for ROOT), so a linear probe has the same number of parameters.

Features are standardized with statistics from the training data, so linear
weights are comparable across heads (effect of one standard deviation).
A relation-conditioned linear probe ("relation") has one weight vector per
dependency relation of the dependent (the gold relation is given: it is an
analysis tool, not a parser).
"""

import collections

import numpy as np
import torch
from torch import nn

N_POSITION = 144  # offsets -144..144 (excluding 0) -> 288 pair features


###############################################################################
# Features
###############################################################################

def _maps(example, key, layers=None):
  m = np.asarray(example[key], dtype=np.float32)
  if layers is not None:
    m = m[layers]
  return m.reshape(-1, m.shape[-2], m.shape[-1])  # [K, n + 2, n + 2]


def example_features(example, key="attns", feature_type="attention",
                     layers=None):
  """phi [n, n, F] and sigma [n, Fr] for one sentence (float32 numpy)."""
  n = len(example["words"])
  if feature_type == "attention":
    m = _maps(example, key, layers)
    words = m[:, 1:-1, 1:-1]  # [K, n, n]
    phi = np.concatenate([words, words.transpose(0, 2, 1)], 0)  # [2K, n, n]
    sigma = np.concatenate([m[:, 1:-1, 0], m[:, 1:-1, -1]], 0)  # [2K, n]
    return phi.transpose(1, 2, 0), sigma.T
  if feature_type == "position":
    offsets = np.arange(n)[None, :] - np.arange(n)[:, None]  # j - i
    offsets = np.clip(offsets, -N_POSITION, N_POSITION)
    idx = np.where(offsets < 0, offsets + N_POSITION, offsets + N_POSITION - 1)
    phi = np.zeros((n, n, 2 * N_POSITION), dtype=np.float32)
    rows, cols = np.nonzero(offsets != 0)
    phi[rows, cols, idx[rows, cols]] = 1.0
    sigma = np.zeros((n, 2 * N_POSITION), dtype=np.float32)
    sigma[np.arange(n), np.minimum(np.arange(n), 2 * N_POSITION - 1)] = 1.0
    return phi, sigma
  raise ValueError("Unknown feature type", feature_type)


class Standardizer(object):
  """Per-feature mean/std over valid (i != j) pairs and over tokens.
  identity=True leaves features unchanged (used for the one-hot positional
  control, where standardizing would blow up rare offsets)."""

  def __init__(self, data, key, feature_type, layers=None, max_sentences=3000,
               seed=0, identity=False):
    self.identity = identity
    if identity:
      return
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(data))[:max_sentences]
    s1 = s2 = r1 = r2 = 0.0
    n_pairs = n_tokens = 0
    for i in idx:
      phi, sigma = example_features(data[i], key, feature_type, layers)
      n = phi.shape[0]
      mask = ~np.eye(n, dtype=bool)
      p = phi[mask].astype(np.float64)
      s1, s2, n_pairs = s1 + p.sum(0), s2 + (p ** 2).sum(0), n_pairs + len(p)
      r = sigma.astype(np.float64)
      r1, r2, n_tokens = r1 + r.sum(0), r2 + (r ** 2).sum(0), n_tokens + len(r)
    self.phi_mean = s1 / n_pairs
    self.phi_std = np.sqrt(np.maximum(s2 / n_pairs - self.phi_mean ** 2, 0)) + 1e-6
    self.sigma_mean = r1 / n_tokens
    self.sigma_std = np.sqrt(np.maximum(r2 / n_tokens - self.sigma_mean ** 2, 0)) + 1e-6

  def __call__(self, phi, sigma):
    if self.identity:
      return phi, sigma
    return ((phi - self.phi_mean) / self.phi_std).astype(np.float32), \
        ((sigma - self.sigma_mean) / self.sigma_std).astype(np.float32)


class Batcher(object):
  """Builds padded batches:
    phi [B, N, N, F], sigma [B, N, Fr], pair_mask [B, N, N + 1] (valid
    candidates, ROOT at 0), heads [B, N] (-1 = padding), rel [B, N],
    punct [B, N]."""

  def __init__(self, data, key, feature_type, standardizer, relations=None,
               layers=None):
    self.data = data
    self.key, self.feature_type, self.layers = key, feature_type, layers
    self.std = standardizer
    self.relations = relations or {}

  def rel_id(self, r):
    return self.relations.get(r, self.relations.get("<other>", 0))

  def batches(self, batch_size=32, shuffle=False, rng=None, device="cpu",
              feature_mask=None):
    order = np.arange(len(self.data))
    if shuffle:
      rng.shuffle(order)
    for start in range(0, len(order), batch_size):
      exs = [self.data[i] for i in order[start:start + batch_size]]
      yield self.make_batch(exs, device, feature_mask)

  def make_batch(self, exs, device="cpu", feature_mask=None):
    feats = [self.std(*example_features(e, self.key, self.feature_type,
                                        self.layers)) for e in exs]
    N = max(len(e["words"]) for e in exs)
    F, Fr = feats[0][0].shape[-1], feats[0][1].shape[-1]
    B = len(exs)
    phi = np.zeros((B, N, N, F), dtype=np.float32)
    sigma = np.zeros((B, N, Fr), dtype=np.float32)
    mask = np.zeros((B, N, N + 1), dtype=bool)
    heads = np.full((B, N), -1, dtype=np.int64)
    rel = np.zeros((B, N), dtype=np.int64)
    punct = np.zeros((B, N), dtype=bool)
    for b, (e, (p, s)) in enumerate(zip(exs, feats)):
      n = len(e["words"])
      phi[b, :n, :n] = p
      sigma[b, :n] = s
      mask[b, :n, 0] = True
      mask[b, :n, 1:n + 1] = ~np.eye(n, dtype=bool)
      heads[b, :n] = e["heads"]
      rel[b, :n] = [self.rel_id(r) for r in e["relns"]]
      punct[b, :n] = [r == "punct" for r in e["relns"]]
    if feature_mask is not None:  # ablation: set features to their mean (0)
      phi[..., ~feature_mask[0]] = 0.0
      sigma[..., ~feature_mask[1]] = 0.0
    t = lambda x: torch.as_tensor(x, device=device)
    return t(phi), t(sigma), t(mask), t(heads), t(rel), t(punct)


###############################################################################
# Models
###############################################################################

class ParentProbe(nn.Module):

  def __init__(self, n_features, n_root_features, kind="linear", hidden=256,
               n_relations=1):
    super().__init__()
    self.kind = kind
    if kind == "linear":
      self.pair = nn.Linear(n_features, 1, bias=False)
      self.root = nn.Linear(n_root_features, 1)
    elif kind == "mlp":
      self.pair = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(),
                                nn.Linear(hidden, 1))
      self.root = nn.Sequential(nn.Linear(n_root_features, hidden), nn.ReLU(),
                                nn.Linear(hidden, 1))
    elif kind == "relation":
      self.pair_w = nn.Parameter(torch.zeros(n_relations, n_features))
      self.root_w = nn.Parameter(torch.zeros(n_relations, n_root_features))
      self.root_b = nn.Parameter(torch.zeros(n_relations))
    else:
      raise ValueError("Unknown probe kind", kind)

  def forward(self, phi, sigma, mask, rel=None):
    if self.kind == "relation":
      w = self.pair_w[rel]  # [B, N, F]
      pair = torch.einsum("bijf,bif->bij", phi, w)
      root = torch.einsum("bif,bif->bi", sigma, self.root_w[rel]) + self.root_b[rel]
    else:
      pair = self.pair(phi).squeeze(-1)  # [B, N, N]
      root = self.root(sigma).squeeze(-1)  # [B, N]
    logits = torch.cat([root.unsqueeze(-1), pair], -1)  # ROOT at index 0
    return logits.masked_fill(~mask, float("-inf"))

  def weights(self):
    """(pair weights, root weights) of a linear probe as numpy arrays."""
    if self.kind == "linear":
      return (self.pair.weight.detach().cpu().numpy()[0],
              self.root.weight.detach().cpu().numpy()[0])
    if self.kind == "relation":
      return (self.pair_w.detach().cpu().numpy(),
              self.root_w.detach().cpu().numpy())
    raise ValueError("weights() is only defined for linear probes")


###############################################################################
# Training / evaluation
###############################################################################

def evaluate(probe, batcher, device="cpu", feature_mask=None, batch_size=64):
  """UAS excluding punctuation, root accuracy, and per-relation accuracy."""
  probe.eval()
  correct = collections.Counter()
  total = collections.Counter()
  with torch.no_grad():
    for phi, sigma, mask, heads, rel, punct in batcher.batches(
        batch_size, device=device, feature_mask=feature_mask):
      pred = probe(phi, sigma, mask, rel).argmax(-1)
      valid = heads >= 0
      ok = (pred == heads) & valid
      scored = valid & ~punct
      correct["UAS"] += int((ok & scored).sum())
      total["UAS"] += int(scored.sum())
      is_root = valid & (heads == 0)
      correct["root"] += int((ok & is_root).sum())
      total["root"] += int(is_root.sum())
  return {k: correct[k] / max(total[k], 1) for k in total}


def per_relation_accuracy(probe, batcher, data, device="cpu", batch_size=64):
  """Accuracy per gold relation (same token order as `data`)."""
  probe.eval()
  correct, total = collections.Counter(), collections.Counter()
  with torch.no_grad():
    start = 0
    for phi, sigma, mask, heads, rel, punct in batcher.batches(
        batch_size, device=device):
      pred = probe(phi, sigma, mask, rel).argmax(-1).cpu().numpy()
      for b in range(pred.shape[0]):
        e = data[start + b]
        for i, r in enumerate(e["relns"]):
          total[r] += 1
          correct[r] += int(pred[b, i] == e["heads"][i])
      start += pred.shape[0]
  return {r: correct[r] / total[r] for r in total}


def train_probe(train_batcher, val_batcher, kind="linear", seed=0,
                device="cpu", lr=None, max_epochs=10, patience=2,
                batch_size=32, hidden=256, n_relations=1, verbose=True):
  """Adam, mean token cross-entropy, early stopping on validation UAS."""
  torch.manual_seed(seed)
  rng = np.random.RandomState(seed)
  phi, sigma, *_ = train_batcher.make_batch(train_batcher.data[:1])
  probe = ParentProbe(phi.shape[-1], sigma.shape[-1], kind, hidden,
                      n_relations).to(device)
  lr = lr or (1e-3 if kind == "mlp" else 1e-2)
  opt = torch.optim.Adam(probe.parameters(), lr=lr)
  best, best_state, bad = -1.0, None, 0
  for epoch in range(max_epochs):
    probe.train()
    for phi, sigma, mask, heads, rel, punct in train_batcher.batches(
        batch_size, shuffle=True, rng=rng, device=device):
      logits = probe(phi, sigma, mask, rel)
      valid = heads >= 0
      loss = nn.functional.cross_entropy(logits[valid], heads[valid])
      opt.zero_grad()
      loss.backward()
      opt.step()
    val = evaluate(probe, val_batcher, device)["UAS"]
    if verbose:
      print("  epoch {:d}: val UAS {:.1f}".format(epoch + 1, 100 * val))
    if val > best:
      best, bad = val, 0
      best_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}
    else:
      bad += 1
      if bad >= patience:
        break
  probe.load_state_dict(best_state)
  return probe


###############################################################################
# Analyses
###############################################################################

def head_ablation(probe, batcher, n_layers=12, n_heads=12, device="cpu"):
  """UAS drop when all features of one head (both directions and both root
  features) are set to their training mean. Returns [layers, heads]."""
  base = evaluate(probe, batcher, device)["UAS"]
  K = n_layers * n_heads
  drops = np.zeros((n_layers, n_heads))
  for k in range(K):
    fm_pair = np.ones(2 * K, dtype=bool)
    fm_root = np.ones(2 * K, dtype=bool)
    fm_pair[[k, K + k]] = False
    fm_root[[k, K + k]] = False
    uas = evaluate(probe, batcher, device, feature_mask=(fm_pair, fm_root))["UAS"]
    drops[k // n_heads, k % n_heads] = base - uas
  return base, drops


def layer_ablation(probe, batcher, n_layers=12, n_heads=12, device="cpu"):
  base = evaluate(probe, batcher, device)["UAS"]
  K = n_layers * n_heads
  drops = np.zeros(n_layers)
  for l in range(n_layers):
    ks = np.arange(l * n_heads, (l + 1) * n_heads)
    fm = np.ones(2 * K, dtype=bool)
    fm[np.concatenate([ks, K + ks])] = False
    drops[l] = base - evaluate(probe, batcher, device,
                               feature_mask=(fm, fm.copy()))["UAS"]
  return base, drops


def weight_grid(pair_weights, n_layers=12, n_heads=12):
  """Linear pair weights [2K] -> [2 (d->h, h<-d), layers, heads]."""
  return np.asarray(pair_weights).reshape(2, n_layers, n_heads)


def seed_stability(weight_vectors):
  """Mean pairwise Pearson correlation between weight vectors of different
  seeds, and the mean Jaccard overlap of their top-10 |weight| features."""
  W = np.asarray(weight_vectors)
  corrs, jacc = [], []
  for a in range(len(W)):
    for b in range(a + 1, len(W)):
      corrs.append(np.corrcoef(W[a], W[b])[0, 1])
      ta = set(np.argsort(-np.abs(W[a]))[:10])
      tb = set(np.argsort(-np.abs(W[b]))[:10])
      jacc.append(len(ta & tb) / len(ta | tb))
  return float(np.mean(corrs)), float(np.mean(jacc))


def make_batcher(data, key="attns", feature_type="attention", layers=None,
                 standardizer_data=None, relations=None):
  """Batcher with a standardizer fitted on `standardizer_data` (default:
  `data`). Pass the training data as standardizer_data for dev batchers."""
  std = Standardizer(standardizer_data if standardizer_data is not None
                     else data, key, feature_type, layers,
                     identity=(feature_type == "position"))
  return Batcher(data, key, feature_type, std, relations, layers)


def relation_vocab(data, min_count=200):
  """Relations with >= min_count tokens get their own weights; the rest share
  "<other>". "root" is always mapped to "<other>": its own weights would tell
  the probe that the head is ROOT."""
  counts = collections.Counter(r for e in data for r in e["relns"])
  rels = [r for r, c in counts.most_common() if c >= min_count and r != "root"]
  vocab = {r: i for i, r in enumerate(rels)}
  vocab["<other>"] = len(vocab)
  return vocab
