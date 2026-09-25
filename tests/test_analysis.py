"""Sanity checks of analysis_utils on synthetic maps with a known answer.

  python -m pytest tests/test_analysis.py
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analysis_utils as au  # noqa: E402


def random_tree(n, rng):
  # word i (1-based) gets a head among 0..n, one root, no cycles: attach each
  # word to an earlier word or to the root word
  root = rng.randint(1, n + 1)
  heads, order = [0] * n, [root] + [i for i in range(1, n + 1) if i != root]
  for k, w in enumerate(order[1:], 1):
    heads[w - 1] = order[rng.randint(0, k)]
  return heads


def make_data(n_examples=60, layers=2, heads=3, seed=0):
  """Head (1, 2) attends from each dependent to its gold head, everything else
  is noise; rows sum to one."""
  rng = np.random.RandomState(seed)
  data = []
  for _ in range(n_examples):
    n = rng.randint(4, 12)
    tree = random_tree(n, rng)
    maps = rng.rand(layers, heads, n + 2, n + 2)
    oracle = np.full((n + 2, n + 2), 0.01)
    for i, h in enumerate(tree):
      oracle[i + 1, h if h > 0 else n + 1] = 1.0  # root -> [SEP]
    maps[1, 2] = oracle
    maps /= maps.sum(-1, keepdims=True)
    relns = ["root" if h == 0 else ("det" if i % 2 else "nsubj")
             for i, h in enumerate(tree)]
    data.append({"words": ["w%d" % i for i in range(n)], "heads": tree,
                 "relns": relns, "attns": maps.astype(np.float16),
                 "norms": (3 * maps).astype(np.float16),
                 "tokens": ["[CLS]"] + ["w%d" % i for i in range(n)] +
                           ["[SEP]"]})
  return data


def test_oracle_head_is_found():
  data = make_data()
  for key in ["attns", "norms"]:
    scores = au.get_head_scores(data, key)
    acc, layer, head, direction = au.get_all_scores(scores, "det")[0]
    assert (layer, head, direction) == (1, 2, "dep->head")
    assert acc == 1.0
    # non-root words are all recovered; root words never are
    n_root = len(data)
    n_words = sum(len(e["words"]) for e in data)
    np.testing.assert_allclose(scores["dep->head"][1][2]["all"],
                               (n_words - n_root) / float(n_words))


def test_offset_baseline():
  example = {"words": ["a", "b", "c"], "heads": [2, 0, 2],
             "relns": ["det", "root", "obj"]}
  assert au.offset_predictor(0)(example) == [1, 2, 3]
  assert au.offset_predictor(1)(example) == [2, 3, 3]
  assert au.offset_predictor(-1)(example) == [0, 1, 2]


def test_evaluate_keeps_perfect_relations():
  example = {"words": ["a", "b"], "heads": [2, 0], "relns": ["det", "root"]}
  scores = au.evaluate_predictor([example], lambda e: [2, 1])
  assert scores["det"] == 1.0 and scores["root"] == 0.0


def test_probe_learns_oracle_map():
  torch.manual_seed(0)
  train, dev = make_data(300, seed=1), make_data(50, seed=2)
  probe = au.attn_linear_combo(n_maps=6)
  uas = au.run_training(probe, train, dev, n_epochs=2, log_every=0)
  assert uas > 0.95


def test_token_stats_rows():
  data = make_data(5)
  stats = au.token_stats(data, "attns")
  # attention to the last position ([SEP]) + everything else sums to one
  total = sum(stats[k] for k in ["sep", "cls"]) 
  assert np.all(total <= 1.0 + 1e-3)
  norm_stats = au.token_stats(data, "norms", normalize=True)
  np.testing.assert_allclose(norm_stats["sep"], stats["sep"], atol=1e-2)


def test_head_distances_torch_matches_numpy():
  import head_distances
  rng = np.random.RandomState(0)
  maps = rng.rand(20, 7, 7)
  maps /= maps.sum(-1, keepdims=True)
  np.testing.assert_allclose(head_distances.js_torch(maps, "cpu", chunk_size=6),
                             head_distances.js_numpy(maps), rtol=1e-4,
                             atol=1e-5)


def _brute_force_mst(scores):
  import itertools
  n = scores.shape[0]
  best, best_heads = -np.inf, None
  for heads in itertools.product(range(n), repeat=n - 1):
    heads = (-1,) + heads
    if any(h == d for d, h in enumerate(heads) if d > 0):
      continue
    ok = True
    for d in range(1, n):  # every node must reach the root
      seen, node = set(), d
      while node != 0:
        if node in seen:
          ok = False
          break
        seen.add(node)
        node = heads[node]
      if not ok:
        break
    if not ok:
      continue
    s = sum(scores[heads[d], d] for d in range(1, n))
    if s > best:
      best, best_heads = s, heads
  return best, best_heads


def test_chu_liu_edmonds_matches_brute_force():
  import syntax_extended as se
  rng = np.random.RandomState(0)
  for trial in range(300):
    n = rng.randint(2, 7)
    scores = rng.rand(n, n)
    best, _ = _brute_force_mst(scores)
    heads = se.chu_liu_edmonds(scores)
    assert se._find_cycle(heads) is None
    got = sum(scores[heads[d], d] for d in range(1, n))
    np.testing.assert_allclose(got, best, err_msg=str((trial, n)))


def test_tree_decoding_on_oracle_maps():
  import syntax_extended as se
  data = make_data(40)
  for ex in data:  # the oracle head maps the root word to [SEP]; ok for MST
    for direction, key in [("dep->head", "attns")]:
      pred = se.decode_directed(ex, key, 1, 2, direction)
      assert list(pred) == ex["heads"]
    edges = se.decode_undirected(ex, "attns", 1, 2)
    c, t = se.uuas(ex, edges, ignore_punct=False)
    assert c == t


def test_vectorized_head_accuracies_match_reference():
  import syntax_extended as se
  data = make_data(30)
  acc, _ = se.head_accuracies(data, "attns")
  ref = au.get_head_scores(data, "attns")
  for d, direction in enumerate(se.DIRECTIONS):
    for l in range(2):
      for h in range(3):
        for reln, a in ref[direction][l][h].items():
          np.testing.assert_allclose(acc[reln][d, l, h], a)


def test_cross_fit_and_stats():
  import syntax_extended as se
  data = make_data(60)
  cf = se.cross_fit_correctness(data, "attns", ["all", "det"])
  assert cf["det"][0].all()  # the oracle head is selected in every fold
  assert len(cf["all"][0]) == sum(len(e["words"]) for e in data)
  a = np.array([True] * 50 + [False] * 50)
  assert se.mcnemar(a, a) == 1.0
  assert se.mcnemar(a, np.ones(100, bool)) < 1e-10
  adj = se.holm({"x": 0.01, "y": 0.04, "z": 0.5})
  assert abs(adj["x"] - 0.03) < 1e-12 and abs(adj["y"] - 0.08) < 1e-12


def test_exclude_punct():
  import syntax_extended as se
  ex = make_data(1)[0]
  ex["words"][1] = ","
  preds = se.all_head_predictions(ex, "attns", exclude_punct=True)
  assert not (preds == 2).any()  # word 2 is "," -> never predicted as head


def test_tree_counts_and_table():
  import syntax_extended as se
  data = make_data(20)
  counts = se.tree_counts_all_heads(data, "attns")
  table = se.tree_table(counts)
  score, (l, h), held = table["UAS dep->head"]
  assert (l, h) == (1, 2) and score == 1.0 and held == 1.0
