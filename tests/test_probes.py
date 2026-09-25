"""Checks of probes.py on synthetic maps with a known answer.

  python -m pytest tests/test_probes.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import probes  # noqa: E402
from test_analysis import make_data  # noqa: E402


def test_features_shapes_and_positions():
  ex = make_data(1)[0]
  n = len(ex["words"])
  phi, sigma = probes.example_features(ex, "attns")
  assert phi.shape == (n, n, 12) and sigma.shape == (n, 12)  # 2 x 2 x 3 heads
  m = np.asarray(ex["attns"], dtype=np.float32).reshape(6, n + 2, n + 2)
  np.testing.assert_allclose(phi[0, 1, :6], m[:, 1, 2])  # word 1 -> word 2
  np.testing.assert_allclose(phi[0, 1, 6:], m[:, 2, 1])  # transpose
  np.testing.assert_allclose(sigma[0, 6:], m[:, 1, -1])  # word 1 -> [SEP]
  phi, sigma = probes.example_features(ex, feature_type="position")
  assert phi.shape == (n, n, 288)
  assert phi[0, 1].argmax() == probes.N_POSITION  # offset +1
  assert phi[1, 0].argmax() == probes.N_POSITION - 1  # offset -1
  assert phi[2, 2].sum() == 0 and np.allclose(phi.sum(-1), 1 - np.eye(n))


def test_linear_probe_learns_oracle_and_ablation_finds_it():
  train, dev = make_data(200, seed=1), make_data(40, seed=2)
  tb = probes.make_batcher(train)
  db = probes.make_batcher(dev, standardizer_data=train)
  probe = probes.train_probe(tb, db, "linear", max_epochs=5, verbose=False)
  assert probes.evaluate(probe, db)["UAS"] > 0.95
  base, drops = probes.head_ablation(probe, db, n_layers=2, n_heads=3)
  assert np.unravel_index(np.argmax(drops), drops.shape) == (1, 2)
  grid = probes.weight_grid(probe.weights()[0], 2, 3)
  assert np.unravel_index(np.argmax(grid[0]), grid[0].shape) == (1, 2)


def test_mlp_relation_and_position_probes_run():
  train, dev = make_data(200, seed=1), make_data(30, seed=2)
  for kind in ["mlp", "relation"]:
    rels = probes.relation_vocab(train, min_count=1)
    tb = probes.make_batcher(train, relations=rels)
    db = probes.make_batcher(dev, standardizer_data=train, relations=rels)
    p = probes.train_probe(tb, db, kind, max_epochs=6, verbose=False,
                           n_relations=len(rels), hidden=16, batch_size=8,
                           patience=10)
    assert probes.evaluate(p, db)["UAS"] > 0.9
  tb = probes.make_batcher(train, feature_type="position")
  db = probes.make_batcher(dev, feature_type="position")
  p = probes.train_probe(tb, db, "linear", max_epochs=2, verbose=False)
  assert 0.0 <= probes.evaluate(p, db)["UAS"] <= 1.0


def test_seed_stability():
  w = np.random.RandomState(0).randn(20)
  corr, jacc = probes.seed_stability([w, w, w])
  assert abs(corr - 1) < 1e-9 and jacc == 1.0
