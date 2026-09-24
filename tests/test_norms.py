"""Checks norm_utils / extract_norms.py.

  python -m pytest tests/test_norms.py

The comparison against Kobayashi et al.'s own implementation (BertNormOutput in
their fork of transformers 3.0) runs only if KOBAYASHI_SRC points to
norm-analysis-of-transformer/emnlp2020/transformers/src, e.g.

  git clone https://github.com/gorokoba560/norm-analysis-of-transformer
  KOBAYASHI_SRC=norm-analysis-of-transformer/emnlp2020/transformers/src \
      python -m pytest tests/test_norms.py

(the fork additionally needs `sentencepiece` and `sacremoses` installed).
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import torch
from transformers import BertConfig, BertModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import bpe_utils  # noqa: E402
import extract_norms  # noqa: E402
import norm_utils  # noqa: E402

CONFIG = dict(vocab_size=200, hidden_size=96, num_hidden_layers=3,
              num_attention_heads=12, intermediate_size=128,
              max_position_embeddings=64)


def random_layer(n=9, hidden=96, heads=12, seed=0):
  g = torch.Generator().manual_seed(seed)
  attn = torch.softmax(torch.randn(heads, n, n, generator=g), -1)
  value = torch.randn(n, hidden, generator=g)
  dense = torch.randn(hidden, hidden, generator=g) / hidden ** 0.5
  return attn, value, dense


def direct_vectors(attn, value, dense, heads):
  """alpha_ij f(x_j) as explicit [h, n, n, hidden] vectors (as in Kobayashi's
  BertNormOutput)."""
  n, hidden = value.shape
  d = hidden // heads
  v = value.view(n, heads, 1, d)
  w = dense.view(hidden, heads, d).permute(1, 2, 0)  # [h, d, hidden]
  fx = v.matmul(w).view(n, heads, hidden).permute(1, 0, 2)  # [h, n, hidden]
  return fx, torch.einsum("hks,hsd->hksd", attn, fx)


def test_token_level_matches_direct_computation():
  attn, value, dense = random_layer()
  attn, value, dense = attn.double(), value.double(), dense.double()
  out = norm_utils.attention_and_norms(attn, value, dense, 12)
  fx, weighted = direct_vectors(attn, value, dense, 12)
  torch.testing.assert_close(out["attns"], attn)
  torch.testing.assert_close(out["fx_norms"], fx.norm(dim=-1))
  torch.testing.assert_close(out["norms"], weighted.norm(dim=-1))
  torch.testing.assert_close(out["summed_norms"],
                             weighted.sum(0).norm(dim=-1))
  # ||alpha_ij f(x_j)|| = alpha_ij ||f(x_j)||
  torch.testing.assert_close(out["norms"],
                             attn * out["fx_norms"].unsqueeze(1))


def test_word_level():
  attn, value, dense = random_layer(n=9)
  attn, value, dense = attn.double(), value.double(), dense.double()
  # [CLS] w1 w2(3 pieces) w3 w4(2 pieces) [SEP]
  w2t = [[0], [1], [2, 3, 4], [5], [6, 7], [8]]
  out = norm_utils.attention_and_norms(attn, value, dense, 12,
                                       words_to_tokens=w2t)
  # attention: identical to Clark's conversion
  clark = np.stack([bpe_utils.get_word_word_attention(a, w2t, mode="mean")
                    for a in attn.numpy()])
  np.testing.assert_allclose(out["attns"].numpy(), clark, rtol=1e-10)
  np.testing.assert_allclose(out["attns"].sum(-1).numpy(), 1.0, rtol=1e-6)

  # norms: norm of the merged vectors
  _, weighted = direct_vectors(attn, value, dense, 12)
  R, C = norm_utils.word_merge_matrices(w2t, 9, torch.float64)
  merged = torch.einsum("Ia,habd,bJ->hIJd", R, weighted, C)
  torch.testing.assert_close(out["norms"], merged.norm(dim=-1))
  torch.testing.assert_close(out["summed_norms"], merged.sum(0).norm(dim=-1))

  # "sum" mode is Clark's conversion applied to the token-level norm matrix
  out_sum = norm_utils.attention_and_norms(attn, value, dense, 12,
                                           words_to_tokens=w2t,
                                           word_norm_mode="sum")
  clark_norms = np.stack([
      bpe_utils.get_word_word_attention(a, w2t, mode="mean")
      for a in weighted.norm(dim=-1).numpy()])
  np.testing.assert_allclose(out_sum["norms"].numpy(), clark_norms,
                             rtol=1e-10)
  # triangle inequality
  assert torch.all(out["norms"] <= out_sum["norms"] + 1e-12)


def _extract(model, input_ids, outputs=("attns", "norms", "summed_norms",
                                        "fx_norms")):
  class E(object):
    pass
  examples = []
  for ids in input_ids:
    e = E()
    e.input_ids, e.segment_ids = list(ids), [0] * len(ids)
    examples.append(e)
  extractor = extract_norms.NormMapExtractor(model, "cpu", list(outputs))
  return extractor.get_maps(examples)


def test_padding_does_not_change_maps():
  torch.manual_seed(0)
  model = BertModel(BertConfig(**CONFIG, attn_implementation="eager")).eval()
  short, long = [1, 5, 6, 7, 2], [1, 8, 9, 10, 11, 12, 13, 2]
  alone = _extract(model, [short])[0]
  batched = _extract(model, [short, long])[0]
  for k in alone:
    np.testing.assert_allclose(alone[k], batched[k], atol=1e-5)


KOBAYASHI_SCRIPT = r"""
import sys, torch
sys.path.insert(0, sys.argv[1])
from transformers import BertConfig, BertModel
torch.manual_seed(0)
model = BertModel(BertConfig(**{config})).eval()
ids = torch.tensor({ids})
mask = torch.tensor({mask})
with torch.no_grad():
  _, _, attentions, norms = model(ids, attention_mask=mask,
                                  token_type_ids=torch.zeros_like(ids),
                                  output_attentions=True, output_norms=True)
torch.save({{"state_dict": model.state_dict(),
            "attentions": attentions, "norms": norms}}, sys.argv[2])
"""


@pytest.mark.skipif("KOBAYASHI_SRC" not in os.environ,
                    reason="set KOBAYASHI_SRC to compare with Kobayashi et al.")
def test_matches_kobayashi_implementation():
  ids = [[1, 5, 6, 7, 8, 9, 2, 0, 0], [1, 10, 11, 12, 13, 14, 15, 16, 2]]
  mask = [[1] * 7 + [0] * 2, [1] * 9]
  with tempfile.TemporaryDirectory() as tmp:
    out = os.path.join(tmp, "kobayashi.pt")
    script = KOBAYASHI_SCRIPT.format(config=repr(CONFIG), ids=ids, mask=mask)
    subprocess.run([sys.executable, "-c", script,
                    os.environ["KOBAYASHI_SRC"], out], check=True, cwd=tmp)
    ref = torch.load(out, weights_only=False)

  model = BertModel(BertConfig(**CONFIG, attn_implementation="eager"))
  missing, unexpected = model.load_state_dict(ref["state_dict"], strict=False)
  assert not [k for k in missing if "position_ids" not in k], missing
  assert not [k for k in unexpected if "position_ids" not in k], unexpected
  model.eval()

  maps = _extract(model, [row[:sum(m)] for row, m in zip(ids, mask)])
  for b, m in enumerate(mask):
    n = sum(m)
    for l in range(CONFIG["num_hidden_layers"]):
      fx_norm, afx_norm, summed_afx_norm = ref["norms"][l]
      np.testing.assert_allclose(
          maps[b]["attns"][l], ref["attentions"][l][b, :, :n, :n].numpy(),
          atol=1e-5)
      np.testing.assert_allclose(maps[b]["fx_norms"][l],
                                 fx_norm[b, :, :n].numpy(), rtol=1e-4)
      np.testing.assert_allclose(maps[b]["norms"][l],
                                 afx_norm[b, :, :n, :n].numpy(),
                                 rtol=1e-4, atol=1e-6)
      np.testing.assert_allclose(maps[b]["summed_norms"][l],
                                 summed_afx_norm[b, :n, :n].numpy(),
                                 rtol=1e-4, atol=1e-6)
