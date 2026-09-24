"""Norm-based analysis of attention (Kobayashi et al., EMNLP 2020) for BERT.

Kobayashi et al. rewrite the output of attention head h at position i as
    sum_j alpha^h_ij f^h(x_j),    f^h(x) = (x W_V^h + b_V^h) W_O^h,
and propose ||alpha^h_ij f^h(x_j)|| as a measure of how much token j contributes
to token i, instead of the attention weight alpha^h_ij alone. The bias of the
output projection (b_O) is shared by all positions and is not part of f.

This module computes the same quantities as `BertNormOutput` in
https://github.com/gorokoba560/norm-analysis-of-transformer (emnlp2020), i.e.
  fx_norms      ||f^h(x_j)||                         [n_heads, n]
  norms         ||alpha^h_ij f^h(x_j)||              [n_heads, n, n]
  summed_norms  ||sum_h alpha^h_ij f^h(x_j)||        [n, n]
but without materialising the [n_heads, n, n, hidden] tensor. Since
f^h(x) = v^h W_O^h with v^h the (head_size-dimensional) value vector,
||y W_O^h||^2 = y (W_O^h W_O^h^T) y^T, so every norm can be computed in the
head_size-dimensional value space with a small Gram matrix. This is exact (up
to floating point) and is checked against Kobayashi's code in
tests/test_norms.py.

Word-level maps. Clark et al. convert token-level attention to word-level
attention by summing the columns (attention *to* a word) and averaging the
rows (attention *from* a word) of the pieces of each word. Written with a
row-averaging matrix R [n_words, n] and a column-summing matrix C [n, n_words],
word-level attention is R alpha C. For norms there are two options:
  "vector": ||sum_a sum_b R_Ia alpha_ab C_bJ f(x_b)||, i.e. the norm of the
            word-level weighted vector obtained by applying Clark's merge to the
            vectors alpha_ab f(x_b). This is the norm-based analogue of Clark's
            word-level attention (default).
  "sum":    R ||alpha f|| C, i.e. Clark's merge applied to the token-level norm
            matrix. By the triangle inequality this is an upper bound on
            "vector" and ignores cancellation between the pieces of a word.
"""

import numpy as np
import torch


def word_merge_matrices(words_to_tokens, n_tokens, dtype=torch.float32):
  """Returns (R, C) such that R @ attn @ C reproduces
  bpe_utils.get_word_word_attention(attn, words_to_tokens, mode="mean")."""
  n_words = len(words_to_tokens)
  R = torch.zeros(n_words, n_tokens, dtype=dtype)
  C = torch.zeros(n_tokens, n_words, dtype=dtype)
  for w, toks in enumerate(words_to_tokens):
    R[w, toks] = 1.0 / len(toks)
    C[toks, w] = 1.0
  return R, C


def _quadratic_norm(y, gram):
  """sqrt(y G y^T) over the last dimension of y, batched over leading dims of
  gram (y: [h, ..., d], gram: [h, d, d])."""
  sq = torch.einsum("h...d,hde,h...e->h...", y, gram, y)
  return torch.sqrt(torch.clamp(sq, min=0.0))


def attention_and_norms(attn, value, dense_weight, num_heads,
                        words_to_tokens=None, word_norm_mode="vector"):
  """Computes attention maps and norm-based maps for one example and layer.

  Args:
    attn: [n_heads, n, n] attention weights (after softmax, no padding).
    value: [n, hidden] output of the value projection, x W_V + b_V.
    dense_weight: [hidden, hidden] weight of the attention output projection
      (torch.nn.Linear convention, i.e. [out_features, in_features]).
    num_heads: number of attention heads.
    words_to_tokens: optional list of lists giving the token indices of each
      word (as returned by bpe_utils.tokenize_and_align). If given, all maps
      are converted to word level.
    word_norm_mode: "vector" or "sum"; see the module docstring.

  Returns:
    dict with "attns" [h, m, m], "norms" [h, m, m], "summed_norms" [m, m] and,
    for token-level maps only, "fx_norms" [h, n]; m = n or n_words.
  """
  n, hidden = value.shape
  head_size = hidden // num_heads
  attn = attn.to(value.dtype)
  v = value.view(n, num_heads, head_size).permute(1, 0, 2)  # [h, n, d]
  # W_O^h: rows h*d:(h+1)*d of W_O = dense_weight^T -> [h, d, hidden]
  w_o = dense_weight.t().reshape(num_heads, head_size, hidden)
  gram = w_o @ w_o.transpose(1, 2)  # [h, d, d]
  gram_full = dense_weight.t() @ dense_weight  # [hidden, hidden]

  # alpha_ij v_j: [h, n, n, d]
  weighted = attn.unsqueeze(-1) * v.unsqueeze(1)

  outputs = {}
  if words_to_tokens is None:
    outputs["attns"] = attn
    outputs["fx_norms"] = _quadratic_norm(v, gram)
    outputs["norms"] = _quadratic_norm(weighted, gram)
    y = weighted
  else:
    R, C = word_merge_matrices(words_to_tokens, n, value.dtype)
    R, C = R.to(value.device), C.to(value.device)
    outputs["attns"] = R @ attn @ C
    y = torch.einsum("Ia,habd,bJ->hIJd", R, weighted, C)
    if word_norm_mode == "vector":
      outputs["norms"] = _quadratic_norm(y, gram)
    elif word_norm_mode == "sum":
      outputs["norms"] = R @ _quadratic_norm(weighted, gram) @ C
    else:
      raise ValueError("Unknown word_norm_mode", word_norm_mode)

  # ||sum_h alpha^h_ij f^h(x_j)||: concatenating the per-head value-space vectors
  # gives a [hidden]-dimensional vector that W_O maps to the summed vector.
  m = y.shape[1]
  y_cat = y.permute(1, 2, 0, 3).reshape(m, m, hidden)
  outputs["summed_norms"] = _quadratic_norm(y_cat.unsqueeze(0),
                                            gram_full.unsqueeze(0))[0]
  return outputs


def normalize_rows(maps, eps=1e-12):
  """Rescales the last axis to sum to one (e.g. to turn norm-based maps into
  distributions before computing entropies or JS divergences)."""
  maps = np.asarray(maps, dtype=np.float64)
  return maps / np.maximum(maps.sum(-1, keepdims=True), eps)
