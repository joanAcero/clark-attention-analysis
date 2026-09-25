"""Computes the average Jenson-Shannon Divergence between attention heads."""

import argparse
import numpy as np

import norm_utils
import utils


def js_numpy(attns_flat):
  """The original implementation (one head against all heads at a time).
  Note that it computes in the dtype of the maps (float16 for the paper's
  data)."""
  n_heads = attns_flat.shape[0]
  js_distances = np.zeros([n_heads, n_heads])
  for head in range(n_heads):
    head_attns = np.expand_dims(attns_flat[head], 0)
    head_attns_smoothed = (0.001 / head_attns.shape[1]) + (head_attns * 0.999)
    attns_flat_smoothed = (0.001 / attns_flat.shape[1]) + (attns_flat * 0.999)
    m = (head_attns_smoothed + attns_flat_smoothed) / 2
    js = -head_attns_smoothed * np.log(m / head_attns_smoothed)
    js += -attns_flat_smoothed * np.log(m / attns_flat_smoothed)
    js /= 2
    js = js.sum(-1).sum(-1)
    js_distances[head] += js
  return js_distances


def js_torch(attns_flat, device, chunk_size=16):
  """Same computation as js_numpy, vectorized over pairs of heads (float32)."""
  import torch
  p = torch.as_tensor(np.asarray(attns_flat, dtype=np.float32), device=device)
  p = (0.001 / p.shape[1]) + p * 0.999
  p_log_p = (p * torch.log(p)).sum((-1, -2))  # [n_heads]
  q = p.unsqueeze(0)
  out = []
  for start in range(0, p.shape[0], chunk_size):
    h = p[start:start + chunk_size].unsqueeze(1)
    m = (h + q) / 2
    log_m = torch.log(m)
    # -p log(m / p) - q log(m / q), summed over positions
    js = (-(h * log_m).sum((-1, -2)) - (q * log_m).sum((-1, -2)) +
          p_log_p[start:start + chunk_size, None] + p_log_p[None, :])
    out.append(js / 2)
  return torch.cat(out).double().cpu().numpy()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--attn-data-file", required=True,
    help="Pickle file containing extracted attention maps.")
  parser.add_argument("--outfile", required=True,
                      help="Where to write out the distances between heads.")
  parser.add_argument(
      "--key", default="attns",
      help="Which maps to compare: 'attns' (default) or e.g. 'norms'. Maps "
           "other than 'attns' are row-normalized into distributions first.")
  parser.add_argument(
      "--device", default=None,
      help="'numpy' for the original (slow) implementation, or a torch device "
           "(default: cuda if available, else cpu).")
  args = parser.parse_args()

  if args.device is None:
    try:
      import torch
      args.device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
      args.device = "numpy"

  print("Loading attention data")
  data = utils.load_pickle(args.attn_data_file)

  print("Computing head distances")
  js_distances = None
  for doc in utils.logged_loop(data):
    if args.key not in doc:
      continue
    attns = np.array(doc[args.key])
    if args.key != "attns":
      attns = norm_utils.normalize_rows(attns)

    n_heads = attns.shape[0] * attns.shape[1]
    attns_flat = attns.reshape([n_heads, attns.shape[2], attns.shape[3]])
    if args.device == "numpy":
      js = js_numpy(attns_flat)
    else:
      js = js_torch(attns_flat, args.device)
    js_distances = js if js_distances is None else js_distances + js

  utils.write_pickle(js_distances, args.outfile)


if __name__ == "__main__":
  main()
