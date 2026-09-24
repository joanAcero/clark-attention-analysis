"""Computes the average Jenson-Shannon Divergence between attention heads."""

import argparse
import numpy as np

import norm_utils
import utils


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
  args = parser.parse_args()

  print("Loading attention data")
  data = utils.load_pickle(args.attn_data_file)

  print("Computing head distances")
  js_distances = np.zeros([144, 144])
  for doc in utils.logged_loop(data, n_steps=None):
    if args.key not in doc:
      continue
    attns = np.array(doc[args.key])
    if args.key != "attns":
      attns = norm_utils.normalize_rows(attns)

    n_heads = attns.shape[0] * attns.shape[1]
    if js_distances.shape[0] != n_heads:
      js_distances = np.zeros([n_heads, n_heads])
    attns_flat = attns.reshape([n_heads, attns.shape[2], attns.shape[3]])
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

    utils.write_pickle(js_distances, args.outfile)


if __name__ == "__main__":
  main()
