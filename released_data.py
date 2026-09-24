"""Helpers for the data released with Clark et al. (2019).

  # 1. Recover the exact inputs (wordpiece tokens, incl. [CLS]/[SEP]) of the
  #    released attention maps so they can be re-run through extract_norms.py
  python released_data.py tokens --released $DATA/unlabeled_attn.pkl \
      --outfile $DATA/unlabeled.json

  # 2. Compare the released (TensorFlow) attention maps with re-extracted ones
  python released_data.py compare --released $DATA/unlabeled_attn.pkl \
      --extracted $DATA/unlabeled_norms.pkl
"""

import argparse

import numpy as np

import utils


def to_tokens(args):
  data = utils.load_pickle(args.released)
  examples = []
  for doc in data:
    examples.append({k: v for k, v in doc.items()
                     if not isinstance(v, np.ndarray)})
  utils.write_json(examples, args.outfile)
  print("Wrote {:} examples to {:}".format(len(examples), args.outfile))


def compare(args):
  released = utils.load_pickle(args.released)
  extracted = utils.load_pickle(args.extracted)
  assert len(released) == len(extracted), (len(released), len(extracted))
  max_diffs, mean_diffs, argmax_agreement = [], [], []
  for r, e in zip(released, extracted):
    assert list(r["tokens"]) == list(e["tokens"])
    a = np.asarray(r[args.key], dtype=np.float32)
    b = np.asarray(e[args.key], dtype=np.float32)
    assert a.shape == b.shape, (a.shape, b.shape)
    diff = np.abs(a - b)
    max_diffs.append(diff.max())
    mean_diffs.append(diff.mean())
    argmax_agreement.append((a.argmax(-1) == b.argmax(-1)).mean())
  print("examples:                    {:}".format(len(released)))
  print("max |released - extracted|:  {:.2e} (worst example)".format(
      np.max(max_diffs)))
  print("mean |released - extracted|: {:.2e}".format(np.mean(mean_diffs)))
  print("argmax agreement per row:    {:.4f}".format(
      np.mean(argmax_agreement)))


def main():
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawTextHelpFormatter)
  sub = parser.add_subparsers(dest="command", required=True)
  p = sub.add_parser("tokens", help="released pickle -> JSON of inputs")
  p.add_argument("--released", required=True)
  p.add_argument("--outfile", required=True)
  p.set_defaults(func=to_tokens)
  p = sub.add_parser("compare", help="compare released and extracted maps")
  p.add_argument("--released", required=True)
  p.add_argument("--extracted", required=True)
  p.add_argument("--key", default="attns")
  p.set_defaults(func=compare)
  args = parser.parse_args()
  args.func(args)


if __name__ == "__main__":
  main()
