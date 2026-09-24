"""Converts Universal Dependencies (CoNLL-U) treebanks into the JSON format used
by preprocess_depparse.py / extract_attention.py / extract_norms.py.

Each sentence becomes {"words": [...], "heads": [...], "relns": [...],
"upos": [...]}, where heads use the same convention as the paper's data
(0 for ROOT, 1 for the first word). Multi-word token lines (e.g. "1-2 don't")
and empty nodes (e.g. "8.1") are skipped, so "words" are the syntactic words
the tree is defined over.

Example:
  python preprocess_conllu.py --conllu en_ewt-ud-dev.conllu --outfile dev.json
"""

import argparse

import utils


def read_conllu(path, strip_subtypes=False):
  examples = []
  current = {"words": [], "heads": [], "relns": [], "upos": []}
  with open(path, encoding="utf-8") as f:
    for line in f:
      line = line.rstrip("\n")
      if not line:
        if current["words"]:
          examples.append(current)
        current = {"words": [], "heads": [], "relns": [], "upos": []}
        continue
      if line.startswith("#"):
        continue
      fields = line.split("\t")
      if "-" in fields[0] or "." in fields[0]:
        continue
      assert int(fields[0]) == len(current["words"]) + 1, line
      reln = fields[7]
      if strip_subtypes:
        reln = reln.split(":")[0]
      current["words"].append(fields[1])
      current["upos"].append(fields[3])
      current["heads"].append(int(fields[6]))
      current["relns"].append(reln)
  if current["words"]:
    examples.append(current)
  return examples


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--conllu", required=True, help="Input .conllu file.")
  parser.add_argument("--outfile", required=True, help="Output .json file.")
  parser.add_argument("--strip_subtypes", default=False, action="store_true",
                      help="Map relation subtypes to their universal type "
                           "(e.g. nsubj:pass -> nsubj).")
  args = parser.parse_args()
  examples = read_conllu(args.conllu, args.strip_subtypes)
  print("Read {:} sentences, {:} words".format(
      len(examples), sum(len(e["words"]) for e in examples)))
  utils.write_json(examples, args.outfile)


if __name__ == "__main__":
  main()
