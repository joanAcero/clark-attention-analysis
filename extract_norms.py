"""Runs BERT (PyTorch / HuggingFace) over input data and writes out its
attention maps together with the norm-based maps of Kobayashi et al. (2020).

This is a PyTorch counterpart of extract_attention.py. Input and output formats
are the same (see the README), and the following fields can be added to each
example dict (select them with --outputs):
  "attns":        [n_layers, n_heads, n, n] attention weights alpha
  "norms":        [n_layers, n_heads, n, n] ||alpha f(x)||
  "summed_norms": [n_layers, n, n]          ||sum_h alpha f(x)|| (layer level)
  "fx_norms":     [n_layers, n_heads, n]    ||f(x)|| (token level only)
where n is the number of tokens, or of words (+ [CLS]/[SEP]) with --word_level.
"""

import argparse
import os

import numpy as np
import torch
from transformers import BertConfig, BertModel

import bpe_utils
import norm_utils
import utils
from bert import tokenization

OUTPUT_KEYS = ["attns", "norms", "summed_norms", "fx_norms"]


class Example(object):
  """Represents a single input sequence to be passed into BERT."""

  def __init__(self, features, tokenizer, segment_ids="zeros"):
    self.features = features

    if "tokens" in features:
      self.tokens = features["tokens"]
    else:
      if "text" in features:
        text = features["text"]
      else:
        text = " ".join(features["words"])
      self.tokens = ["[CLS]"] + tokenizer.tokenize(text) + ["[SEP]"]

    self.input_ids = tokenizer.convert_tokens_to_ids(self.tokens)
    if segment_ids == "zeros":
      # what extract_attention.py (and hence the paper's data) does
      self.segment_ids = [0] * len(self.tokens)
    else:
      # standard BERT sentence-pair segment ids (as in Kobayashi et al.)
      first_sep = self.tokens.index("[SEP]")
      self.segment_ids = ([0] * (first_sep + 1) +
                          [1] * (len(self.tokens) - first_sep - 1))


def examples_in_batches(examples, batch_size):
  for i in utils.logged_loop(range(1 + ((len(examples) - 1) // batch_size))):
    yield examples[i * batch_size:(i + 1) * batch_size]


def get_vocab_file(bert_dir):
  if os.path.isdir(bert_dir):
    return os.path.join(bert_dir, "vocab.txt")
  from huggingface_hub import hf_hub_download
  return hf_hub_download(bert_dir, "vocab.txt")


def load_model(bert_dir, random_init=False, debug=False, seed=0):
  if debug:
    config = BertConfig(num_hidden_layers=3, hidden_size=144,
                        num_attention_heads=12, intermediate_size=576)
    random_init = True
  else:
    config = BertConfig.from_pretrained(bert_dir)
  config._attn_implementation = "eager"  # needed to return attention weights
  if random_init:
    torch.manual_seed(seed)
    model = BertModel(config)
  else:
    model = BertModel.from_pretrained(bert_dir, config=config)
  model.eval()
  return model


class NormMapExtractor(object):
  """Runs BERT over examples to get its attention and norm-based maps."""

  def __init__(self, model, device, outputs, word_norm_mode="vector"):
    self._model = model.to(device)
    self._device = device
    self._outputs = outputs
    self._word_norm_mode = word_norm_mode
    self._num_heads = model.config.num_attention_heads
    self._values = []
    for layer in model.encoder.layer:
      layer.attention.self.value.register_forward_hook(
          lambda module, inputs, output: self._values.append(output))

  @torch.no_grad()
  def get_maps(self, examples, words_to_tokens=None):
    max_len = max(len(e.input_ids) for e in examples)
    pad = lambda xs: xs + [0] * (max_len - len(xs))
    input_ids = torch.tensor([pad(e.input_ids) for e in examples])
    segment_ids = torch.tensor([pad(e.segment_ids) for e in examples])
    input_mask = torch.tensor([pad([1] * len(e.input_ids)) for e in examples])

    self._values = []
    attentions = self._model(
        input_ids=input_ids.to(self._device),
        attention_mask=input_mask.to(self._device),
        token_type_ids=segment_ids.to(self._device),
        output_attentions=True).attentions
    layers = self._model.encoder.layer
    assert len(self._values) == len(layers)

    results = []
    for b, e in enumerate(examples):
      n = len(e.input_ids)
      per_layer = []
      for l, layer in enumerate(layers):
        per_layer.append(norm_utils.attention_and_norms(
            attentions[l][b, :, :n, :n],
            self._values[l][b, :n],
            layer.attention.output.dense.weight,
            self._num_heads,
            words_to_tokens=(None if words_to_tokens is None
                             else words_to_tokens[b]),
            word_norm_mode=self._word_norm_mode))
      results.append({
          k: torch.stack([m[k] for m in per_layer]).cpu().numpy()
          for k in self._outputs if k in per_layer[0]})
    return results


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--preprocessed-data-file", required=True,
      help="Location of preprocessed data (JSON file); see the README for "
           "expected data format.")
  parser.add_argument(
      "--bert-dir", required=True,
      help="HuggingFace BERT model: a local directory in HuggingFace format "
           "(config.json, vocab.txt, weights) or a hub name such as "
           "bert-base-uncased.")
  parser.add_argument("--cased", default=False, action='store_true',
                      help="Don't lowercase the input.")
  parser.add_argument("--max_sequence_length", default=128, type=int,
                      help="Maximum input sequence length after tokenization "
                           "(default=128).")
  parser.add_argument("--batch_size", default=16, type=int,
                      help="Batch size when running BERT (default=16).")
  parser.add_argument("--debug", default=False, action='store_true',
                      help="Use tiny randomly initialized model for fast "
                           "debugging.")
  parser.add_argument("--word_level", default=False, action='store_true',
                      help="Get word-level rather than token-level maps.")
  parser.add_argument(
      "--word_norm_mode", default="vector", choices=["vector", "sum"],
      help="How to merge norm-based maps to word level (see norm_utils.py).")
  parser.add_argument(
      "--outputs", default="attns,norms",
      help="Comma-separated subset of {:} to write (default=attns,norms)."
           .format(",".join(OUTPUT_KEYS)))
  parser.add_argument(
      "--segment_ids", default="zeros", choices=["zeros", "pair"],
      help="'zeros' (default) sets all segment ids to 0, as extract_attention.py "
           "does; 'pair' uses 1 after the first [SEP] (standard BERT, and what "
           "Kobayashi et al. use).")
  parser.add_argument("--random_init", default=False, action='store_true',
                      help="Use a randomly initialized BERT with the same "
                           "config (control model).")
  parser.add_argument("--seed", default=0, type=int,
                      help="Seed for --random_init.")
  parser.add_argument("--outfile", default=None,
                      help="Where to write the maps (default: "
                           "<preprocessed-data-file>_norms.pkl).")
  parser.add_argument("--device", default=None,
                      help="torch device (default: cuda if available).")
  args = parser.parse_args()

  outputs = args.outputs.split(",")
  for k in outputs:
    if k not in OUTPUT_KEYS:
      raise ValueError("Unknown output", k)
  device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

  print("Creating examples...")
  tokenizer = tokenization.FullTokenizer(
      vocab_file=get_vocab_file(args.bert_dir),
      do_lower_case=not args.cased)
  examples, n_skipped = [], 0
  for features in utils.load_json(args.preprocessed_data_file):
    example = Example(features, tokenizer, args.segment_ids)
    if len(example.input_ids) <= args.max_sequence_length:
      examples.append(example)
    else:
      n_skipped += 1
  print("Skipped {:} examples longer than {:} tokens".format(
      n_skipped, args.max_sequence_length))

  print("Building BERT model...")
  model = load_model(args.bert_dir, args.random_init, args.debug, args.seed)
  extractor = NormMapExtractor(model, device, outputs, args.word_norm_mode)

  print("Extracting attention and norm maps...")
  feature_dicts = []
  for batch_of_examples in examples_in_batches(examples, args.batch_size):
    words_to_tokens = None
    if args.word_level:
      words_to_tokens = []
      for e in batch_of_examples:
        w2t = bpe_utils.tokenize_and_align(
            tokenizer, e.features["words"], args.cased)
        assert sum(len(word) for word in w2t) == len(e.tokens)
        words_to_tokens.append(w2t)
    maps = extractor.get_maps(batch_of_examples, words_to_tokens)
    for e, e_maps in zip(batch_of_examples, maps):
      for k, v in e_maps.items():
        e.features[k] = v.astype("float16")
      e.features["tokens"] = e.tokens
      feature_dicts.append(e.features)

  outpath = args.outfile
  if outpath is None:
    outpath = args.preprocessed_data_file.replace(".json", "") + "_norms.pkl"
  print("Writing maps to {:}...".format(outpath))
  utils.write_pickle(feature_dicts, outpath)
  print("Done!")


if __name__ == "__main__":
  main()
