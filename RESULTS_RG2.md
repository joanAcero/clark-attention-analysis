# RG2: does norm-based attention recover more syntax? Results

**Question.** Kobayashi et al. (2020) show that the attention weight α is only one factor in a
head's output. The other is the norm of the transformed input ‖f(x)‖, and ‖αf(x)‖ measures how
much each token actually contributes. We asked whether syntax is more recoverable from ‖αf(x)‖
than from α, the object studied by Clark et al. (2019) and the rest of the literature.

**Setup.** Model: BERT-base-uncased. The norms were computed with a port of Kobayashi et al.'s
implementation that matches their code number for number (`tests/test_norms.py`). Data:

- Clark's 992 Wikipedia segments.
- UD English EWT: train for the probes, dev for evaluation.
- UD English PUD: evaluated with probes trained on EWT.

Every analysis was run on α and on ‖αf‖, with a randomly initialized BERT as a control.
Best heads were selected on held-out data (2-fold cross-fitting). Comparisons use paired tests:
exact McNemar, a sentence-level bootstrap and Holm correction.

## Reproduction

- Our attention maps match Clark's released ones up to float16 precision
  (max |Δ| = 4.9e-4, argmax agreement 1.000).
- The match holds only with **all-zero segment ids**, which is what Clark's code uses, whereas
  Kobayashi et al. use sentence-pair ids.
- Clark's syntactic heads reappear on UD: determiner 7-10, object 7-9, possessive 6-5 and
  passive auxiliary 3-9 (0-indexed).
- His attention-only probe reproduces: 61.4 UAS on EWT dev, vs 61 on PTB-SD.

## Results (EWT dev / PUD)

| Analysis | α | ‖αf‖ | Random init |
| --- | --- | --- | --- |
| Share going to [SEP], layers 6–10 (Wikipedia) | 50–57% | 10–17% | – |
| Best single head, all relations (held-out) | 41.9 / 43.1 | 43.4 / 44.6 | 9.0 / 6.3 |
| … with punctuation removed from the candidates | 43.1 / 43.4 | 43.8 / 44.8 | – |
| MST from the best head, UUAS (rows renormalized) | 61.9 / 62.3 | 62.5 / 63.4 | 15.1 / 11.5 |
| Attention-only linear probe, UAS | 61.4 / 63.7 | 62.1 / 64.4 | 15.7 / 11.4 |
| Attention + GloVe probe, UAS (alignment fixed) | 74.7 / 73.9 | 74.1 / 73.0 | – |

Baselines: adjacent-word chain UUAS 41.6 / 40.9; right-branching UAS 32.1 / 33.3.

- **Single heads.** The gain from norms is small but significant: +1.5 overall, p < 1e-8.
  It is concentrated in *case*, *obj* and *obl* (+1.5 to +5.4). These gains replicate on
  both treebanks and remain after punctuation is removed. The other relations don't change,
  and the selected heads are the same.
- **Sinks.** Removing punctuation from α's candidates accounts for all of the overall gain on
  EWT (norms vs. α without punctuation: +0.2, not significant) and part of it on PUD (+1.2).
- **Trees.** Renormalizing rows after dropping [CLS]/[SEP] helps more (+2.9 UUAS) than
  switching α → ‖αf‖ (+0.6 to +1.1).
- **Probes.** α and ‖αf‖ are within about 1 point. The random-initialization control shows the
  probe's accuracy comes from the representation, not from probe capacity.

## Conclusion

**Most of what norms add is discounting sinks: [SEP], [CLS] and, in the upper layers,
punctuation.** They are the tokens that receive large α but transmit small ‖f(x)‖.
Clark's protocol already removes [CLS]/[SEP] explicitly before decoding. The earlier
literature was therefore not looking "from the wrong angle": it had already corrected
implicitly most of what norms correct. The norm-based view does change the account of
where attention goes (special tokens, entropy, clustering of heads). It barely changes how
much syntax can be decoded from attention. RG2 is closed as a negative result.

## Side findings

- **Root alignment in Clark's GloVe probes** (`fix_root_alignment` in `analysis_utils.Probe`).
  In the original code, the word embedding used for candidate head *j* is that of word *j+1*.
  - Fixing it raises the attention + GloVe probe by 3.1 / 1.5 UAS, and the words + distances
    baseline by 7.1 / 4.5.
  - As a result, attention's advantage over the baseline shrinks from 17.8 / 18.6 to
    13.8 / 15.6. If the paper's numbers (77 vs. 58) came from this code, they overstate that
    advantage.
  - This has not been checked on PTB-SD.
- **Figure 2 legend in Clark's `General_Analysis.ipynb`.** The "next token" and "prev token"
  labels are swapped (fixed in `Norm_General_Analysis.ipynb`).
- **`poss` special case in the per-head evaluation.** It raises an IndexError when the `poss`
  word is the last in the sentence. Relations that were never predicted wrongly also dropped
  out of the results (both fixed in `analysis_utils.py`).

## Limitations

- BERT-base only, English only, UD only (no PTB-SD).
- Only the per-head ‖αf‖ was tested, not the residual- and LayerNorm-aware analysis of
  Kobayashi et al. (2021).
- The probes use a single seed for GloVe and 3 seeds for the rest.
- The frequency explanation for the gains in *case*, *obj* and *obl* (‖f(x)‖ being smaller for
  frequent words) is untested.

Notebooks: `Norm_General_Analysis.ipynb`, `Norm_Syntax_Analysis.ipynb` and
`Norm_Syntax_Extended.ipynb`. How to run them: [REPRODUCE.md](REPRODUCE.md) and
[slurm/README.md](slurm/README.md).
