# Is dependency syntax recoverable from BERT's attention? Summary of results

We reproduced Clark et al. (2019), "What Does BERT Look At?", and tested two research
questions from the proposal:

- **RG2:** does norm-based attention ‖αf(x)‖ (Kobayashi et al., 2020) recover more syntax than
  the attention weights α?
- **RG1:** what does an attention-only parent probe actually use?

Setup:

- **Model:** BERT-base-uncased.
- **Syntax data:** UD English EWT (train for probes, dev for evaluation) and UD English PUD
  (evaluated with probes trained on EWT).
- **Attention data:** Clark's 992 Wikipedia segments.
- **Control in every experiment:** a randomly initialized BERT with the same architecture.

Head/layer indices are 0-based. Numbers are given as EWT dev / PUD.

## 1. Reproduction

- A PyTorch/HuggingFace pipeline (`extract_norms.py`) writes α and ‖αf(x)‖ at token or word
  level. The norms match Kobayashi et al.'s own implementation number for number
  (`tests/test_norms.py`).
- Our attention matches Clark's released maps up to float16 precision
  (max |Δ| = 4.9e-4, argmax agreement 1.000). This only holds with **all-zero segment ids**,
  which is what Clark's code uses; Kobayashi et al. use sentence-pair ids.
- Clark's results reproduce on UD:
  - The same syntactic heads reappear: determiner/case 7-10, object 7-9, possessive 6-5 and
    passive auxiliary 3-9.
  - Over half of the attention in layers 6–10 goes to [SEP].
  - His attention-only probe gives 61.4 UAS, vs. 61 on PTB-SD.

## 2. RG2: attention weights α vs. norms ‖αf(x)‖

All comparisons select heads on held-out data (2-fold) and use paired tests (McNemar,
sentence bootstrap, Holm correction).

| Analysis | α | ‖αf‖ | Random init |
| --- | --- | --- | --- |
| Share going to [SEP], layers 6–10 (Wikipedia) | 50–57% | 10–17% | – |
| Best single head, all relations | 41.9 / 43.1 | 43.4 / 44.6 | 9.0 / 6.3 |
| … punctuation removed from the candidates | 43.1 / 43.4 | 43.8 / 44.8 | – |
| MST from the best head, UUAS (rows renormalized) | 61.9 / 62.3 | 62.5 / 63.4 | 15.1 / 11.5 |
| Attention-only linear probe (Clark), UAS | 61.4 / 63.7 | 62.1 / 64.4 | 15.7 / 11.4 |
| Attention + GloVe probe, UAS (alignment fixed) | 74.7 / 73.9 | 74.1 / 73.0 | – |

Baselines: adjacent-word chain UUAS 41.6 / 40.9; right-branching UAS 32.1 / 33.3.

- **Where α puts mass vs. what gets transmitted.** ‖f([SEP])‖ is about 0.3 against 5–6 for
  other tokens, so the heads that attend to [SEP] transmit almost nothing from it. This
  reproduces Kobayashi et al. The same happens, more weakly, for punctuation in layers 11–12.
  With norms, the [SEP] and punctuation clusters of heads disappear.
- **Single heads.** Norms give a small but significant gain: +1.5 overall, p < 1e-8.
  - It is concentrated in *case*, *obj* and *obl* (+1.5 to +5.4), on both treebanks.
  - Removing punctuation from α's candidates accounts for all of the overall gain on EWT and
    part of it on PUD.
  - The gains in *case*, *obj* and *obl* remain after that ablation.
- **Trees.** Renormalizing rows after dropping [CLS]/[SEP] helps more (+2.9 UUAS) than
  switching α → ‖αf‖ (+0.6 to +1.1).
- **Probes.** α and ‖αf‖ are within about 1 point.

**Conclusion (RG2).** Most of what norms add is discounting sinks: [SEP], [CLS] and, in the
upper layers, punctuation. Clark's protocol already removes [CLS]/[SEP] before decoding. The
earlier literature was therefore not looking "from the wrong angle": it had already corrected
implicitly most of what norms correct. The norm-based view changes the account of where
attention goes, but barely changes how much syntax can be decoded from it.

## 3. RG1: what does the attention-only parent probe use?

**The probe (`probes.py`).** For each dependent i, it scores every candidate parent j and ROOT,
and takes a softmax.

- **Pair features:** φ_ij, each head's attention i→j and j→i (288 features).
- **ROOT features:** each head's attention from i to [CLS] and to [SEP] (288 features).
- **Scoring function:** linear, or an MLP with one hidden layer.
- Decisions are independent per token. Features are standardized. 5 seeds.

**Controls.**

- The same probes on random-init BERT.
- One-hot position features with the same dimensionality.

| Features | Linear | MLP |
| --- | --- | --- |
| BERT attention | **65.0** / 67.4 | **80.9** / 82.4 |
| Random-init attention | 18.5 / 13.0 | 36.6 / 35.1 |
| Position (matched parameters) | 32.2 / 33.3 | 32.2 / 33.3 |

- **BERT vs. probe.** BERT's attention adds about 45 UAS over the random-init model with both
  probes (+46.5 linear, +44.3 MLP).
  - The position control can only learn a prior over offsets, so it ends up at "attach to the
    next word" (= right-branching).
  - The linear probe beats Clark's 61.4. The differences: separate ROOT features,
    standardization, early stopping.
- **Weighted vote vs. interactions.** The MLP gains +15.9 over the linear probe on BERT, but
  +18.2 on random init. The extra capacity is generic: most likely positional structure, since
  the MLP depends more on layer 0. It is not evidence that syntax needs interactions between
  heads. The syntactic information is essentially linearly accessible.
- **Which heads.**
  - Linear weights are stable across seeds (correlation 0.97), but they are not importances.
    The largest weights are in layer 11, yet ablating layers 10–11 costs 0 UAS: correlated
    features cancel each other.
  - Ablation points to Clark's syntactic heads: 3-9 (−2.9 UAS), 7-10 (−2.7), 4-5 (−1.7), and
    7-9 for the MLP.
  - No single head is essential (redundancy). Layer 7 matters most (−7.1).
- **Which layers.**
  - Single-layer probes peak at layers 7 (50.9) and 3–4 (about 50), and fall to 25–31 in
    layers 10–11.
  - Layers 0–7 already give 63.7 of the 65.2 UAS.
  - Syntax in attention sits in the middle layers, consistent with Clark, Htut et al. and
    Hewitt & Manning.
- **By relation.** A probe with one weight vector per (gold) relation improves the relations
  where the global probe fails:

  | Relation | Global probe | Relation-conditioned |
  | --- | --- | --- |
  | ccomp | 16.3 | 87.6 |
  | xcomp | 39.1 | 91.6 |
  | obj | 66.9 | 97.1 |

  Its weight vectors cluster in linguistically coherent groups: nominal premodifiers, verbal
  arguments/adjuncts, subjects, nominal postmodifiers, coordination, function words. But the
  gold relation also reveals the usual position of the head. Without a relation-conditioned
  positional control, this is not evidence of relation-specific heads. We did not run that
  control.

**Conclusion (RG1).** The probe confirms the literature with better controls:

- The parent information in attention is largely linear.
- It relies redundantly on the same mid-layer syntactic heads identified per head.
- The last layers contribute nothing.

We found nothing new, so RG1 was closed without further runs.

## 4. Side findings

- **Root alignment in Clark's GloVe probes** (`fix_root_alignment` in `analysis_utils.Probe`).
  In the original code, the word embedding used for candidate head *j* is that of word *j+1*.
  - Fixing it raises the attention + GloVe probe by 3.1 / 1.5 UAS, and the words + distances
    baseline by 7.1 / 4.5.
  - Attention's advantage over the baseline therefore drops from 17.8 / 18.6 to 13.8 / 15.6.
  - If the paper's numbers (77 vs. 58) came from this code, they overstate that advantage.
    This has not been checked on PTB-SD.
- **Bugs in Clark's notebooks** (fixed in our versions):
  - The "next"/"prev token" labels in the Figure 2 legend are swapped.
  - The `poss` special case raises an IndexError when the `poss` word ends the sentence.
  - Relations that were never predicted wrongly dropped out of the per-head results.

## 5. Limitations

- BERT-base only, English only, UD only (no PTB-SD).
- Only the per-head ‖αf‖ was tested, not the residual- and LayerNorm-aware analysis of
  Kobayashi et al. (2021).
- The GloVe probes use a single seed.
- The frequency explanation for the norm gains in *case*, *obj* and *obl* is untested.
- The Q4 positional control is missing.

**Overall.** Neither angle, norm-weighted attention (RG2) or a closer look at the attention
probe (RG1), shows that attention contains more syntax than the literature already reported.
The contributions are:

- the controlled negative result and its mechanism (sinks);
- the exact reproduction of Clark's maps;
- the root-alignment bug in Clark's probe.

## Files

- **Code:** `extract_norms.py`, `norm_utils.py`, `analysis_utils.py`, `syntax_extended.py`,
  `probes.py`.
- **Notebooks:** `Norm_General_Analysis.ipynb`, `Norm_Syntax_Analysis.ipynb`,
  `Norm_Syntax_Extended.ipynb`, `RG1_Probe_Analysis.ipynb` (generated by `make_notebooks.py`).
- **How to run:** [REPRODUCE.md](REPRODUCE.md), [slurm/README.md](slurm/README.md).
