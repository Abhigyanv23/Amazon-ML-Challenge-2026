# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Sunshine
**Team Members:** Abhigyan Varma, Cristiano Fernandes, Enrique Dias, Rayirth Deolalkar
**Submission Date:** 27 Sep 2026

---

## 1. Executive Summary

We resolve Source 1 entities against Source 2/3 with a six-channel, label-free blocking stage feeding a
three-part matcher: a pairwise LightGBM (stage 1), two fine-tuned transformer cross-encoders (GPT-2 small
and XLM-RoBERTa-base) that re-read the raw text of the uncertain pairs, and a second LightGBM (stage 2)
that combines the stage-1 probability, group-consistency evidence from each S1's other candidates and the
cross-encoder probabilities. All decisions are tuned for macro F0.5 on out-of-fold predictions and checked
once per version on a frozen holdout. The foundation is normalization and blocking that recover
cross-script and cross-format matches without any external data — an Indic-to-Latin consonant skeleton, a
state-inference map learned from the data's own text, house-number-aware features, and a reverse blocking
channel — reaching 99.1% candidate recall. The two cross-encoders together were the largest late gain:
holdout 0.9773 → 0.9857 over the LightGBM-only pipeline on the same candidates (GPT-2 +0.0062, XLM-R +0.0022).
Final submission **v009**: holdout F0.5 **0.9857**, public leaderboard **0.980893**.

---

## 2. Methodology

### 2.1 Problem Analysis

- **Scale:** 2.2M train S1 / 5.0M S2 / 5.3M S3 records; 1.7M test S1 / 4.9M S2 / 5.1M S3. A naive
  same-country cross join would be trillions of comparisons.
- **Countries:** train is US 60% / India 40%; test adds **France at 15%, with zero training labels**.
  Every true match is same-country (confirmed on a 300K-pair sample), so blocking and decisions never
  cross countries. Country is treated as an open set of labels; there is no country feature.
- **Ground truth structure:** each S2/S3 record belongs to at most one S1 (0 exceptions in the full
  ground truth) — a hard constraint used directly in the decision rule. Matches per S1: 5.6% singletons,
  up to 11, mean 3.46. S2/S3 are not deduplicated (up to 5–6 copies of one business per source);
  about 74% of S2/S3 records match some S1, the rest are distractors.
- **Noise patterns found in EDA:** native-script names (Devanagari, Gujarati, Tamil, …), injected accents,
  typos, word reordering, appended junk ("Center", "#35740", ".com" forms), truncated addresses,
  state-name variants (UP / Uttar Pradesh / Devanagari), house-number variants (3906 vs 3906a,
  "0071/1" vs "71/1"), digits inside words ("co1onial"), postcodes fused into house numbers, letters glued
  to digits ("fl13"). Postcodes are rare (US ~11%, India <1%). Chain-like names repeat (one name occurs
  253 times in S1).
- **Missing data:** S1 has no blank names/addresses; S2/S3 have ~3% blank addresses, which proved to be the
  weakest matching segment.

### 2.2 Solution Strategy

**Approach Type:** unsupervised multi-channel blocking → stage-1 pairwise classifier (LightGBM) →
transformer cross-encoders on uncertain pairs (fine-tuned GPT-2 small and XLM-RoBERTa-base) → stage-2
re-scorer (LightGBM over stage-1 probability, group-consistency and cross-encoder features) → rule-based
decision layer. Every stage is trained on out-of-fold outputs of the previous one, grouped by S1.

**Core Innovation:**
1. **Consonant skeleton for Indic scripts.** Major Indian scripts share one Unicode block layout, so a
   single Devanagari consonant table maps Gujarati/Tamil/Telugu/… names onto the same skeleton as their
   Latin transliteration. This drives a cross-script blocking channel and a similarity feature with no
   dictionary or translation.
2. **Learned state fill without labels.** Address components (city, locality) are counted against the
   states they co-occur with in the same split's S1 records (which all carry a state); components seen
   ≥20 times with ≥98% purity fill missing S2/S3 states. France S2/S3 state coverage rose from 67% to 97%.
3. **Reverse blocking channel.** Each S2/S3 record finds its own best S1s; its score is the most
   important stage-1 feature.
4. **Cross-encoders only where the tree model is unsure.** GPT-2 and XLM-R read the raw text of both
   records, but only for the ~2% of pairs with stage-1 probability in [0.02, 0.98]. The multilingual
   XLM-R reads Devanagari/Tamil and French as real sub-words, which a character-similarity model cannot.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** six unsupervised channels, unioned, each restricted to a (`country`, `state`)
  block (records with no inferred state join every block of their country). Final configuration (v005w):
  1. **Word channel (top 30)** — IDF-weighted cosine over normalized name words, a no-space name token,
     address words, address numbers and a composite house-number token (e.g. "71/1").
  2. **Trigram channel (top 10)** — character trigrams of the no-space name (typos, concatenations).
  3. **Skeleton channel (top 10)** — trigrams of the Indic-to-Latin consonant skeleton (cross-script).
  4. **Address-only channel (top 5)** — address tokens/numbers alone, for names that diverge completely.
  5. **Non-Latin address channel (top 5)** — address similarity against pool records with non-Latin names.
  6. **Reverse channel (top 5)** — each S2/S3 record's own top-5 S1s, added back as candidates.
  Tokens with document frequency above 2% of a block are dropped; top-k via sparse matrix top-n
  multiplication. Blocks are merged where sources mix labels (India: AP+TG, JK+LA).
  A supervised meta-blocking pruner was also developed (v006) but the final version uses the
  **unpruned** candidate set: pruning lowered the public score despite a slightly higher holdout.
- **Candidate pairs generated:** test **115.4M** pairs (66.6 per S1); holdout 30.0M (68.0 per S1).
- **How true matches were not lost:** candidate recall is tracked on the frozen holdout as the primary
  blocking metric, separately from F0.5. Final blocking reaches **99.11%** candidate recall (India 98.73%,
  US 99.37%), an oracle F0.5 ceiling of **0.9971**. Each version's misses were categorized (non-Latin name,
  address-only overlap, state mismatch, outranked, blank address) and the next channel targeted the largest
  category: 94.8% (one channel, v001) → 96.8% (v002) → 98.8% (v005) → 99.1% (v005w, wider k).
  The file `output/candidate_pairs.tsv` is exactly the set scored by the models.

---

## 4. Matching Model

**Features used:**
- **Stage 1 (71 features per pair; no country feature):**
  - *Name:* exact / no-space exact, Levenshtein ratio, token-set / token-sort / partial ratios,
    Jaro-Winkler, token Jaccard, overlap coefficient (acronyms, subset names), first-token match,
    length ratio, legal-suffix Jaccard, IDF-weighted token alignment (share of each side's rare-word mass
    matched), name frequency in S1 and pool (chain indicators), character 2-/3-gram cosine, skeleton cosine.
  - *Address:* fuzzy ratio, token-set ratio, token Jaccard, length ratio, number Jaccard, first-number
    match, state and zip agreement, house-number edit distance and log numeric distance, composite
    house-number equality, address-word overlap without numbers.
  - *Blocking/context:* each channel's score and rank, score relative to the S1's best candidate,
    candidate count, S2/S3 flag, blank-address / non-Latin / inferred-state flags.
- **Cross-encoder inputs:** the raw business name and address of both records, serialized as
  `name: … addr: …` for each side (GPT-2: `a <eos> b`; XLM-R: its text-pair format
  `<s> a </s></s> b </s>`), max 96 tokens.
- **Stage 2 (24 features):** group consistency — for each candidate, agreement with the S1's other
  confident candidates (top 6 with stage-1 p ≥ 0.3) on name, no-space name, address, numbers, zip and state
  (max, probability-weighted mean, strong-link count); stage-1 probability, its rank, gap to the S1's best,
  second-best, group counts (≥0.5, ≥0.3), sum, candidate count, S3 flag; and for each cross-encoder its
  probability and gap to the S1's best cross-encoder score (missing outside the scored band).

**Model type:**
1. **Stage 1 — LightGBM** (MIT): learning rate 0.1, 127 leaves, min 100 rows per leaf, feature/bagging
   fraction 0.9/0.8; 5-fold group-by-S1 early stopping (best iterations 1048–1226); final model on all
   training pairs with 1.1 × mean best iteration. Training data: 300K train-fold S1, 20.4M candidate pairs.
   Out-of-fold stage-1 probabilities are saved for the next stages.
2. **Cross-encoders — GPT-2 small** (MIT, 124M parameters) **and XLM-RoBERTa-base** (MIT, 278M), each
   fully fine-tuned as a binary pair classifier (linear head; GPT-2 reads the last token) on the 346K
   uncertain train-fold pairs (stage-1 p in [0.02, 0.98], top 8 per S1; 32.7% positive). 2-fold
   GroupKFold by S1 gives out-of-fold scores for stage 2; holdout and test use the mean logit of the two
   fold models. 2 epochs, batch 64, AdamW with one-cycle schedule (GPT-2: lr 3e-5, fp16; XLM-R: lr 2e-5,
   bf16). On the band pairs, out-of-fold AUC: stage 1 0.9256, GPT-2 0.9430, **XLM-R 0.9679**.
3. **Stage 2 — LightGBM** (same settings, 3-fold group-by-S1, best iterations 100–108) trained on the
   out-of-fold outputs above. Gain importance: stage-1 probability 0.931, its rank 0.033, XLM-R
   probability 0.027, GPT-2 probability 0.001 (XLM-R absorbs most of GPT-2's signal). The earlier
   fallback rule (use stage 1 when an S1 has ≤1 confident candidate) is no longer selected on OOF, most
   likely because the cross-encoders supply the evidence that single-candidate S1s lacked.

**Threshold selection method:** grid search (0.05 steps, refined to 0.01) on **out-of-fold** stage-2
predictions to maximize macro F0.5 — never on the holdout. Three rules, in order: (1) keep pairs with
probability ≥ *t*; (2) each S2/S3 record is assigned to at most one S1, the highest-probability one
(justified by the ground truth's uniqueness property; holdout F0.5 is higher with the rule than without);
(3) singleton gate: an S1 keeps its matches only if its best probability ≥ *t_top*. Final: *t* = *t_top* =
**0.73** (OOF F0.5 0.9852).

**Data and licences:** only competition data is used; there are no lookups, APIs, geocoding or scraping.
The two pretrained checkpoints are MIT-licensed and far below 8B parameters; their weights are downloaded
once at setup and fine-tuned only on the provided training pairs.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), final model v009:** holdout **0.9857** (out-of-fold 0.9852); public leaderboard
  **0.980893**. Micro precision 0.997, micro recall 0.964; singleton accuracy 0.991. By country (holdout):
  India 0.9848, US 0.9864. By true match count: 0 → 0.991, 1 → 0.952, 2 → 0.983, 3+ → 0.989.
- **Where the remaining loss is (v009 holdout, pairs among candidates):** 45,900 wrong pairs —
  41,458 missed and 4,442 false — plus 13,545 true pairs never generated by blocking.
  77% of the wrong pairs lie in the cross-encoder band (stage-1 p 0.02–0.98), 16% just outside it
  (0.005–0.02 and 0.98–0.995), 7% far outside. The model is now precision-heavy, as F0.5 rewards.
- **Common false positives (wrong merges):** distinct businesses sharing a building/street address or a
  near-identical chain-style name at different addresses; house-number distance, IDF "unmatched token"
  features and the cross-encoders target these.
- **Common false negatives (missed matches):** blank-address candidates, where the name alone must decide;
  heavily abbreviated or transliterated names with thin addresses; and pairs where stage 1 is confident
  the other way (p < 0.02), which the cross-encoders never see. Blocking misses concentrate in non-Latin
  names with very short addresses.
- **France (no labels):** judged by public-LB movement and prediction statistics. v009 test predictions are
  balanced across countries (average matches France 3.28 / India 3.33 / US 3.36; empty rows 5.8% / 5.9% /
  5.8%), and the holdout-to-LB gap (driven by France) shrank from 0.012 (v004) to 0.009 (v005) to
  0.005 (v009): the multilingual cross-encoder helps France as well as India.

---

## 6. Conclusion

Candidate generation was treated as the lever for coverage (six label-free channels, 99.1% candidate
recall) and precision as an explicit objective of the matching stage, in line with F0.5. The biggest late
gain came from adding a multilingual transformer only where the tree model was uncertain: it reads
native-script and French text directly, lifting India from 0.971 to 0.985 on the holdout, and it costs
minutes of GPU time because it scores under 2% of pairs. Lessons: measure blocking recall separately from
F0.5; train every stage on out-of-fold outputs of the previous one; and compare submissions that differ in
one component only, since the public LB contains a country (France) that the holdout cannot measure.

---

## Appendix

### A. Code Artefacts

The runnable pipeline is under `code/business_entity_resolution/` (`src/`, `README.md`,
`requirements.txt`); `src/run_all.sh` runs everything below in order with the exact flags:

1. `src/make_split.py` — frozen 80/20 holdout split of train S1 (seed 42).
2. `src/normalization.py --split train|test` — normalized caches.
3. `src/blocking.py` — six-channel candidate generation (holdout, 300K train-fold sample, test).
4. `src/matcher.py train|holdout|test --tag v005ws1 --cand-tag v005w` — stage 1; saves stage-1 probabilities.
5. `src/llm_rescore.py train|holdout|test --tag g001` (GPT-2) and `--tag x001 --model xlm-roberta-base`
   (XLM-R) — cross-encoders on the uncertain band.
6. `src/stage2.py train|holdout|test --tag v009 --llm g001,x001` — stage 2; writes
   `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
7. `src/check_submission.py`, `utils/validate_submission.py` — format checks.

Diagnostics (`error_analysis.py`, `diagnose_blocking.py`, `diagnose_country.py`) read ground truth for
analysis only. Version history and every intermediate metric: `experiments/EXPERIMENTS.md`.

### B. Additional Results

| Version | Change | Candidate recall | Holdout F0.5 | Public LB |
|---|---|---|---|---|
| v001 | Baseline: word-only blocking + LightGBM | 94.8% | 0.9481 | 0.9310 |
| v002 | + learned state fill, 3-channel blocking | 96.8% | 0.9545 | 0.9436 |
| v004 | + house-number / IDF features, stage 2 with fallback | 96.8% | 0.9667 | 0.9550 |
| v005 | + skeleton / address / reverse channels | 98.8% | 0.9773 | 0.9683 |
| v006 | + supervised candidate pruning (10.4/S1) | 98.0% | 0.9778 | 0.9652 |
| v005w | wider channels (k 30 / skeleton 10 / reverse 5) + name overlap feature | 99.1% | 0.9773 | — |
| v008 | + GPT-2 cross-encoder feature in stage 2 | 99.1% | 0.9835 | 0.9768 |
| **v009** | + XLM-RoBERTa cross-encoder (final) | **99.1%** | **0.9857** | **0.9809** |

Cross-encoder band sizes (pairs scored): train 346K of 20.4M, holdout 509K of 30.0M, test 2.44M of 115.4M.
Country breakdown (holdout): India 0.9225 (v001) → 0.9712 (v005) → 0.9848 (v009);
US 0.9652 → 0.9813 → 0.9864.
