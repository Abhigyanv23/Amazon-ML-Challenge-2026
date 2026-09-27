# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Sunshine
**Team Members:** Abhigyan Varma, Cristiano Fernandes, Enrique Dias, Rayirth Deolalkar
**Submission Date:** 27 Sep 2026

---

## 1. Executive Summary

We resolve Source 1 entities against Source 2/3 with a six-channel blocking stage feeding a
two-stage LightGBM matcher: a pairwise classifier followed by a group-consistency re-scorer, with
decisions tuned end-to-end for macro F0.5 on a frozen out-of-sample holdout. Our core innovation is
normalization that recovers cross-script and cross-format matches without any lookup table or
external data — an Indic-to-Latin consonant skeleton (derived from the shared Unicode block layout
of Indian scripts), a learned state-inference model built only from the training data's own text,
and house-number-aware features — which closed most of the India/France gap versus a plain
fuzzy-matching baseline. Best confirmed public leaderboard score: **0.9683** (holdout F0.5 0.9773).

---

## 2. Methodology

### 2.1 Problem Analysis

- **Scale:** 2.2M train S1 / 5.0M S2 / 5.3M S3 records; 1.7M test S1 / 4.9M S2 / 5.1M S3. All
  same-country pairs in a naive cross join would be 6.7 trillion comparisons.
- **Countries:** train is US 60% / India 40%; test adds **France at 15%, with zero training
  labels**. Every true match is same-country (confirmed on a 300K-pair sample), so blocking and
  decisions never need to cross countries.
- **Ground truth structure:** each S2/S3 record belongs to at most one S1 (0 exceptions in the
  full ground truth) — a hard constraint we exploit directly in the decision rule. Match counts per
  S1: 5.6% singletons, up to 11 matches, mean 3.46. S2/S3 are not deduplicated (up to 5–6 copies of
  one business per source).
- **Noise patterns found in EDA:** native-script names (Devanagari, Gujarati, Tamil, ...), injected
  accents, typos, word reordering, appended junk ("Center", "#35740", ".com" suffixes), truncated
  addresses, state-name variants (UP / Uttar Pradesh / Devanagari script), house-number variants
  (3906 vs 3906a, "0071/1" vs "71/1"), digits spelled into words ("co1onial"), postcodes fused into
  house numbers, and letters glued to digits ("fl13"). Postcodes are rare overall (US ~11%, India
  <1%). Chain-like repeated names are common (e.g. one name occurs 253 times in S1).
- **Missing data:** S1 has zero blank names/addresses; S2/S3 have ~3% blank addresses. Blank-address
  candidates turned out to be our single weakest matching segment (Section 5).

### 2.2 Solution Strategy

**Approach Type:** Blocking (multi-channel, unsupervised) → supervised candidate pruning →
two-stage classifier (LightGBM pairwise + LightGBM group-consistency) → rule-based decision layer,
all thresholds tuned on out-of-fold predictions.

**Core Innovation:** Normalization and blocking that close the India/France gap without external
data or translation: (1) an Indic-to-Latin **consonant skeleton** exploiting the fact that major
Indian scripts share one Unicode block layout, so a single Devanagari consonant table maps
Gujarati/Tamil/Telugu/etc. names onto the same skeleton as their Latin transliteration, enabling
cross-script blocking and a `skel_cos3` similarity feature with no dictionary; (2) an **unsupervised
learned state-fill**: address components (city/locality) are counted against the states that always
co-occur with them in the same split's own S1 records (which all carry an explicit state), and used
to fill missing S2/S3 states purely from text statistics — no labels, no geocoding; (3) a **reverse
blocking channel** (each S2/S3 record finds its own best-matching S1, not just the other way
around) whose score became our single most important matching feature.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** six unsupervised channels, unioned, each restricted to a
  (`country`, `state`) block (records with no inferred state join every block of their country):
  1. **Word channel** — IDF-weighted cosine over normalized name words, address words, address
     numbers, and a composite house-number token (e.g. "71/1").
  2. **Trigram channel** — character trigrams of the no-space name, for typos/concatenations.
  3. **Skeleton channel** — trigrams of the Indic-to-Latin consonant skeleton, for cross-script
     matches.
  4. **Address-only channel** — address tokens/numbers alone, for names that diverge completely.
  5. **Non-Latin address channel** — address-only similarity restricted to pool records with
     non-Latin names.
  6. **Reverse channel** — each S2/S3 record's own top-k best S1 matches, added back as candidates.
  A **supervised meta-blocking pruner** (a small LightGBM using only blocking-stage signals —
  channel scores/ranks, hit count, score relative to the S1's best, candidate count) then trims the
  unioned set to a cap per S1, thresholds chosen on out-of-fold data.
- **Candidate pairs generated:** 63.5M for the submitted best version (v005, unpruned, 36.7/S1
  average, max 6,243 for one pathological S1); a pruned variant reduces this to 19.0M (11.0/S1,
  capped at 20) at a small, measured recall cost.
- **How true matches were not lost:** candidate recall (share of true matches present in the
  candidate set) is tracked as the primary blocking metric on the frozen holdout, separately from
  final F0.5 — v005 reaches **98.8%** candidate recall (oracle ceiling F0.5 = 0.996, i.e. a perfect
  matcher on these candidates would score 0.996). Every blocking version's miss set was
  categorized (non-Latin name, address-only overlap, state mismatch, outranked, blank address) and
  the next channel was built to target the largest remaining category, taking candidate recall from
  94.8% (single-channel baseline) to 98.8%.

---

## 4. Matching Model

**Features used:**
- **Name features:** exact/no-space-exact match, Levenshtein ratio, token-set/token-sort/partial
  ratio, Jaro-Winkler, token Jaccard, first-token match, length ratio, legal-suffix Jaccard,
  IDF-weighted token alignment (share of each side's rare-word mass matched), name-frequency-based
  chain indicators, character 2-/3-gram cosine, and the cross-script skeleton cosine.
- **Address features:** fuzzy ratio, token-set ratio, token Jaccard, length ratio, number Jaccard,
  first-number match, state agreement, zip agreement, **house-number edit distance and log-scaled
  numeric distance** (the single largest feature-gain contributor after the initial blocking
  scores), and a composite-house-number equality flag.
- **Blocking/context features:** each channel's score and rank, score relative to the S1's best
  candidate, candidate count, source (S2/S3) flag, blank-address and non-Latin-name flags.
- **Group-consistency (stage 2) features:** for each candidate, agreement with the S1's other
  confident candidates (top 6 with stage-1 probability ≥ 0.3) on name, address, numbers, zip and
  state — plus the stage-1 probability's rank and gap to the best/second-best.

**Model type:** LightGBM (MIT-licensed) binary classifier, two stages:
1. **Stage 1 (pairwise):** trained on candidate pairs, using string/address/blocking features above.
2. **Stage 2 (group re-scorer):** a second LightGBM that takes stage-1 probabilities plus the
   group-consistency features and rescores each candidate against its own S1's other candidates,
   with a **fallback rule**: an S1 with too few confident co-candidates falls back to the stage-1
   score rather than trusting an unsupported group signal (this fixed a measured regression on
   single-match S1 entities). A CatBoost (Apache 2.0) blend on stage 1 was tested; gain was small
   (+0.0002 OOF) and not used in the final submitted model.

**Threshold selection method:** grid search (coarse 0.05, refined to 0.01) on **out-of-fold**
predictions to maximize macro F0.5, never on the held-out validation set. Three rules, in order:
(1) keep pairs with probability ≥ *t*; (2) each S2/S3 record is assigned to at most one S1 — the
highest-probability one — directly justified by the ground truth's own uniqueness property, and
confirmed empirically (holdout F0.5 is higher with this rule than without it); (3) a singleton gate:
an S1 keeps its matches only if its best candidate's probability ≥ *t_top*. No pretrained models or
external data are used anywhere in the pipeline.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), best submitted model (v005):** holdout **0.9773**; public leaderboard
  **0.9683** (out-of-fold estimate 0.9763, so holdout tracked the out-of-fold score to within
  0.001 in every version, confirming no overfitting to the holdout).
- **Common false positives (wrong merges):** distinct businesses that share a street address or a
  near-identical name — e.g. two different companies at the same building number, or the same
  chain-style name appearing at different addresses. These account for roughly a third of the
  remaining error on holdout (extra false positives + singleton false positives + mixed cases).
  House-number-distance and IDF-based "unmatched token" features were added specifically to target
  this pattern.
- **Common false negatives (missed matches):** the dominant category by far is **blank-address
  candidates** — pairs where one side's address is empty and the name alone must decide;
  recall on this segment was only ~51% versus ~99% when an address is present, because near-identical
  names can't be confirmed without any address signal. The remaining blocking-stage misses (candidates
  never generated at all) are a small and now well-characterized share of total loss (~0.004 of 0.023
  on the ceiling), concentrated in non-Latin names with very thin addresses.
- **Note on France:** France has no training labels, so it cannot be error-analyzed directly. A
  confidence-profile diagnostic on stage-1 probabilities shows the model is not more *uncertain* on
  France than on India/US, meaning any remaining gap there is made of confident errors rather than
  a coverage problem — consistent with normalization/blocking improvements (which are
  country-agnostic by construction) closing most of an initially large French gap over successive
  versions.

---

## 6. Conclusion

Our solution treats candidate generation as the primary lever for coverage (six complementary,
label-free blocking channels reaching 98.8% candidate recall) and treats precision as an explicit,
measured objective throughout the matching stage, in line with F0.5's 2x precision weighting — via
house-number and IDF-based disagreement features, a data-justified one-S1-per-record uniqueness
rule, and a group-consistency re-scorer with a fallback safeguard for sparse cases. The main lesson
learned was that our internal validation holdout, while reliable for measuring the matching model,
under-represents one real competitive dynamic of the full test set (records compete across many
more Source 1 entities than in a partial holdout), which we discovered by comparing several
submissions that differed only in which population supplied which rows — a diagnostic technique we
would apply earlier in a future iteration.

---

## Appendix

### A. Code Artefacts

The complete, runnable pipeline ships under `code/business_entity_resolution/` (`src/`, `README.md`,
`requirements.txt`). Entry points, run in order from the package root:

1. `src/normalization.py --split train` / `--split test` — builds normalized caches.
2. `src/blocking.py --split train --s1-set holdout|train --tag <v>` and `--split test --tag <v>` —
   six-channel candidate generation.
3. `src/prune.py train --tag <v>` then `apply --split ... --s1-set ...` — supervised meta-blocking
   candidate pruning (optional).
4. `src/matcher.py train|holdout|test --tag <v>s1 --cand-tag <v>` — stage-1 pairwise classifier.
5. `src/stage2.py train|holdout|test --stage1 <v>s1 --tag <v>` — stage-2 group re-scorer; writes
   `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
6. `src/check_submission.py` and `utils/validate_submission.py` — format validation before
   submission.

Diagnostics used throughout development (`error_analysis.py`, `diagnose_blocking.py`,
`diagnose_country.py`) read ground truth only for analysis; no labels are used for training or
threshold tuning beyond the frozen train-fold / holdout split. Full version history, exact
parameters and every intermediate metric are in `experiments/EXPERIMENTS.md`.

### B. Additional Results

| Version | Change | Candidate recall | Holdout F0.5 | Public LB |
|---|---|---|---|---|
| v001 | Baseline: word-only blocking + LightGBM | 94.8% | 0.9481 | 0.9310 |
| v002 | + state-fill normalization, 3-channel blocking | 96.8% | 0.9545 | 0.9436 |
| v004 | + house-number/IDF features, stage-2 fallback | 96.8% | 0.9667 | 0.9550 |
| **v005** | + skeleton/address/reverse channels (best submitted) | **98.8%** | **0.9773** | **0.9683** |
| v006 | + supervised candidate pruning, feature/model tuning | 98.0% (pruned to 10.4/S1) | 0.9778 | 0.9652 |

Country breakdown (holdout, v005): India F0.5 0.9712, US F0.5 0.9813 — a gap that narrowed from
0.043 in v001 to 0.010 in v005 as normalization and blocking improved.