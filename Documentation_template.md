# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Sunshine
**Team Members:** Abhigyan Varma, Cristiano Fernandes, Enrique Dias, Rayirth Deolalkar
**Submission Date:** 27 Sep 2026

---

## 1. Executive Summary

We resolve Source 1 entities against Source 2/3 with a six-channel blocking stage feeding a
two-stage classifier: a LightGBM pairwise matcher, a LightGBM group-consistency re-scorer, and
[CONFIRM: a final GPT-2-based re-scoring/augmentation stage — one-sentence description of what it
adds, once confirmed], with decisions tuned end-to-end for macro F0.5 on a frozen out-of-sample
holdout. Our core innovation is normalization that recovers cross-script and cross-format matches
without any lookup table or external data — an Indic-to-Latin consonant skeleton (derived from the
shared Unicode block layout of Indian scripts), a learned state-inference model built only from the
training data's own text, and house-number-aware features. **Best confirmed public leaderboard
score: 0.9770** (LightGBM-only pipeline alone: 0.9690, holdout F0.5 for that pipeline 0.9773+).

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
- **Noise patterns found in EDA:** native-script names (Devanagari, Gujarati script observed
  directly; the normalization is written to generalize to the wider family of Unicode-block-aligned
  Indic scripts), injected accents, typos, word reordering, appended junk ("Center", "#35740",
  ".com" suffixes), truncated addresses, state-name variants (UP / Uttar Pradesh / Devanagari
  script), house-number variants (3906 vs 3906a, "0071/1" vs "71/1"), digits spelled into words
  ("co1onial"), postcodes fused into house numbers, and letters glued to digits ("fl13"). Postcodes
  are rare overall (US ~11%, India <1%). Chain-like repeated names are common (e.g. one name occurs
  253 times in S1).
- **Missing data:** S1 has zero blank names/addresses; S2/S3 have ~3% blank addresses. Blank-address
  candidates were our single weakest matching segment throughout development (Section 5).

### 2.2 Solution Strategy

**Approach Type:** Blocking (multi-channel, unsupervised) → two-stage classifier (LightGBM pairwise
+ LightGBM group-consistency) → [CONFIRM: GPT-2 stage — describe as an additional rescoring pass,
a blended score, or a feature into a further model] → rule-based decision layer, thresholds tuned
on out-of-fold predictions throughout.

**Core Innovation:** Normalization and blocking that close the India/France gap without external
data or translation: (1) an Indic-to-Latin **consonant skeleton** exploiting the fact that major
Indian scripts share one Unicode block layout, so a single Devanagari consonant table maps
Gujarati/Tamil/Telugu/etc. names onto the same skeleton as their Latin transliteration, enabling
cross-script blocking and a `skel_cos3` similarity feature with no dictionary; (2) an **unsupervised
learned state-fill**: address components (city/locality) are counted against the states that always
co-occur with them in the same split's own S1 records (which all carry an explicit state), and used
to fill missing S2/S3 states purely from text statistics — no labels, no geocoding; (3) a **reverse
blocking channel** (each S2/S3 record finds its own best-matching S1, not just the other way
around) whose score became our single most important matching feature; (4) validating every
significant pipeline change against the **leaderboard, not only the internal holdout** — a
frozen-holdout validation set, however carefully constructed, does not reproduce the full
competitive dynamics of the test set (Section 5 discusses a concrete case where this mattered).

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
  A supervised meta-blocking pruner (a small LightGBM using only blocking-stage signals) was
  developed to shrink the candidate set, but is **not used in the submitted pipeline** — it
  measurably regressed leaderboard performance despite improving the internal holdout score
  (Section 5 explains why, and why we trust the leaderboard result over holdout here).
- **Candidate pairs generated:** 63.5M for the submitted pipeline (unpruned, 36.7/S1 average, max
  6,243 for one pathological S1).
- **How true matches were not lost:** candidate recall (share of true matches present in the
  candidate set) is tracked as the primary blocking metric on the frozen holdout, separately from
  final F0.5 — our blocking reaches **98.8%** candidate recall (oracle ceiling F0.5 = 0.996, i.e. a
  perfect matcher on these candidates would score 0.996). Every blocking iteration's miss set was
  categorized (non-Latin name, address-only overlap, state mismatch, outranked, blank address) and
  the next channel was built to target the largest remaining category, taking candidate recall from
  94.8% (single-channel baseline) to 98.8%.

---

## 4. Matching Model

**Features used (LightGBM stages):**
- **Name features:** exact/no-space-exact match, Levenshtein ratio, token-set/token-sort/partial
  ratio, Jaro-Winkler, token Jaccard, an **overlap coefficient** (min-based, so acronym/subset name
  pairs like "IBM" vs "IBM International Business Machines" score highly without the length penalty
  a plain Jaccard applies), first-token match, length ratio, legal-suffix Jaccard, IDF-weighted
  token alignment (share of each side's rare-word mass matched, and its inverse — the IDF mass of
  tokens that do *not* match, a direct signal for "these share a common word but are otherwise
  unrelated"), name-frequency-based chain indicators, character 2-/3-gram cosine, and the
  cross-script skeleton cosine.
- **Address features:** fuzzy ratio, token-set ratio, token Jaccard, length ratio, number Jaccard,
  first-number match, state agreement, zip agreement, **house-number edit distance and log-scaled
  numeric distance** (the single largest feature-gain contributor after the blocking scores), a
  composite-house-number equality flag, a hard binary house-number-**conflict** flag (both sides
  have a number and it differs, independent of how close the edit distance looks), and an
  address-word-overlap check specifically when house numbers agree (same building number, different
  street name → likely a different business).
- **Blocking/context features:** each channel's score and rank, score relative to the S1's best
  candidate, candidate count, source (S2/S3) flag, blank-address and non-Latin-name flags.
- **Group-consistency (stage 2) features:** for each candidate, agreement with the S1's other
  confident candidates (top 6 with stage-1 probability ≥ 0.3) on name, address, numbers, zip and
  state — plus the stage-1 probability's rank and gap to the best/second-best.

**Model type:**
1. **Stage 1 (pairwise, LightGBM, MIT-licensed):** trained on candidate pairs using the string/
   address/blocking features above.
2. **Stage 2 (group re-scorer, LightGBM):** takes stage-1 probabilities plus the group-consistency
   features and rescores each candidate against its own S1's other candidates, with a **fallback
   rule**: an S1 with at most one confident co-candidate falls back to the stage-1 score rather than
   trusting an unsupported group signal (this fixed a measured regression on single-match S1
   entities found during error analysis).
3. **[CONFIRM: Stage 3 — GPT-2.]** A GPT-2 checkpoint ([CONFIRM: exact variant, e.g. `gpt2` /
   `gpt2-medium`; parameter count; MIT license confirmed on the model card] — well under the
   competition's 8B-parameter cap) was fine-tuned [CONFIRM: entirely on our own labelled India/US
   pairs, with no external data] as [CONFIRM: a match/no-match sequence classifier / a reranker],
   run [CONFIRM: locally, with no network calls at inference — confirm this explicitly] on
   [CONFIRM: which candidates — all of them, or a restricted uncertain band?]. Its output is
   [CONFIRM: combined with the stage-2 LightGBM score by — replacing it / blended with it / fed in
   as an additional feature]. This stage improved the confirmed public leaderboard score from
   0.9690 (LightGBM-only) to **0.9770**.
   A CatBoost (Apache 2.0) blend was also tested on stage 1 in an earlier iteration; its gain was
   small (+0.0002 on out-of-fold data) and it is not used in the submitted pipeline.

**Threshold selection method:** grid search (coarse 0.05, refined to 0.01) on **out-of-fold**
predictions to maximize macro F0.5, never on the held-out validation set. Three rules, applied in
order: (1) keep pairs with probability ≥ *t*; (2) each S2/S3 record is assigned to at most one S1 —
the highest-probability one — directly justified by the ground truth's own uniqueness property, and
confirmed empirically (holdout F0.5 is higher with this rule than without it, 0.9773 vs 0.9769 in
our reference LightGBM-only pipeline); (3) a singleton gate: an S1 keeps its matches only if its
best candidate's probability ≥ *t_top*. No pretrained models are used in the LightGBM stages; the
GPT-2 stage is the one component with pretrained weights, fine-tuned entirely on our own data (see
Section 4's stage-3 note above for the compliance details to confirm before this ships).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):**
  - Public leaderboard, **best confirmed: 0.9770** (with the GPT-2 stage-3 rescoring).
  - Public leaderboard, LightGBM-only pipeline: **0.9690**.
  - Holdout F0.5, LightGBM-only pipeline (reference version, `v005`/`v005w`): **0.9773**,
    tracking its own out-of-fold estimate (0.9763) to within 0.001 — the two-stage LightGBM
    pipeline does not overfit its own validation split.
- **A genuine, load-bearing finding from our validation process:** partway through development we
  observed that **holdout F0.5 stopped reliably predicting leaderboard direction** for certain
  classes of change. Two concrete examples: (a) a supervised candidate-pruning step improved
  holdout F0.5 (+0.0005) but *reduced* the leaderboard score (−0.0031); (b) a further round of
  added matching features, evaluated on the exact same unpruned candidate set with no pruning
  involved at all, produced our single best holdout score of the entire project (0.9784) yet still
  scored *worse* on the leaderboard than the simpler feature set it was compared against (0.9640
  vs 0.9690). Root cause, best evidence to date: our internal holdout, while a fair random sample
  of Source 1 entities, under-represents the amount of competition for shared Source 2/3 records
  that occurs across the *full* test set (every Source 1 entity in test genuinely competes for
  Source 2/3 matches against many more entities than a partial holdout can simulate, because
  candidates it would compete against are disproportionately concentrated in the *train* fold,
  which never enters the holdout's competitive pool). **Practical consequence for our process:**
  from this point in development onward, only confirmed leaderboard scores were used to select
  between competing pipeline variants; holdout was still used as a sanity floor (to catch a
  version that's obviously broken) but never to rank two reasonable candidates against each other.
  We regard this as the most important methodological lesson of the project and would build
  multiple independent holdout-style validation constructions earlier in any future iteration.
- **Common false positives (wrong merges):** distinct businesses that share a street address or a
  near-identical name — e.g. two different companies at the same building number, or the same
  chain-style name appearing at different addresses. House-number-distance/conflict features and
  IDF-based "unmatched token" features were added specifically to target this pattern.
- **Common false negatives (missed matches):** the dominant category by far is **blank-address
  candidates** — pairs where one side's address is empty and the name alone must decide; recall on
  this segment was only ~51% versus ~99% when an address is present, because near-identical names
  can't be confirmed without any address signal. This remained only partially addressed at
  submission time. The remaining blocking-stage misses (candidates never generated at all) are a
  small and well-characterized share of total loss (~0.004 of the total ceiling gap), concentrated
  in non-Latin names with very thin addresses.
- **Note on France:** France has no training labels, so it cannot be error-analyzed directly. A
  confidence-profile diagnostic on stage-1 probabilities showed the model was not more *uncertain*
  on France than on India/US at any version checked, meaning any remaining France-specific gap is
  made of confident errors rather than a coverage problem — consistent with normalization/blocking
  improvements (which are country-agnostic by construction) closing most of an initially large
  French gap over successive iterations.

---

## 6. Conclusion

Our solution treats candidate generation as the primary lever for coverage (six complementary,
label-free blocking channels reaching 98.8% candidate recall) and treats precision as an explicit,
measured objective throughout the matching stage, in line with F0.5's 2x precision weighting — via
house-number and IDF-based disagreement features, a data-justified one-S1-per-record uniqueness
rule, a group-consistency re-scorer with a fallback safeguard for sparse cases, and [CONFIRM:
one sentence on what the GPT-2 stage contributes to precision/recall specifically, once confirmed].
The main lesson learned, and the one we would apply earliest in any future iteration, is that our
internal validation holdout — while reliable for catching obviously broken changes — systematically
under-represents the full test set's cross-entity competition dynamics, and so cannot be trusted
alone to rank competing model variants; we discovered this by comparing several submissions that
differed only in which population or feature set supplied which predictions, a diagnostic technique
we now consider essential rather than optional for this class of problem.

---

## Appendix

### A. Code Artefacts

The complete, runnable pipeline ships under `code/business_entity_resolution/` (`src/`, `README.md`,
`requirements.txt`). Entry points, run in order from the package root:

1. `src/normalization.py --split train` / `--split test` — builds normalized caches.
2. `src/blocking.py --split train --s1-set holdout|train --tag <v>` and `--split test --tag <v>` —
   six-channel candidate generation.
3. `src/matcher.py train|holdout|test --tag <v>s1 --cand-tag <v>` — stage-1 pairwise classifier.
4. `src/stage2.py train|holdout|test --stage1 <v>s1 --tag <v>` — stage-2 group re-scorer.
5. [CONFIRM: entry point script(s) for the GPT-2 stage-3 fine-tuning and inference, to be added
   under `src/` and listed here with their exact invocation, before this package is finalized.]
6. `src/check_submission.py` and `utils/validate_submission.py` — format validation before
   submission; writes/checks `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

Diagnostics used throughout development (`error_analysis.py`, `diagnose_blocking.py`,
`diagnose_country.py`, `diagnose_france.py`) read ground truth only for analysis; no labels are used
for training or threshold tuning beyond the frozen train-fold / holdout split. Full version history,
exact parameters and every intermediate metric are in `experiments/EXPERIMENTS.md`.

### B. Additional Results

| Version | Change | Candidate recall | Holdout F0.5 | Public LB |
|---|---|---|---|---|
| v001 | Baseline: word-only blocking + LightGBM | 94.8% | 0.9481 | 0.9310 |
| v002 | + state-fill normalization, 3-channel blocking | 96.8% | 0.9545 | 0.9436 |
| v004 | + house-number/IDF features, stage-2 fallback | 96.8% | 0.9667 | 0.9550 |
| v005 | + skeleton/address/reverse channels | **98.8%** | 0.9773 | 0.9683 |
| v006 | + supervised candidate pruning, feature/model tuning | 98.0% (pruned to 10.4/S1) | 0.9778 | 0.9652 (regressed — pruning under test-time competition; not used) |
| v005w | + overlap-coefficient name feature, unpruned | 98.8% | ~0.9773 | 0.9690 |
| v006np | + further feature additions, unpruned, no pruner | **98.8%** | **0.9784 (best ever)** | 0.9640 (regressed — see Section 5's central finding; not used) |
| **v005w + GPT-2 stage 3** | + fine-tuned GPT-2 rescoring | 98.8% | [CONFIRM] | **0.9770 (submitted)** |

Country breakdown (holdout, LightGBM-only pipeline): India F0.5 0.9712–0.9728, US F0.5 0.9813–0.9821
depending on exact feature version — a gap that narrowed from 0.043 in v001 to ~0.010 as
normalization and blocking improved.