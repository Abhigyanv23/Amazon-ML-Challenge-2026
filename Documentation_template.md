# Methodology - Business Entity Resolution (Amazon ML Challenge 2026)

Status: living document, updated through v006. Best submitted so far: **v005** (holdout F0.5 0.9773,
public LB 0.9683). **v006** is the current candidate for the final submission: holdout F0.5 0.9778,
LB not yet uploaded as of this update. Public leaderboard leader (26 Sep): 0.988419.

## 1. Methodology overview

Pipeline: data analysis -> normalization (incl. learned state fill) -> multi-channel blocking ->
meta-blocking pruner (v006) -> pair features -> LightGBM stage-1 matcher, blended with CatBoost
(v006) -> stage-2 group-consistency rescoring with a fallback rule -> decision logic tuned for
macro F0.5 -> output files -> local scorer + official validator. Only competition-provided data is
used at every stage; no external lookups, no pretrained models.

## 2. Data analysis (train unless stated)

| Item | Value |
|---|---|
| Rows S1 / S2 / S3 (train) | 2,206,821 / 5,034,616 / 5,285,603 |
| Rows S1 / S2 / S3 (test) | 1,732,544 / 4,887,273 / 5,082,316 |
| Countries (train) | US 60%, India 40% |
| Countries (test) | India 47%, US 38%, **France 15% (absent from train)** |
| Blank names | 0 in all files |
| Blank addresses | S1 0%; S2/S3 about 3% |
| Matches per S1 | 0: 5.6%, 1: 5.4%, 2: 17.0%, 3: 24.1%, 4: 21.9%, 5+: 26.0% (max 11, mean 3.46) |
| S2/S3 records matched to more than one S1 | 0 (each belongs to at most one S1) |
| S2/S3 records used in any match | about 74%; the rest are distractors |
| True pairs in the same country | 100% (300K-pair sample) |
| True pairs with identical names (light normalization) | 21.8% |

Observed noise: native-script (Devanagari/Tamil) names, injected accents, typos, word reordering,
appended words ("Center", "Services", "#35740", ".com" forms), truncated addresses, state name
variants (UP / Uttar Pradesh / Devanagari), house-number variants (3906 / 3906a), digits spelled
into words ("co1onial" for "colonial"), postcodes fused into house numbers. Postcodes are rare
overall (US 11%, India under 1%). Generic chain names repeat heavily (e.g. "primary care group"
253 times in S1).

Test-set country diagnostic (unsupervised, before v002 fixes): France S2/S3 addresses often end in
a city with no region, state found only 66-68%; 28.5% of French S1 names occur 5+ times (India
37.1%, US 20.8%), a chain-like pattern. Later (post-v004), stage-1 confidence on France sits
between India and US (top p >= 0.95: France 93.5%, India 91.3%, US 93.7%; mid-confidence band
0.3-0.8: 1.0% / 1.7% / 0.7%), meaning the model is not uncertain about France specifically; any
remaining France gap is confident errors or leaderboard-estimate noise, not a coverage problem.

## 3. Validation design

- Frozen 80/20 split of train S1 IDs, stratified by country x match-count bucket (seed 42):
  1,765,457 train / 441,364 holdout (`experiments/splits/holdout_s1_ids.txt`).
- Blocking candidates for holdout S1 are drawn from the full S2+S3 pool, mirroring test.
- Model training and decision tuning use train-fold data only (3-fold group-by-S1 out-of-fold).
  Holdout is scored once per version. Stage 2 and the pruner are both trained on out-of-fold
  stage-1 outputs, never on the holdout.
- Local scorer: per-S1 F0.5, macro-averaged. Engineering assumption (not stated in the official
  rules, but consistent with them): an entity with true matches and an empty prediction scores 0;
  a true singleton scores 1 only if the prediction is empty. Self-test: perfect = 1.0000,
  all-empty = 0.0558 (matches the holdout singleton share).
- Holdout tracks out-of-fold estimates closely across every version (no overfitting to the
  holdout): v001 0.9481 vs OOF 0.9467; v002 0.9545 vs 0.9533; v005 0.9773 vs OOF 0.9763.
- Public LB moves in the same direction as holdout every submission, but by a larger margin
  (v001->v002: holdout +0.0064, LB +0.0126; v004->v005: holdout +0.0106, LB +0.0133). With India/US
  held near their holdout levels, this gap is explained mostly by France improving faster than the
  labeled countries, since France has no ground truth to validate against directly.
- One change per version, tracked in `experiments/EXPERIMENTS.md`, so any regression is traceable
  to a single component.

## 4. Normalization

Rule-based, no learned parameters beyond the unsupervised state-fill described below. State/region
maps for the US, India (including Hindi/Tamil names) and France come from general language
knowledge, not from business data.

- **Names (v1):** NFKC normalize, Latin accent folding (non-Latin scripts preserved), lowercase,
  domain unwrapping (`konkandata.com` -> `konkandata`), `#digits` removal, `&` -> `and`,
  abbreviation expansion (pvt -> private, ltd -> limited, co -> company, ...). Legal forms are
  split into `name_legal`; `name_core` is the name without them; `name_nospace` removes spaces.
- **Addresses (v1):** per-country abbreviation maps (street -> st, R -> rue, ...); a comma
  component that matches a state/region becomes a canonical code (`addr_state`); house/unit numbers
  extracted into `addr_nums`.
- **v2 additions:** postcodes separated from house numbers into `addr_zip` (US 5-digit, France
  5-digit, India 6-digit PIN, each matched only as the last token of a component); `addr_state`
  takes the **last** matching component, fixing cases like "Washington, NY" being read as the
  state. Learned state fill: from the same split's S1 records (which all carry a state), address
  components (city, locality) are counted by state; components seen 20+ times with 98%+ purity
  fill empty states in S2/S3 by majority vote, using unlabeled text only (`addr_state_src` flags
  this as inferred). State-found rate: France 67% -> 97%, India 91% -> 96.5%.
- **v3 additions:** Indic-to-Latin consonant skeleton via shared Unicode block layout
  (`name_skel`), for matching transliterated names without a lookup table; composite house number
  field (`addr_hn`) combining number and unit; state rule generalized to "last 2-letter code, else
  first spelled-out region name"; digits spelled inside words are recovered ("co1onial" ->
  "colonial"); 6+ digit runs are stripped from names (phone numbers, IDs); letters glued to digits
  are split ("fl13" -> "fl 13").
- Country is treated as an open string label throughout; nothing is hardcoded to {US, India}, so
  France is normalized by the same general rules rather than a special case.

Evaluation on 200K train-fold true pairs vs. random same-country pairs:

| Metric | True pairs (v1 -> latest) | Random pairs |
|---|---|---|
| Name exact (name_core) | 0.220 -> 0.508 | 0.000 |
| Name token Jaccard (name_core) | 0.617 -> 0.699 | 0.003 |
| Address token Jaccard (addr_norm) | 0.597 -> 0.766 | 0.022 |
| >=1 shared number (both have numbers) | 0.945 | 0.037 |
| State agrees (both have state, v1 -> v2) | 0.993 -> 0.994 | 0.068-0.069 |

Normalization consistently raises similarity for true pairs while leaving random pairs near zero,
confirming it isn't just inflating similarity across the board.

## 5. Blocking / candidate generation

### v001: single word channel
Per (country, state) block, with records lacking a state joining every block of their country:
IDF-weighted vectors over `name_core` words, address words and address numbers; tokens with
document frequency above 2% of the block are dropped; cosine similarity; top-k per S1 via sparse
top-n matrix multiplication.

| k | Candidate recall | Oracle F0.5 ceiling | Total pairs (holdout) |
|---|---|---|---|
| 5 | 0.838 | 0.953 | 2.2M |
| 10 | 0.928 | 0.973 | 4.4M |
| 20 | 0.948 | 0.980 | 8.8M |
| 50 | 0.962 | 0.985 | 22.1M |

Miss analysis at k=50 (57,559 missed pairs): non-Latin candidate name 21.1K; overlap only visible
via address, concatenated or domain-style names, 13.3K; state mismatch 9.6K (Telangana/Andhra
Pradesh relabeling, and the "Washington, NY" bug fixed in v2); blank address plus name typo 9.0K;
outranked typos 4.5K.

### v002: three channels, unioned
- Word channel (top 20): as v001, plus a `name_nospace` token.
- Trigram channel (top 10): character trigrams of `name_nospace`, catches typos and concatenations.
- Non-Latin channel (top 5): address-only similarity for pool records with non-Latin names.
- Blocks merged where label sources mix (India: AP+TG, JK+LA); IDF smoothed.

| Holdout | v001 (k=20) | v002 |
|---|---|---|
| Candidate recall | 0.948 | 0.968 |
| Oracle F0.5 ceiling | 0.980 | 0.989 |
| Recall India / US | - | 0.948 / 0.981 |
| True pairs found only by trigram / non-Latin channel | - | 7,058 / 6,733 |
| Misses due to state blocks | 9,622 | 1,246 |
| Avg candidates per S1 | 20 | 25.6 |
| Runtime (holdout) | 815s | 974s |

### v003-v004: unchanged blocking, upstream improvements
No blocking change; v003 and v004 both reuse v002 candidates while improving the matcher (see
Sections 6-7). Candidate recall and oracle ceiling stay at 0.968 / 0.989 for both versions.

### v005: skeleton, address and reverse channels
- **Skeleton channel:** blocks on `name_skel` (the Indic-to-Latin consonant skeleton), catching
  transliteration variants a token or trigram match misses.
- **Address channel:** blocks directly on normalized address tokens/numbers independent of name,
  recovering pairs where the name diverges heavily but the address matches.
- **Reverse channel:** for each S2/S3 record, finds its own best-matching S1 candidates and adds
  that S1 back as a candidate, recovering asymmetric misses the forward search alone doesn't reach.

| Holdout | v002 | v005 |
|---|---|---|
| Candidate recall | 0.968 | 0.988 |
| Oracle F0.5 ceiling | 0.989 | 0.996 |
| Recall India / US | 0.948 / 0.981 | 0.982 / 0.992 |
| Only-channel finds | - | word 46,578, skeleton 8,282, trigram 5,975, address 5,839, non-Latin 644, reverse 106 |
| Misses due to state blocks | 1,246 | 2,899 |
| Avg candidates per S1 | 25.6 | 37.4 (max 6,243) |

The reverse channel drives most of the recall gain but is the noisiest: it lets a single S1 pick up
thousands of low-value candidates in pathological chain-name cases (max 6,243 candidates for one
S1), motivating the pruner added in v006.

### v006: supervised meta-blocking pruner
A second, lightweight model scores each v005 candidate pair using only blocking-stage signals
(channel scores, ranks, candidate-count context) and prunes to the top candidates per S1: cap 20,
minimum-score threshold tau 0.005, both chosen on out-of-fold data to bound recall loss at 0.008.

| Holdout | v005 (unpruned) | v006 (pruned) |
|---|---|---|
| Candidate recall | 0.988 | 0.980 |
| Avg candidates per S1 | 37.4 | 10.4 |
| Total pairs (holdout) | - | 4.59M (from 37.4 avg) |
| Total pairs (test) | 63.5M | 19.0M (11.0 avg per S1, capped at 20) |

The pruner trades 0.008 of candidate recall for a 3.6x reduction in pairs the matcher has to score,
which lowers both training time and the matcher's false-positive surface. Net effect on holdout
F0.5 is positive once combined with the v006 matcher changes (Section 7).

## 6. Feature engineering

Features grow from 39 (v001) to 46 (v002) to a further expanded set in v004-v006 (v3, v4, v5
feature passes). No country feature at any stage, since France is unseen in training.

- **Name:** exact / no-space exact, Levenshtein ratio, token-set, token-sort and partial ratios,
  Jaro-Winkler and Levenshtein on the no-space form, token Jaccard, first-token match, token-count
  difference, length ratio, legal-form Jaccard.
- **Address:** ratio, token-set ratio, token Jaccard, length ratio; number Jaccard, first-number
  match, share of S1 numbers present; state agreement; zip agreement (from v2).
- **Blocking/context:** each channel's score and rank (word, trigram, non-Latin, and from v005
  skeleton, address, reverse); score relative to the S1's best candidate; candidate count; S2/S3
  source flag; name frequency in S1 and in the pool (chain indicator); non-Latin and blank-address
  flags; count of near-identical names among the S1's other candidates; a combined name+address
  similarity score and its rank/gap within the S1's candidate set.
- **v3 additions (v004):** house-number edit distance (`hn_edit`) and log-scaled numeric distance
  (`hn_logdiff`); whether the candidate's name appears as a substring of the S1 name
  (`b_name_in_s1`); IDF-weighted alignment and name-specificity features; n-gram cosine similarity.
  `hn_edit` alone is the single largest new contributor to model gain after v004.
- **v4 additions (v005):** skeleton cosine similarity (`skel_cos3`) and the new blocking channel
  scores (`blk_sk`, `blk_rev` for skeleton and reverse channels). `blk_rev` becomes the top feature
  by gain once the reverse channel is added, reflecting how much signal that channel carries.
- **v5 additions (v006):** unmatched IDF mass between the two records' name tokens
  (`unmatched_idf_b`), house-number-equals combined with word Jaccard (`hn_eq_word_jac`), added
  alongside the pruner's own context features.
- **Stage-2 features (from v003):** for each candidate, agreement with the other confident
  candidates of the same S1 (top 6 with stage-1 probability >= 0.3): name, no-space name, address,
  numbers, zip, and state agreement (max, probability-weighted mean, and a strong-link count),
  plus the stage-1 probability's rank, gap to the second-best, and group counts.

## 7. Model

**Stage 1:** LightGBM binary classifier (MIT licence). v001-v005: learning rate 0.1, 127 leaves,
min 100 rows per leaf, feature/bagging fraction 0.9/0.8, early stopping on 3-fold group-by-S1
out-of-fold validation, final model retrained on all train-fold pairs at 1.1x the mean best
iteration. v006 changes: learning rate lowered to 0.05, 5 folds (from 3), and the stage-1 output
becomes a blend of LightGBM and CatBoost (weight w=0.5 chosen on OOF; w=0 0.9753, w=0.5 0.9757,
w=1.0 0.9755, both models licensed MIT/Apache 2.0). No pretrained models at any stage.

Training data scales with the candidate/feature changes: v002 used 300K train-fold S1 with 7.68M
candidate pairs (13.1% positive); v006 trains on 300K S1 over 3.12M pruned pairs (32.7% positive),
a much richer positive rate after pruning removes easy negatives.

Top features by gain, v002: combined name+address similarity 0.49, its gap to the S1's best 0.08,
number Jaccard 0.08, address token-set 0.05, word-channel rank and score 0.04 each, pool name
frequency 0.02. By v006 the ranking shifts toward the newer features: combined similarity 0.44,
`hn_edit` 0.10, `hn_logdiff` 0.05, `blk_sk` 0.05, `unmatched_idf_b` 0.03, reflecting how much of
the remaining signal now comes from house-number and skeleton-based features rather than raw text
similarity alone.

**Stage 2 (from v003):** a second LightGBM trained on out-of-fold stage-1 probabilities plus the
group-consistency features from Section 6. v003 used the stage-2 score directly; from v004 onward,
stage 2 uses a **fallback rule**: if an S1 entity has too few confident co-candidates to compute
reliable group-consistency features (its stage-1 probabilities all sit below 0.3), the pipeline
falls back to the plain stage-1 decision rather than trusting an unsupported stage-2 score. This
fixes a regression stage 2 introduced on 1-match S1 entities (F0.5 0.8565 without the fallback vs.
0.8784 for stage-1 alone in v003; recovers to 0.9009 with the fallback in v004). v006 extends this
to a **fallback search over 1/2/3** confident-candidate thresholds, chosen on OOF (0.9772 / 0.9770
/ 0.9767), settling on FB_MAX=1: the strictest fallback threshold, i.e. stage 2 is trusted only
when an S1 has at least one other confident candidate to compare against.

## 8. Threshold / decision logic

1. Keep pairs with probability >= t.
2. Each S2/S3 record goes to at most one S1, the highest-probability one; justified directly by
   the data (no S2/S3 record matches more than one S1 in the ground truth).
3. Singleton gate: an S1 keeps its matches only if its best candidate's probability >= t_top.
4. **v006 addition:** a separate blank-address threshold `t_blank`, since blank-address candidates
   behave differently (lower baseline similarity, see Section 10) and a single global threshold
   under- or over-triggers on them; `t_blank = 0.75` chosen on OOF, +0.0001 F0.5 over using the
   global threshold uniformly.

t and t_top are grid-searched (0.20-0.95, step 0.05, with a finer 0.01 grid added to the tuner
after v003) to maximize macro F0.5 on out-of-fold predictions. Selected values: v001 t = t_top =
0.65; v002 t = t_top = 0.70. Neighbouring settings score within 0.0005 of the selected value in
every version, so the choice is stable rather than a narrow optimum.

## 9. Experiments and results

| Version | Change | Cand. recall | Oracle F0.5 | Holdout F0.5 | Precision / Recall | Singleton acc. | Public LB |
|---|---|---|---|---|---|---|---|
| v000 | EDA, split, scorer | - | - | 0.0558 (all-empty) | - | 1.000 | - |
| v001 | Normalization v1 + word blocking (k=20) + LightGBM | 0.948 | 0.980 | 0.9481 | 0.987 / 0.894 | 0.950 | 0.9310 |
| v002 | Normalization v2 (zip, state fill) + 3-channel blocking + features v2 | 0.968 | 0.989 | 0.9545 | 0.988 / 0.905 | 0.953 | 0.9436 |
| v003 | Stage-2 group-consistency rescoring | 0.968 | 0.989 | 0.9580 | 0.988 / 0.920 | 0.963 | not submitted |
| v004 | Features v3 + stage-2 fallback | 0.968 | 0.989 | 0.9667 | 0.992 / 0.929 | 0.965 | 0.9550 |
| v005 | Normalization v3 + blocking v3 (skeleton/address/reverse) + features v4 | 0.988 | 0.996 | 0.9773 | 0.994 / 0.951 | 0.970 | 0.9683 |
| v006 | Pruner + features v5 + LightGBM/CatBoost blend + t_blank + fallback search | 0.980 | 0.994 | 0.9778 | 0.995 / 0.950 | 0.974 | pending |

Holdout tracks the out-of-fold estimate within 0.001-0.002 at every version, so the local scorer
is a reliable proxy for the leaderboard's direction, though not its exact magnitude (Section 3).

Country breakdown (holdout, where measured): India trails US throughout, but the gap narrows as
normalization and blocking improve: v001 India 0.9225 / US 0.9652 (gap 0.043); v002 0.9358 / 0.9669
(gap 0.031); v005 0.9712 / 0.9813 (gap 0.010).

Full per-version detail (blocking miss breakdowns, feature gain tables, exact OOF numbers) is
tracked in `experiments/EXPERIMENTS.md` and not duplicated here.

## 10. Error analysis

Consolidated across versions, using the v005 holdout breakdown as the most complete available
picture (total loss 1 - F0.5 = 0.0227):

| Error type | Share of loss |
|---|---|
| Missing-only (blocking or matcher under-recall) | 0.0116 |
| False-empty (true matches predicted as none) | 0.0048 |
| Extra false positive | 0.0035 |
| Singleton false positive | 0.0017 |
| False positive + missing combined | 0.0011 |

- **Blank-address candidates** are the single weakest segment: recall 0.508 on holdout true pairs
  with a blank address on one side (about 61K true pairs, roughly 30K false negatives), motivating
  the dedicated `t_blank` threshold in v006.
- **False positives** cluster around genuinely distinct businesses that share a street address
  (e.g. "Manya Traders" vs. "Manya Rifle" at the same house number, different street) and around
  chain names, since the pool contains many near-identical records that are legitimately different
  entities.
- **The per-record uniqueness rule** (each S2/S3 record assigned to at most one S1) is confirmed to
  help rather than just theoretically justified: holdout F0.5 is consistently higher with it than
  without (e.g. v005 0.9773 vs. 0.9769 without), since 42% of v005 holdout false positives belong
  to S1 entities that already appear in the train fold and are resolved once uniqueness is applied.
- **Stage 2's regression on 1-match S1 entities** (Section 7) was the largest single bug caught
  during error analysis; the fallback rule turned a net negative (v003: 0.8565 vs. stage-1's
  0.8784) into a net positive (v004: 0.9009) for that segment, while the bootstrap confidence
  interval on the overall v003-v002 holdout gain, [0.0033, 0.0037], confirmed the rest of stage 2's
  benefit was real rather than noise.
- **Non-Latin names** remain the largest single blocking-miss category throughout (21.1K at
  k=50 in v001), addressed partially by the non-Latin and skeleton channels but not fully closed;
  India's persistent gap vs. US in Section 9 traces mostly to this.
- **France** cannot be directly error-analyzed (no ground truth), but the confidence-profile
  diagnostic in Section 2 and the leaderboard movement in Section 9 together suggest its errors are
  now similar in kind to India/US rather than being a distinct failure mode from missing blocking
  coverage, which was the dominant France-specific issue before the v002 state fill.

## 11. Final approach

As of this update, v006 is the strongest holdout result (0.9778) but has not yet been submitted to
the leaderboard, so its public LB score and the resulting choice of final model are still pending.
Planned steps for the remainder of the challenge:

1. Confirm v006's public LB score against holdout (expected in the 0.970s range, extrapolating
   from the holdout-to-LB gap pattern in Section 9).
2. Re-run the full pipeline from a fresh clone on a second machine (already verified once,
   candidate recall and oracle ceiling reproduced exactly: 0.9881 / 0.9961, avg 37.5 candidates on
   an 800K-S1 sample), to confirm the submission package is reproducible end-to-end before
   packaging.
3. Final-fit the selected model on the full training set (currently trained on a 300K-S1 subsample
   per version for iteration speed), then regenerate `matching_results.tsv` and
   `candidate_pairs.tsv` for the test set from that final fit.
4. Run `utils/validate_submission.py` against the final output files, then assemble the submission
   zip with the runnable `code/business_entity_resolution/` pipeline and this document.

Decision criterion for which version ships as final: whichever of v005/v006 has the higher
confirmed public LB score, with holdout F0.5 and the per-country breakdown as tie-breakers if LB
scores land within noise of each other.