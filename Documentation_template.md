# Methodology - Business Entity Resolution (Amazon ML Challenge 2026)

Status: updated through v006 and the per-country hybrids. Best public leaderboard so far:
**v005** (git tag `day2-sub2`; holdout F0.5 0.9773, public LB 0.9683). Public leaderboard leader
(26 Sep): 0.988419.

## 1. Methodology overview

Pipeline: data analysis -> normalization (incl. learned state fill and cross-script skeletons) ->
multi-channel blocking (six channels) -> pair features -> LightGBM stage-1 matcher -> stage-2
group-consistency rescoring with a fallback rule -> decision logic tuned for macro F0.5 -> output
files -> local scorer + official validator. Only competition-provided data is used at every stage;
no external lookups, no pretrained models, no network calls at run time.

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

Observed noise: native-script names (Devanagari, Gujarati, Tamil, ...), injected accents, typos, word
reordering, appended words ("Center", "Services", "#35740", ".com" forms), truncated addresses, state
name variants (UP / Uttar Pradesh / Devanagari), house-number variants (3906 / 3906a, 0071/1 vs 71/1),
digits inside words ("co1onial"), postcodes fused with house numbers. Postcodes are rare (US 11%, India
under 1%). Generic chain names repeat heavily (e.g. "primary care group" 253 times in S1).

Test-set country diagnostic (unsupervised): France S2/S3 addresses often end in a city with no region
(state found only 66-68% before the v002 state fill); 28.5% of French S1 names occur 5+ times (India
37.1%, US 20.8%). Post-v004, stage-1 confidence on France sits between India and US (top p >= 0.95:
France 93.5%, India 91.3%, US 93.7%; top p in 0.3-0.8: 1.0% / 1.7% / 0.7%): the model is not
uncertain on France specifically.

## 3. Validation design

- Frozen 80/20 split of train S1 IDs, stratified by country x match-count bucket (seed 42):
  1,765,457 train / 441,364 holdout.
- Blocking candidates for holdout S1 are drawn from the full S2+S3 pool, mirroring test.
- Model training and decision tuning use train-fold data only (group-by-S1 out-of-fold). Stage 2 and
  the pruner are trained on out-of-fold outputs; the holdout is scored once per version.
- Local scorer: per-S1 F0.5, macro-averaged. Engineering assumption (consistent with the official
  rules): an entity with true matches and an empty prediction scores 0; a true singleton scores 1 only
  if the prediction is empty. Self-test: perfect = 1.0000, all-empty = 0.0558 (holdout singleton share).
- Holdout tracks out-of-fold within 0.001-0.002 in every version (v001 0.9481 vs 0.9467; v002 0.9545 vs
  0.9533; v004 0.9667 vs 0.9658; v005 0.9773 vs 0.9763). Bootstrap CIs over S1 entities (e.g. v003-v002
  difference [0.0033, 0.0037]) confirm version differences of ~0.001+ are real.
- **Known limitation, found in v006:** through v005 the public LB moved in the same direction as holdout.
  v006 broke this (holdout +0.0005, LB -0.0031). Test S1 records all compete for the same pool records
  under the one-S1-per-candidate rule, while holdout S1 mostly do not (their competitors are train-fold
  S1). Changes that interact with that competition (candidate pruning) are therefore not fully
  measurable on the holdout. From v006 onward, such changes are judged on the public LB.
- France has no labels; its quality is inferred from LB movement with India/US held at holdout levels,
  and from per-country hybrid submissions (Section 9).
- Versions, submissions and tags are tracked in `experiments/EXPERIMENTS.md`.

## 4. Normalization

Rule-based. State/region maps (US, India incl. Hindi/Tamil names, France regions/departments) and
street abbreviations come from general language knowledge, not business data.

- **Names (v1):** NFKC, Latin accent folding (non-Latin scripts preserved), lowercase, domain unwrapping
  (`konkandata.com` -> `konkandata`), `#digits` removal, `&` -> `and`, abbreviation expansion (pvt ->
  private, ltd -> limited, co -> company, ...). Legal forms go to `name_legal`; `name_core` is the name
  without them; `name_nospace` removes spaces.
- **Addresses (v1):** per-country abbreviation maps (street -> st, R -> rue, ...); a comma component that
  is a state/region becomes a canonical code (`addr_state`); house/unit numbers extracted (`addr_nums`).
- **v2:** postcodes split into `addr_zip` (US 5-digit ZIP only as the last token of a component, France
  5-digit, India 6-digit PIN); learned state fill: from the same split's S1 records (which all carry a
  state), address components are counted by state; components seen 20+ times with 98%+ purity fill empty
  S2/S3 states by majority vote (unlabeled text only; `addr_state_src` flags inferred states). State
  found: France S2/S3 67% -> 97%, India 91% -> 96.5%.
- **v3:** Indic-to-Latin consonant skeleton (`name_skel`): Indic scripts (Devanagari, Bengali, Gurmukhi,
  Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam) share one Unicode block layout, so a single
  Devanagari consonant table maps all of them onto the same consonant classes as Latin names (e.g.
  "Lotus Consulting" and its Devanagari form both -> `lts knsltnk`); composite house number `addr_hn`
  ("H No 71/1" and "0071/1" -> `71/1`); state rule = last 2-letter code component, else first spelled-out
  name ("OH, Delaware" -> oh; "District of Columbia, ..., Washington" -> dc); digits inside words mapped to
  letters ("co1onial" -> "colonial"); runs of 6+ digits removed from names; letter->digit split
  ("fl13" -> "fl 13").
- Country is an open string label; nothing is hardcoded to {US, India}.

Evaluation on 200K train-fold true pairs vs. random same-country pairs:

| Metric | True pairs (baseline -> normalized) | Random pairs |
|---|---|---|
| Name exact (name_core) | 0.220 -> 0.508 | 0.000 |
| Name token Jaccard (name_core) | 0.617 -> 0.699 | 0.003 |
| Address token Jaccard (addr_norm) | 0.597 -> 0.766 | 0.022 |
| >= 1 shared number (both have numbers) | 0.945 | 0.037 |
| State agrees (both have state, v1 -> v2) | 0.993 -> 0.994 | 0.068 |

## 5. Blocking / candidate generation

All channels run per (country, state) block; pool records without a state join every block of their
country. Each channel uses IDF-weighted sparse vectors, cosine similarity and top-k per S1 via sparse
top-n matrix multiplication; channel outputs are unioned and each channel's score and rank are kept as
matcher features. Reduction vs. all same-country pairs on test (6.72 trillion): > 99.999%.

### v001: single word channel
IDF-weighted name words, address words and numbers; tokens with document frequency above 2% of the block
dropped.

| k | Candidate recall | Oracle F0.5 ceiling | Total pairs (holdout) |
|---|---|---|---|
| 5 | 0.838 | 0.953 | 2.2M |
| 10 | 0.928 | 0.973 | 4.4M |
| 20 | 0.948 | 0.980 | 8.8M |
| 50 | 0.962 | 0.985 | 22.1M |

Miss analysis at k=50 (57,559 pairs): non-Latin names 21.1K; overlap only via address (concatenated or
domain names) 13.3K; state mismatch 9.6K; blank address + name typo 9.0K; outranked typos 4.5K.

### v002: three channels
Word (top 20, plus a `name_nospace` token), character trigrams of `name_nospace` (top 10), address-only
for pool records with non-Latin names (top 5); AP+TG and JK+LA blocks merged; smoothed IDF.

### v005: six channels (used by the best submission)
- **Skeleton channel (top 5):** skeleton trigrams + address tokens, against pool records with non-Latin
  names (cross-script matching).
- **Address-only channel (top 5):** against all pool records (names that diverge completely).
- **Reverse channel (top 2):** each pool record keeps its top-2 S1 by the word-channel vectors, computed
  against ALL S1 of the split (so holdout mirrors test).
- Word channel also gets the composite house-number token.

| Holdout | v001 (k=20) | v002 | v005 |
|---|---|---|---|
| Candidate recall | 0.948 | 0.968 | 0.988 |
| Oracle F0.5 ceiling | 0.980 | 0.989 | 0.996 |
| Recall India / US | - | 0.948 / 0.981 | 0.982 / 0.992 |
| Misses | 57,559 (k=50) | 49,259 | 18,114 |
| Avg candidates per S1 | 20 | 25.6 | 37.4 (max 6,243) |

True pairs found by only one channel (v005): word 46,578, skeleton 8,282, trigram 5,975, address 5,839,
non-Latin address 644, reverse 106. The reverse channel adds few unique candidates, but its score
(whether this S1 is among the pool record's top choices) became the strongest matcher feature. It also
lets some S1 collect very many candidates (max 6,243).

### v006: supervised meta-blocking pruner (evaluated; not in the best submission)
A small LightGBM scores each blocked pair from blocking outputs only (channel scores/ranks, number of
channels that found the pair, score relative to the S1's best, candidate count) and keeps pairs with
score >= tau, at most `cap` per S1 (supervised meta-blocking, Papadakis et al. 2014). cap=20, tau=0.005
chosen on out-of-fold data (smallest candidate set within 0.008 recall of unpruned).

| | v005 (unpruned) | v006 (pruned) |
|---|---|---|
| Holdout candidate recall | 0.988 | 0.980 (OOF predicted 0.980) |
| Avg / max candidates per S1 | 37.4 / 6,243 | 10.4 / 20 |
| Test pairs | 63.5M | 19.0M |
| Holdout F0.5 (full pipeline) | 0.9773 | 0.9778 |
| Public LB | 0.9683 | 0.9652 |

Pruning reduced the candidate set 3.6x with a slightly higher holdout score, but lowered the public LB
on India/US (Section 9).

## 6. Feature engineering

No country feature at any stage (France is unseen in training).

- **Name:** exact / no-space exact, Levenshtein ratio, token-set, token-sort and partial ratios,
  Jaro-Winkler and Levenshtein on the no-space form, token Jaccard, first-token match, token-count
  difference, length ratio, legal-form Jaccard.
- **Address:** ratio, token-set ratio, token Jaccard, length ratio; number Jaccard, first-number match,
  share of S1 numbers present; state agreement; zip agreement (v2).
- **Blocking/context:** each channel's score and rank; score relative to the S1's best candidate;
  candidate count; S2/S3 flag; name frequency in S1 and in the pool (chain indicator); non-Latin,
  blank-address and inferred-state flags; count of near-identical names among the S1's candidates;
  combined name+address similarity and its rank/gap within the S1's candidates.
- **v3 (v004):** IDF-weighted name-token alignment (share of each side's IDF mass matched, matched IDF
  sum); name specificity (`b_name_in_s1`: how many S1 records share the candidate's name;
  `a_name_in_pool`); house-number edit distance (`hn_edit`) and log numeric distance (`hn_logdiff`);
  address-word overlap without numbers; character 2/3-gram cosine for name and address.
- **v4 (v005):** skeleton trigram cosine (`skel_cos3`, cross-script name match), composite house-number
  equality (`hn_comp_eq`), and the skeleton/address/reverse channel scores (`blk_rev` top by gain).
- **v5 (v006 only):** IDF mass of unmatched name tokens (`unmatched_idf_a/b`), address-word overlap when
  house numbers are equal (`hn_eq_word_jac`).
- **Stage 2 (from v003):** per candidate, agreement with the other confident candidates of the same S1
  (top 6 with stage-1 p >= 0.3) on name, no-space name, address, numbers, zip, state (max,
  probability-weighted mean, strong-link count), plus the stage-1 probability's rank, gap to the best,
  second-best and group counts.

## 7. Model

**Stage 1 (v005):** LightGBM binary classifier (MIT), learning rate 0.1, 127 leaves, min 100 rows per
leaf, feature/bagging fraction 0.9/0.8, early stopping on 3-fold group-by-S1 validation; final model
retrained on all training pairs at 1.1x the mean best iteration. Training: 300K train-fold S1, 11.25M
candidate pairs (9.1% positive). Top gain: `blk_rev` 0.45, combined similarity 0.20, `hn_edit` 0.04,
number Jaccard 0.04, `blk_sk` 0.02, `skel_cos3` 0.02.

**v006 variant:** learning rate 0.05, 5 folds, CatBoost (Apache 2.0, GPU) blended with LightGBM (blend
weight chosen on OOF: w=0 0.9753, 0.5 0.9757, 1.0 0.9755 -> +0.0002), trained on 3.12M pruned pairs.

**Stage 2 (from v003):** a second LightGBM on out-of-fold stage-1 probabilities plus the group-consistency
features. **Fallback rule (from v004):** for an S1 with at most FB_MAX candidates having stage-1 p >= 0.3,
the stage-1 probability is used instead of stage 2 (a single confident candidate has nothing to be
compared against). This fixed stage 2's regression on 1-match S1 (v003 0.8565 vs stage-1 0.8784; v004
0.9009 with the fallback). FB_MAX=1 was best in v006's 1/2/3 search (OOF 0.9772 / 0.9770 / 0.9767).

No pretrained models at any stage; all models trained from scratch on the provided training data.

## 8. Threshold / decision logic

1. Keep pairs with probability >= t.
2. Each S2/S3 record goes to at most one S1 (the highest-probability one); justified by the data (no S2/S3
   record matches more than one S1 in the ground truth); confirmed on holdout (v005 0.9773 with, 0.9769
   without).
3. Singleton gate: an S1 keeps its matches only if its best probability >= t_top.
4. (v006 only) separate threshold `t_blank` for blank-address candidates (0.75, +0.0001 OOF).

t and t_top are grid-searched (0.20-0.95 step 0.05, then 0.01 around the best) to maximize macro F0.5 on
out-of-fold predictions. Selected: v001 0.65/0.65; v002-v005 about 0.70/0.70. Neighbouring settings score
within 0.0005, so the optimum is flat and stable.

## 9. Experiments and results

| Version | Change | Cand. recall | Oracle F0.5 | Holdout F0.5 | Precision / Recall | Cands/S1 (test) | Public LB |
|---|---|---|---|---|---|---|---|
| v001 | Normalization v1 + word blocking + LightGBM | 0.948 | 0.980 | 0.9481 | 0.987 / 0.894 | 20.0 | 0.9310 |
| v002 | State fill, zip, 3-channel blocking, features v2 | 0.968 | 0.989 | 0.9545 | 0.988 / 0.905 | 25.6 | 0.9436 |
| v003 | Stage-2 rescoring | 0.968 | 0.989 | 0.9580 | 0.988 / 0.920 | 25.6 | not submitted |
| v004 | Features v3 + stage-2 fallback | 0.968 | 0.989 | 0.9667 | 0.992 / 0.929 | 25.6 | 0.9550 |
| **v005** | Normalization v3 + 6-channel blocking + features v4 | 0.988 | 0.996 | 0.9773 | 0.994 / 0.951 | 36.7 | **0.9683** |
| v006 | Pruner + features v5 + CatBoost blend + t_blank | 0.980 | 0.994 | 0.9778 | 0.995 / 0.950 | 11.0 | 0.9652 |
| v006fr | France rows from v005, India/US from v006 | - | - | - | - | mixed | 0.9642 |

Country breakdown (holdout): India / US = v001 0.9225 / 0.9652; v002 0.9358 / 0.9669; v004 0.9498 / 0.9780;
v005 0.9712 / 0.9813; v006 0.9720 / 0.9816. Implied France F0.5 (from LB with India/US at holdout levels,
test mix 47% / 38% / 15%): v001 ~0.87, v002 ~0.91, v004 ~0.91, v005 ~0.93.

**Per-country hybrids isolate where v006 lost points.** v006fr differs from v006 only in its France rows,
and from v005 only in its India/US rows:
- v006fr - v006 = -0.0010 -> v006's France rows are about 0.007 better than v005's.
- v005 - v006fr = +0.0041 -> v005's India/US rows are about 0.005 better than v006's on test, although
  v006 was slightly better on the India/US holdout. This is the competition effect described in Section 3.

## 10. Error analysis

v005 holdout (total loss 1 - F0.5 = 0.0227):

| Error type | Loss |
|---|---|
| Missing only (some true matches not predicted) | 0.0116 |
| False empty (true matches, nothing predicted) | 0.0048 |
| Extra false positive only | 0.0035 |
| Singleton false positive | 0.0017 |
| False positive + missing | 0.0011 |

- **Blank-address candidates** are the weakest segment: recall 0.508 on 61K true pairs (~30K false
  negatives). With no address, near-identical names cannot be confirmed, and some blank look-alikes are
  real distractors.
- **False positives** are mostly distinct businesses sharing an address or name ("Manya Traders" vs
  "Manya Rifle" in the same building; the same house number on a different street) and chain names.
- **Uniqueness rule:** 42% of holdout false positives are records that truly belong to a train-fold S1;
  on test those S1 compete, so the rule can assign such records correctly.
- **Stage 2** adds +0.0016 overall (CI [0.0015, 0.0017]) but can add false positives for 1-match S1
  (254 new vs 24 removed in v005); the fallback rule limits this.
- **Blocking misses** are now small (18,114 pairs, ceiling 0.996): remaining non-Latin names with thin
  addresses, blank-address typos, and 2,899 caused by state blocks.
- **France** cannot be error-analyzed directly (no labels); the confidence profile shows no special
  uncertainty, and the hybrids show v006's France rows beat v005's.

## 11. Final approach

The final submission is chosen by confirmed public LB, with candidate-set size as the second criterion.

1. **Current best:** v005 (LB 0.9683; 36.7 candidates per S1).
2. **v005fr** (built and validated, not yet submitted): India/US rows from v005, France rows from v006;
   expected ~0.969 (v005 + ~0.001 from v006's better France rows).
3. **v010** (planned): v005 code (tag `day2-sub2`) trained on 600K S1 including the holdout (final fit,
   unpruned), with France rows from v006. Expected +0.001 to +0.002 over v005fr. Judged by OOF (no
   holdout possible for a final fit) and the public LB.
4. The best-scoring version is uploaded last (the private leaderboard uses the final submission); the zip's
   `output/` holds exactly that version's two files, re-checked with both validators.

Related work consulted: supervised meta-blocking (Papadakis et al., PVLDB 2014), blocking surveys
(Papadakis et al., PVLDB 2016; Christophides et al., ACM CSUR 2020), domain adaptation for ER (DADER,
PVLDB 2022), LLM-based entity linking (LELA, 2026) — the latter judged infeasible at this data size on
the available hardware.