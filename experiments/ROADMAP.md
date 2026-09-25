# Roadmap — Amazon ML Challenge 2026 (Business Entity Resolution)

Last updated: 26 Sep 2026 (Day 2). Estimated gains are engineering estimates, not measurements;
every version is verified on the frozen holdout before submission.

## Current position

| Item | Value |
|---|---|
| Best submitted | v002 — public LB 0.9436 (holdout 0.9545) |
| Best evaluated | v003 — holdout 0.9580 (not submitted; held to combine with v004) |
| Leader (public LB) | 0.9859 |
| Submissions used / left | 2 / 10 (assuming unused Day 1 slots do not carry over) |

### Where the remaining loss is (v003 holdout, 1 − F0.5 = 0.0420)

| Source of loss | Size | Meaning |
|---|---|---|
| Matcher / decision | ≈ 0.031 | Ceiling 0.9886 − 0.9580: candidates wrongly rejected or accepted |
| Blocking | ≈ 0.011 | True matches never reach the model (49,259 pairs; India recall 0.948 vs US 0.981) |
| France (LB only) | ≈ 0.006 on LB | Implied France F0.5 ≈ 0.91 at v002; 15% of test, no labels |
| Known regression | — | v003 lowered F0.5 for 1-match S1 (0.878 → 0.857): no other candidates to give support |

## Version plan

| Version | What | Targets | Est. gain | Status / when |
|---|---|---|---|---|
| v001 | Baseline: normalization, word blocking, LightGBM | — | — | Done (LB 0.9310) |
| v002 | Normalization v2, 3-channel blocking, features v2 | Blocking, France | — | Done (LB 0.9436) |
| v003 | Stage-2 group-consistency rescoring | Matcher recall | — | Done (holdout 0.9580) |
| **v004** | Error analysis → decision fixes + new pair features | Matcher loss, 1-match regression | +0.004 to +0.012 | Day 2 morning |
| **v005** | Blocking v3: candidate expansion + fixes from v002 misses | Blocking ceiling, India | +0.003 to +0.008 | Day 2 |
| **v006** | Non-Latin names: learned transliteration or multilingual embeddings | India recall/precision | +0.002 to +0.006 | Day 2 (if time) |
| **v007** | Model capacity + hard-negative mining | General matcher loss | +0.002 to +0.005 | Day 2 night |
| **v008** | Polish track: small verified gains | Everything, a little | +0.002 to +0.008 combined | Day 2 night / Day 3 morning |
| **Final** | Best version; clean run; reproducibility; docs; zip | Package | — | Day 3 |

### v004 — Error analysis → targeted fixes + new features
Error analysis (holdout; labels used for analysis only):
- Classify errors: blocking miss, matcher false negative, false positive, singleton failure —
  by country, true match count, non-Latin flag, blank address, candidate rank.
- **Stage 1 → Stage 2 decomposition:** true positives recovered, false positives introduced,
  false negatives removed / newly introduced, by the same slices. Explains the 1-match regression.
- **Bootstrap confidence interval** of holdout F0.5 (resampling S1 entities) — confirms version
  differences are beyond noise without extra retraining.
- **Uniqueness A/B** on saved probabilities: (A) no one-S1-per-candidate rule, (B) current rule,
  (C) rule only when the winning p is high. Ground truth shows 0 S2/S3 records matched to >1 S1,
  so (B) is expected to hold; measured for documentation.

Fixes chosen by the largest error bucket, likely:
- 1-match regression: pass Stage-1 probability through when Stage 2 has no support
  (k_size = 0), or a separate threshold for S1s with a single confident candidate.
- Relative threshold: keep a candidate if p ≥ t and p ≥ r × (S1's best p).

New pair features (stage 1):
- **Token alignment:** share of S1 name tokens matched, share of candidate tokens matched,
  IDF-weighted count of rare matched tokens, unmatched-token count.
- **Pair-level character n-gram cosine** (2- and 3-grams) for name and address on every pair
  (today only the trigram channel's top 10 carry a trigram score).
- **Address component agreement:** house number, street name, city compared separately
  (instead of one fuzzy address score).

### v005 — Blocking v3
- Input: `experiments/v002/blocking_misses.txt` (category counts for the 49,259 remaining misses).
- **Candidate expansion via S2/S3 duplicate clustering:** for each confidently matched candidate,
  add its near-duplicates in the pool (same address + numbers; similar or non-Latin name), then
  rescore. The expanded set is written to `candidate_pairs.tsv` (rule: matches ⊆ candidates).
- Small **relaxed-state channel** (same country, outside the S1's state block, top 5) — only
  ~1.2K state-block misses remain, so low priority.
- Tune per-channel k (word / trigram / non-Latin) against recall and candidate count.

### v006 — Non-Latin names (pick one after a quick test)
- Option A: learn a Devanagari/Tamil → Latin character mapping from training pairs (train-fold labels only).
- Option B: multilingual sentence embeddings only for non-Latin records and their candidates
  (~1M names); runs on the laptop GPU. Verify licence (MIT/Apache 2.0) and size (≤ 8B) on the model card.
- Note: the non-Latin names are genuine Devanagari/Tamil text, not encoding corruption
  (the garbled characters seen earlier came from PowerShell output encoding only).

### v007 — Model capacity + hard-negative mining
- Train on 600K+ train-fold S1 instead of 300K (watch 16 GB RAM).
- **Hard-negative mining:** after a first model, add back S1 groups containing high-scoring
  negatives (same common name + different address, same address + different business,
  high blocking score + negative label) and hard positives; retrain.
- Lower learning rate, more rounds.

### v008 — Polish track (every measured gain counts)

| Item | Cost | Expected |
|---|---|---|
| Finer threshold grid (0.05 then 0.01) | minutes | +0.000 to +0.001 — **done in `matcher.tune()`; applies from the next retrain** (v003 used the coarse grid) |
| 5-fold instead of 3-fold OOF | +10–15 min training | +0.000 to +0.001 |
| Phonetic token in the word blocking channel | 1 h blocking re-run | +0.000 to +0.002 |
| Lower learning rate (0.03) + more rounds | ~3× training time | +0.001 to +0.003 |
| CatBoost (Apache 2.0) blended with LightGBM on the same folds | ~1 h | +0.001 to +0.003 |
| LightGBM on GPU (`device_type="gpu"`, `max_bin=63`) | setup test | speed only; compare OOF vs CPU |

### Stretch (parked)
- **Transformer cascade (Ditto-style)** on uncertain pairs only (stage-2 p ≈ 0.2–0.9), with a
  multilingual checkpoint via Hugging Face `transformers`. Parked: v004/v005 have higher expected
  gain per hour. If revived: check SageMaker GPU quotas first; prototype locally on a sample.

### Reviewed and not adopted (reason)
- Blocking on country + first 4 name characters — breaks on reordering, typos, non-Latin; trigram channel covers typos.
- Postcode-prefix blocking — postcodes present in ~1% of records.
- Source-specific thresholds (S2 vs S3) — each pair already must pass `t`; `is_s3` is a feature.
- Suffix/street regex, numeric column, corpus TF-IDF cosine, numeric-mismatch flag — already implemented.
- Multi-pass / character n-gram blocking, relaxed state partition — already implemented in v002.
- Repairing "mojibake" names — not present in the data (display artifact).
- Multiple independent validation splits — replaced by a bootstrap CI on the 441K-entity holdout;
  holdout ≈ OOF in every version, so no evidence of overfitting.
- Splink — unsupervised Fellegi-Sunter; we have 7.6M labelled pairs, so supervised LightGBM is stronger; integration = rewrite.
- DeepMatcher — older and weaker than Ditto.

## France (unlabeled, 15% of test)
- Keep features language-agnostic (no country feature).
- For every version compare France prediction statistics (avg matches, empty rate) with India/US.
- Public-LB movement is the only direct France signal.

## Submission budget

| Day | Left | Plan |
|---|---|---|
| Day 1 (25 Sep) | 3 | Not used (assumed not to carry over) |
| Day 2 (26 Sep) | 5 | v004 (includes v003) if holdout > 0.9580 + 0.002; then v005–v007 winners |
| Day 3 (27 Sep) | 5 | Final + 1–2 safety submissions; finish by afternoon, well before 11:59 PM IST |

## Working rules
1. Holdout first, test second: run test blocking + prediction (~1 h) only for holdout winners.
2. One major change per version, logged in `experiments/EXPERIMENTS.md`.
3. Holdout labels never used for training or tuning; stage 2 trains on out-of-fold probabilities.
4. Before every submission: `check_submission.py` + official validator with `--check-ids`,
   copy to `experiments/<tag>/submission/`, git tag `dayN-subM`, record LB score.
   Evaluated-but-not-submitted versions get tags like `v003-holdout-0.9580`.
5. Parallel work across teammates: v004 and v005 can be built on v003 outputs simultaneously.
6. If behind schedule, drop v006/v007/v008 items before cutting final packaging time.
7. No network calls in the pipeline at run time (model/library downloads are one-time setup only).

## Compute notes
- Laptop: 12 CPU cores, 16 GB RAM, NVIDIA RTX 3050 Laptop (4 GB). Stay local; SageMaker free tier is weaker.
- CPU-bound (GPU cannot help): blocking (sparse top-k), feature building (rapidfuzz).
- GPU can help: LightGBM training (modest), embeddings / transformer options (large).