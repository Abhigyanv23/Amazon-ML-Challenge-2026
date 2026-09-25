# Roadmap — Amazon ML Challenge 2026 (Business Entity Resolution)

Last updated: 25 Sep 2026, 23:30 (Day 1). Estimated gains are engineering estimates, not measurements;
every version is verified on the frozen holdout before submission.

## Current position

| Item | Value |
|---|---|
| Best version | v002 |
| Holdout F0.5 | 0.9545 (India 0.9358, US 0.9669) |
| Public LB | 0.9436 |
| Leader (public LB) | 0.9859 |
| Submissions used / left | 2 / 10 (assuming the 3 unused Day 1 slots do not carry over) |

### Where the remaining loss is (holdout, 1 − F0.5 = 0.0455)

| Source of loss | Size | Meaning |
|---|---|---|
| Matcher | ≈ 0.035 | True matches among candidates but rejected, or wrong matches accepted |
| Blocking | ≈ 0.011 | True matches never entered the candidate set (mostly India, non-Latin names) |
| France (LB only) | ≈ 0.01 on LB | Implied France F0.5 ≈ 0.91 vs ≈ 0.95 elsewhere; 15% of test, no labels |

## Version plan

| Version | What | Targets | Est. gain | When |
|---|---|---|---|---|
| **v003** | Stage-2 group consistency: rescore each candidate by its agreement with the other confident candidates of the same S1 | Matcher loss; non-Latin and typo records | +0.005 to +0.015 | Day 1 night (running) |
| **v004** | Error analysis → targeted fixes (relative threshold, better tie-breaks between competing S1s, chain-name features) | Precision on chains; singletons | +0.003 to +0.010 | Day 2 morning |
| **v005** | Candidate expansion via S2/S3 duplicate clustering | Blocking ceiling; India non-Latin misses | +0.003 to +0.008 | Day 2 |
| **v006** | Learned non-Latin → Latin character mapping from training pairs | India recall and precision | +0.002 to +0.006 | Day 2 (if time) |
| **v007** | Model capacity: larger training sample, tuned LightGBM, extra pair features | General matcher loss | +0.002 to +0.005 | Day 2 night |
| **v008** | Polish track: small verified gains (see below) | Everything, a little | +0.002 to +0.008 combined | Day 2 night / Day 3 morning |
| **Final** | Best version by holdout; clean full run; reproducibility check; docs; zip | Submission package | — | Day 3 |

### v003 — Stage-2 group consistency (in progress)
- Input: stage-1 (v002) out-of-fold probabilities for train, stage-1 predictions for holdout/test.
- Features per candidate: max / probability-weighted agreement with the top-6 confident candidates
  (p ≥ 0.3) of the same S1 on name, no-space name, address, numbers, zip, state; strong-link count;
  stage-1 probability rank, gap, second-best, group counts.
- Why: true matches of one business resemble each other (e.g. a Devanagari-named record sharing an
  address with confident Latin-named matches).
- Success check: stage-2 OOF > 0.9533 (stage-1 on same rows), holdout > 0.9545.

### v004 — Error analysis → targeted fixes
- Classify holdout errors: blocking miss, matcher false negative, false positive, singleton failure,
  normalization failure — by country and true match count.
- Likely fixes (chosen by the largest error bucket):
  - Relative threshold: keep a candidate if p ≥ t and p ≥ r × (S1's best p).
  - Tie-breaking when two S1s compete for one candidate (currently: highest p wins).
  - Chain-name features (many S1s with the same name in one city → address must decide).
- Can be developed in parallel with v003 using saved holdout probabilities.

### v005 — Candidate expansion by duplicate clustering
- S2/S3 are not deduplicated (up to 5–6 records per source for one S1).
- For each confidently matched candidate, add its near-duplicates from the pool
  (same address and numbers; similar or non-Latin name) as extra candidates, then rescore.
- Raises the recall ceiling beyond blocking. The expanded set is what the final model sees,
  so it is written to `candidate_pairs.tsv` (rule: final matches ⊆ candidates).

### v006 — Non-Latin names (two options; pick one after a quick test)
- Option A: learn a Devanagari/Tamil → Latin character mapping from training pairs (native-script
  S2/S3 names paired with Latin S1 names via train-fold labels only). Computed from provided data.
- Option B: multilingual sentence-embedding model, applied ONLY to non-Latin records and their
  candidates (~1M names), as a name-similarity feature (and optionally a blocking channel).
  Must verify licence (MIT/Apache 2.0) and parameter count (≤ 8B) on the model card before use.
  Runs on the laptop GPU (RTX 3050, 4 GB) — minutes instead of 1–2 h on CPU.
- Either adds name similarity for non-Latin records (currently address-only).

### v007 — Model capacity
- Train on 600K+ train-fold S1 instead of 300K (watch the 16 GB RAM limit).
- Lower learning rate, more rounds; extra features (IDF-weighted name overlap, character n-gram
  cosine per pair).

### v008 — Polish track (every measured gain counts)
Ordered by gain per hour; each item verified on holdout, kept only if it helps.

| Item | Cost | Expected |
|---|---|---|
| Finer threshold grid (coarse 0.05, then 0.01 around the winner) | minutes | +0.000 to +0.001 — **done in `matcher.tune()`, applies from v003** |
| 5-fold instead of 3-fold OOF | +10–15 min training | +0.000 to +0.001 |
| Phonetic token in the word blocking channel | 1 h blocking re-run | +0.000 to +0.002 |
| Lower learning rate (0.03) + more rounds | ~3× training time | +0.001 to +0.003 |
| CatBoost (Apache 2.0) blended with LightGBM on the same folds | ~1 h | +0.001 to +0.003 |
| LightGBM on GPU (`device_type="gpu"`, `max_bin=63`) | setup test only | speed, not score — only if the GPU test passes; compare OOF vs CPU |

Reviewed and **not** adopted (reason):
- Blocking on country + first 4 name characters — breaks on reordering, typos, non-Latin; trigram channel covers typos.
- Postcode-prefix blocking — postcodes present in ~1% of records.
- Source-specific thresholds (S2 vs S3) — each pair already must pass `t`; `is_s3` is a feature.
- Suffix/street regex, numeric column, corpus TF-IDF cosine, numeric-mismatch flag — already implemented.

### Final — Day 3
- Select the final version by holdout F0.5, sanity-checked against the public LB trend
  (the private LB is hidden until after the challenge).
- Full clean run from a fresh `git clone` to prove reproducibility.
- Complete `Documentation_template.md` (error analysis, final approach).
- Build `<team_name>_submission.zip` with the required structure; run both validators on the
  exact files in the zip.

## France (unlabeled, 15% of test)

- Keep features language-agnostic (no country feature).
- For every version compare France prediction statistics (avg matches, empty rate) with India/US.
- Read public-LB movement as the only direct France signal.

## Submission budget

| Day | Left | Plan |
|---|---|---|
| Day 1 (25 Sep) | 3 | Not used — v003 not verified before midnight (assumed not to carry over: limit is per day) |
| Day 2 (26 Sep) | 5 | v003 first (if holdout > 0.9545), then v004–v007; only versions that beat the previous holdout by ≥ 0.002 |
| Day 3 (27 Sep) | 5 | Final + 1–2 safety submissions; finish by afternoon, well before 11:59 PM IST |

## Working rules

1. Holdout first, test second: run test blocking + prediction (~1 h) only for holdout winners.
2. One major change per version, logged in `experiments/EXPERIMENTS.md`.
3. Holdout labels never used for training or tuning; stage 2 trains on out-of-fold probabilities.
4. Before every submission: `check_submission.py` + official validator with `--check-ids`,
   copy to `experiments/<tag>/submission/`, git tag, record LB score.
5. Parallel work across teammates: v004 and v005 can be built on v003 outputs simultaneously.
6. If behind schedule, drop v006/v007/v008 items before cutting final packaging time.

## Compute notes
- Laptop: 12 CPU cores, 16 GB RAM, NVIDIA RTX 3050 Laptop (4 GB). Stay local; SageMaker free tier is weaker.
- CPU-bound (GPU cannot help): blocking (sparse top-k), feature building (rapidfuzz).
- GPU can help: LightGBM training (modest), embeddings for v006 option B (large).