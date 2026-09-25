# Methodology — Business Entity Resolution (Amazon ML Challenge 2026)

Status: living document, updated per version. Items marked **TODO** are pending results.

## 1. Methodology overview

Pipeline: data analysis → normalization → blocking (candidate generation) → pair features →
LightGBM matcher → decision logic tuned for macro F0.5 → output files → official validator.
Only competition-provided data is used.

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
appended words ("Center", "Services", "#35740", ".com" forms), truncated addresses,
state name variants (UP / Uttar Pradesh / Devanagari), house-number variants (3906 / 3906a).
Postcodes are rare (US 11%, India under 1%). Generic chain names repeat (e.g. "primary care group" 253 times in S1).

## 3. Validation design

- Frozen 80/20 split of train S1 IDs, stratified by country × match-count bucket (seed 42):
  1,765,457 train / 441,364 holdout.
- Blocking candidates for holdout S1 are drawn from the full S2+S3 pool, mirroring test.
- Model training and decision tuning use train-fold data only (3-fold group out-of-fold by S1).
  Holdout is scored once per version.
- Local scorer: per-S1 F0.5, macro-averaged. Assumption (engineering, not official): an entity with
  true matches and an empty prediction scores 0; a true singleton scores 1 only if prediction is empty.
  Self-test: perfect = 1.0000, all-empty = 0.0558 (= holdout singleton share).

## 4. Normalization

Rule-based, no learned parameters. Maps for US/Indian states (including Hindi/Tamil names) and
French regions/departments come from general language knowledge, not business data.

- Names: NFKC, Latin accent folding (non-Latin scripts preserved), lowercase, domain unwrapping
  (`konkandata.com` → `konkandata`), `#digits` removal, `&`→`and`, abbreviation expansion
  (pvt→private, ltd→limited, co→company, …). Legal forms moved to `name_legal`; `name_core` is the
  name without them; `name_nospace` removes spaces.
- Addresses: per-country abbreviation maps (street→st, R→rue, …); a comma component that is a
  state/region becomes a canonical code (`addr_state`); house/unit numbers extracted (`addr_nums`).
- Country: open set, never hard-coded.

Evaluation on 200K train-fold true pairs vs random same-country pairs:

| Metric | True pairs | Random pairs |
|---|---|---|
| Name exact (EDA baseline → name_core) | 0.220 → 0.508 | 0.000 |
| Name token Jaccard (baseline → name_core) | 0.617 → 0.699 | 0.003 |
| Address token Jaccard (baseline → addr_norm) | 0.597 → 0.766 | 0.022 |
| ≥1 shared number (both have numbers) | 0.945 | 0.037 |
| State agrees (both have state) | 0.993 | 0.069 |

Normalization raises similarity of true pairs without making random pairs identical.

## 5. Blocking / candidate generation (v001)

Per (country, state) block (pool records without a state join every block of their country):
TF-IDF-style vectors over name_core words, address words and address numbers; tokens with document
frequency above 2% of the block are dropped; cosine similarity; top-k per S1 via sparse top-n
matrix multiplication.

Holdout (441,364 S1, 1,527,218 true pairs):

| k | Candidate recall | Oracle F0.5 ceiling | Total pairs |
|---|---|---|---|
| 5 | 0.838 | 0.953 | 2.2M |
| 10 | 0.928 | 0.973 | 4.4M |
| 20 | 0.948 | 0.980 | 8.8M |
| 30 | 0.956 | 0.983 | 13.2M |
| 50 | 0.962 | 0.985 | 22.1M |

Miss analysis at k=50 (57,559 pairs): non-Latin candidate name 21.1K; overlap only via address
(concatenated/domain names) 13.3K; state mismatch 9.6K (Telangana/Andhra Pradesh relabeling, and a
normalization bug where city "Washington" was read as the state); blank address + name typo 9.0K;
outranked typos 4.5K.

Planned v002: add `name_nospace` token; character-trigram name channel; address-only channel for
non-Latin names; merge AP/TG blocks; take the last state-like component. **TODO: results.**

## 6. Feature engineering

41 features per pair, computed from normalized fields and blocking output (no country feature,
because France is unseen in training):
- Name: exact / no-space exact, Levenshtein ratio, token-set, token-sort and partial ratios,
  Jaro-Winkler and Levenshtein on no-space form, token Jaccard, first-token match, token-count
  difference, length ratio, legal-form Jaccard.
- Address: ratio, token-set ratio, token Jaccard, length ratio; number Jaccard, first-number match,
  share of S1 numbers present; state agreement.
- Blocking/context: blocking score and rank, score relative to the S1's best candidate, candidate
  count, S2/S3 source flag, name frequency in S1 and in the pool (chain indicator), non-Latin and
  blank-address flags, and each candidate's rank and gap to the best within its S1 for key similarities.

## 7. Model

LightGBM binary classifier (MIT licence), learning rate 0.1, 127 leaves, early stopping on
3-fold group-by-S1 out-of-fold validation; final model retrained on all train-fold pairs with
1.1 × mean best iteration. No pretrained models. **TODO: feature importance.**

## 8. Threshold / decision logic

1. Keep pairs with probability ≥ t.
2. Each S2/S3 record goes to at most one S1 (the highest probability); justified by the data
   (no S2/S3 record matches more than one S1).
3. Singleton gate: an S1 keeps its matches only if its best probability ≥ t_top.

t and t_top are grid-searched to maximize macro F0.5 on out-of-fold predictions. **TODO: values.**

## 9. Experiments and results

See `experiments/EXPERIMENTS.md`. **TODO: holdout F0.5, leaderboard scores.**

## 10. Error analysis

**TODO** after v001 holdout: blocking failures, matcher false negatives, false positives,
singleton failures, normalization failures — by country.

## 11. Final approach

**TODO.**# Methodology — Business Entity Resolution (Amazon ML Challenge 2026)

Status: living document, updated per version. Current best submitted: **v002** (holdout F0.5 0.9545,
public LB 0.9436). Items marked **TODO** are pending results.

## 1. Methodology overview

Pipeline: data analysis → normalization (incl. learned state fill) → multi-channel blocking →
pair features → LightGBM matcher (stage 1) → [v003: group-consistency rescoring, stage 2] →
decision logic tuned for macro F0.5 → output files → local + official validators.
Only competition-provided data is used.

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
appended words ("Center", "Services", "#35740", ".com" forms), truncated addresses,
state name variants (UP / Uttar Pradesh / Devanagari), house-number variants (3906 / 3906a).
Postcodes are rare. Generic chain names repeat (e.g. "primary care group" 253 times in S1).

Test-set country diagnostic (unsupervised): France S2/S3 addresses often end in a city with no
region (state found 66–68% before v002); 28.5% of French S1 names occur ≥5 times (India 37.1%, US 20.8%).

## 3. Validation design

- Frozen 80/20 split of train S1 IDs, stratified by country × match-count bucket (seed 42):
  1,765,457 train / 441,364 holdout.
- Blocking candidates for holdout S1 are drawn from the full S2+S3 pool, mirroring test.
- Model training and decision tuning use train-fold data only (3-fold group-by-S1 out-of-fold).
  Holdout is scored once per version. Stage 2 is trained on out-of-fold stage-1 probabilities.
- Local scorer: per-S1 F0.5, macro-averaged. Assumption (engineering, not official): an entity with
  true matches and an empty prediction scores 0; a true singleton scores 1 only if prediction is empty.
  Self-test: perfect = 1.0000, all-empty = 0.0558 (= holdout singleton share).
- Holdout vs OOF agreement (no overfitting): v001 0.9481 vs 0.9467; v002 0.9545 vs 0.9533.
- Limitation: France has no labels; its quality is inferred from public-LB movement and
  prediction statistics (match counts, empty rates) compared with India/US.

## 4. Normalization

Rule-based. Maps for US/Indian states (including Hindi/Tamil names) and French regions/departments
come from general language knowledge, not business data.

- Names: NFKC, Latin accent folding (non-Latin scripts preserved), lowercase, domain unwrapping
  (`konkandata.com` → `konkandata`), `#digits` removal, `&`→`and`, abbreviation expansion
  (pvt→private, ltd→limited, co→company, …). Legal forms moved to `name_legal`; `name_core` is the
  name without them; `name_nospace` removes spaces.
- Addresses: per-country abbreviation maps (street→st, R→rue, q→quai, …); a comma component that is a
  state/region becomes a canonical code (`addr_state`; v2: the **last** such component wins);
  house/unit numbers extracted (`addr_nums`); v2: postcodes separated into `addr_zip`
  (US 5-digit only as last token of a component, France 5-digit, India 6-digit PIN).
- v2 learned state fill: from the same split's S1 records (all have a state), count address
  components (city, locality) by state; keep components seen ≥20 times with ≥98% purity; fill empty
  states in S2/S3 by majority vote (`addr_state_src` = 2). Uses unlabeled text only.
  State found in S2/S3: France 67% → 97%, India 91% → 96.5%.
- Country: open set, never hard-coded.

Evaluation on 200K train-fold true pairs vs random same-country pairs:

| Metric | True pairs | Random pairs |
|---|---|---|
| Name exact (EDA baseline → name_core) | 0.220 → 0.508 | 0.000 |
| Name token Jaccard (baseline → name_core) | 0.617 → 0.699 | 0.003 |
| Address token Jaccard (baseline → addr_norm) | 0.597 → 0.765 | 0.022 |
| ≥1 shared number (both have numbers) | 0.945 | 0.037 |
| State agrees (both have state; v1 → v2) | 0.993 → 0.994 | 0.068 |

Normalization raises similarity of true pairs without making random pairs identical.

## 5. Blocking / candidate generation

### v001: single word channel
Per (country, state) block (pool records without a state join every block of their country):
IDF-weighted vectors over name_core words, address words and numbers; tokens with document frequency
above 2% of the block dropped; cosine similarity; top-k per S1 via sparse top-n matrix multiplication.

| k | Candidate recall | Oracle F0.5 ceiling | Total pairs (holdout) |
|---|---|---|---|
| 5 | 0.838 | 0.953 | 2.2M |
| 10 | 0.928 | 0.973 | 4.4M |
| 20 | 0.948 | 0.980 | 8.8M |
| 50 | 0.962 | 0.985 | 22.1M |

Miss analysis at k=50 (57,559 pairs): non-Latin candidate name 21.1K; overlap only via address
(concatenated/domain names) 13.3K; state mismatch 9.6K (Telangana/Andhra Pradesh relabeling and the
"Washington, NY" state bug); blank address + name typo 9.0K; outranked typos 4.5K.

### v002: three channels, unioned
- Word channel (top 20): as v001 plus a `name_nospace` token.
- Trigram channel (top 10): character trigrams of `name_nospace` (typos, concatenations).
- Non-Latin channel (top 5): address-only similarity against pool records with non-Latin names.
- Blocks merged where sources mix labels (India: AP+TG, JK+LA); smoothed IDF.

| Holdout | v001 (k=20) | v002 |
|---|---|---|
| Candidate recall | 0.948 | 0.968 |
| Oracle F0.5 ceiling | 0.980 | 0.989 |
| Recall India / US | — | 0.948 / 0.981 |
| True pairs found only by trigram / non-Latin channel | — | 7,058 / 6,733 |
| Misses due to state blocks | 9,622 | 1,246 |
| Avg candidates per S1 | 20 | 25.6 |

Every channel's score and rank are passed to the matcher as features.

## 6. Feature engineering

v002: 46 features per pair (v001: 39), from normalized fields and blocking output. No country
feature, because France is unseen in training.
- Name (13): exact / no-space exact, Levenshtein ratio, token-set, token-sort and partial ratios,
  Jaro-Winkler and Levenshtein on no-space form, token Jaccard, first-token match, token-count
  difference, length ratio, legal-form Jaccard.
- Address (9): ratio, token-set ratio, token Jaccard, length ratio; number Jaccard, first-number
  match, share of S1 numbers present; state agreement; zip agreement (v2).
- Blocking (6): word score/rank, trigram score/rank, non-Latin score/rank.
- Context (18): score relative to the S1's best candidate, candidate count, S2/S3 flag, name
  frequency in S1 and in the pool (chain indicator), non-Latin / blank-address / inferred-state flags,
  count of near-identical names among the S1's candidates, a combined name+address similarity, and
  each candidate's rank and gap to the best within its S1 for four key similarities.

## 7. Model

Stage 1: LightGBM binary classifier (MIT licence), learning rate 0.1, 127 leaves, min 100 rows per
leaf, feature/bagging fraction 0.9/0.8; early stopping on 3-fold group-by-S1 validation; final model
retrained on all train-fold pairs with 1.1 × mean best iteration (v002: best iterations 1277/1217/1134).
Training data: 300K train-fold S1, 7.68M candidate pairs (13.1% positive). No pretrained models.

Top features by gain (v002): combined name+address similarity 0.49, its gap to the S1's best 0.08,
number Jaccard 0.08, address token-set 0.05, word-channel rank 0.04 and score 0.04, pool name frequency 0.02.

Stage 2 (v003, in progress): for each candidate, agreement with the other confident candidates of
the same S1 (top 6 with stage-1 p ≥ 0.3): name, no-space name, address, numbers, zip, state
(max, probability-weighted mean, strong-link count), plus stage-1 probability rank, gap and group
counts. A second LightGBM is trained on out-of-fold stage-1 probabilities. **TODO: results.**

## 8. Threshold / decision logic

1. Keep pairs with probability ≥ t.
2. Each S2/S3 record goes to at most one S1 (the highest probability); justified by the data
   (no S2/S3 record matches more than one S1).
3. Singleton gate: an S1 keeps its matches only if its best probability ≥ t_top.

t and t_top are grid-searched (0.20–0.95, step 0.05) to maximize macro F0.5 on out-of-fold
predictions. Selected: v001 t = t_top = 0.65; v002 t = t_top = 0.70. Neighbouring settings
score within 0.0005, so the choice is stable.

## 9. Experiments and results

| Version | Candidate recall | Oracle ceiling | OOF F0.5 | Holdout F0.5 | India / US (holdout) | Public LB |
|---|---|---|---|---|---|---|
| v001 | 0.948 | 0.980 | 0.9467 | 0.9481 | 0.9225 / 0.9652 | 0.9310 |
| v002 | 0.968 | 0.989 | 0.9533 | 0.9545 | 0.9358 / 0.9669 | 0.9436 |
| v003 | 0.968 | 0.989 | TODO | TODO | TODO | TODO |

Holdout v002: precision 0.988, recall 0.905 (micro), singleton accuracy 0.953.
Public LB rose more than holdout (+0.0126 vs +0.0064); with India/US at holdout levels, implied
France F0.5 rose from about 0.87 to about 0.91 — consistent with the learned state fill fixing
French blocking. Full log: `experiments/EXPERIMENTS.md`.

## 10. Error analysis

Loss decomposition on holdout (1 − F0.5):
- v001: blocking 0.020 (1 − ceiling), matcher 0.032.
- v002: blocking 0.011, matcher 0.035 → the matcher is now the main bottleneck (motivates stage 2).
- India trails US in both candidate recall (0.948 vs 0.981) and F0.5 (0.936 vs 0.967); remaining
  India misses are mainly non-Latin names with sparse addresses.
**TODO:** false positives vs false negatives by country and match count for v003.

## 11. Final approach

**TODO** (to be fixed on Day 3 based on holdout F0.5, checked against public LB).