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

**TODO.**