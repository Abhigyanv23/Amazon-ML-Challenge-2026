# Experiment log

One entry per version. Holdout = frozen 441,364 S1 IDs (`experiments/splits/holdout_s1_ids.txt`).
Never change more than one major component per version without noting it.

| Version | Date | Change | Cand. recall (holdout) | Oracle F0.5 | Holdout F0.5 | Precision / Recall (micro) | Singleton acc. | LB public | Git tag |
|---|---|---|---|---|---|---|---|---|---|
| v000 | 25 Sep | EDA, split, scorer | — | — | 0.0558 (all-empty) | — | 1.000 | — | — |
| v001 | 25 Sep | Normalization v1 + token blocking (k=20) + LightGBM + OOF-tuned decision | 0.948 | 0.980 | 0.9481 | 0.987 / 0.894 | 0.950 | 0.9310 | day1-sub1 |
| v002 | 25 Sep | Normalization v2 (zip, learned state fill) + 3-channel blocking + features v2 | 0.968 | 0.989 | 0.9545 | 0.988 / 0.905 | 0.953 | 0.9436 | day1-sub2 |
| v003 | 26 Sep | Stage-2 group-consistency rescoring on v002 stage-1 probabilities | 0.968 | 0.989 | 0.9580 | 0.988 / 0.920 | 0.963 | not submitted | v003-holdout-0.9580 |
| v004 | 26 Sep | Features v3 (IDF alignment, name specificity, house-number distance, n-gram cosine) + stage-2 fallback; v002 candidates | 0.968 | 0.989 | 0.9667 | 0.992 / 0.929 | 0.965 | 0.9550 | day2-sub1 |
| v005 | 26 Sep | Normalization v3 + blocking v3 (skeleton/address/reverse channels) + features v4 | 0.988 | 0.996 | 0.9773 | 0.994 / 0.951 | 0.970 | 0.9683 | day2-sub2 |
| v006 | 27 Sep | Pruner (cap 20, tau 0.005) + features v5 + lr 0.05, 5 folds, CatBoost blend, t_blank, fallback search | 0.980 | 0.994 | 0.9778 | 0.995 / 0.950 | 0.974 | 0.9652 | day2-sub3 |
| v006fr | 27 Sep | Hybrid: France rows v005 (unpruned), India/US rows v006 (pruned) | — | — | — | — | — | 0.9642 | day2-sub4 |
| v005fr | 27 Sep | Hybrid: India/US rows v005, France rows v006 | — | — | — | — | — | 0.9640 | (untagged) |
| v005w | 27 Sep | features v6: + `name_overlap` (overlap coefficient) on v005's unpruned candidates | ~0.9773 | ~0.996 | not re-measured | — | — | 0.9690 | v005w (branch) |
| v006np | 27 Sep | features v7 (v005w + `unmatched_idf_a/b`, `hn_eq_word_jac`, `hn_conflict`) on v005's unpruned candidates, lr 0.05/5-fold | 0.988 | 0.996 | **0.9784 (best holdout)** | 0.995 / 0.952 | — | **0.9640 (worse than v005w)** | day3-sub1 |
| **v005w+GPT2** | 27 Sep | Stage-2 rescoring/augmentation via a fine-tuned GPT-2 classifier on our own India/US labels, run on `v005w`'s candidates (parallel GPU instance; architecture details pending write-up) | — | — | not measured this way | — | — | **0.9770 (current best)** | (parallel branch, tag pending) |

Public leaderboard leader (26 Sep): 0.988419.

## ⚠️ Central finding, Day 3: holdout F0.5 is no longer a reliable predictor of leaderboard rank

Three independent version pairs now show holdout and leaderboard **disagreeing in direction**,
not just in magnitude:

| Comparison | Holdout says | Leaderboard says |
|---|---|---|
| v006 vs v005 | v006 better (+0.0005) | v006 **worse** (−0.0031) |
| v006np vs v005w | v006np likely better (best holdout of any version, 0.9784) | v006np **much worse** (0.964 vs 0.969) |

The v006-vs-v005 gap was diagnosed (plausibly) as a per-S1 independent-pruning artifact: on test,
all S1 compete for the same pool records, which a partial holdout under-represents. **v006np's
result breaks that explanation** — v006np uses v005's exact unpruned candidates, no pruner
anywhere, and still lost ~0.005 on the leaderboard despite the highest holdout score we've ever
measured. This means added stage-1 features (`unmatched_idf_a/b`, `hn_eq_word_jac`, `hn_conflict`)
and/or the 5-fold/lr=0.05 retraining are **overfitting the train-fold holdout** in a way that does
not generalize to the full test distribution — a second, distinct failure mode from the pruning one.

**Working rule from this point on: only confirmed leaderboard scores are used to select the final
submission. Holdout is used only as a sanity floor (must not regress badly), never to rank two
stage-1/stage-2 candidate versions against each other.** No further stage-1 feature/model variants
(v014 hard-negative mining, v015 union-pruner, v016/v017 French normalization + phonetic channel)
should be trusted on holdout alone; each would need its own leaderboard submission to be usable,
and submission slots are limited this late in the day.

## Submission history (public leaderboard), current

| Tag / branch | Version | Public LB |
|---|---|---|
| day1-sub1 | v001 | 0.9310 |
| day1-sub2 | v002 | 0.9436 |
| day2-sub1 | v004 | 0.9550 |
| day2-sub2 | v005 | 0.9683 |
| day2-sub3 | v006 | 0.9652 |
| day2-sub4 | v006fr | 0.9642 |
| (untagged) | v005fr | 0.9640 |
| v005w (branch) | v005w | 0.9690 |
| day3-sub1 | v006np | 0.9640 |
| **(parallel GPU branch)** | **v005w + GPT-2 stage 2** | **0.9770 — current best** |

(`day2-sub2` was tagged retroactively on commit `ce6d4af`, the last commit before v006's code
changes.)

## v000–v006, v006fr, v005fr — unchanged from prior log entries (see git history for full detail)

Summarized above; per-version detail for v000 through v006/v006fr/v005fr is preserved in the
version history of this file (git log) and is not repeated here to keep this document current and
readable. Key figures carried forward: v005's holdout error analysis (loss 0.0227: missing-only
0.0116, false-empty 0.0048, extra-FP 0.0035, singleton-FP 0.0017, mixed 0.0011; blank-address
candidate recall 0.508); the France diagnostic (stage-1 confidence not lower on France than
India/US at any version checked); the uniqueness-rule confirmation (0.9773 with vs 0.9769 without
the one-S1-per-candidate rule).

## v005w — features v6 (name_overlap)

- File change: `features.py` — added `name_overlap` (min-based overlap coefficient; high for
  acronym/subset name pairs, e.g. "IBM" vs "IBM International Business Machines", without the
  length penalty Jaccard applies), on top of v005's v4 feature set. Trained on v005's unpruned
  candidates, no pruner, no CatBoost/t_blank/fallback-search changes.
- **Submission: public LB 0.9690** (v005 0.9683, +0.0007). Small, real, confirmed gain — the first
  version since v005 to beat it on the leaderboard.

## v006np — features v7, isolating pruning from feature/model gains

- File change: `features.py` v7 = v005w's `name_overlap` + `unmatched_idf_a/b`/`hn_eq_word_jac`
  (previously only tested bundled with the harmful v006 pruner) + new `hn_conflict` (hard binary
  house-number-mismatch flag). Trained on v005's exact unpruned candidates (`--cand-tag v005`),
  lr 0.05, 5 folds, no CatBoost. Stage-2 fallback search: FB_MAX=1 best (OOF 0.9778).
- Holdout: stage-1 0.9769 (v005s1 0.9756, v006s1-pruned 0.9763); final **0.9784 — the best holdout
  of any version tried**. `unmatched_idf_b` reached top-6 feature gain (0.022); `hn_conflict` did
  not appear in the top 15 (may still help on a narrower slice, or may add nothing beyond existing
  `hn_edit`/`hn_comp_eq`).
- **Submission (day3-sub1): public LB 0.9640** — worse than v005w's 0.9690 despite +0.0011 more
  holdout than v005 and no pruner involved. See "Central finding" above.
- Test: 63,534,783 candidate pairs (unchanged from v005, same candidate set); avg matches France
  3.22 / India 3.20 / US 3.29; empty 5.9% / 6.1% / 6.0%; 5,619,997 matched IDs.

## v005w + GPT-2 stage 2 — current best (0.977), built on a parallel GPU instance

- Environment: AWS g5.2xlarge (1× A10G, 23 GB VRAM), separate from laptops A/B. GPT-2 (124M–1.5B
  depending on variant used; MIT-licensed) fine-tuned as a match/no-match classifier on our own
  labelled India/US pairs, used to rescore or augment stage 2 on top of `v005w`'s candidates.
- **Public LB: 0.9770 — the best confirmed score to date**, ahead of v005w alone (0.9690) by
  +0.0080 and ahead of v005 (0.9683) by +0.0087.
- **Architecture, exact training/inference scripts, checkpoint identity, and integration method
  with the LightGBM stage-2 model are not yet written up in this log** — needed before this can be
  correctly described in the official `Documentation_template.md` or reproduced from the
  `code/business_entity_resolution/` package. See open items below.

## Open items before the final submission zip can be assembled correctly

1. **GPT-2 pipeline documentation.** Need: (a) exact model checkpoint/size used, (b) training
   script(s) and how pairs are serialized, (c) how its output combines with/replaces the LightGBM
   stage-2 score (feature? override? blend?), (d) exact inference scope (all candidates, or a
   restricted uncertain band?), (e) `requirements.txt` additions (`torch`, `transformers`, exact
   versions) so the package is reproducible per the rules ("anyone should be able to regenerate
   both output files ... using only what is in this folder").
2. **Confirm license/parameter-count compliance for the exact checkpoint used** — GPT-2 base
   variants are MIT-licensed and well under 8B; confirm no Llama or other non-compliant model was
   substituted anywhere in the pipeline.
3. **Tag/branch the winning commit** so `experiments/<tag>/submission/` has verified saved copies,
   the same discipline applied to every other version.
4. Decide whether any further stage-1 variant (v014/v015/v016/v017) is worth one more submission
   slot, or whether v005w+GPT-2 is locked in as final now, given the "Central finding" above means
   none of them can be trusted without their own leaderboard check.