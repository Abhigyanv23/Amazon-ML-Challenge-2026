# Experiment log

One entry per version. Holdout = frozen 441,364 S1 IDs (`experiments/splits/holdout_s1_ids.txt`).
Never change more than one major component per version without noting it.

**Two environments are now submitting independently:** laptop A (this repo's `main`) and a
parallel system (its own local tracking; teammate updates its docs separately). Version names are
the join key between the two — **use version name, not `dayN-subM` tag number**, to compare
results, since each environment numbers its own submissions locally and the same tag name can
point to different things in each place.

| Version | Environment | Date | Change | Holdout F0.5 | LB public |
|---|---|---|---|---|---|
| v000 | Laptop A | 25 Sep | EDA, split, scorer | 0.0558 (all-empty) | — |
| v001 | Laptop A | 25 Sep | Normalization v1 + token blocking (k=20) + LightGBM | 0.9481 | 0.9310 |
| v002 | Laptop A | 25 Sep | Normalization v2 (zip, state fill) + 3-channel blocking + features v2 | 0.9545 | 0.9436 |
| v003 | Laptop A | 26 Sep | Stage-2 group-consistency rescoring | 0.9580 | not submitted |
| v004 | Laptop A | 26 Sep | Features v3 + stage-2 fallback | 0.9667 | 0.9550 |
| v005 | Laptop A | 26 Sep | Normalization v3 + blocking v3 (6 channels) + features v4 | 0.9773 | 0.9683 |
| v006 | Laptop A | 27 Sep | Pruner + features v5 + CatBoost blend + t_blank + fallback search | 0.9778 | 0.9652 (regressed) |
| v006fr | Laptop A | 27 Sep | Hybrid: France rows v005, India/US rows v006 | — | 0.9642 |
| v005fr | Laptop A | 27 Sep | Hybrid: India/US rows v005, France rows v006 | — | **0.9640** |
| v006np | Laptop A | 27 Sep | Features v7 (v6 + unmatched_idf_a/b, hn_eq_word_jac, hn_conflict) on v005's unpruned candidates | **0.9784 (best holdout of any version)** | **0.9640 (regressed)** |
| v005w | **Parallel system** | 27 Sep | Features v6 (+ name_overlap) on v005's unpruned candidates | ~0.9773 (not independently re-measured) | **0.9690** |
| v005w + GPT-2 stage 3 | **Parallel system** | 27 Sep | [CONFIRM: exact GPT-2 checkpoint/architecture] rescoring on top of v005w | [CONFIRM] | **0.9770 (current best overall)** |

Public leaderboard leader (26 Sep): 0.988419.

## Submission tag mapping (each environment numbers its own `dayN-subM` locally — do not assume the same tag means the same thing across environments)

### Laptop A (this repo, `main`)
| Tag | Version | Public LB |
|---|---|---|
| day1-sub1 | v001 | 0.9310 |
| day1-sub2 | v002 | 0.9436 |
| day2-sub1 | v004 | 0.9550 |
| day2-sub2 | v005 | 0.9683 |
| day2-sub3 | v006 | 0.9652 |
| day2-sub4 | v006fr | 0.9642 |
| day3-sub2 | v005fr | 0.9640 |
| day3-sub3 | v006np | 0.9640 |

Note: `day3-sub1` in this repo's git tags currently points to the v006np commit (a labeling slip
during tagging on 27 Sep) — v006np's actual submission slot was `day3-sub3`. Fix pending:
```
git tag day3-sub3 7b84568   # if not already correctly tagged
git tag -d day3-sub1        # remove the mislabeled tag, or repoint it once v005fr's real hash is found
```

### Parallel system (own local tracking — teammate to confirm/update independently)
| Local tag | Version | Public LB |
|---|---|---|
| (their) day3-sub1 | v005w | 0.9690 |
| (unlogged) | v005w + GPT-2 stage 3 | 0.9770 |
| **[CONFIRM: second parallel-account submission — still not reported to this log]** | ? | ? |

## ⚠️ Central finding, Day 3: holdout F0.5 is no longer a reliable predictor of leaderboard rank

Two independent laptop-A version pairs show holdout and leaderboard disagreeing in **direction**:

| Comparison | Holdout says | Leaderboard says |
|---|---|---|
| v006 vs v005 | v006 better (+0.0005) | v006 **worse** (−0.0031) |
| v006np vs v005w | v006np likely better (best holdout of any version, 0.9784) | v006np **much worse** (0.9640 vs 0.9690) |

The v006-vs-v005 gap was diagnosed as a per-S1 independent-pruning artifact (test-time cross-S1
competition a partial holdout under-represents). **v006np's result breaks that specific
explanation** — no pruner is involved at all, same unpruned v005 candidate set as v005w — yet it
still lost on the leaderboard despite the best holdout score of the entire project. This points to
the added stage-1 features and/or the 5-fold/lr=0.05 retraining **overfitting the holdout** in a
way that doesn't generalize, a second and distinct failure mode from the pruning one.

**Working rule: only confirmed leaderboard scores are used to select the final submission.**
Holdout is a sanity floor only, never used to rank two stage-1/stage-2 variants against each other.

## Per-version detail (laptop A)

Preserved from prior versions of this file (git history) for v000 through v006/v006fr; summarized
figures: v005's holdout error analysis (loss 0.0227: missing-only 0.0116, false-empty 0.0048,
extra-FP 0.0035, singleton-FP 0.0017, mixed 0.0011; blank-address candidate recall 0.508); the
uniqueness-rule confirmation (0.9773 with vs 0.9769 without the one-S1-per-candidate rule); the
France confidence-profile diagnostic (never more uncertain on France than India/US at any version
checked).

**v005fr:** India/US rows from v005 (unpruned), France rows from v006 (pruned). Built via
`combine_by_country.py`. LB 0.9640 — worse than pure v005 (0.9683), consistent with the central
finding: even though v006's France rows measured slightly better in isolation, taking any rows
from a pipeline whose stage-1/stage-2 changes don't generalize costs more than it gains.

**v006np:** Isolates feature/model gain from the v006 pruner by training on v005's exact unpruned
candidates. Stage-1 holdout 0.9769 (v005s1 0.9756); final holdout 0.9784, the best of any version.
`unmatched_idf_b` reached top-6 feature gain. Submission LB 0.9640 — see central finding above.

## Per-version detail (parallel system) — [CONFIRM/EXPAND: teammate to complete independently]

**v005w:** features v6 = v005's v4 feature set + `name_overlap` (min-based overlap coefficient;
high for acronym/subset name pairs like "IBM" vs "IBM International Business Machines"). Trained
on v005's unpruned candidates, no pruner, no CatBoost/t_blank/fallback-search changes. LB 0.9690
(v005 0.9683, +0.0007) — the first version since v005 to beat it, and (so far) the only
LightGBM-only variant that did.

**v005w + GPT-2 stage 3:** [CONFIRM: checkpoint variant and parameter count; license confirmation;
training data scope (own India/US labels only?); serialization format for pairs; how its output
combines with the LightGBM stage-2 score — replace / blend / feed as feature; inference scope — all
candidates or a restricted uncertain band; new `requirements.txt` entries needed (`torch`,
`transformers`, exact versions); confirmation of no network calls at inference time.] LB 0.9770 —
current best result across both environments.

## Open items before the final submission zip can be assembled

1. **GPT-2 pipeline documentation** (see placeholders above) — blocks `Documentation_template.md`
   Sections 1/2.2/4/6/Appendix A and the `code/business_entity_resolution/` package's ability to
   actually reproduce the 0.9770 result.
2. **License/parameter-count compliance confirmation** for the exact GPT-2 checkpoint used.
3. **Second parallel-account submission** — version and score not yet reported to this log.
4. **Git tag cleanup on laptop A** — `day3-sub1` currently mislabeled (see mapping table above);
   `v005fr`'s exact commit hash not yet located (`git log --all --grep="v005fr"` pending).
5. **Final decision:** lock in v005w+GPT-2 (0.9770) as the submission, pending items 1–3 above, or
   fall back to a fully-documented LightGBM-only version (v005w at 0.9690, or v005 at 0.9683) if
   the GPT-2 pipeline can't be fully verified/reproduced in the code package before the deadline.