# Roadmap — Amazon ML Challenge 2026 (Business Entity Resolution)

Last updated: 27 Sep 2026 (Day 3, packaging phase). **We are now in final-assembly mode**: the
priority is locking in the best confirmed result and building the submission zip, not further
feature engineering. See "Central finding" in `EXPERIMENTS.md` for why holdout-only validation is
no longer trusted for new stage-1/stage-2 changes.

## Current position

| Item | Value |
|---|---|
| **Best confirmed (use this as the final, pending open items below)** | **v005w + GPT-2 stage 2 — public LB 0.9770** |
| Best confirmed, LightGBM-only pipeline | v005w — public LB 0.9690 |
| Leader (public LB) | 0.988419 |
| Gap to leader | 0.0114 |

### Submission history (public leaderboard)

| Version | Public LB |
|---|---|
| v001 | 0.9310 |
| v002 | 0.9436 |
| v004 | 0.9550 |
| v005 | 0.9683 |
| v006 (pruned) | 0.9652 |
| v006fr (hybrid) | 0.9642 |
| v005fr (hybrid) | 0.9640 |
| v005w (+name_overlap) | 0.9690 |
| v006np (+more features, no pruner) | 0.9640 |
| **v005w + GPT-2 stage 2** | **0.9770 — current best** |

## Central finding (moved from EXPERIMENTS.md, repeated here since it governs everything below)

Holdout F0.5 no longer reliably predicts leaderboard rank for stage-1/stage-2 changes, not just
for pruning changes as first diagnosed at v006. v006np (no pruner, v005's exact unpruned
candidates, best holdout ever at 0.9784) still lost to v005w on the leaderboard (0.964 vs 0.969).
**No further stage-1/stage-2 variant should be trusted without its own leaderboard confirmation.**
This is why the roadmap below shifts from "build more versions" to "package what's confirmed."

## What's done and what's not going further

| Version | Status | Verdict |
|---|---|---|
| v001–v005 | Done | Solid, monotonic improvement; v005 was best for a full day |
| v006 (pruner) | Done | Regressed on LB; root-caused to per-S1 pruning under test-time competition |
| v006fr / v005fr (hybrids) | Done | Both worse than v005; the hybrid-isolation approach itself is sound methodology but didn't find a net win here |
| v007 / v008 (final-fits on v006 config) | Abandoned | Inherited v006's regression before it was understood; correctly stopped |
| Large-k blocking experiment | Abandoned | Marginal ceiling gain for a 6x candidate-set cost, OOM'd on test |
| **v005w (+name_overlap)** | **Done — confirmed +0.0007 real gain** | Kept; base for GPT-2 layer |
| **v006np (+more features)** | **Done — confirmed regression despite best-ever holdout** | Not used; proves holdout is unreliable for this class of change |
| **v005w + GPT-2 stage 2** | **Done — confirmed best result, 0.9770** | **Current leading candidate for final submission** |
| v013 (cost-sensitive weight search) | Not built | Deprioritized — even if positive, expected gain (+0.0005–0.002) is smaller than the risk of another holdout-only-validated change going the wrong way on LB |
| v014 (hard-negative oversampling) | Not built | Same reasoning — deprioritized given the central finding |
| v015 (union-rule pruner v2) | Not built | Deprioritized — would need its own LB check to trust, and v005w+GPT-2 already beats every pruned version tried |
| v016/v017 (French normalization + `blk_ph` phonetic channel) | Code written, never run | Deprioritized for the same reason; code is preserved in `src/` for future use if a submission slot and time remain |
| v011 (GraLMatch group-coherence) | Not built | Deprioritized |
| v012 (AnyMatch GPT-2, original spec) | **Superseded** | This is effectively what the parallel GPU instance built and confirmed at 0.977 — the idea worked |

## Immediate priorities (in order)

1. **Document the GPT-2 stage-2 pipeline properly.** See the "Open items" list in
   `EXPERIMENTS.md` — exact checkpoint, training script, serialization format, how its output
   combines with the LightGBM stage-2 score, inference scope, and new `requirements.txt` entries
   (`torch`, `transformers`, versions). This is required both for the official
   `Documentation_template.md` and for `code/business_entity_resolution/` to actually reproduce
   the submitted output, which the rules say gets audited.
2. **Verify license/parameter-count compliance** for the exact GPT-2 checkpoint used (confirm it's
   a standard MIT-licensed GPT-2 variant, not a substituted non-compliant model).
3. **Save and tag the winning submission** the same way every other version was: copy
   `output/*.tsv` to `experiments/<tag>/submission/`, re-verify counts with
   `check_submission.py --out-dir`, tag the commit.
4. **Decide the final call:** is v005w+GPT-2 (0.977) locked in as the submission going up last, or
   is there a specific, already-in-flight change worth one more submission slot? Given the central
   finding, the bar for "worth another slot" should be high — only submit something with either
   (a) a mechanism clearly different from the ones that already failed (pruning, added stage-1
   features), or (b) enough time left to also fall back cleanly to 0.977 if it doesn't pan out.
5. **Build the submission zip.** Structure: `output/` (the two TSVs from whichever version is
   final), `code/business_entity_resolution/` (`src/`, `README.md`, `requirements.txt`),
   `Documentation_template.md` at the zip's top level (not inside `code/`). Run
   `check_submission.py --out-dir` and the official validator on the files **inside the built zip**
   before calling it done, not just on the working `output/` folder.

## France (unlabeled, 15% of test)

- Confidence-profile diagnostic (stage-1 p) showed the model was never more uncertain on France
  than India/US at any version checked — any France gap has been confident errors, not a coverage
  gap.
- Country-hybrid submissions (v006fr, v005fr) were meant to isolate France's contribution cleanly;
  read together they were internally inconsistent (see prior version of this file / git history
  for the full weighted-average reconciliation attempt), most likely because the public leaderboard
  scores only a subset of test, so hybrid submissions built at different times don't necessarily
  share an identical sample. Not pursued further given the central finding above shifted priority
  to packaging.

## Compute notes

- Laptop A: 12 CPU cores, 16 GB RAM — ran v001–v006np.
- Laptop B: available, ran the v006-config final-fit attempts (abandoned) and the large-k blocking
  experiment (abandoned).
- SageMaker (Linux, CPU): set up for a v006-config final fit, stopped — superseded by the GPU
  instance's GPT-2 work.
- **Parallel GPU instance (AWS g5.2xlarge, 1× A10G 23 GB VRAM):** ran the GPT-2 stage-2 fine-tune
  that produced the current best result (0.977). Known risks on this box, not yet fully mitigated:
  root disk only ~1.2 GB free (installs must be redirected to `/data`), and `/data` is wiped on
  instance stop (checkpoint must be copied out before any stop). See
  `gpu_instance_next_model.md` for the full note written when this was first flagged.