# Roadmap — Amazon ML Challenge 2026 (Business Entity Resolution)

Last updated: 27 Sep 2026 (Day 3, packaging phase, two-environment consolidation). Priority is
locking in the best confirmed result and building the submission zip, not further feature
engineering — see "Central finding" in `EXPERIMENTS.md`.

## Current position

| Item | Value |
|---|---|
| **Best confirmed overall** | **v005w + GPT-2 stage 3 (parallel system) — public LB 0.9770** |
| Best confirmed, fully-documented LightGBM-only fallback | v005w (parallel system) — 0.9690, or v005 (laptop A, tag `day2-sub2`) — 0.9683, fully reproducible today |
| Leader (public LB) | 0.988419 |
| Gap to leader | 0.0114 |

**Two environments are now tracking submissions independently.** Laptop A's `dayN-subM` tags and
the parallel system's own local tags are NOT the same numbering — join on version name (v005,
v005w, v006np, etc.), not tag number, when comparing results. See `EXPERIMENTS.md`'s submission
tag mapping section for the current (partially reconciled) picture.

## What's done and what's not going further

| Version | Environment | Status | Verdict |
|---|---|---|---|
| v001–v005 | Laptop A | Done | Solid, monotonic improvement |
| v006 (pruner) | Laptop A | Done | Regressed on LB; root-caused to per-S1 pruning under test-time competition |
| v006fr / v005fr (hybrids) | Laptop A | Done | Both worse than v005 |
| v007 / v008 (final-fits on v006 config) | Laptop B / SageMaker | Abandoned | Inherited v006's regression |
| Large-k blocking experiment | Laptop B | Abandoned | Marginal ceiling gain for 6x candidate cost, OOM'd |
| **v006np (+more features, no pruner)** | Laptop A | **Done — regressed despite best-ever holdout** | Confirms holdout unreliability extends beyond pruning changes |
| **v005w (+name_overlap)** | **Parallel system** | **Done — confirmed +0.0007 real gain** | Kept as GPT-2 base |
| **v005w + GPT-2 stage 3** | **Parallel system** | **Done — confirmed best result, 0.9770** | **Current leading candidate for final submission**, pending documentation |
| Second parallel-account submission | Parallel system | **Unknown — score not yet reported** | Need version + score before finalizing |
| v013–v017 (weight search, hard-negative mining, union pruner, French normalization, phonetic channel) | Not run | Deprioritized | Would each need their own LB check given the central finding; code preserved in `src/` if a slot and time remain |

## Immediate priorities (in order)

1. **Get the second parallel-account submission's version and score.** Still open.
2. **Document the GPT-2 stage-3 pipeline.** Needed from the parallel-system teammate: exact
   checkpoint, training script/serialization format, integration method with stage 2, inference
   scope, new `requirements.txt` entries, and an explicit confirmation of no network calls at
   inference time and MIT/Apache-2.0 + ≤8B compliance for the checkpoint used.
3. **Reconcile git tags on laptop A.** `day3-sub1` currently points to v006np's commit (should be
   `day3-sub3`); v005fr's exact commit is not yet located. Low urgency — the score ledger in
   `EXPERIMENTS.md` is correct independent of tag state, so this doesn't block packaging.
4. **Save and verify the winning submission's folder** the same way every prior version was:
   `check_submission.py --out-dir experiments/<version>/submission`, compare counts against what
   was actually uploaded, before treating any score as final.
5. **Decide the final call.** If GPT-2 documentation/reproducibility can be completed and verified
   in time: package v005w+GPT-2 (0.9770). If not, fall back to v005w (0.9690) or v005 (0.9683) —
   both fully documented and reproducible today with zero open questions.
6. **Build the submission zip.** `output/` (final version's two TSVs), `code/business_entity_resolution/`
   (`src/`, `README.md`, `requirements.txt`), `Documentation_template.md` at the zip's top level
   (sibling of `output/`/`code/`, not inside `code/`). Validate the files **inside the built zip**,
   not just the working `output/` folder, before calling it done.

## France (unlabeled, 15% of test)

Unchanged from prior updates: model confidence on France was never lower than on India/US at any
version checked; any France gap has been confident errors, not a coverage gap. Country-hybrid
submissions (v006fr, v005fr) did not produce a net win; not pursued further.

## Compute notes

- Laptop A: 12 CPU cores, 16 GB RAM — ran v001–v006np.
- Laptop B: ran v006-config final-fit attempts (abandoned) and the large-k blocking experiment
  (abandoned).
- SageMaker (Linux, CPU): set up for a v006-config final fit, stopped.
- **Parallel GPU system (AWS g5.2xlarge, 1× A10G, 23 GB VRAM):** ran v005w and the GPT-2 stage-3
  fine-tune that produced the current best result. Known risks flagged when first set up, not yet
  confirmed resolved: root disk only ~1.2 GB free (installs must redirect to `/data`), and `/data`
  is wiped on instance stop (checkpoint must be copied out before any stop) — see
  `gpu_instance_next_model.md`.