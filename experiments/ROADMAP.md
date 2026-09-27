# Roadmap — Amazon ML Challenge 2026 (Business Entity Resolution)

Last updated: 27 Sep 2026 (Day 3). Estimated gains are engineering estimates, not measurements;
every version is verified on the frozen holdout before submission, but from v006 onward the
holdout no longer reliably predicts leaderboard direction (see "Known limitation" below) — treat
holdout as necessary, not sufficient.

## Current position

| Item | Value |
|---|---|
| **Best submitted (use this as the fallback final)** | **v005 — public LB 0.9683 (holdout 0.9773)** |
| Leader (public LB, 26 Sep) | 0.988419 |
| Submissions used | 6 (day1-sub1/2, day2-sub1/2/3/4) + v005fr (this update) |

### Submission history (public leaderboard)

| Tag | Version | Public LB | Candidates/S1 (test) |
|---|---|---|---|
| day1-sub1 | v001 | 0.9310 | 20.0 |
| day1-sub2 | v002 | 0.9436 | 25.6 |
| day2-sub1 | v004 | 0.9550 | 25.6 |
| **day2-sub2** | **v005** | **0.9683 (best)** | 36.7 (max 6,243) |
| day2-sub3 | v006 | 0.9652 | 11.0 (max 20) |
| day2-sub4 | v006fr (India/US v006, France v005) | 0.9642 | mixed |
| (unlogged) | v005fr (India/US v005, France v006) | **0.9640** | mixed |

### Open question: the four hybrid scores are inconsistent with a simple weighted-average model

Treating LB ≈ w·S(India/US rows) + (1−w)·S(France rows) for some fixed w, the four submissions
(v005, v006, v006fr, v005fr) over-determine the four unknowns and **do not solve consistently**:
comparing (v005 − v006fr) implies France-v005 beats France-v006 by ~0.029; comparing
(v006 − v005fr) implies the opposite (France-v006 better by ~0.007) — a sign flip, not just a
different magnitude. Likely causes, in order of suspicion given our history of copy/build
mistakes on this exact step (Section "known incidents" below): (a) the public leaderboard scores
only a **subset** of test, so the four submissions may not share an identical France/India/US
row sample; (b) a build error in `v005fr` (verify with
`check_submission.py --out-dir experiments/v005fr/submission` against the expected combine-table
counts — France ≈36.7 cands/S1 from v005, India/US ≈11.0 cands/S1 from v006 — before trusting
0.9640 as a real measurement); (c) genuine per-submission noise from the LB being a subset.
**Action before drawing any France conclusion from these four numbers: re-verify the v005fr
submission folder counts.** Until resolved, do not use v006's France rows over v005's — v005 pure
remains the best confirmed result and needs no hybrid.

### Where the remaining loss is (v005 holdout, 1 − F0.5 = 0.0227)

| Source of loss | Size | Meaning |
|---|---|---|
| Blocking | ≈ 0.004 | Ceiling 0.996 vs unpruned candidate recall 98.8%; small and well-characterized |
| Matcher / decision | ≈ 0.019 | Blank-address recall only ~51%; false positives on same-address/chain names |
| France (LB only) | ≈ 0.007–0.03 | See open question above — magnitude currently unclear |

## Version plan — status

| Version | What | Result | Status |
|---|---|---|---|
| v001–v004 | Baseline through stage-2 fallback + features v3 | LB 0.9310 → 0.9550 | Done |
| **v005** | Blocking v3 (skeleton/address/reverse channels) + features v4 | **holdout 0.9773, LB 0.9683** | **Done — current best** |
| v006 | Supervised meta-blocking pruner + features v5 + CatBoost blend + t_blank | holdout 0.9778, **LB 0.9652 (worse)** | Done — not used |
| v006fr / v005fr | Per-country hybrids of v005/v006 | LB 0.9642 / 0.9640 (both worse than v005) | Done — not used; see open question above |
| v007 (laptop B) | v006 configuration, final fit 800K S1 | Stopped: inherits v006's regression | Abandoned |
| v008 (SageMaker) | v006 configuration, full final fit | Stopped: same reason | Abandoned |
| Large-k blocking experiment | k=30/k_sk=10/k_rev=5, 68 cands/S1 | Ceiling +0.0009 for 6x v006's candidate count; test run OOM'd | Abandoned |
| **v010** | v005 code (tag `day2-sub2`) final-fit on 600K+ S1 incl. holdout, **unpruned** | Not yet run | **Candidate for final submission if time allows** |
| **v011** | Group-coherence post-filter (GraLMatch-inspired, Section below) | Not yet built | **Candidate for final submission if time allows** |
| v012 (parked) | Fine-tuned small LM (AnyMatch-style, GPT-2/124M) trained on India/US labels, applied zero-shot to France | Not started | Stretch only if v010/v011 land early and time remains |

### v010 — final fit on more data (same v005 code, no pruning)
- Rationale: v005 is trained on only 300K of 1.77M available train-fold S1. More data, same exact
  configuration that is proven on the leaderboard, is the lowest-risk remaining lever.
- Run at git tag `day2-sub2` (the exact code that produced v005), with `--final-fit` on 600K–800K
  S1 including the holdout (no holdout scoring possible for this run — judge by OOF only).
  **Do not add the v006 pruner, feature v5, or CatBoost/t_blank/fallback-search changes** — those
  are exactly what regressed on test in v006; keep the proven v005 configuration unchanged and
  only add data.
- Test candidates: reuse v005's saved unpruned test blocking output directly (`cand_test_all_v005`).
- Expected gain: +0.001 to +0.003 over v005, unconfirmed until submitted.

### v011 — group-coherence post-filter (new idea, from GraLMatch, PVLDB-adjacent 2024)
- For every S1 with 2+ predicted matches, compute pairwise name/address similarity **between the
  assigned candidates themselves** (not just each candidate vs. S1). A true business's several
  duplicate records should resemble each other; a false positive sharing only S1's address/name
  typically looks unlike the *other* accepted candidates.
- Implementation: a group-coherence feature (min/mean candidate-candidate similarity within the
  predicted group) added to stage 2, or a post-hoc filter dropping a candidate whose similarity to
  the rest of its assigned group is far below the group's internal average. Threshold tuned on
  out-of-fold data only.
- Targets the largest remaining precision-loss bucket (false positives on shared-address/chain
  names) with a mechanism not yet tried. Expected +0.001 to +0.004, concentrated on multi-match S1
  precision. Build on top of whichever base (v005 or v010) is currently best.

### v012 — parked stretch (AnyMatch-style zero-shot for France)
- Fine-tune GPT-2 (124M params, MIT-licensed, well under the 8B cap) as a match/no-match sequence
  classifier on our own labelled India/US pairs (Ditto-style serialization); apply zero-shot to
  French candidate pairs, especially the stage-1 uncertain band (p in 0.3–0.8).
- Only pursue if v010/v011 land early and a GPU is free; needs a training pipeline and inference
  over millions of pairs that we have not built. No external data — labels are entirely our own.

## Reviewed and not adopted (reason) — additions since last update
- **EXC-style reciprocal/mutual-best-choice decision rule** (from Papadakis et al., VLDB J. 2023,
  one-to-one matching algorithms survey) — does not transfer: that paper's setting is strict
  one-to-one on both sides, while ours is many-to-one (one S1 can have several true matches). A
  mutual-best-choice requirement would wrongly reject an S1's legitimate 2nd/3rd match.
- **Optimal (Hungarian-style) bipartite assignment for the decision step** — same survey found the
  empirically best-performing rule (UMC: highest-score-wins, unique per S2/S3 record) is exactly
  what we already do, and that the "principled" optimal-assignment approximation (BAH) performs
  *worse* in practice. Treated as validation that the decision layer is not the remaining
  bottleneck; no further engineering effort planned there.

## France (unlabeled, 15% of test)
- Keep features language-agnostic (no country feature) — unchanged.
- Confidence-profile diagnostic (stage-1 p) showed the model is not more uncertain on France than
  India/US at any version checked; any remaining France gap is confident errors, not a coverage gap.
- Country-hybrid submissions (this update) were meant to isolate France's contribution cleanly but
  produced an internally inconsistent result (see "Open question" above) — resolve this before
  acting on it further, rather than trusting either hybrid's implied France score.

## Submission budget

| Day | Status |
|---|---|
| Day 1 (25 Sep) | 2 used (v001, v002) |
| Day 2 (26 Sep) | 4 used (v004, v005, v006, v006fr) + v005fr this update |
| Day 3 (27 Sep) | Remaining slots: v010 and/or v011 if ready; final upload must be the best-scoring version, submitted last, well before 11:59 PM IST |

## Working rules (unchanged) + one addition
1–7. Unchanged from Day 2 (holdout-first where holdout is informative; one change per version;
no holdout leakage; validators before every submission; parallel work; drop lower-priority items
if behind; no network calls at run time).
8. **New:** after building any per-country or per-version hybrid/combined submission, re-run
   `check_submission.py --out-dir <saved folder>` and compare its printed counts against the
   combine script's own table **before** treating the resulting leaderboard score as a clean
   measurement — this exact class of copy/build error has happened twice already (v006fr's first
   attempt, and possibly v005fr per the open question above).

## Compute notes (unchanged)
- Laptop A: 12 CPU cores, 16 GB RAM. Laptop B: available for v010/v011 in parallel. SageMaker:
  stopped (v006-config final fit abandoned); restart only for v012 if pursued.
- CPU-bound (GPU cannot help): blocking (sparse top-k), feature building (rapidfuzz).
- GPU can help: LightGBM/CatBoost training (modest), any v012 fine-tuning (large).