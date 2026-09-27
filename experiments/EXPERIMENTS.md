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
| v006 | 27 Sep | Pruner (cap 20, tau 0.005) + features v5 + lr 0.05, 5 folds, CatBoost blend, t_blank, fallback search | 0.980 | 0.994 | 0.9778 | 0.995 / 0.950 | 0.974 | TODO | TODO |

Public leaderboard leader (26 Sep): 0.988419.

## v000 — data analysis and scaffolding
- EDA report: `experiments/v000_eda/eda_report.txt`.
- Split: 1,765,457 train / 441,364 holdout S1.
- Scorer self-test passed (perfect 1.0000, all-empty 0.0558).

## v001 — baseline
- Normalization eval (200K train-fold true pairs): name_core exact 0.508 (EDA baseline 0.220);
  addr Jaccard 0.766 (baseline 0.597); random pairs stay at 0.000 exact.
- Blocking: per (country, state), IDF cosine, max_df 2%, weights name/addr/num = 1/1/1.
  Recall@20 0.948, @50 0.962. Runtime 815 s for holdout.
- Blocking misses @50: 57,559 (non-Latin 21.1K, address-only overlap 13.3K, state mismatch 9.6K,
  no shared number 9.0K, outranked 4.5K). Details: `experiments/v001/blocking_misses.txt`.
- Matcher: 300K train-fold S1, 6.0M pairs (16.4% positive); OOF F0.5 0.9467 at t=0.65, t_top=0.65;
  best iters 1110/934/953. Top features by gain: comb 0.43, comb_gap 0.15, num_jac 0.07, blk_score 0.06.
- Holdout: F0.5 0.9481 (India 0.9225, US 0.9652); precision 0.987, recall 0.894; singleton acc 0.950.
- Submission day1-sub1: public LB 0.9310 (holdout 0.9481). Country-mix reweighting explains ~0.006;
  implied France F0.5 ~0.87 if India/US match holdout -> France is the main unmeasured gap.
- Test predictions: avg matches France 2.96 / India 3.01 / US 3.26; empty 6.3% / 7.5% / 6.0%.

## v002 — normalization v2 + blocking v2 + features v2
- Country diagnostic (test, unsupervised): France S2/S3 state found only 66-68% (addresses end in a city);
  postcodes mixed into house numbers; 28.5% of French S1 names repeat >= 5 times (chain-like).
- Normalization v2: zip split from house numbers; last state component wins (fixes "Washington, NY");
  learned component->state fill from same-split S1, no labels (state found: France S2/S3 67%->97%,
  India 91%->96.5%); FR abbreviations. Eval: state agrees 0.9941 (0.9929); random-pair exact rows stay 0.
- Blocking v2: word (+nospace token) k=20, trigram name k=10, non-Latin address k=5; AP/TG, JK/LA merged; smoothed IDF.
  Holdout: recall 0.9677 (v001 0.948), oracle F0.5 0.9886 (0.980), avg cands 25.6; India 0.9475 / US 0.9812;
  only-trigram 7,058, only-non-Latin 6,733; state-block misses 1,246 (9,622). Runtime 974 s.
- Features v2: + zip_eq, b_state_inferred, n_name_close; all blk_* channel scores/ranks.
- Matcher: 300K S1, 7.68M pairs (13.1% positive); OOF F0.5 0.9533 at t=0.70, t_top=0.70.
- Holdout: F0.5 0.9545 (v001 0.9481); India 0.9358, US 0.9669; precision 0.988, recall 0.905; singleton acc 0.953.
- Loss split: blocking 0.011, matcher 0.035 -> matcher is now the bottleneck.
- Test: 44.27M candidate pairs; avg matches France 3.24 / India 3.08 / US 3.28; empty 5.5% / 6.8% / 6.0%.
- Submission day1-sub2: public LB 0.9436 (v001 0.9310, +0.0126 vs holdout +0.0064);
  implied France ~0.91 (from ~0.87) -> learned state fill fixed French blocks.

## v003 — stage-2 group-consistency rescoring
- For each candidate: agreement with the other confident candidates of the same S1 (top 6 with p1 >= 0.3):
  name, no-space name, address, numbers, zip, state (max, p-weighted mean, strong-link count),
  plus stage-1 probability rank/gap/second-best and group counts. Second LightGBM on OOF stage-1 probabilities.
- OOF: stage-2 0.9565 vs stage-1 0.9533 on the same rows, at t=t_top=0.65; best iters 122/173/196.
  Top gain: p1 0.71, p1_gap 0.21, n_ge50 0.04, sup_best 0.02.
- Holdout: F0.5 0.9580 (v002 0.9545); India 0.9405, US 0.9696; precision 0.988, recall 0.920; singleton acc 0.963.
  Regression: 1-match S1 F0.5 0.8565 (v002 0.8784) -> no other confident candidates means no support signal.
- Test: avg matches France 3.29 / India 3.19 / US 3.40; empty 5.6% / 7.1% / 6.1%.
- Decision tuned on the coarse 0.05 grid only (fine 0.01 grid added to matcher.tune() after this run).
- Not submitted: holdout gain +0.0035 held back to combine with v004 fixes. Test outputs kept in experiments/v003/not_submitted/.

## v004 — error analysis → features v3 + stage-2 fallback
- Error analysis (v003 holdout): loss mostly recall (missing-only 0.0188, false-empty 0.0119); blank-address
  candidates recall 0.52; FPs = near-duplicates with different house numbers; stage 2 hurt 1-match S1;
  uniqueness rule confirmed (0.9580 vs 0.9574 without); bootstrap CI of v003-v002 diff [0.0033, 0.0037].
- Stage 1 (v004s1, v002 candidates): OOF 0.9637, holdout 0.9646 (v002 0.9545). New top features:
  hn_edit 0.063, b_name_in_s1 0.030, hn_logdiff 0.024.
- Stage 2 (v004): OOF plain 0.9655 vs fallback 0.9658 -> fallback. Holdout 0.9667; India 0.9498, US 0.9780;
  precision 0.992, recall 0.929; 1-match S1 0.9009 (v003 0.8565).
- Loss split: matcher ~0.022, blocking ~0.011.
- Test: avg matches France 3.20 / India 3.15 / US 3.35; empty 5.8% / 6.8% / 5.9%; 5,599,754 matched IDs.
- France diagnostic (test, unsupervised, stage-1 p): France confidence profile between US and India
  (top p>=0.95: FR 93.5% / IN 91.3% / US 93.7%; top p in 0.3-0.8: 1.0% / 1.7% / 0.7%). Model is not
  uncertain on France -> any France gap is confident errors (likely FPs on chain names) or LB-estimate noise.
  No France-specific change in v005.

## v005 — normalization v3 + blocking v3 + features v4
- Normalization v3: Indic->Latin consonant skeleton via shared Unicode block layout (name_skel);
  composite house number (addr_hn); state rule = last 2-letter code else first spelled-out name;
  digits-in-words (co1onial->colonial); 6+ digit runs removed from names; letter->digit split (fl13->fl 13).
- Blocking v3 (holdout): recall 0.9881 (v002 0.9677), oracle 0.9962 (0.9886), misses 18,114 (49,259),
  India 0.9821 / US 0.9922, avg cands 37.4 (25.6). Only-channel: word 46,578, skeleton 8,282,
  trigram 5,975, address 5,839, non-Latin addr 644, reverse 106. State-block misses 2,899 (1,246).
- Stage 1 (v005s1): 300K S1, 11.25M pairs (9.1% positive); OOF 0.9747; holdout 0.9756.
  Top gain: blk_rev 0.450, comb 0.204, hn_edit 0.041, num_jac 0.039, blk_sk 0.024, skel_cos3 0.021.
- Stage 2 (v005, fallback): OOF 0.9763; holdout 0.9773 (v004 0.9667); India 0.9712 (+0.021), US 0.9813;
  precision 0.994, recall 0.951; 1-match S1 0.926; false-empty 0.0051.
- Issue: reverse channel lets one S1 collect many candidates (max 6,243; avg 37.4) -> pruner in v006.
- Submission (day2-sub2): public LB 0.9683 (v004 0.9550, +0.0133; holdout +0.0106).
  Implied France F0.5 ~0.927 (from ~0.912): first France gain; holdout-LB gap 0.009 (was 0.012).
- Error analysis (holdout): loss 0.0227 = missing-only 0.0116, false-empty 0.0048, extra-FP 0.0035,
  singleton-FP 0.0017, FP+missing 0.0011. Blank-address candidates recall 0.508 (61K true pairs, ~30K FNs).
  FPs: distinct businesses at same address (Manya Traders vs Manya Rifle; same number, different street).
  42% of holdout FPs belong to train-fold S1 (resolvable by uniqueness on test). Stage-2 gain +0.0016
  (CI [0.0015, 0.0017]); stage 2 adds FPs for 1-match S1. Uniqueness rule confirmed (0.9773 vs 0.9769).
  -> v006: t_blank threshold, features v5 (unmatched IDF mass, hn_eq_word_jac), fallback 1/2/3 search.

## v006 — supervised meta-blocking pruner + model upgrades + features v5
- Pruner (blocking outputs only; cap=20, tau=0.005 chosen on OOF with max recall loss 0.008):
  OOF recall 0.9803 @ 10.4 cands; holdout 0.9881 -> 0.9802, 37.4 -> 10.4 cands (4.59M pairs);
  test 63.5M -> 19.0M pairs (36.7 -> 11.0 per S1, max 20).
- Laptop B (final-fit prep, 800K S1): blocking recall 0.9881, oracle 0.9961, avg 37.5 -> identical to laptop A
  (reproducible from fresh clone).
  - Stage 1 (v006s1, 300K S1, 3.12M pruned pairs, 32.7% positive): LGB best iters 1625-2083; CatBoost hit 6000 cap.
  Blend OOF: w=0 0.9753, 0.5 0.9757, 1.0 0.9755 -> blend +0.0002. t_blank=0.75 +0.0001. Holdout 0.9763 (v005s1 0.9756).
  Top gain: comb 0.44, hn_edit 0.10, hn_logdiff 0.05, blk_sk 0.05, unmatched_idf_b 0.03.
- Stage 2 (v006): fallback 1/2/3 OOF 0.9772/0.9770/0.9767 -> FB_MAX=1. Holdout 0.9778 (v005 0.9773);
  precision 0.9945, recall 0.9498; singleton 0.9735; India 0.9720, US 0.9816.
- Candidates: 10.4/S1 (max 20) vs v005 37.4 (max 6,243): 3.6x smaller at +0.0005 F0.5.
- Submission day2-sub3 (v006): public LB 0.9652 (v005 0.9683, -0.0031) despite holdout +0.0005.
  Implied France ~0.902 (v005 ~0.927): pruner trained on India/US labels drops true French candidates.
- v006fr: hybrid submission — France rows from v005 (unpruned candidates), India/US from v006 (pruned).# Experiment log

One entry per version. Holdout = frozen 441,364 S1 IDs (`experiments/splits/holdout_s1_ids.txt`).
Never change more than one major component per version without noting it.

| Version | Date | Change | Cand. recall (holdout) | Oracle F0.5 | Holdout F0.5 | Precision / Recall (micro) | Singleton acc. | LB public | Git tag |
|---|---|---|---|---|---|---|---|---|---|
| v000 | 25 Sep | EDA, split, scorer | — | — | 0.0558 (all-empty) | — | 1.000 | — | — |
| v001 | 25 Sep | Normalization v1 + token blocking (k=20) + LightGBM + OOF-tuned decision | 0.948 | 0.980 | 0.9481 | 0.987 / 0.894 | 0.950 | 0.9310 | day1-sub1 |
| v002 | 25 Sep | Normalization v2 (zip, learned state fill) + 3-channel blocking + features v2 | 0.968 | 0.989 | 0.9545 | 0.988 / 0.905 | 0.953 | 0.9436 | day1-sub2 |
| v003 | 26 Sep | Stage-2 group-consistency rescoring on v002 stage-1 probabilities | 0.968 | 0.989 | 0.9580 | 0.988 / 0.920 | 0.963 | not submitted | v003-holdout-0.9580 |
| v004 | 26 Sep | Features v3 (IDF alignment, name specificity, house-number distance, n-gram cosine) + stage-2 fallback; v002 candidates | 0.968 | 0.989 | 0.9667 | 0.992 / 0.929 | 0.965 | 0.9550 | day2-sub1 |
| **v005** | 26 Sep | Normalization v3 + blocking v3 (skeleton/address/reverse channels) + features v4 | 0.988 | 0.996 | 0.9773 | 0.994 / 0.951 | 0.970 | **0.9683** (best) | day2-sub2 |
| v006 | 26 Sep | Pruner (cap 20, tau 0.005) + features v5 + lr 0.05, 5 folds, CatBoost blend, t_blank, fallback search | 0.980 | 0.994 | 0.9778 | 0.995 / 0.950 | 0.974 | 0.9652 | day2-sub3 |
| v006fr | 26 Sep | Hybrid: France rows from v005, India/US rows from v006 (no training) | — | — | — | — | — | 0.9642 | day2-sub4 |
| v005fr | 26 Sep | Hybrid: India/US rows from v005, France rows from v006 (no training) | — | — | — | — | — | built, not submitted | — |

Public leaderboard leader (26 Sep): 0.988419.

## Submission history (public leaderboard)

| Tag | Version | Public LB | Candidate pairs (test) | Matched IDs |
|---|---|---|---|---|
| day1-sub1 | v001 | 0.9310 | 34.65M (20.0/S1) | 5,367,942 |
| day1-sub2 | v002 | 0.9436 | 44.27M (25.6/S1) | 5,511,125 |
| day2-sub1 | v004 | 0.9550 | 44.27M (25.6/S1) | 5,599,754 |
| day2-sub2 | v005 | **0.9683** | 63.53M (36.7/S1, max 6,243) | 5,761,795 |
| day2-sub3 | v006 | 0.9652 | 19.01M (11.0/S1, max 20) | 5,652,386 |
| day2-sub4 | v006fr | 0.9642 | France unpruned + India/US pruned | 5,685,011 |

(`day2-sub2` was tagged retroactively on commit `ce6d4af`, the last commit before v006's code changes.)

## v000 — data analysis and scaffolding
- EDA report: `experiments/v000_eda/eda_report.txt`.
- Split: 1,765,457 train / 441,364 holdout S1.
- Scorer self-test passed (perfect 1.0000, all-empty 0.0558).

## v001 — baseline
- Normalization eval (200K train-fold true pairs): name_core exact 0.508 (EDA baseline 0.220);
  addr Jaccard 0.766 (baseline 0.597); random pairs stay at 0.000 exact.
- Blocking: per (country, state), IDF cosine, max_df 2%, weights name/addr/num = 1/1/1.
  Recall@20 0.948, @50 0.962. Runtime 815 s for holdout.
- Blocking misses @50: 57,559 (non-Latin 21.1K, address-only overlap 13.3K, state mismatch 9.6K,
  no shared number 9.0K, outranked 4.5K). Details: `experiments/v001/blocking_misses.txt`.
- Matcher: 300K train-fold S1, 6.0M pairs (16.4% positive); OOF F0.5 0.9467 at t=0.65, t_top=0.65;
  best iters 1110/934/953. Top features by gain: comb 0.43, comb_gap 0.15, num_jac 0.07, blk_score 0.06.
- Holdout: F0.5 0.9481 (India 0.9225, US 0.9652); precision 0.987, recall 0.894; singleton acc 0.950.
- Submission day1-sub1: public LB 0.9310 (holdout 0.9481). Country-mix reweighting explains ~0.006;
  implied France F0.5 ~0.87 if India/US match holdout -> France is the main unmeasured gap.
- Test predictions: avg matches France 2.96 / India 3.01 / US 3.26; empty 6.3% / 7.5% / 6.0%.

## v002 — normalization v2 + blocking v2 + features v2
- Country diagnostic (test, unsupervised): France S2/S3 state found only 66-68% (addresses end in a city);
  postcodes mixed into house numbers; 28.5% of French S1 names repeat >= 5 times (chain-like).
- Normalization v2: zip split from house numbers; last state component wins (fixes "Washington, NY");
  learned component->state fill from same-split S1, no labels (state found: France S2/S3 67%->97%,
  India 91%->96.5%); FR abbreviations. Eval: state agrees 0.9941 (0.9929); random-pair exact rows stay 0.
- Blocking v2: word (+nospace token) k=20, trigram name k=10, non-Latin address k=5; AP/TG, JK/LA merged; smoothed IDF.
  Holdout: recall 0.9677 (v001 0.948), oracle F0.5 0.9886 (0.980), avg cands 25.6; India 0.9475 / US 0.9812;
  only-trigram 7,058, only-non-Latin 6,733; state-block misses 1,246 (9,622). Runtime 974 s.
- Features v2: + zip_eq, b_state_inferred, n_name_close; all blk_* channel scores/ranks.
- Matcher: 300K S1, 7.68M pairs (13.1% positive); OOF F0.5 0.9533 at t=0.70, t_top=0.70.
- Holdout: F0.5 0.9545 (v001 0.9481); India 0.9358, US 0.9669; precision 0.988, recall 0.905; singleton acc 0.953.
- Loss split: blocking 0.011, matcher 0.035 -> matcher is now the bottleneck.
- Test: 44.27M candidate pairs; avg matches France 3.24 / India 3.08 / US 3.28; empty 5.5% / 6.8% / 6.0%.
- Submission day1-sub2: public LB 0.9436 (v001 0.9310, +0.0126 vs holdout +0.0064);
  implied France ~0.91 (from ~0.87) -> learned state fill fixed French blocks.

## v003 — stage-2 group-consistency rescoring
- For each candidate: agreement with the other confident candidates of the same S1 (top 6 with p1 >= 0.3):
  name, no-space name, address, numbers, zip, state (max, p-weighted mean, strong-link count),
  plus stage-1 probability rank/gap/second-best and group counts. Second LightGBM on OOF stage-1 probabilities.
- OOF: stage-2 0.9565 vs stage-1 0.9533 on the same rows, at t=t_top=0.65; best iters 122/173/196.
  Top gain: p1 0.71, p1_gap 0.21, n_ge50 0.04, sup_best 0.02.
- Holdout: F0.5 0.9580 (v002 0.9545); India 0.9405, US 0.9696; precision 0.988, recall 0.920; singleton acc 0.963.
  Regression: 1-match S1 F0.5 0.8565 (v002 0.8784) -> no other confident candidates means no support signal.
- Test: avg matches France 3.29 / India 3.19 / US 3.40; empty 5.6% / 7.1% / 6.1%.
- Decision tuned on the coarse 0.05 grid only (fine 0.01 grid added to matcher.tune() after this run).
- Not submitted: holdout gain +0.0035 held back to combine with v004 fixes.

## v004 — error analysis → features v3 + stage-2 fallback
- Error analysis (v003 holdout): loss mostly recall (missing-only 0.0188, false-empty 0.0119); blank-address
  candidates recall 0.52; FPs = near-duplicates with different house numbers; stage 2 hurt 1-match S1;
  uniqueness rule confirmed (0.9580 vs 0.9574 without); bootstrap CI of v003-v002 diff [0.0033, 0.0037].
- Stage 1 (v004s1, v002 candidates): OOF 0.9637, holdout 0.9646 (v002 0.9545). New top features:
  hn_edit 0.063, b_name_in_s1 0.030, hn_logdiff 0.024.
- Stage 2 (v004): OOF plain 0.9655 vs fallback 0.9658 -> fallback. Holdout 0.9667; India 0.9498, US 0.9780;
  precision 0.992, recall 0.929; 1-match S1 0.9009 (v003 0.8565).
- Loss split: matcher ~0.022, blocking ~0.011.
- Test: avg matches France 3.20 / India 3.15 / US 3.35; empty 5.8% / 6.8% / 5.9%; 5,599,754 matched IDs.
- Submission day2-sub1: public LB 0.9550 (v002 0.9436, +0.0114; holdout +0.0122).
  Implied France F0.5 ~0.912 (unchanged from v002).
- France diagnostic (test, unsupervised, stage-1 p): France confidence profile between US and India
  (top p>=0.95: FR 93.5% / IN 91.3% / US 93.7%; top p in 0.3-0.8: 1.0% / 1.7% / 0.7%). Model is not
  uncertain on France -> any France gap is confident errors or LB-estimate noise.

## v005 — normalization v3 + blocking v3 + features v4  (best public LB)
- Normalization v3: Indic->Latin consonant skeleton via shared Unicode block layout (name_skel);
  composite house number (addr_hn); state rule = last 2-letter code else first spelled-out name;
  digits-in-words (co1onial->colonial); 6+ digit runs removed from names; letter->digit split (fl13->fl 13).
- Blocking v3 (holdout): recall 0.9881 (v002 0.9677), oracle 0.9962 (0.9886), misses 18,114 (49,259),
  India 0.9821 / US 0.9922, avg cands 37.4 (25.6). Only-channel: word 46,578, skeleton 8,282,
  trigram 5,975, address 5,839, non-Latin addr 644, reverse 106. State-block misses 2,899 (1,246).
- Stage 1 (v005s1): 300K S1, 11.25M pairs (9.1% positive); OOF 0.9747; holdout 0.9756.
  Top gain: blk_rev 0.450, comb 0.204, hn_edit 0.041, num_jac 0.039, blk_sk 0.024, skel_cos3 0.021.
- Stage 2 (v005, fallback): OOF 0.9763; holdout 0.9773 (v004 0.9667); India 0.9712 (+0.021), US 0.9813;
  precision 0.994, recall 0.951; 1-match S1 0.926; false-empty 0.0051.
- Issue: reverse channel lets one S1 collect many candidates (max 6,243; avg 37.4) -> pruner in v006.
- Test: 63,534,783 candidate pairs (36.7/S1); avg matches France 3.35 / India 3.30 / US 3.36;
  empty 5.4% / 6.0% / 5.8%; 5,761,795 matched IDs.
- Submission day2-sub2: public LB 0.9683 (v004 0.9550, +0.0133; holdout +0.0106).
  Implied France F0.5 ~0.927 (from ~0.912); holdout-LB gap 0.009 (was 0.012).
- Error analysis (holdout): loss 0.0227 = missing-only 0.0116, false-empty 0.0048, extra-FP 0.0035,
  singleton-FP 0.0017, FP+missing 0.0011. Blank-address candidates recall 0.508 (61K true pairs, ~30K FNs).
  FPs: distinct businesses at same address (Manya Traders vs Manya Rifle; same number, different street).
  42% of holdout FPs belong to train-fold S1 (on test those S1 compete, so the uniqueness rule can resolve them).
  Stage-2 gain +0.0016 (CI [0.0015, 0.0017]); stage 2 adds FPs for 1-match S1. Uniqueness rule confirmed
  (0.9773 vs 0.9769 without).

## v006 — supervised meta-blocking pruner + model upgrades + features v5
- Pruner (blocking outputs only; cap=20, tau=0.005 chosen on OOF with max recall loss 0.008):
  OOF recall 0.9803 @ 10.4 cands; holdout 0.9881 -> 0.9802, 37.4 -> 10.4 cands (4.59M pairs);
  test 63.5M -> 19.0M pairs (36.7 -> 11.0 per S1, max 20).
- Stage 1 (v006s1, 300K S1, 3.12M pruned pairs, 32.7% positive): LGB best iters 1625-2083; CatBoost hit 6000 cap.
  Blend OOF: w=0 0.9753, 0.5 0.9757, 1.0 0.9755 -> blend +0.0002. t_blank=0.75 +0.0001. Holdout 0.9763 (v005s1 0.9756).
  Top gain: comb 0.44, hn_edit 0.10, hn_logdiff 0.05, blk_sk 0.05, unmatched_idf_b 0.03.
- Stage 2 (v006): fallback 1/2/3 OOF 0.9772/0.9770/0.9767 -> FB_MAX=1. Holdout 0.9778 (v005 0.9773);
  precision 0.9945, recall 0.9498; singleton 0.9735; India 0.9720, US 0.9816.
- Test: 19,011,103 candidate pairs (11.0/S1, 20 S1 with none); 5,652,386 matched IDs;
  avg matches France 3.22 / India 3.23 / US 3.32 (v005: 3.35 / 3.30 / 3.36).
- Submission day2-sub3: public LB 0.9652 (v005 0.9683, -0.0031) despite holdout +0.0005.

## v006fr / v005fr — per-country hybrids (no training)
- v006fr (France rows from v005 unpruned, India/US rows from v006): public LB 0.9642 (day2-sub4).
- Reading the three submissions together (they differ only in which version supplies which countries):
  - v006fr vs v006 (only France rows differ): -0.0010 -> v006's France rows are ~0.007 BETTER than v005's.
  - v005 vs v006fr (only India/US rows differ): +0.0041 -> v005's India/US rows are ~0.005 better than v006's.
  - The earlier hypothesis "the pruner hurts France" was WRONG. The loss is on India/US test, which the holdout
    did not show. Most likely cause: on test all S1 compete for each pool record under the one-S1-per-candidate
    rule; per-S1 pruning can drop a record from its true owner's list while keeping it in another S1's list.
    The holdout cannot show this (42% of its FPs belong to non-competing train-fold S1).
- v005fr (India/US rows from v005, France rows from v006): built and validated, NOT submitted.
  Expected LB ~0.969 (v005 + ~0.001 from v006's France rows).

## Final-fit runs (other machines)
- Laptop B, v007 (v006 configuration, 800K S1 incl. holdout): blocking recall 0.9881, oracle 0.9961, avg 37.5
  (identical to laptop A -> pipeline reproducible from a fresh clone); pruner reproduced (cap 20, tau 0.005,
  recall 0.9803, 10.4/S1). Stopped: v006 configuration underperforms v005 on test.
- SageMaker, v008 (v006 configuration, all S1): set up (Linux), stopped for the same reason.
- Laptop B, larger-k blocking experiment (k=30, k_sk=10, k_rev=5): holdout recall 0.9911, oracle 0.9971,
  68.0 cands/S1; test blocking ran out of memory (~118M pairs). Abandoned: +0.0009 ceiling for 6x v006's candidates.
- Planned v010: v005 code (tag day2-sub2), final fit on 600K S1 incl. holdout, unpruned; France rows from v006.