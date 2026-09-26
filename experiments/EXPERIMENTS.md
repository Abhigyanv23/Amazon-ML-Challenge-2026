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
| v005 | 26 Sep | Normalization v3 + blocking v3 (skeleton/address/reverse channels) + features v4 | 0.988 | 0.996 | 0.9773 | 0.994 / 0.951 | 0.970 | TODO | TODO |

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