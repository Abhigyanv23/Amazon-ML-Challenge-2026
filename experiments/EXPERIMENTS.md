# Experiment log

One entry per version. Holdout = frozen 441,364 S1 IDs (`experiments/splits/holdout_s1_ids.txt`).
Never change more than one major component per version without noting it.

**Final submission: v009** (public LB **0.980893**, holdout 0.9857).

| Version | Date | Change | Cand. recall (holdout) | Oracle F0.5 | Holdout F0.5 | Precision / Recall (micro) | Singleton acc. | LB public | Git tag |
|---|---|---|---|---|---|---|---|---|---|
| v000 | 25 Sep | EDA, split, scorer | n/a | n/a | 0.0558 (all-empty) | n/a | 1.000 | n/a | n/a |
| v001 | 25 Sep | Normalization v1 + token blocking (k=20) + LightGBM + OOF-tuned decision | 0.948 | 0.980 | 0.9481 | 0.987 / 0.894 | 0.950 | 0.9310 | day1-sub1 |
| v002 | 25 Sep | Normalization v2 (zip, learned state fill) + 3-channel blocking + features v2 | 0.968 | 0.989 | 0.9545 | 0.988 / 0.905 | 0.953 | 0.9436 | day1-sub2 |
| v003 | 26 Sep | Stage-2 group-consistency rescoring on v002 stage-1 probabilities | 0.968 | 0.989 | 0.9580 | 0.988 / 0.920 | 0.963 | not submitted | v003-holdout-0.9580 |
| v004 | 26 Sep | Features v3 (IDF alignment, name specificity, house-number distance, n-gram cosine) + stage-2 fallback; v002 candidates | 0.968 | 0.989 | 0.9667 | 0.992 / 0.929 | 0.965 | 0.9550 | day2-sub1 |
| v005 | 26 Sep | Normalization v3 + blocking v3 (skeleton/address/reverse channels) + features v4 | 0.988 | 0.996 | 0.9773 | 0.994 / 0.951 | 0.970 | 0.9683 | day2-sub2 |
| v005w | 27 Sep | Wider blocking (k30/sk10/rev5) + name_overlap feature | 0.991 | 0.997 | 0.9773 | 0.993 / 0.952 | 0.969 | ~0.969 (exact value to confirm) | day3-sub1 |
| v008 | 27 Sep | GPT-2 cross-encoder score (g001) as stage-2 feature, on v005w/v005ws1 | 0.991 | 0.997 | 0.9835 | 0.996 / 0.962 | 0.986 | 0.9768 | v008-lb-0.9768 |
| **v009** | 27 Sep | + XLM-R-base cross-encoder score (x001) next to GPT-2 in stage 2 | 0.991 | 0.997 | **0.9857** | 0.997 / 0.964 | 0.991 | **0.9809 (final)** | day3-sub4, v009-holdout-0.9857 |
| v010a | 27 Sep | x001 scores widened to p1 0.005-0.995 (x001w, no retraining); stage-2 base-feature cache; test with --p1-min 0.001 | 0.991 | 0.997 | 0.9860 | 0.997 / 0.965 | 0.992 | not submitted | v010-holdout-0.9860 |
| v006 | 27 Sep | Pruner (cap 20, tau 0.005, 10.4 candidates/S1) + features v5 + lr 0.05, 5 folds, CatBoost blend, t_blank, fallback search | 0.980 | 0.994 | 0.9778 | 0.995 / 0.950 | 0.974 | 0.9652 (regressed) | day2-sub3 |
| v006np | 27 Sep | Features v7 (v005w + `unmatched_idf_a/b`, `hn_eq_word_jac`, `hn_conflict`) on v005's unpruned candidates, lr 0.05/5-fold | 0.988 | 0.996 | 0.9784 | 0.995 / 0.952 | n/a | 0.9640 (regressed, worse than v005w) | day3-sub3 |
| v005fr | 27 Sep | Per-country hybrid of v005 and v006 (version supplying France vs India/US rows swapped) | n/a | n/a | n/a (France unlabeled) | n/a | n/a | 0.9640 (regressed) | day2-sub4 |

Rows v006, v006np and v005fr come from a parallel branch (laptop A). All three were submitted to the
public LB and all three scored below v005 (0.9683). v006fr was prepared but never submitted. They are not
part of the final lineage. Details are in the last section.


## v000: data analysis and scaffolding
- EDA report: `experiments/v000_eda/eda_report.txt`.
- Split: 1,765,457 train / 441,364 holdout S1.
- Scorer self-test passed (perfect 1.0000, all-empty 0.0558).

## v001: baseline
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
  implied France F0.5 ~0.87 if India/US match holdout, so France is the main unmeasured gap.
- Test predictions: avg matches France 2.96 / India 3.01 / US 3.26; empty 6.3% / 7.5% / 6.0%.

## v002: normalization v2 + blocking v2 + features v2
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
- Test: 44.27M candidate pairs; avg matches France 3.24 / India 3.08 / US 3.28; empty 5.5% / 6.8% / 6.0%.
- Submission day1-sub2: public LB 0.9436 (v001 0.9310, +0.0126 vs holdout +0.0064);
  implied France ~0.91 (from ~0.87), so learned state fill fixed French blocks.

## v003: stage-2 group-consistency rescoring
- For each candidate: agreement with the other confident candidates of the same S1 (top 6 with p1 >= 0.3):
  name, no-space name, address, numbers, zip, state (max, p-weighted mean, strong-link count),
  plus stage-1 probability rank/gap/second-best and group counts. Second LightGBM on OOF stage-1 probabilities.
- OOF: stage-2 0.9565 vs stage-1 0.9533 on the same rows, at t=t_top=0.65; best iters 122/173/196.
- Holdout: F0.5 0.9580 (v002 0.9545); regression on 1-match S1 (0.8565 vs v002's 0.8784), fixed in v004.
- Not submitted: holdout gain held back to combine with v004 fixes.

## v004: error analysis → features v3 + stage-2 fallback
- Error analysis (v003 holdout): loss mostly recall; blank-address candidates recall 0.52; FPs =
  near-duplicates with different house numbers; stage 2 hurt 1-match S1; uniqueness rule confirmed
  (0.9580 vs 0.9574 without); bootstrap CI of v003-v002 diff [0.0033, 0.0037].
- Stage 1 (v004s1, v002 candidates): OOF 0.9637, holdout 0.9646. New top features: hn_edit 0.063,
  b_name_in_s1 0.030, hn_logdiff 0.024.
- Stage 2 (v004, fallback rule added): Holdout 0.9667; 1-match S1 0.9009 (v003 0.8565); the fallback
  fixed the regression.
- Submission day2-sub1: public LB 0.9550 (v002 0.9436, +0.0114; holdout +0.0122).
- France diagnostic (test, unsupervised, stage-1 p): confidence profile between US and India at every
  version checked, so any France gap is confident errors, not a coverage problem.

## v005: normalization v3 + blocking v3 + features v4
- Normalization v3: Indic->Latin consonant skeleton via shared Unicode block layout (name_skel);
  composite house number (addr_hn); state rule = last 2-letter code else first spelled-out name;
  digits-in-words (co1onial->colonial); 6+ digit runs removed from names; letter->digit split (fl13->fl 13).
- Blocking v3 (holdout): recall 0.9881 (v002 0.9677), oracle 0.9962 (0.9886), misses 18,114 (49,259),
  India 0.9821 / US 0.9922, avg cands 37.4 (25.6).
- Stage 1 (v005s1): 300K S1, 11.25M pairs (9.1% positive); OOF 0.9747; holdout 0.9756.
  Top gain: blk_rev 0.450, comb 0.204, hn_edit 0.041, num_jac 0.039, blk_sk 0.024, skel_cos3 0.021.
- Stage 2 (v005, fallback): OOF 0.9763; holdout 0.9773; India 0.9712 (+0.021), US 0.9813.
- Submission day2-sub2: public LB 0.9683 (v004 0.9550, +0.0133; holdout +0.0106).
  Implied France F0.5 ~0.927 (from ~0.912); holdout-LB gap 0.009 (was 0.012).

## v005w: wider blocking + name_overlap feature
- Blocking: word channel widened to top 30, skeleton to top 10, reverse to top 5 (from 20/5/2);
  candidate recall 0.9881 -> 0.9911 (India 0.9873, US 0.9937), oracle ceiling 0.9962 -> 0.9971.
- Features: + `name_overlap` (min-based overlap coefficient, high for acronym/subset name pairs
  like "IBM" vs "IBM International Business Machines", without Jaccard's length penalty).
- Holdout: F0.5 0.9773 (unchanged from v005 at this stage; the real payoff is wider recall feeding
  the cross-encoder band added next, not a holdout jump by itself). Candidates: 99.11M train-sample
  equivalent scale; test 115.4M pairs (66.6/S1).
- Submission day3-sub1 (27 Sep): public LB about 0.969 (exact value to confirm), roughly +0.001 over v005 (0.9683) against an unchanged holdout.

## v008: GPT-2 cross-encoder in stage 2 (AWS g5.2xlarge, A10G)
- `src/llm_rescore.py` (tag g001): GPT-2 small (124M, MIT) fine-tuned as a pair classifier on raw
  "name: .. addr: .." text of both records. Only uncertain pairs: stage-1 p in [0.02, 0.98], top 8
  per S1 (train 346K of 20.4M pairs, 32.7% positive; holdout 509K; test 2.44M). 2-fold GroupKFold by
  S1, 2 epochs, batch 64, lr 3e-5, fp16; ~455 pairs/s training, ~2K pairs/s inference; 30 min train.
- OOF on band pairs: AUC stage-1 0.9256 / GPT-2 0.9430 / mean 0.9623; logloss 0.3108 / 0.2815.
- Stage 2 (`stage2.py --llm g001`, features p_llm + llm_gap, NaN outside band): OOF 0.9827
  (v005w 0.9765); plain stage 2 now beats the earlier sparse-group fallback rule.
  Gain: p1 0.841, p1_rank 0.114, p_llm 0.024.
- Holdout: F0.5 0.9835 (v005w 0.9773); precision 0.996, recall 0.962; singleton acc 0.986 (0.969);
  1-match S1 0.941 (0.927); India 0.9800 (+0.009), US 0.9857 (+0.005).
- **Submission: public LB 0.9768 (v005 0.9683, +0.0085; holdout +0.0062 vs v005w). Tag: day3-sub3.**
  Holdout-LB gap 0.0067 (was 0.009).
- Remaining holdout errors (pairs among candidates, 51,230): 79% inside the GPT-2 band (35.8K FN /
  4.7K FP), 14% just outside it (p1 0.005-0.02 / 0.98-0.995), 7% far outside; plus 13,545 true
  pairs never blocked. -> next: a second, multilingual cross-encoder on the same band.

## v009: XLM-RoBERTa-base as a second cross-encoder (final submission)
- x001: `xlm-roberta-base` (MIT, 278M, multilingual), same band and folds as g001; text-pair input
  (`<s> a </s></s> b </s>`), 2 epochs, lr 2e-5, bf16; ~530 pairs/s training, ~4.8K pairs/s inference.
  First test run was OOM-killed (20.5 GB: tokenizing 2.4M pairs in one call while stage 2 used
  10 GB); fixed by encoding in 50K-row chunks (~8 GB).
- OOF on band pairs: AUC stage-1 0.9256 / GPT-2 0.9430 / XLM-R 0.9679; logloss 0.3108 / 0.2815 / 0.2121.
- Stage 2 (`--llm g001,x001`): OOF 0.9852 (v008 0.9827); plain stage 2, t = t_top = 0.73.
  Gain: p1 0.931, p1_rank 0.033, p_llm_x001 0.027, p_llm (GPT-2) 0.001, so XLM-R absorbs most of
  GPT-2's signal (multilingual sub-word reading of Devanagari/Tamil/French beats GPT-2's
  English-only tokenizer on this band).
- Holdout: F0.5 0.9857 (v008 0.9835); precision 0.997, recall 0.964; singleton acc 0.991 (0.986);
  1-match S1 0.952 (0.941); India 0.9848 (+0.005), US 0.9864 (+0.001).
- Test: avg matches France 3.28 / India 3.33 / US 3.36; empty 5.8% / 5.9% / 5.8%;
  5,780,837 matched IDs.
- **Submission: public LB 0.980893 (v008 0.9768, +0.0041; holdout +0.0022). Holdout-LB gap 0.0048
  (v008 0.0067); the multilingual cross-encoder narrowed the gap further, consistent with it
  helping France as well as India. Tag: day3-sub4.**
  **Final submission package is built from v009**: `output/` = these files, validator PASS with
  `--check-ids`.

## v010 — wide-band XLM-R scores + stage-2 feature cache (after v009; not in the final package)
- Stage-2 base features cached per split (`stage2.py cache`); reruns cost minutes. Verified identical to v009 features.
- x001w: XLM-R x001 fold models also score p1 in [0.005, 0.02) and (0.98, 0.995] (`llm_rescore.py extend`,
  no retraining; train rows scored only by the fold model that did not see the S1; OOF reproduction check
  mean |diff| 0.0006). Extra pairs: train 381K, holdout 562K, test 2.1M.
- v010a = stage 2 `--llm g001,x001w`: OOF 0.9855 (v009 0.9852); holdout 0.9860 (v009 0.9857; full stage 2, no --p1-min);
  precision 0.9971, recall 0.9649; India 0.9853, US 0.9865.
- Test with `--p1-min 0.001` (stage 2 only on the 10.2% of pairs with p1 >= 0.001; features verified identical,
  skipped pairs get p2 = 0): validator PASS; outputs in /data/submissions/v010a.
- Not finished: v010b (--group-llm) and x003 (xlm-roberta-large; stopped in fold 0 epoch 2 to free RAM).

## Superseded branches (laptop A): submitted, not in the final lineage

These were explored on laptop A in parallel with the GPT-2/XLM-R work above, before that work's
results were known. v006, v005fr and v006np were submitted to the public LB, and all three scored below v005 (0.9683).
v006fr was never submitted. The submitted ones are listed in the main table. All are superseded by v009.

- **v006** (supervised meta-blocking pruner, cap 20/tau 0.005, 10.4 candidates/S1, plus features v5, lr 0.05, 5 folds, CatBoost blend, t_blank, fallback search): holdout 0.9778
  (slightly above v005's 0.9773) but **public LB 0.9652, a regression** versus v005's 0.9683.
  Root cause (best evidence): per-S1 independent top-K pruning can drop a candidate from its true
  owner's (crowded) list while a less-contested S1 keeps it; the holdout under-represents this
  because many fewer S1s compete per record there than on the full test set.
- **v005fr** (per-country hybrid swapping which version supplies France vs. India/US
  rows; submitted as day3-sub2): public LB 0.9640, worse than pure v005. A second hybrid, v006fr, was
  prepared but not submitted. The hybrid-isolation technique was sound
  methodology but did not find a net win from this particular pair of versions.
- **v006np** (features v7 (v005w features plus `unmatched_idf_a/b`, `hn_eq_word_jac`, `hn_conflict`) run on
  v005's unpruned candidates, isolating feature gain from pruning damage): holdout **0.9784, the
  best holdout score achieved on laptop A**, but **public LB 0.9640, still a regression (and below v005w)**, despite no
  pruner being involved at all. This showed the holdout-vs-LB gap is not limited to pruning effects;
  some feature/retraining changes can overfit the holdout in ways that don't generalize to the full
  test-time candidate competition. This finding directly motivated treating confirmed leaderboard
  scores, not holdout alone, as the deciding signal for the v008/v009 cross-encoder work above.