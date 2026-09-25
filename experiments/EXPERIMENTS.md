# Experiment log

One entry per version. Holdout = frozen 441,364 S1 IDs (`experiments/splits/holdout_s1_ids.txt`).
Never change more than one major component per version without noting it.

| Version | Date | Change | Cand. recall (holdout) | Oracle F0.5 | Holdout F0.5 | Precision / Recall (micro) | Singleton acc. | LB public | Git tag |
|---|---|---|---|---|---|---|---|---|---|
| v000 | 25 Sep | EDA, split, scorer | — | — | 0.0558 (all-empty) | — | 1.000 | — | — |
| v001 | 25 Sep | Normalization v1 + token blocking (k=20) + LightGBM + OOF-tuned decision | 0.948 | 0.980 | TODO | TODO | TODO | TODO | TODO |

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
- Matcher: TODO (OOF F0.5, t, t_top, top features).
- Holdout: TODO.
- Submission: TODO.

## Planned
- v002: blocking fixes (nospace token, trigram channel, non-Latin address channel, AP/TG merge,
  last-state-component rule). Requires re-running normalization → blocking → train → holdout.