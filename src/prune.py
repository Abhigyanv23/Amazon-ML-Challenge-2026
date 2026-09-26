"""
src/prune.py  — supervised meta-blocking (comparison cleaning), part of candidate generation.
A small LightGBM scores each blocked pair using ONLY blocking outputs (channel scores/ranks,
number of channels that found the pair, score relative to the S1's best, candidate count).
Pairs below a threshold (and beyond a per-S1 cap) are dropped. The kept set is what the
matcher sees and what goes into candidate_pairs.tsv.
Reference: Papadakis et al., Supervised Meta-blocking, PVLDB 2014 (survey: Christophides et al. 2020, sec. 4.2).

Threshold/cap are chosen on train-fold OUT-OF-FOLD scores: smallest average candidate count whose
candidate recall is within --max-recall-loss of the unpruned recall. Holdout labels are never used.

  python src/prune.py train --tag v005                       # fit on cand_train_train_v005 -> writes ..._v005p
  python src/prune.py apply --tag v005 --split train --s1-set holdout
  python src/prune.py apply --tag v005 --split test  --s1-set all
Output: experiments/cache/cand_<split>_<s1set>_<tag>p.parquet (same columns as input)
"""
import argparse
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from data_loading import CACHE_DIR, load_split_ids
from matcher import true_pairs

CH = [("score", "rank"), ("blk_tri", "blk_tri_rank"), ("blk_nl", "blk_nl_rank"),
      ("blk_sk", "blk_sk_rank"), ("blk_ad", "blk_ad_rank"), ("blk_rev", "blk_rev_rank")]
PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1,
              num_threads=os.cpu_count(), verbose=-1, seed=0)
TAUS = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
CAPS = [0, 20, 15, 12]            # 0 = no cap


def cpath(split, s1set, tag):
    return os.path.join(CACHE_DIR, f"cand_{split}_{s1set}_{tag}.parquet")


def features(c):
    X = pd.DataFrame(index=c.index)
    hits = np.zeros(len(c), np.float32)
    for col, rk in CH:
        if col not in c:
            continue
        X[col] = c[col]
        X[rk] = c[rk]
        X[col + "_rel"] = c[col] / c.groupby("s1_id")[col].transform("max").clip(lower=1e-6)
        hits += (c[col] > 0).to_numpy()
    X["n_hits"] = hits
    X["n_cands"] = c.groupby("s1_id").s1_id.transform("size")
    X["is_s3"] = c.cand_id.str.startswith("S3-").to_numpy()
    return X.astype(np.float32)


def keep_mask(c, p, tau, cap):
    k = p >= tau
    if cap:
        r = pd.Series(p).groupby(c.s1_id.to_numpy()).rank(ascending=False, method="first").to_numpy()
        k &= r <= cap
    return k


def curve(c, p, n_true):
    tot_true = n_true.sum()
    rows = []
    for cap in CAPS:
        for tau in TAUS:
            k = keep_mask(c, p, tau, cap)
            tp_k = c.label.to_numpy()[k].sum()
            per = pd.Series(c.label.to_numpy() * k).groupby(c.s1_id.to_numpy()).sum().reindex(n_true.index, fill_value=0)
            r = np.where(n_true > 0, per / n_true.clip(lower=1), 1.0)
            f = np.where(n_true > 0, np.where(per > 0, 1.25 * r / (0.25 + r), 0.0), 1.0)
            rows.append({"cap": cap, "tau": tau, "recall": tp_k / tot_true, "oracle_F0.5": f.mean(),
                         "avg_cands": k.sum() / len(n_true)})
    return pd.DataFrame(rows)


def run_train(a):
    s1set = "all" if a.final_fit else "train"
    c = pd.read_parquet(cpath("train", s1set, a.tag))
    scope = c.s1_id.unique()
    if not a.final_fit and set(scope) & set(load_split_ids("holdout")):
        raise SystemExit("[STOP] holdout IDs in training candidates")
    gt, tp = true_pairs(scope)
    tp["label"] = 1
    c = c.merge(tp, how="left")
    c["label"] = c.label.fillna(0).astype("int8")
    c = c.sort_values(["s1_id", "rank"]).reset_index(drop=True)
    n_true = gt.set_index("source1_entity_id").matched_ids.map(len).reindex(scope)
    X = features(c)
    y, g = c.label.to_numpy(), c.s1_id.to_numpy()
    oof, iters = np.zeros(len(c)), []
    for tr, va in GroupKFold(n_splits=3).split(X, y, g):
        dtr = lgb.Dataset(X.iloc[tr], y[tr])
        b = lgb.train(PARAMS, dtr, 1000, valid_sets=[lgb.Dataset(X.iloc[va], y[va], reference=dtr)],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
        oof[va] = b.predict(X.iloc[va], num_iteration=b.best_iteration)
        iters.append(b.best_iteration)
    cur = curve(c, oof, n_true)
    base = cur.recall.max()
    ok = cur[cur.recall >= base - a.max_recall_loss].sort_values("avg_cands")
    best = ok.iloc[0]
    print(f"unpruned: recall {c.label.sum() / n_true.sum():.4f}, avg cands {len(c) / len(n_true):.1f}")
    print(cur.round(4).to_string(index=False))
    print(f"\nchosen (OOF): cap={int(best.cap)} tau={best.tau} -> recall {best.recall:.4f}, "
          f"oracle F0.5 {best['oracle_F0.5']:.4f}, avg cands {best.avg_cands:.1f}")

    final = lgb.train(PARAMS, lgb.Dataset(X, y), int(np.mean(iters) * 1.1))
    os.makedirs(os.path.join("models", a.tag), exist_ok=True)
    final.save_model(os.path.join("models", a.tag, "pruner.txt"))
    cfg = {"tau": float(best.tau), "cap": int(best.cap), "features": list(X.columns),
           "oof_recall": float(best.recall), "oof_avg_cands": float(best.avg_cands), "iters": iters}
    os.makedirs(os.path.join("experiments", a.tag), exist_ok=True)
    with open(os.path.join("experiments", a.tag, "pruner.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    # training candidates for the matcher are pruned with OOF scores (no in-sample optimism)
    k = keep_mask(c, oof, cfg["tau"], cfg["cap"])
    c.loc[k].drop(columns="label").to_parquet(cpath("train", s1set, a.tag + "p"), index=False)
    print(f"wrote {cpath('train', s1set, a.tag + 'p')} ({int(k.sum())} pairs)")


def run_apply(a):
    with open(os.path.join("experiments", a.tag, "pruner.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    b = lgb.Booster(model_file=os.path.join("models", a.tag, "pruner.txt"))
    c = pd.read_parquet(cpath(a.split, a.s1_set, a.tag)).sort_values(["s1_id", "rank"]).reset_index(drop=True)
    p = b.predict(features(c)[cfg["features"]])
    k = keep_mask(c, p, cfg["tau"], cfg["cap"])
    out = c.loc[k]
    out.to_parquet(cpath(a.split, a.s1_set, a.tag + "p"), index=False)
    n_s1 = c.s1_id.nunique()
    print(f"{a.split}/{a.s1_set}: {len(c)} -> {len(out)} pairs | avg cands {len(c) / n_s1:.1f} -> {len(out) / n_s1:.1f}")
    if a.split == "train":
        ids = load_split_ids(a.s1_set)
        gt, tp = true_pairs(ids)
        hit = tp.merge(out[["s1_id", "cand_id"]].assign(h=1), how="left").h.notna()
        hit0 = tp.merge(c[["s1_id", "cand_id"]].assign(h=1), how="left").h.notna()
        print(f"candidate recall {hit0.mean():.4f} -> {hit.mean():.4f}")
    print(f"wrote {cpath(a.split, a.s1_set, a.tag + 'p')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "apply"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--s1-set", default="holdout")
    ap.add_argument("--max-recall-loss", type=float, default=0.002)
    ap.add_argument("--final-fit", action="store_true", help="train on cand_train_all_<tag> (includes holdout)")
    a = ap.parse_args()
    {"train": run_train, "apply": run_apply}[a.mode](a)


if __name__ == "__main__":
    main()