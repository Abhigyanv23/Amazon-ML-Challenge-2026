"""
src/matcher.py
LightGBM pair classifier + decision logic tuned for per-entity macro F0.5.

  python src/matcher.py train   --tag v001 --k 20   # train-fold candidates only; OOF-tuned decision
  python src/matcher.py holdout --tag v001 --k 20   # score untouched holdout
  python src/matcher.py test    --tag v001 --k 20   # write output/*.tsv
  --cand-tag reuses another version's blocking candidates (e.g. --tag v004s1 --cand-tag v002)

Decision: keep pair if p >= t; each candidate goes to at most one S1 (highest p);
an S1 keeps matches only if its best p >= t_top (singleton gate).
Holdout labels are NEVER used for training or tuning.
"""
import argparse
import csv
import json
import os
import time
from multiprocessing import Pool

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from data_loading import CACHE_DIR, load_ground_truth, load_normalized, load_split_ids
from features import FeatureBuilder, iter_chunks
from validate_local import print_metrics, score

PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=100,
              feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=os.cpu_count(), verbose=-1, seed=0)
CHUNK = 1_000_000


def p1_path(split, s1set, tag):
    return os.path.join(CACHE_DIR, f"p1_{split}_{s1set}_{tag}.parquet")


def save_p(c, split, s1set, tag):
    cols = [x for x in ["s1_id", "cand_id", "p", "label"] if x in c.columns]
    c[cols].to_parquet(p1_path(split, s1set, tag), index=False)
    print(f"  saved probabilities -> {p1_path(split, s1set, tag)}")


def load_cands(split, s1set, tag, k):
    c = pd.read_parquet(os.path.join(CACHE_DIR, f"cand_{split}_{s1set}_{tag}.parquet"))
    if "blk_tri_rank" not in c.columns:          # v001 single-channel file: cut to top-k
        c = c[c["rank"] <= k]
    c = c.rename(columns={"score": "blk_score", "rank": "blk_rank"})
    return c.sort_values(["s1_id", "blk_rank"]).reset_index(drop=True)


def true_pairs(s1_ids):
    gt = load_ground_truth()
    gt = gt[gt.source1_entity_id.isin(set(s1_ids))]
    tp = gt[["source1_entity_id", "matched_ids"]].explode("matched_ids").dropna()
    tp.columns = ["s1_id", "cand_id"]
    return gt, tp


def featurize(fb, c, workers):
    parts = []
    for ch in iter_chunks(c, CHUNK):
        t = time.time()
        parts.append(fb.build(ch, workers))
        print(f"  features: {sum(len(p) for p in parts)}/{len(c)} ({time.time() - t:.0f}s/chunk)")
    return pd.concat(parts, ignore_index=True)


def predict(fb, c, booster, feats, workers):
    ps = []
    for ch in iter_chunks(c, CHUNK):
        X = fb.build(ch, workers)
        ps.append(booster.predict(X[feats].to_numpy(np.float32)))
        print(f"  predicted {sum(len(p) for p in ps)}/{len(c)}")
    c = c.copy()
    c["p"] = np.concatenate(ps).astype(np.float32)
    return c


def decide(c, t, t_top):
    d = c[c.p >= t].sort_values("p", ascending=False, kind="stable").drop_duplicates("cand_id")
    top = c.groupby("s1_id").p.max()
    return d[d.s1_id.map(top) >= t_top]


def macro_f05(d, n_true):
    g = d.groupby("s1_id").label.agg(["sum", "size"])
    tp = g["sum"].reindex(n_true.index, fill_value=0).to_numpy(float)
    npred = g["size"].reindex(n_true.index, fill_value=0).to_numpy(float)
    nt = n_true.to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(npred > 0, tp / npred, 0.0)
        r = np.where(nt > 0, tp / nt, 0.0)
        f = np.where(tp > 0, 1.25 * p * r / (0.25 * p + r), 0.0)
    f = np.where(nt == 0, (npred == 0).astype(float), f)
    return float(f.mean())


def tune(c, n_true):
    """Coarse grid (0.05) then fine grid (0.01) within +-0.05 of the coarse winner."""
    def search(ts, tops_fn):
        out = []
        for t in ts:
            for tt in tops_fn(t):
                out.append((macro_f05(decide(c, t, tt), n_true), float(t), float(tt)))
        return out

    coarse = np.round(np.arange(0.20, 0.96, 0.05), 2)
    res = search(coarse, lambda t: coarse[coarse >= t])
    res.sort(reverse=True)
    _, t0, tt0 = res[0]
    fine_t = np.round(np.arange(max(0.05, t0 - 0.05), min(0.99, t0 + 0.05) + 1e-9, 0.01), 2)
    fine_tt = np.round(np.arange(max(0.05, tt0 - 0.05), min(0.99, tt0 + 0.05) + 1e-9, 0.01), 2)
    res += search(fine_t, lambda t: fine_tt[fine_tt >= t])
    res = sorted(set(res), reverse=True)
    print("  top decision settings (OOF F0.5, t, t_top):")
    for r in res[:5]:
        print(f"    {r[0]:.4f}  t={r[1]:.2f}  t_top={r[2]:.2f}")
    return res[0]


def write_tsv(path, ids, mapping, col2):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writerow(["source1_entity_id", col2])
        for i in ids:
            w.writerow([i, ",".join(mapping.get(i, []))])


def to_map(df):
    return df.groupby("s1_id").cand_id.agg(list).to_dict()


# ------------------------------------------------------------------ modes
def run_train(a, workers):
    c = load_cands("train", "train", a.cand_tag or a.tag, a.k)
    scope = c.s1_id.unique()
    if set(scope) & set(load_split_ids("holdout")):
        raise SystemExit("[STOP] holdout IDs found in training candidates")
    gt, tp = true_pairs(scope)
    tp["label"] = 1
    c = c.merge(tp, how="left")
    c["label"] = c.label.fillna(0).astype("int8")
    c = c.sort_values(["s1_id", "blk_rank"]).reset_index(drop=True)
    n_true = gt.set_index("source1_entity_id").matched_ids.map(len).reindex(scope)
    print(f"train: {len(scope)} S1, {len(c)} pairs, positive rate {c.label.mean():.3f}")

    X = featurize(FeatureBuilder("train"), c, workers)
    feats, y, groups = list(X.columns), c.label.to_numpy(), c.s1_id.to_numpy()
    oof, iters = np.zeros(len(c), np.float32), []
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=3).split(X, y, groups)):
        t = time.time()
        dtr = lgb.Dataset(X.iloc[tr], y[tr])
        dva = lgb.Dataset(X.iloc[va], y[va], reference=dtr)
        b = lgb.train(PARAMS, dtr, 2000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)])
        oof[va] = b.predict(X.iloc[va], num_iteration=b.best_iteration)
        iters.append(b.best_iteration)
        print(f"  fold {fold}: best_iter {b.best_iteration} ({time.time() - t:.0f}s)")
    c["p"] = oof
    save_p(c, "train", "train", a.tag)                  # OOF probabilities (for stage 2)
    best_f, t, t_top = tune(c, n_true)

    final = lgb.train(PARAMS, lgb.Dataset(X, y), int(np.mean(iters) * 1.1))
    os.makedirs(os.path.join("models", a.tag), exist_ok=True)
    final.save_model(os.path.join("models", a.tag, "lgbm.txt"))
    imp = pd.Series(final.feature_importance("gain"), index=feats).sort_values(ascending=False)
    print("  top features (gain):\n" + (imp.head(15) / imp.sum()).round(3).to_string())

    os.makedirs(os.path.join("experiments", a.tag), exist_ok=True)
    dec = {"t": t, "t_top": t_top, "k": a.k, "oof_F0.5": best_f, "features": feats,
           "iters": iters, "params": PARAMS, "n_train_s1": int(len(scope)), "n_train_pairs": int(len(c))}
    with open(os.path.join("experiments", a.tag, "decision.json"), "w", encoding="utf-8") as f:
        json.dump(dec, f, indent=2)
    print(f"saved models/{a.tag}/lgbm.txt and experiments/{a.tag}/decision.json")


def load_model(tag):
    with open(os.path.join("experiments", tag, "decision.json"), encoding="utf-8") as f:
        dec = json.load(f)
    return lgb.Booster(model_file=os.path.join("models", tag, "lgbm.txt")), dec


def run_holdout(a, workers):
    booster, dec = load_model(a.tag)
    c = load_cands("train", "holdout", a.cand_tag or a.tag, a.k)
    c = predict(FeatureBuilder("train"), c, booster, dec["features"], workers)
    save_p(c, "train", "holdout", a.tag)
    d = decide(c, dec["t"], dec["t_top"])

    ids = load_split_ids("holdout")
    gt, _ = true_pairs(ids)
    gtd = {k: set(v) for k, v in zip(gt.source1_entity_id, gt.matched_ids)}
    s1 = load_normalized("train", 1, ["entity_id", "country_key"])
    country = dict(zip(s1.entity_id, s1.country_key))
    pred = {k: set(v) for k, v in to_map(d).items()}
    cand = {k: set(v) for k, v in to_map(c).items()}
    m = score(pred, gtd, ids, country, cand)
    print(f"\nHOLDOUT (t={dec['t']}, t_top={dec['t_top']}, k={a.k}):")
    print_metrics(m)

    out = os.path.join("experiments", a.tag)
    write_tsv(os.path.join(out, "holdout_matching_results.tsv"), ids, to_map(d), "matched_entity_ids")
    write_tsv(os.path.join(out, "holdout_candidate_pairs.tsv"), ids, to_map(c), "candidate_entity_ids")
    with open(os.path.join(out, "holdout_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, default=float)


def run_test(a, workers):
    booster, dec = load_model(a.tag)
    c = load_cands("test", "all", a.cand_tag or a.tag, a.k)
    c = predict(FeatureBuilder("test"), c, booster, dec["features"], workers)
    save_p(c, "test", "all", a.tag)
    d = decide(c, dec["t"], dec["t_top"])

    s1 = load_normalized("test", 1, ["entity_id", "country_key"])
    ids = s1.entity_id.tolist()
    cmap, mmap = to_map(c), to_map(d)
    bad = sum(len(set(v) - set(cmap.get(k, []))) for k, v in mmap.items())
    assert bad == 0, "matches not subset of candidates"
    assert len(set(ids)) == len(ids), "duplicate S1 ids"
    write_tsv(os.path.join("output", "candidate_pairs.tsv"), ids, cmap, "candidate_entity_ids")
    write_tsv(os.path.join("output", "matching_results.tsv"), ids, mmap, "matched_entity_ids")

    n = pd.Series({i: len(mmap.get(i, [])) for i in ids})
    ck = s1.set_index("entity_id").country_key
    print(f"\nTEST: {len(ids)} S1 | empty {(n == 0).mean():.2%} | avg matches {n.mean():.2f}")
    print(n.groupby(ck.reindex(n.index).to_numpy()).agg(["mean", lambda x: (x == 0).mean()]).rename(
        columns={"mean": "avg_matches", "<lambda_0>": "empty_rate"}).round(3).to_string())
    print("wrote output/candidate_pairs.tsv and output/matching_results.tsv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "holdout", "test"])
    ap.add_argument("--tag", default="v001")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--cand-tag", default=None, help="blocking tag to reuse (default: --tag)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    t = time.time()
    with Pool(a.jobs) as workers:
        {"train": run_train, "holdout": run_holdout, "test": run_test}[a.mode](a, workers)
    print(f"done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()