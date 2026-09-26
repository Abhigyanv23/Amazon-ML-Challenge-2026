"""
src/matcher.py
LightGBM pair classifier + decision logic tuned for per-entity macro F0.5.

  python src/matcher.py train   --tag v001 --k 20   # train-fold candidates only; OOF-tuned decision
  python src/matcher.py holdout --tag v001 --k 20   # score untouched holdout
  python src/matcher.py test    --tag v001 --k 20   # write output/*.tsv
  --cand-tag reuses another version's blocking candidates (e.g. --tag v004s1 --cand-tag v002)

v006 options (all optional; defaults reproduce earlier versions):
  --lr 0.05 --leaves 127 --folds 5 --max-rounds 6000   LightGBM settings
  --catboost [--cb-gpu]        also train CatBoost (Apache 2.0) on the same folds; blend weight tuned on OOF
  --cache-features             save feature matrices (train/holdout/test) to experiments/cache/X_*
  --reuse-features             load them instead of recomputing (same candidates + same feature set only)

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
from features import PAIR_FEATS, FeatureBuilder, iter_chunks
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


def xcache_dir(split, s1set, cand_tag):
    return os.path.join(CACHE_DIR, f"X_{split}_{s1set}_{cand_tag}_f{len(PAIR_FEATS)}")


def iter_features(fb, c, workers, split, s1set, cand_tag, cache=False, reuse=False):
    """Yields feature chunks aligned with iter_chunks(c); optionally cached as parquet parts."""
    d = xcache_dir(split, s1set, cand_tag)
    parts = sorted(os.listdir(d)) if (reuse and os.path.isdir(d)) else None
    if parts:
        print(f"  reusing cached features from {d}")
        n = 0
        for f in parts:
            X = pd.read_parquet(os.path.join(d, f))
            n += len(X)
            yield X
        if n != len(c):
            raise SystemExit(f"[STOP] cached features have {n} rows, candidates {len(c)} - delete {d}")
        return
    if fb is None:
        raise SystemExit(f"[STOP] no cached features in {d}")
    if cache:
        os.makedirs(d, exist_ok=True)
    for i, ch in enumerate(iter_chunks(c, CHUNK)):
        X = fb.build(ch, workers)
        if cache:
            X.to_parquet(os.path.join(d, f"part{i:04d}.parquet"), index=False)
        yield X


def featurize(fb, c, workers, split="train", s1set="train", cand_tag="", cache=False, reuse=False):
    parts, t = [], time.time()
    for X in iter_features(fb, c, workers, split, s1set, cand_tag, cache, reuse):
        parts.append(X)
        print(f"  features: {sum(len(p) for p in parts)}/{len(c)} ({time.time() - t:.0f}s)")
    return pd.concat(parts, ignore_index=True)


def predict_proba(models, X):
    booster, cb, w = models
    p = booster.predict(X)
    if cb is not None and w < 1.0:
        p = w * p + (1 - w) * cb.predict_proba(X)[:, 1]
    return p


def predict(fb, c, models, feats, workers, split="train", s1set="holdout", cand_tag="", cache=False, reuse=False):
    ps = []
    for X in iter_features(fb, c, workers, split, s1set, cand_tag, cache, reuse):
        ps.append(predict_proba(models, X[feats].to_numpy(np.float32)))
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
    s1set = "all" if a.final_fit else "train"
    c = load_cands("train", s1set, a.cand_tag or a.tag, a.k)
    scope = c.s1_id.unique()
    if not a.final_fit and set(scope) & set(load_split_ids("holdout")):
        raise SystemExit("[STOP] holdout IDs found in training candidates")
    gt, tp = true_pairs(scope)
    tp["label"] = 1
    c = c.merge(tp, how="left")
    c["label"] = c.label.fillna(0).astype("int8")
    c = c.sort_values(["s1_id", "blk_rank"]).reset_index(drop=True)
    n_true = gt.set_index("source1_entity_id").matched_ids.map(len).reindex(scope)
    print(f"train: {len(scope)} S1, {len(c)} pairs, positive rate {c.label.mean():.3f}")

    cand_tag = a.cand_tag or a.tag
    reuse_ok = a.reuse_features and os.path.isdir(xcache_dir("train", s1set, cand_tag))
    X = featurize(None if reuse_ok else FeatureBuilder("train"), c, workers, "train", s1set, cand_tag,
                  a.cache_features, a.reuse_features)
    params = dict(PARAMS, learning_rate=a.lr, num_leaves=a.leaves)
    feats, y, groups = list(X.columns), c.label.to_numpy(), c.s1_id.to_numpy()
    Xn = X.to_numpy(np.float32)
    del X
    oof, iters = np.zeros(len(c), np.float32), []
    oof_cb, cb_iters = (np.zeros(len(c), np.float32), []) if a.catboost else (None, [])
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=a.folds).split(Xn, y, groups)):
        t = time.time()
        dtr = lgb.Dataset(Xn[tr], y[tr], feature_name=feats)
        dva = lgb.Dataset(Xn[va], y[va], reference=dtr)
        b = lgb.train(params, dtr, a.max_rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(500)])
        oof[va] = b.predict(Xn[va], num_iteration=b.best_iteration)
        iters.append(b.best_iteration)
        msg = f"  fold {fold}: lgb best_iter {b.best_iteration}"
        if a.catboost:
            from catboost import CatBoostClassifier
            cb = CatBoostClassifier(**cb_params(a))
            cb.fit(Xn[tr], y[tr], eval_set=(Xn[va], y[va]), early_stopping_rounds=100, verbose=500)
            oof_cb[va] = cb.predict_proba(Xn[va])[:, 1]
            cb_iters.append(cb.get_best_iteration())
            msg += f", cb best_iter {cb.get_best_iteration()}"
        print(msg + f" ({time.time() - t:.0f}s)")

    w = 1.0
    if a.catboost:
        print("  blend weight search (w * lgb + (1-w) * catboost) on OOF:")
        best = None
        for wi in np.round(np.arange(0.0, 1.01, 0.25), 2):
            c["p"] = wi * oof + (1 - wi) * oof_cb
            f_ = macro_f05(decide(c, 0.7, 0.7), n_true)
            print(f"    w={wi:.2f}  F0.5@0.7 {f_:.4f}")
            if best is None or f_ > best[0]:
                best = (f_, wi)
        w = float(best[1])
        c["p"] = w * oof + (1 - w) * oof_cb
    else:
        c["p"] = oof
    save_p(c, "train", "train", a.tag)                  # OOF probabilities (for stage 2)
    best_f, t, t_top = tune(c, n_true)

    os.makedirs(os.path.join("models", a.tag), exist_ok=True)
    final = lgb.train(params, lgb.Dataset(Xn, y, feature_name=feats), int(np.mean(iters) * 1.1))
    final.save_model(os.path.join("models", a.tag, "lgbm.txt"))
    if a.catboost and w < 1.0:
        from catboost import CatBoostClassifier
        cbp = dict(cb_params(a), iterations=int(np.mean(cb_iters) * 1.1))
        cbf = CatBoostClassifier(**cbp)
        cbf.fit(Xn, y, verbose=500)
        cbf.save_model(os.path.join("models", a.tag, "catboost.cbm"))
    imp = pd.Series(final.feature_importance("gain"), index=feats).sort_values(ascending=False)
    print("  top features (gain):\n" + (imp.head(15) / imp.sum()).round(3).to_string())

    os.makedirs(os.path.join("experiments", a.tag), exist_ok=True)
    dec = {"t": t, "t_top": t_top, "k": a.k, "oof_F0.5": best_f, "features": feats,
           "iters": iters, "params": params, "folds": a.folds, "n_train_s1": int(len(scope)),
           "n_train_pairs": int(len(c)), "cand_tag": cand_tag,
           "catboost": bool(a.catboost and w < 1.0), "blend_w": w, "cb_iters": cb_iters}
    with open(os.path.join("experiments", a.tag, "decision.json"), "w", encoding="utf-8") as f:
        json.dump(dec, f, indent=2)
    print(f"saved models/{a.tag}/ and experiments/{a.tag}/decision.json")


def cb_params(a):
    p = dict(iterations=a.max_rounds, learning_rate=max(a.lr, 0.05), depth=8, loss_function="Logloss",
             random_seed=0, thread_count=os.cpu_count())
    if a.cb_gpu:
        p.update(task_type="GPU", devices="0")
    return p


def load_model(tag):
    with open(os.path.join("experiments", tag, "decision.json"), encoding="utf-8") as f:
        dec = json.load(f)
    booster = lgb.Booster(model_file=os.path.join("models", tag, "lgbm.txt"))
    cb, w = None, float(dec.get("blend_w", 1.0))
    if dec.get("catboost"):
        from catboost import CatBoostClassifier
        cb = CatBoostClassifier()
        cb.load_model(os.path.join("models", tag, "catboost.cbm"))
    return (booster, cb, w), dec


def run_holdout(a, workers):
    models, dec = load_model(a.tag)
    cand_tag = a.cand_tag or a.tag
    c = load_cands("train", "holdout", cand_tag, a.k)
    reuse_ok = a.reuse_features and os.path.isdir(xcache_dir("train", "holdout", cand_tag))
    c = predict(None if reuse_ok else FeatureBuilder("train"), c, models, dec["features"], workers,
                "train", "holdout", cand_tag, a.cache_features, a.reuse_features)
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
    models, dec = load_model(a.tag)
    cand_tag = a.cand_tag or a.tag
    c = load_cands("test", "all", cand_tag, a.k)
    reuse_ok = a.reuse_features and os.path.isdir(xcache_dir("test", "all", cand_tag))
    c = predict(None if reuse_ok else FeatureBuilder("test"), c, models, dec["features"], workers,
                "test", "all", cand_tag, a.cache_features, a.reuse_features)
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
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--leaves", type=int, default=127)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--max-rounds", type=int, default=2000)
    ap.add_argument("--catboost", action="store_true")
    ap.add_argument("--cb-gpu", action="store_true")
    ap.add_argument("--cache-features", action="store_true")
    ap.add_argument("--reuse-features", action="store_true")
    ap.add_argument("--final-fit", action="store_true", help="train on cand_train_all_<tag> (includes holdout)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    t = time.time()
    with Pool(a.jobs) as workers:
        {"train": run_train, "holdout": run_holdout, "test": run_test}[a.mode](a, workers)
    print(f"done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()