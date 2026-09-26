"""
src/stage2.py  (v003)
Stage 2: group-consistency re-scoring. For each candidate, measure agreement with the OTHER
confident candidates of the same S1 (name, no-space name, address, numbers, zip, state).
True matches of one business resemble each other (e.g. a Devanagari-named record sharing an
address with confident Latin-named matches). A second LightGBM combines stage-1 probability
with these support features.

Inputs: stage-1 probabilities written by matcher.py (experiments/cache/p1_*_<stage1>.parquet).
  python src/stage2.py train   --stage1 v002 --tag v003   # train-fold OOF p1 only; OOF-tuned decision
  python src/stage2.py holdout --stage1 v002 --tag v003
  python src/stage2.py test    --stage1 v002 --tag v003
Holdout labels are never used for training or tuning.
v004: optional fallback — S1 groups with <= FB_MAX stage-1 probabilities >= P_MIN use the stage-1
probability instead of stage 2 (stage 2 has no support signal there). Chosen on OOF only.
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from sklearn.model_selection import GroupKFold

from data_loading import load_normalized, load_split_ids
from features import iter_chunks
from matcher import PARAMS, decide, p1_path, to_map, true_pairs, tune, write_tsv
from validate_local import print_metrics, score

POOL_COLS = ["entity_id", "name_core", "name_nospace", "addr_norm", "addr_nums", "addr_zip", "addr_state"]
M_TOP, P_MIN = 6, 0.3
FB_MAX = 1
SUP_FEATS = ["k_size", "sup_name_max", "sup_nos_max", "sup_addr_max", "sup_num_max", "sup_zip_max",
             "sup_state_max", "sup_name_wmean", "sup_addr_wmean", "sup_best", "sup_strong"]
CHUNK = 1_000_000


def _support_chunk(args):
    """recs/p are sorted by s1 then p desc; starts = group offsets (last = len)."""
    recs, p, starts = args
    out = np.full((len(recs), len(SUP_FEATS)), -1.0, dtype=np.float32)
    for g in range(len(starts) - 1):
        s, e = starts[g], starts[g + 1]
        conf = [j for j in range(s, min(e, s + M_TOP + 1)) if p[j] >= P_MIN]
        for i in range(s, e):
            K = [j for j in conf if j != i][:M_TOP]
            if not K:
                out[i, 0] = 0.0
                continue
            ci, ni, ai, mi, zi, sti = recs[i]
            sn, so, sa, sm, sz, ss, w = [], [], [], [], [], [], []
            for j in K:
                cj, nj, aj, mj, zj, stj = recs[j]
                sn.append(fuzz.token_set_ratio(ci, cj) / 100)
                so.append(JaroWinkler.normalized_similarity(ni, nj))
                sa.append(fuzz.token_set_ratio(ai, aj) / 100 if ai and aj else -1.0)
                if mi and mj:
                    x, y = set(mi.split()), set(mj.split())
                    sm.append(len(x & y) / len(x | y))
                else:
                    sm.append(-1.0)
                sz.append(float(bool(set(zi.split()) & set(zj.split()))) if zi and zj else -1.0)
                ss.append(float(sti == stj) if sti and stj else -1.0)
                w.append(p[j])
            wsum = sum(w)
            out[i] = [
                len(K), max(sn), max(so), max(sa), max(sm), max(sz), max(ss),
                sum(a * b for a, b in zip(w, sn)) / wsum,
                sum(a * max(b, 0.0) for a, b in zip(w, sa)) / wsum,
                max(a * max(b, c) for a, b, c in zip(w, sn, sa)),
                float(sum((x >= 0.9) or (y >= 0.9 and z == 1.0) for x, y, z in zip(sn, sa, sm))),
            ]
    return out


class Stage2Builder:
    def __init__(self, split):
        self.pool = pd.concat([load_normalized(split, 2, POOL_COLS), load_normalized(split, 3, POOL_COLS)],
                              ignore_index=True)
        self.idx = pd.Index(self.pool.entity_id)

    def build(self, c, workers, groups_per_job=4000):
        pp = self.idx.get_indexer(c.cand_id)
        if (pp < 0).any():
            raise ValueError("candidate IDs not in caches")
        recs = list(self.pool[POOL_COLS[1:]].take(pp).itertuples(index=False, name=None))
        p = c.p.to_numpy(np.float64)
        s = c.s1_id.to_numpy()
        starts = np.r_[np.flatnonzero(np.r_[True, s[1:] != s[:-1]]), len(s)]
        jobs = []
        for g0 in range(0, len(starts) - 1, groups_per_job):
            g1 = min(g0 + groups_per_job, len(starts) - 1)
            a, b = starts[g0], starts[g1]
            jobs.append((recs[a:b], p[a:b], (starts[g0:g1 + 1] - a).tolist()))
        X = pd.DataFrame(np.vstack(workers.map(_support_chunk, jobs)), columns=SUP_FEATS)

        size = np.diff(starts)
        first = p[starts[:-1]]
        second = np.where(size > 1, p[np.minimum(starts[:-1] + 1, len(p) - 1)], 0.0)
        X["p1"] = p
        X["p1_rank"] = np.concatenate([np.arange(1, n + 1) for n in size])
        X["p1_gap"] = np.repeat(first, size) - p
        X["p1_second"] = np.repeat(second, size)
        X["n_ge50"] = np.repeat(np.add.reduceat(p >= 0.5, starts[:-1]), size)
        X["n_ge30"] = np.repeat(np.add.reduceat(p >= 0.3, starts[:-1]), size)
        X["p1_sum"] = np.repeat(np.add.reduceat(p, starts[:-1]), size)
        X["n_cands"] = np.repeat(size, size)
        X["is_s3"] = c.cand_id.str.startswith("S3-").to_numpy()
        return X.astype(np.float32)


def apply_fallback(c, p2):
    """c has stage-1 p in column p1; returns final probabilities."""
    n_conf = (c.p1 >= P_MIN).groupby(c.s1_id).transform("sum").to_numpy()
    return np.where(n_conf <= FB_MAX, c.p1.to_numpy(), p2).astype(np.float32)


def load_p1(split, s1set, stage1):
    c = pd.read_parquet(p1_path(split, s1set, stage1))
    return c.sort_values(["s1_id", "p"], ascending=[True, False], kind="stable").reset_index(drop=True)


def featurize(sb, c, workers):
    parts = []
    for ch in iter_chunks(c, CHUNK):
        t = time.time()
        parts.append(sb.build(ch, workers))
        print(f"  stage2 features: {sum(len(x) for x in parts)}/{len(c)} ({time.time() - t:.0f}s/chunk)")
    return pd.concat(parts, ignore_index=True)


def run_train(a, workers):
    c = load_p1("train", "train", a.stage1)
    if "label" not in c.columns:
        raise SystemExit("train p1 file has no labels - re-run matcher.py train with the latest matcher.py")
    scope = c.s1_id.unique()
    if not a.final_fit and set(scope) & set(load_split_ids("holdout")):
        raise SystemExit("[STOP] holdout IDs found in training data")
    gt, _ = true_pairs(scope)
    n_true = gt.set_index("source1_entity_id").matched_ids.map(len).reindex(scope)
    print(f"stage2 train: {len(scope)} S1, {len(c)} pairs")

    X = featurize(Stage2Builder("train"), c, workers)
    feats, y, groups = list(X.columns), c.label.to_numpy(), c.s1_id.to_numpy()
    oof, iters = np.zeros(len(c), np.float32), []
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=3).split(X, y, groups)):
        t = time.time()
        dtr = lgb.Dataset(X.iloc[tr], y[tr])
        b = lgb.train(PARAMS, dtr, 2000, valid_sets=[lgb.Dataset(X.iloc[va], y[va], reference=dtr)],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)])
        oof[va] = b.predict(X.iloc[va], num_iteration=b.best_iteration)
        iters.append(b.best_iteration)
        print(f"  fold {fold}: best_iter {b.best_iteration} ({time.time() - t:.0f}s)")
    c1 = c.copy()
    print("stage-1 decision on same rows (reference):")
    tune(c1, n_true)
    c["p1"] = c.p
    c["p"] = oof
    print("stage-2:")
    best_plain = tune(c, n_true)
    cf = c.copy()
    cf["p"] = apply_fallback(c, oof)
    print(f"stage-2 with fallback to stage-1 when <= {FB_MAX} candidate has p1 >= {P_MIN}:")
    best_fb = tune(cf, n_true)
    fallback = best_fb[0] > best_plain[0]
    best_f, t, t_top = best_fb if fallback else best_plain
    print(f"  -> using {'FALLBACK' if fallback else 'plain stage 2'} (OOF {best_f:.4f})")

    final = lgb.train(PARAMS, lgb.Dataset(X, y), int(np.mean(iters) * 1.1))
    os.makedirs(os.path.join("models", a.tag), exist_ok=True)
    final.save_model(os.path.join("models", a.tag, "lgbm_stage2.txt"))
    imp = pd.Series(final.feature_importance("gain"), index=feats).sort_values(ascending=False)
    print("  top features (gain):\n" + (imp.head(12) / imp.sum()).round(3).to_string())
    os.makedirs(os.path.join("experiments", a.tag), exist_ok=True)
    with open(os.path.join("experiments", a.tag, "decision.json"), "w", encoding="utf-8") as f:
        json.dump({"stage1": a.stage1, "t": t, "t_top": t_top, "oof_F0.5": best_f, "features": feats,
                   "fallback": bool(fallback), "FB_MAX": FB_MAX,
                   "oof_plain": best_plain[0], "oof_fallback": best_fb[0],
                   "iters": iters, "M_TOP": M_TOP, "P_MIN": P_MIN}, f, indent=2)
    print(f"saved models/{a.tag}/lgbm_stage2.txt and experiments/{a.tag}/decision.json")


def _predict(a, split, s1set, workers):
    with open(os.path.join("experiments", a.tag, "decision.json"), encoding="utf-8") as f:
        dec = json.load(f)
    booster = lgb.Booster(model_file=os.path.join("models", a.tag, "lgbm_stage2.txt"))
    c = load_p1(split, s1set, dec["stage1"])
    sb = Stage2Builder(split)
    ps = []
    for ch in iter_chunks(c, CHUNK):
        ps.append(booster.predict(sb.build(ch, workers)[dec["features"]].to_numpy(np.float32)))
        print(f"  stage2 predicted {sum(len(x) for x in ps)}/{len(c)}")
    p2 = np.concatenate(ps).astype(np.float32)
    if dec.get("fallback"):
        c["p1"] = c.p
        c["p"] = apply_fallback(c, p2)
        c = c.drop(columns="p1")
    else:
        c["p"] = p2
    return c, decide(c, dec["t"], dec["t_top"]), dec


def run_holdout(a, workers):
    c, d, dec = _predict(a, "train", "holdout", workers)
    ids = load_split_ids("holdout")
    gt, _ = true_pairs(ids)
    gtd = {k: set(v) for k, v in zip(gt.source1_entity_id, gt.matched_ids)}
    s1 = load_normalized("train", 1, ["entity_id", "country_key"])
    m = score({k: set(v) for k, v in to_map(d).items()}, gtd, ids, dict(zip(s1.entity_id, s1.country_key)),
              {k: set(v) for k, v in to_map(c).items()})
    print(f"\nHOLDOUT stage2 (t={dec['t']}, t_top={dec['t_top']}):")
    print_metrics(m)
    out = os.path.join("experiments", a.tag)
    write_tsv(os.path.join(out, "holdout_matching_results.tsv"), ids, to_map(d), "matched_entity_ids")
    with open(os.path.join(out, "holdout_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, default=float)


def run_test(a, workers):
    c, d, dec = _predict(a, "test", "all", workers)
    s1 = load_normalized("test", 1, ["entity_id", "country_key"])
    ids = s1.entity_id.tolist()
    cmap, mmap = to_map(c), to_map(d)
    assert sum(len(set(v) - set(cmap.get(k, []))) for k, v in mmap.items()) == 0
    write_tsv(os.path.join("output", "candidate_pairs.tsv"), ids, cmap, "candidate_entity_ids")
    write_tsv(os.path.join("output", "matching_results.tsv"), ids, mmap, "matched_entity_ids")
    n = pd.Series({i: len(mmap.get(i, [])) for i in ids})
    ck = s1.set_index("entity_id").country_key.reindex(n.index).to_numpy()
    print(f"\nTEST: {len(ids)} S1 | empty {(n == 0).mean():.2%} | avg matches {n.mean():.2f}")
    print(pd.DataFrame({"avg_matches": n.groupby(ck).mean(), "empty_rate": (n == 0).groupby(ck).mean()}).round(3))
    print("wrote output/candidate_pairs.tsv and output/matching_results.tsv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "holdout", "test"])
    ap.add_argument("--stage1", default="v002")
    ap.add_argument("--tag", default="v003")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--final-fit", action="store_true", help="stage-1 OOF came from a final-fit run (includes holdout)")
    a = ap.parse_args()
    t = time.time()
    with Pool(a.jobs) as workers:
        {"train": run_train, "holdout": run_holdout, "test": run_test}[a.mode](a, workers)
    print(f"done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()