"""
src/blocking.py
Candidate generation: per (country, state) block, score S2/S3 records by cosine of
IDF-weighted token vectors (name_core words, address words, address numbers),
keep top-k per S1. Pool records without a state join every block of their country.
No labels are used; labels are only read afterwards to MEASURE recall.

Usage (repo root):
  python src/blocking.py --split train --s1-set holdout --k 50 --tag v001      # measure recall
  python src/blocking.py --split train --s1-set train --sample 300000 --k 30 --tag v001   # model training candidates
  python src/blocking.py --split test --k 30 --tag v001
Output: experiments/cache/cand_<split>_<s1set>_<tag>.parquet  (s1_id, cand_id, score, rank)
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

from data_loading import CACHE_DIR, load_ground_truth, load_normalized, load_split_ids

try:
    from sparse_dot_topn import sp_matmul_topn
except ImportError:
    sp_matmul_topn = None

COLS = ["entity_id", "country_key", "addr_state", "name_core", "addr_norm", "addr_nums"]


def make_docs(df):
    docs = []
    for nc, an, nums in zip(df.name_core, df.addr_norm, df.addr_nums):
        t = ["n_" + x for x in nc.split()]
        t += ["a_" + x for x in an.split() if not x.isdigit()]
        t += ["d_" + x for x in nums.split()]
        docs.append(t)
    return docs


def _identity(x):
    return x


def topk(A, BT, k, n_threads):
    if sp_matmul_topn is not None:
        return sp_matmul_topn(A, BT, top_n=k, threshold=1e-9, sort=True, n_threads=n_threads)
    # fallback: chunked scipy product
    rows, cols, vals = [], [], []
    for s in range(0, A.shape[0], 2000):
        P = (A[s:s + 2000] @ BT).tocsr()
        for i in range(P.shape[0]):
            a, b = P.indptr[i], P.indptr[i + 1]
            if a == b:
                continue
            d, c = P.data[a:b], P.indices[a:b]
            if len(d) > k:
                sel = np.argpartition(-d, k)[:k]
                d, c = d[sel], c[sel]
            rows.append(np.full(len(d), s + i)); cols.append(c); vals.append(d)
    if not rows:
        return sp.csr_matrix((A.shape[0], BT.shape[1]), dtype=np.float32)
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(A.shape[0], BT.shape[1]))


def block_candidates(s_blk, p_blk, k, max_df, w, n_threads):
    vec = CountVectorizer(analyzer=_identity, binary=True, dtype=np.float32)
    B = vec.fit_transform(make_docs(p_blk))
    N = B.shape[0]
    df = np.asarray(B.sum(axis=0)).ravel()
    idf = np.log(N / df)
    keep = df <= max(max_df * N, 100)                    # drop very common tokens
    feats = vec.get_feature_names_out()
    pref = np.array([f[0] for f in feats])
    gw = np.where(pref == "n", w["name"], np.where(pref == "a", w["addr"], w["num"]))
    D = sp.diags((idf * gw * keep).astype(np.float32))
    B = normalize(B @ D, norm="l2", copy=False)
    A = normalize(vec.transform(make_docs(s_blk)) @ D, norm="l2", copy=False)
    M = topk(A.tocsr(), B.T.tocsr(), k, n_threads).tocoo()
    return pd.DataFrame({"s1_id": s_blk.entity_id.to_numpy()[M.row],
                         "cand_id": p_blk.entity_id.to_numpy()[M.col],
                         "score": M.data.astype(np.float32)})


def run_blocking(split, s1_ids, k, max_df, w, n_threads):
    pool = pd.concat([load_normalized(split, 2, COLS), load_normalized(split, 3, COLS)], ignore_index=True)
    s1 = load_normalized(split, 1, COLS)
    if s1_ids is not None:
        s1 = s1[s1.entity_id.isin(set(s1_ids))]
    parts = []
    for ck, s_c in s1.groupby("country_key"):
        t = time.time()
        p_c = pool[pool.country_key == ck]
        groups = p_c.groupby("addr_state").indices
        nostate = p_c.iloc[groups.get("", [])]
        for st, s_blk in s_c.groupby("addr_state"):
            if st:
                p_blk = pd.concat([p_c.iloc[groups[st]], nostate]) if st in groups else nostate
            else:
                p_blk = p_c
            if len(p_blk):
                parts.append(block_candidates(s_blk, p_blk, k, max_df, w, n_threads))
        print(f"  {ck}: {len(s_c)} S1 vs {len(p_c)} pool in {time.time() - t:.0f}s")
    c = pd.concat(parts, ignore_index=True)
    c = c.sort_values(["s1_id", "score"], ascending=[True, False])
    c["rank"] = (c.groupby("s1_id").cumcount() + 1).astype("int16")
    return c, pool


def evaluate(c, s1_ids, pool, ks):
    gt = load_ground_truth()
    gt = gt[gt.source1_entity_id.isin(set(s1_ids))]
    n_true = gt.set_index("source1_entity_id").matched_ids.map(len)
    tp = gt[["source1_entity_id", "matched_ids"]].explode("matched_ids").dropna()
    tp.columns = ["s1_id", "cand_id"]
    tp = tp.merge(c[["s1_id", "cand_id", "rank"]], how="left")
    n_c = c.groupby("s1_id").size().reindex(n_true.index, fill_value=0)
    res = {"n_s1": int(len(n_true)), "n_true_pairs": int(len(tp))}
    print(f"\n  {'k':>4} {'cand_recall':>12} {'oracle_F0.5':>12} {'avg_cands':>10} {'total_pairs':>12}")
    for k in ks:
        hit = tp["rank"].le(k)
        rec = float(hit.mean())
        tp_k = hit.groupby(tp.s1_id).sum().reindex(n_true.index, fill_value=0)
        r = np.where(n_true > 0, tp_k / n_true.clip(lower=1), 1.0)
        f = np.where(n_true > 0, np.where(tp_k > 0, 1.25 * r / (0.25 + r), 0.0), 1.0)
        avg = float(np.minimum(n_c, k).mean())
        res[f"k{k}"] = {"candidate_recall": rec, "oracle_F0.5": float(f.mean()), "avg_cands": avg,
                        "total_pairs": int(np.minimum(n_c, k).sum())}
        print(f"  {k:>4} {rec:>12.4f} {f.mean():>12.4f} {avg:>10.1f} {int(np.minimum(n_c, k).sum()):>12}")

    miss = tp[tp["rank"].isna()]
    st = pool.set_index("entity_id").addr_state
    s1st = load_normalized("train", 1, ["entity_id", "addr_state"]).set_index("entity_id").addr_state
    ms, mc = s1st.reindex(miss.s1_id).to_numpy(), st.reindex(miss.cand_id).to_numpy()
    lost_state = int(((mc != "") & (mc != ms)).sum())
    res["missed_pairs_at_max_k"] = int(len(miss))
    res["missed_due_to_state_partition"] = lost_state
    print(f"\n  missed at k={max(ks)}: {len(miss)} pairs, of which {lost_state} are in a different state block")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--s1-set", choices=["holdout", "train", "all"], default="all")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--max-df", type=float, default=0.02)
    ap.add_argument("--w-name", type=float, default=1.0)
    ap.add_argument("--w-addr", type=float, default=1.0)
    ap.add_argument("--w-num", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--tag", default="v001")
    a = ap.parse_args()
    if a.split == "test" and a.s1_set != "all":
        raise SystemExit("test split uses --s1-set all")
    if sp_matmul_topn is None:
        print("[WARN] sparse_dot_topn not installed -> slow fallback. pip install sparse_dot_topn")

    s1_ids = None
    if a.split == "train":
        if a.s1_set == "all":
            raise SystemExit("for train use --s1-set holdout or train (keeps holdout separate)")
        s1_ids = load_split_ids(a.s1_set)
        if a.sample:
            rng = np.random.default_rng(0)
            s1_ids = list(rng.choice(s1_ids, size=min(a.sample, len(s1_ids)), replace=False))

    t = time.time()
    w = {"name": a.w_name, "addr": a.w_addr, "num": a.w_num}
    c, pool = run_blocking(a.split, s1_ids, a.k, a.max_df, w, a.threads)
    out = os.path.join(CACHE_DIR, f"cand_{a.split}_{a.s1_set}_{a.tag}.parquet")
    c.to_parquet(out, index=False)
    print(f"\n{len(c)} candidate pairs in {time.time() - t:.0f}s -> {out}")

    if a.split == "train":
        ks = [k for k in (5, 10, 20, 30, 50, 100) if k <= a.k]
        res = evaluate(c, s1_ids, pool, ks)
        res["params"] = vars(a)
        os.makedirs(os.path.join("experiments", a.tag), exist_ok=True)
        jp = os.path.join("experiments", a.tag, f"blocking_{a.s1_set}.json")
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"  saved -> {jp}")


if __name__ == "__main__":
    main()