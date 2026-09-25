"""
src/blocking.py  (v2)
Candidate generation per (country, state-block). Pool records without a state join every
block of their country. Three channels, unioned:
  word : cosine of IDF-weighted tokens (name words, no-space name, address words, numbers), top --k
  tri  : cosine of IDF-weighted character trigrams of the no-space name (typos), top --k-tri
  nl   : address-only cosine against pool records with NON-LATIN names, top --k-nl
State blocks merged where source data mixes labels (India: AP+TG, JK+LA).
No labels are used; labels are only read afterwards to MEASURE recall.

Usage (repo root):
  python src/blocking.py --split train --s1-set holdout --tag v002
  python src/blocking.py --split train --s1-set train --sample 300000 --tag v002
  python src/blocking.py --split test --tag v002
Output: experiments/cache/cand_<split>_<s1set>_<tag>.parquet
  columns: s1_id, cand_id, score, rank (word channel; 0 / k+1 if absent), blk_tri, blk_tri_rank, blk_nl, blk_nl_rank
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

COLS = ["entity_id", "country_key", "addr_state", "name_core", "name_nospace", "name_nonlatin",
        "addr_norm", "addr_nums"]
BLOCK_MERGE = {"india": {"tg": "ap", "la": "jk"}}


def block_key(ck, st):
    return BLOCK_MERGE.get(ck, {}).get(st, st)


def make_docs(df):
    docs = []
    for nc, ns, an, nums in zip(df.name_core, df.name_nospace, df.addr_norm, df.addr_nums):
        t = ["n_" + x for x in nc.split()]
        if ns:
            t.append("s_" + ns)
        t += ["a_" + x for x in an.split() if not x.isdigit()]
        t += ["d_" + x for x in nums.split()]
        docs.append(t)
    return docs


def make_tri_docs(df):
    docs = []
    for ns, nl in zip(df.name_nospace, df.name_nonlatin):
        if nl or not ns:
            docs.append([])
            continue
        x = "#" + ns + "#"
        docs.append([x[i:i + 3] for i in range(len(x) - 2)])
    return docs


def make_addr_docs(df):
    return [["a_" + x for x in an.split() if not x.isdigit()] + ["d_" + x for x in nums.split()]
            for an, nums in zip(df.addr_norm, df.addr_nums)]


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


def _vectorize(p_docs, s_docs, max_df, w=None):
    vec = CountVectorizer(analyzer=_identity, binary=True, dtype=np.float32)
    B = vec.fit_transform(p_docs)
    N = B.shape[0]
    df = np.asarray(B.sum(axis=0)).ravel()
    idf = np.log((1 + N) / (1 + df)) + 1.0          # smoothed: stays >0 in tiny pools
    keep = df <= max(max_df * N, 100)
    colw = idf * keep
    if w is not None:
        pref = np.array([f[0] for f in vec.get_feature_names_out()])
        colw = colw * np.select([pref == "n", pref == "s", pref == "a"], [w["name"], w["name"], w["addr"]], w["num"])
    D = sp.diags(colw.astype(np.float32))
    B = normalize(B @ D, norm="l2", copy=False)
    A = normalize(vec.transform(s_docs) @ D, norm="l2", copy=False)
    return A.tocsr(), B.T.tocsr()


def _channel(s_blk, p_blk, s_docs, p_docs, k, max_df, n_threads, col, w=None):
    if len(p_blk) == 0 or k <= 0 or not any(p_docs):
        return None
    A, BT = _vectorize(p_docs, s_docs, max_df, w)
    M = topk(A, BT, k, n_threads).tocoo()
    return pd.DataFrame({"s1_id": s_blk.entity_id.to_numpy()[M.row],
                         "cand_id": p_blk.entity_id.to_numpy()[M.col],
                         col: M.data.astype(np.float32)})


def block_candidates(s_blk, p_blk, a, w):
    out = {}
    out["score"] = _channel(s_blk, p_blk, make_docs(s_blk), make_docs(p_blk), a.k, a.max_df, a.threads, "score", w)
    out["blk_tri"] = _channel(s_blk, p_blk, make_tri_docs(s_blk), make_tri_docs(p_blk), a.k_tri, a.max_df,
                              a.threads, "blk_tri")
    p_nl = p_blk[p_blk.name_nonlatin == 1]
    out["blk_nl"] = _channel(s_blk, p_nl, make_addr_docs(s_blk), make_addr_docs(p_nl), a.k_nl, a.max_df,
                             a.threads, "blk_nl")
    return out


def run_blocking(split, s1_ids, a, w):
    pool = pd.concat([load_normalized(split, 2, COLS), load_normalized(split, 3, COLS)], ignore_index=True)
    s1 = load_normalized(split, 1, COLS)
    if s1_ids is not None:
        s1 = s1[s1.entity_id.isin(set(s1_ids))]
    pool["bk"] = [block_key(c, s) for c, s in zip(pool.country_key, pool.addr_state)]
    s1["bk"] = [block_key(c, s) for c, s in zip(s1.country_key, s1.addr_state)]
    parts = {"score": [], "blk_tri": [], "blk_nl": []}
    for ck, s_c in s1.groupby("country_key"):
        t = time.time()
        p_c = pool[pool.country_key == ck]
        groups = p_c.groupby("bk").indices
        nostate = p_c.iloc[groups.get("", [])]
        for bk, s_blk in s_c.groupby("bk"):
            if bk:
                p_blk = pd.concat([p_c.iloc[groups[bk]], nostate]) if bk in groups else nostate
            else:
                p_blk = p_c
            for col, df in block_candidates(s_blk, p_blk, a, w).items():
                if df is not None:
                    parts[col].append(df)
        print(f"  {ck}: {len(s_c)} S1 vs {len(p_c)} pool in {time.time() - t:.0f}s")

    c = None
    for col, lim in [("score", a.k), ("blk_tri", a.k_tri), ("blk_nl", a.k_nl)]:
        if not parts[col]:
            continue
        d = pd.concat(parts[col], ignore_index=True)
        d = d.sort_values(["s1_id", col], ascending=[True, False])
        rk = "rank" if col == "score" else f"{col}_rank"
        d[rk] = (d.groupby("s1_id").cumcount() + 1).astype("int16")
        c = d if c is None else c.merge(d, on=["s1_id", "cand_id"], how="outer")
    for col, rk, lim in [("score", "rank", a.k), ("blk_tri", "blk_tri_rank", a.k_tri), ("blk_nl", "blk_nl_rank", a.k_nl)]:
        if col not in c:
            c[col], c[rk] = 0.0, lim + 1
        c[col] = c[col].fillna(0).astype(np.float32)
        c[rk] = c[rk].fillna(lim + 1).astype("int16")
    c = c.sort_values(["s1_id", "rank", "blk_tri_rank"]).reset_index(drop=True)
    return c, pool, s1


def evaluate(c, s1_ids, pool, s1, a):
    gt = load_ground_truth()
    gt = gt[gt.source1_entity_id.isin(set(s1_ids))]
    n_true = gt.set_index("source1_entity_id").matched_ids.map(len)
    tp = gt[["source1_entity_id", "matched_ids"]].explode("matched_ids").dropna()
    tp.columns = ["s1_id", "cand_id"]
    tp = tp.merge(c[["s1_id", "cand_id", "rank", "blk_tri_rank", "blk_nl_rank"]], how="left")
    found = tp["rank"].notna()
    by_w = tp["rank"].le(a.k)
    by_t = tp["blk_tri_rank"].le(a.k_tri)
    by_n = tp["blk_nl_rank"].le(a.k_nl)
    n_c = c.groupby("s1_id").size().reindex(n_true.index, fill_value=0)
    tp_k = found.groupby(tp.s1_id).sum().reindex(n_true.index, fill_value=0)
    r = np.where(n_true > 0, tp_k / n_true.clip(lower=1), 1.0)
    f = np.where(n_true > 0, np.where(tp_k > 0, 1.25 * r / (0.25 + r), 0.0), 1.0)
    res = {
        "n_s1": int(len(n_true)), "n_true_pairs": int(len(tp)),
        "candidate_recall_union": float(found.mean()),
        "oracle_F0.5": float(f.mean()),
        "avg_cands": float(n_c.mean()), "total_pairs": int(len(c)),
        "recall_word": float(by_w.mean()), "recall_tri": float(by_t.mean()), "recall_nl": float(by_n.mean()),
        "only_tri": int((by_t & ~by_w & ~by_n).sum()), "only_nl": int((by_n & ~by_w & ~by_t).sum()),
    }
    for k in (5, 10, 20):
        if k <= a.k:
            res[f"recall_word@{k}"] = float(tp["rank"].le(k).mean())
    cn = s1.set_index("entity_id").country_key
    ctry = cn.reindex(tp.s1_id).to_numpy()
    res["recall_by_country"] = found.groupby(ctry).mean().round(4).to_dict()
    miss = tp[~found]
    pbk = pool.set_index("entity_id").bk
    sbk = s1.set_index("entity_id").bk
    mc, ms = pbk.reindex(miss.cand_id).to_numpy(), sbk.reindex(miss.s1_id).to_numpy()
    res["missed_pairs"] = int(len(miss))
    res["missed_due_to_state_block"] = int(((mc != "") & (mc != ms)).sum())
    for kk, v in res.items():
        print(f"  {kk:<28} {v:.4f}" if isinstance(v, float) else f"  {kk:<28} {v}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--s1-set", choices=["holdout", "train", "all"], default="all")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--k-tri", type=int, default=10)
    ap.add_argument("--k-nl", type=int, default=5)
    ap.add_argument("--max-df", type=float, default=0.02)
    ap.add_argument("--w-name", type=float, default=1.0)
    ap.add_argument("--w-addr", type=float, default=1.0)
    ap.add_argument("--w-num", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--tag", default="v002")
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
    c, pool, s1 = run_blocking(a.split, s1_ids, a, w)
    out = os.path.join(CACHE_DIR, f"cand_{a.split}_{a.s1_set}_{a.tag}.parquet")
    c.to_parquet(out, index=False)
    print(f"\n{len(c)} candidate pairs in {time.time() - t:.0f}s -> {out}")

    if a.split == "train":
        res = evaluate(c, s1_ids, pool, s1, a)
        res["params"] = vars(a)
        os.makedirs(os.path.join("experiments", a.tag), exist_ok=True)
        jp = os.path.join("experiments", a.tag, f"blocking_{a.s1_set}.json")
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"  saved -> {jp}")


if __name__ == "__main__":
    main()