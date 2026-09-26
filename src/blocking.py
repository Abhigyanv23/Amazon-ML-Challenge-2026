"""
src/blocking.py  (v3; v2 description below still applies, plus:)
  sk   : skeleton channel — consonant-skeleton trigrams of the name + address tokens, against pool
         records with NON-LATIN names (cross-script: Devanagari/Gujarati/... vs Latin), top --k-sk
  ad   : address-only channel against ALL pool records, top --k-ad
  rev  : reverse direction — each pool record keeps its top --k-rev S1 (word-channel vectors).
         For the train split this runs against ALL train S1 so holdout mirrors test.
  word channel also gets the composite house-number token (h_71/1).

(v2)
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
        "addr_norm", "addr_nums", "name_skel", "addr_hn"]
CHANNELS = [("score", "rank", "k"), ("blk_tri", "blk_tri_rank", "k_tri"), ("blk_nl", "blk_nl_rank", "k_nl"),
            ("blk_sk", "blk_sk_rank", "k_sk"), ("blk_ad", "blk_ad_rank", "k_ad"), ("blk_rev", "blk_rev_rank", "k_rev")]
BLOCK_MERGE = {"india": {"tg": "ap", "la": "jk"}}


def block_key(ck, st):
    return BLOCK_MERGE.get(ck, {}).get(st, st)


def make_docs(df):
    docs = []
    for nc, ns, an, nums, hn in zip(df.name_core, df.name_nospace, df.addr_norm, df.addr_nums, df.addr_hn):
        t = ["n_" + x for x in nc.split()]
        if ns:
            t.append("s_" + ns)
        t += ["a_" + x for x in an.split() if not x.isdigit()]
        t += ["d_" + x for x in nums.split()]
        if hn:
            t.append("h_" + hn)
        docs.append(t)
    return docs


def make_skel_docs(df):
    docs = []
    for sk, an, nums, hn in zip(df.name_skel, df.addr_norm, df.addr_nums, df.addr_hn):
        x = "#" + sk.replace(" ", "#") + "#" if sk else ""
        t = ["k_" + x[i:i + 3] for i in range(len(x) - 2)]
        t += ["a_" + y for y in an.split() if not y.isdigit()] + ["d_" + y for y in nums.split()]
        if hn:
            t.append("h_" + hn)
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
            + (["h_" + hn] if hn else []) for an, nums, hn in zip(df.addr_norm, df.addr_nums, df.addr_hn)]


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
    """Returns (A: S1 x V, B: pool x V), both L2-normalized CSR, IDF from the pool."""
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
    B = normalize(B @ D, norm="l2", copy=False).tocsr()
    A = normalize(vec.transform(s_docs) @ D, norm="l2", copy=False).tocsr()
    return A, B


def _pairs(M, s_ids, p_ids, col):
    M = M.tocoo()
    return pd.DataFrame({"s1_id": s_ids[M.row], "cand_id": p_ids[M.col], col: M.data.astype(np.float32)})


def _channel(s_blk, p_blk, make, k, max_df, n_threads, col, w=None):
    if len(p_blk) == 0 or len(s_blk) == 0 or k <= 0:
        return None
    p_docs = make(p_blk)
    if not any(p_docs):
        return None
    A, B = _vectorize(p_docs, make(s_blk), max_df, w)
    return _pairs(topk(A, B.T.tocsr(), k, n_threads), s_blk.entity_id.to_numpy(), p_blk.entity_id.to_numpy(), col)


def block_candidates(s_all, is_target, p_blk, a, w):
    """s_all: all S1 of the block (targets + references for the reverse channel)."""
    s_tgt = s_all[is_target]
    out = {}
    # word channel (forward) + reverse channel share the same vectors
    if len(p_blk) and len(s_all):
        A, B = _vectorize(make_docs(p_blk), make_docs(s_all), a.max_df, w)
        s_ids, p_ids = s_all.entity_id.to_numpy(), p_blk.entity_id.to_numpy()
        tgt_rows = np.flatnonzero(is_target)
        if len(tgt_rows):
            out["score"] = _pairs(topk(A[tgt_rows], B.T.tocsr(), a.k, a.threads), s_ids[tgt_rows], p_ids, "score")
        if a.k_rev > 0:
            R = topk(B, A.T.tocsr(), a.k_rev, a.threads).tocoo()      # pool -> S1
            rev = pd.DataFrame({"s1_id": s_ids[R.col], "cand_id": p_ids[R.row], "blk_rev": R.data.astype(np.float32)})
            out["blk_rev"] = rev[is_target[R.col]]
    out["blk_tri"] = _channel(s_tgt, p_blk, make_tri_docs, a.k_tri, a.max_df, a.threads, "blk_tri")
    p_nl = p_blk[p_blk.name_nonlatin == 1]
    out["blk_nl"] = _channel(s_tgt, p_nl, make_addr_docs, a.k_nl, 1.0, a.threads, "blk_nl")     # no token cap
    out["blk_sk"] = _channel(s_tgt, p_nl, make_skel_docs, a.k_sk, 1.0, a.threads, "blk_sk")
    out["blk_ad"] = _channel(s_tgt, p_blk, make_addr_docs, a.k_ad, a.max_df, a.threads, "blk_ad")
    return out


def run_blocking(split, s1_ids, a, w):
    pool = pd.concat([load_normalized(split, 2, COLS), load_normalized(split, 3, COLS)], ignore_index=True)
    s1 = load_normalized(split, 1, COLS)                       # ALL S1 (reverse channel references)
    tgt = np.ones(len(s1), bool) if s1_ids is None else s1.entity_id.isin(set(s1_ids)).to_numpy()
    s1["is_target"] = tgt
    pool["bk"] = [block_key(c, s) for c, s in zip(pool.country_key, pool.addr_state)]
    s1["bk"] = [block_key(c, s) for c, s in zip(s1.country_key, s1.addr_state)]
    parts = {col: [] for col, _, _ in CHANNELS}
    for ck, s_c in s1.groupby("country_key"):
        if not s_c.is_target.any():
            continue
        t = time.time()
        p_c = pool[pool.country_key == ck]
        groups = p_c.groupby("bk").indices
        nostate = p_c.iloc[groups.get("", [])]
        for bk, s_blk in s_c.groupby("bk"):
            if not s_blk.is_target.any():
                continue
            if bk:
                p_blk = pd.concat([p_c.iloc[groups[bk]], nostate]) if bk in groups else nostate
            else:
                p_blk = p_c
            for col, df in block_candidates(s_blk, s_blk.is_target.to_numpy(), p_blk, a, w).items():
                if df is not None and len(df):
                    parts[col].append(df)
        print(f"  {ck}: {int(s_c.is_target.sum())} target S1 ({len(s_c)} total) vs {len(p_c)} pool in {time.time() - t:.0f}s")

    c = None
    for col, rk, karg in CHANNELS:
        if not parts[col]:
            continue
        d = pd.concat(parts[col], ignore_index=True).drop_duplicates(["s1_id", "cand_id"])
        d = d.sort_values(["s1_id", col], ascending=[True, False])
        d[rk] = (d.groupby("s1_id").cumcount() + 1).astype("int16")
        c = d if c is None else c.merge(d, on=["s1_id", "cand_id"], how="outer")
    for col, rk, karg in CHANNELS:
        lim = getattr(a, karg)
        if col not in c:
            c[col], c[rk] = 0.0, lim + 1
        c[col] = c[col].fillna(0).astype(np.float32)
        c[rk] = c[rk].fillna(lim + 1).astype("int16")
    c = c.sort_values(["s1_id", "rank", "blk_tri_rank"]).reset_index(drop=True)
    return c, pool, s1[s1.is_target]


def evaluate(c, s1_ids, pool, s1, a):
    gt = load_ground_truth()
    gt = gt[gt.source1_entity_id.isin(set(s1_ids))]
    n_true = gt.set_index("source1_entity_id").matched_ids.map(len)
    tp = gt[["source1_entity_id", "matched_ids"]].explode("matched_ids").dropna()
    tp.columns = ["s1_id", "cand_id"]
    tp = tp.merge(c[["s1_id", "cand_id"] + [rk for _, rk, _ in CHANNELS]], how="left")
    found = tp["rank"].notna()
    by = {col: tp[rk].le(getattr(a, karg)) for col, rk, karg in CHANNELS}
    n_c = c.groupby("s1_id").size().reindex(n_true.index, fill_value=0)
    tp_k = found.groupby(tp.s1_id).sum().reindex(n_true.index, fill_value=0)
    r = np.where(n_true > 0, tp_k / n_true.clip(lower=1), 1.0)
    f = np.where(n_true > 0, np.where(tp_k > 0, 1.25 * r / (0.25 + r), 0.0), 1.0)
    res = {"n_s1": int(len(n_true)), "n_true_pairs": int(len(tp)),
           "candidate_recall_union": float(found.mean()), "oracle_F0.5": float(f.mean()),
           "avg_cands": float(n_c.mean()), "total_pairs": int(len(c))}
    for col, _, _ in CHANNELS:
        others = np.logical_or.reduce([by[o] for o, _, _ in CHANNELS if o != col])
        res[f"recall_{col}"] = float(by[col].mean())
        res[f"only_{col}"] = int((by[col] & ~others).sum())
    cn = s1.set_index("entity_id").country_key
    res["recall_by_country"] = found.groupby(cn.reindex(tp.s1_id).to_numpy()).mean().round(4).to_dict()
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
    ap.add_argument("--k-sk", type=int, default=5)
    ap.add_argument("--k-ad", type=int, default=5)
    ap.add_argument("--k-rev", type=int, default=2)
    ap.add_argument("--max-df", type=float, default=0.02)
    ap.add_argument("--w-name", type=float, default=1.0)
    ap.add_argument("--w-addr", type=float, default=1.0)
    ap.add_argument("--w-num", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--tag", default="v005")
    ap.add_argument("--final-fit", action="store_true", help="allow train --s1-set all (uses holdout for training)")
    a = ap.parse_args()
    if a.split == "test" and a.s1_set != "all":
        raise SystemExit("test split uses --s1-set all")
    if a.split == "train" and a.s1_set == "all" and not a.final_fit:
        raise SystemExit("train --s1-set all includes HOLDOUT labels: only allowed with --final-fit")
    if sp_matmul_topn is None:
        print("[WARN] sparse_dot_topn not installed -> slow fallback. pip install sparse_dot_topn")

    s1_ids = None
    if a.split == "train":
        if a.s1_set == "all":
            s1_ids = load_split_ids("train") + load_split_ids("holdout")      # final fit only
        else:
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