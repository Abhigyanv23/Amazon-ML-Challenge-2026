"""
src/features.py  (v2: + zip_eq, b_state_inferred, n_name_close; passes through all blk_* columns)
Pair features for (S1, candidate). Inputs: normalized caches + candidate table
(s1_id, cand_id, blk_*). No labels are used. No country feature (France is unseen in train).
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from data_loading import load_normalized

STR_COLS = ["name_core", "name_nospace", "name_legal", "addr_norm", "addr_nums", "addr_state", "addr_zip"]
REC_COLS = ["entity_id"] + STR_COLS + ["name_nonlatin", "addr_blank", "addr_state_src"]

PAIR_FEATS = [
    "name_exact", "nospace_exact", "name_ratio", "name_tset", "name_tsort", "name_partial",
    "nospace_jw", "nospace_lev", "name_jac", "name_first_tok_eq", "name_tok_diff", "name_len_ratio",
    "legal_jac",
    "addr_ratio", "addr_tset", "addr_jac", "addr_len_ratio",
    "num_jac", "num_first_eq", "num_a_in_b",
    "state_eq", "zip_eq",
]


def _jac(a, b):
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def _pair_feats(a, b):
    ac, an, al, aa, anum, ast, az = a
    bc, bn, bl, ba, bnum, bst, bz = b
    at, bt = ac.split(), bc.split()
    f = [
        float(ac == bc), float(an == bn),
        fuzz.ratio(ac, bc) / 100, fuzz.token_set_ratio(ac, bc) / 100,
        fuzz.token_sort_ratio(ac, bc) / 100, fuzz.partial_ratio(ac, bc) / 100,
        JaroWinkler.normalized_similarity(an, bn), Levenshtein.normalized_similarity(an, bn),
        _jac(set(at), set(bt)), float(bool(at) and bool(bt) and at[0] == bt[0]),
        float(abs(len(at) - len(bt))), min(len(an), len(bn)) / max(len(an), len(bn), 1),
    ]
    la, lb = set(al.split()), set(bl.split())
    f.append(_jac(la, lb) if la and lb else -1.0)
    if aa and ba:
        f += [fuzz.ratio(aa, ba) / 100, fuzz.token_set_ratio(aa, ba) / 100,
              _jac(set(aa.split()), set(ba.split())), min(len(aa), len(ba)) / max(len(aa), len(ba))]
    else:
        f += [-1.0, -1.0, -1.0, -1.0]
    na, nb = anum.split(), bnum.split()
    if na and nb:
        sa, sb = set(na), set(nb)
        f += [_jac(sa, sb), float(na[0] == nb[0]), len(sa & sb) / len(sa)]
    else:
        f += [-1.0, -1.0, -1.0]
    f.append(-1.0 if not (ast and bst) else float(ast == bst))
    f.append(-1.0 if not (az and bz) else float(bool(set(az.split()) & set(bz.split()))))
    return f


def _feat_chunk(args):
    A, B = args
    out = np.empty((len(A), len(PAIR_FEATS)), dtype=np.float32)
    for i, (a, b) in enumerate(zip(A, B)):
        out[i] = _pair_feats(a, b)
    return out


def iter_chunks(c, size):
    """Slices of c (sorted by s1_id) that never split one S1's candidates."""
    s = c["s1_id"].to_numpy()
    start, n = 0, len(c)
    while start < n:
        end = min(start + size, n)
        while end < n and s[end] == s[end - 1]:
            end += 1
        yield c.iloc[start:end]
        start = end


class FeatureBuilder:
    def __init__(self, split):
        self.s1 = load_normalized(split, 1, REC_COLS)
        self.pool = pd.concat([load_normalized(split, 2, REC_COLS), load_normalized(split, 3, REC_COLS)],
                              ignore_index=True)
        self.s1_idx = pd.Index(self.s1.entity_id)
        self.pool_idx = pd.Index(self.pool.entity_id)
        self.s1_freq = np.log1p(self.s1.name_core.map(self.s1.name_core.value_counts()).to_numpy(np.float32))
        self.pool_freq = np.log1p(self.pool.name_core.map(self.pool.name_core.value_counts()).to_numpy(np.float32))
        self.pool_nonlatin = self.pool.name_nonlatin.to_numpy(np.float32)
        self.pool_blank = self.pool.addr_blank.to_numpy(np.float32)
        self.pool_state_inferred = (self.pool.addr_state_src.to_numpy() == 2).astype(np.float32)

    def build(self, c, workers=None, sub=20_000):
        ps = self.s1_idx.get_indexer(c.s1_id)
        pp = self.pool_idx.get_indexer(c.cand_id)
        if (ps < 0).any() or (pp < 0).any():
            raise ValueError("candidate IDs not found in normalized caches")
        A = list(self.s1[STR_COLS].take(ps).itertuples(index=False, name=None))
        B = list(self.pool[STR_COLS].take(pp).itertuples(index=False, name=None))
        jobs = [(A[i:i + sub], B[i:i + sub]) for i in range(0, len(A), sub)]
        parts = workers.map(_feat_chunk, jobs) if workers else [_feat_chunk(j) for j in jobs]
        X = pd.DataFrame(np.vstack(parts), columns=PAIR_FEATS)

        g = c.s1_id.to_numpy()
        for col in [x for x in c.columns if x.startswith("blk_")]:
            X[col] = c[col].to_numpy(np.float32)
        X["blk_rel"] = X.blk_score / X.groupby(g).blk_score.transform("max").clip(lower=1e-6)
        X["n_cands"] = X.groupby(g).blk_score.transform("size").astype(np.float32)
        X["is_s3"] = c.cand_id.str.startswith("S3-").to_numpy(np.float32)
        X["a_name_freq"] = self.s1_freq[ps]
        X["b_name_freq"] = self.pool_freq[pp]
        X["b_nonlatin"] = self.pool_nonlatin[pp]
        X["b_addr_blank"] = self.pool_blank[pp]
        X["b_state_inferred"] = self.pool_state_inferred[pp]
        X["n_name_close"] = (X.name_tset >= 0.9).groupby(g).transform("sum").astype(np.float32)
        X["comb"] = 0.5 * X.name_tset + 0.5 * X.addr_tset.clip(lower=0)
        for col in ["name_tset", "nospace_jw", "addr_tset", "comb"]:
            X[f"{col}_rank"] = X.groupby(g)[col].rank(ascending=False, method="min").astype(np.float32)
            X[f"{col}_gap"] = (X.groupby(g)[col].transform("max") - X[col]).astype(np.float32)
        return X.astype(np.float32)