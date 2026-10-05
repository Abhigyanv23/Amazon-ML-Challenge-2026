"""
src/features.py
Updated for v006: Added Overlap Coefficient (name_overlap) to handle acronyms
and subset entity names without penalizing length differences.
"""
import math
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from data_loading import load_normalized

STR_COLS = ["name_core", "name_nospace", "name_legal", "addr_norm", "addr_nums", "addr_state", "addr_zip",
            "name_skel", "addr_hn"]
REC_COLS = ["entity_id"] + STR_COLS + ["name_nonlatin", "addr_blank", "addr_state_src"]

PAIR_FEATS = [
    "name_exact", "nospace_exact", "name_ratio", "name_tset", "name_tsort", "name_partial",
    "nospace_jw", "nospace_lev", "name_jac", "name_overlap", "name_first_tok_eq", "name_tok_diff", "name_len_ratio",
    "legal_jac",
    "addr_ratio", "addr_tset", "addr_jac", "addr_len_ratio",
    "num_jac", "num_first_eq", "num_a_in_b",
    "state_eq", "zip_eq",
    "idf_match_a", "idf_match_b", "idf_matched_sum", "idf_b_total", "unmatched_a", "unmatched_b",
    "hn_edit", "hn_logdiff", "hn_len_eq", "addr_word_jac",
    "name_cos2", "name_cos3", "addr_cos3",
    "skel_cos3", "hn_comp_eq",
]
MAX_IDF = 16.0


def _jac(a, b):
    u = a | b
    return len(a & b) / len(u) if u else 0.0

def _overlap_coef(a, b):
    if not a or not b: return 0.0
    return len(a & b) / min(len(a), len(b))

def _ngrams(s, n):
    s = "#" + s + "#"
    return Counter(s[i:i + n] for i in range(len(s) - n + 1)) if len(s) >= n else Counter()


def _cos(ca, cb):
    if not ca or not cb:
        return -1.0
    dot = sum(v * cb.get(k, 0) for k, v in ca.items())
    return dot / math.sqrt(sum(v * v for v in ca.values()) * sum(v * v for v in cb.values()))


def _pair_feats(a, b):
    ac, an, al, aa, anum, ast, az, ask, ahn, aidf = a
    bc, bn, bl, ba, bnum, bst, bz, bsk, bhn, bidf = b
    at, bt = ac.split(), bc.split()
    f = [
        float(ac == bc), float(an == bn),
        fuzz.ratio(ac, bc) / 100, fuzz.token_set_ratio(ac, bc) / 100,
        fuzz.token_sort_ratio(ac, bc) / 100, fuzz.partial_ratio(ac, bc) / 100,
        JaroWinkler.normalized_similarity(an, bn), Levenshtein.normalized_similarity(an, bn),
        _jac(set(at), set(bt)), _overlap_coef(set(at), set(bt)), float(bool(at) and bool(bt) and at[0] == bt[0]),
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

    wa = dict(zip(at, aidf))
    wb = dict(zip(bt, bidf))
    common = set(wa) & set(wb)
    ta, tb = sum(wa.values()), sum(wb.values())
    ms = sum(wa[x] for x in common)
    f += [ms / ta if ta else 0.0, ms / tb if tb else 0.0, ms, tb,
          float(len(set(wa) - common)), float(len(set(wb) - common))]

    if na and nb:
        ha, hb = na[0], nb[0]
        f += [float(Levenshtein.distance(ha, hb)),
              math.log1p(abs(int(ha[:9]) - int(hb[:9]))), float(len(ha) == len(hb))]
    else:
        f += [-1.0, -1.0, -1.0]

    if aa and ba:
        wa_ = {x for x in aa.split() if not any(ch.isdigit() for ch in x)}
        wb_ = {x for x in ba.split() if not any(ch.isdigit() for ch in x)}
        f.append(_jac(wa_, wb_) if (wa_ or wb_) else -1.0)
    else:
        f.append(-1.0)

    f += [_cos(_ngrams(an, 2), _ngrams(bn, 2)), _cos(_ngrams(an, 3), _ngrams(bn, 3)),
          _cos(_ngrams(aa.replace(" ", ""), 3), _ngrams(ba.replace(" ", ""), 3)) if aa and ba else -1.0]
    f += [_cos(_ngrams(ask.replace(" ", ""), 3), _ngrams(bsk.replace(" ", ""), 3)) if ask and bsk else -1.0,
          float(ahn == bhn) if ahn and bhn else -1.0]
    return f


def _feat_chunk(args):
    A, B = args
    out = np.empty((len(A), len(PAIR_FEATS)), dtype=np.float32)
    for i, (a, b) in enumerate(zip(A, B)):
        out[i] = _pair_feats(a, b)
    return out


def iter_chunks(c, size):
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

        df = Counter()
        for x in self.pool.name_core:
            df.update(set(x.split()))
        n = len(self.pool)
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}

        s1_counts = self.s1.name_core.value_counts()
        pool_counts = self.pool.name_core.value_counts()
        self.b_name_in_s1 = np.log1p(self.pool.name_core.map(s1_counts).fillna(0).to_numpy(np.float32))
        self.a_name_in_pool = np.log1p(self.s1.name_core.map(pool_counts).fillna(0).to_numpy(np.float32))

    def _with_idf(self, rows):
        idf, mx = self.idf, MAX_IDF
        return [r + (tuple(idf.get(t, mx) for t in r[0].split()),) for r in rows]

    def build(self, c, workers=None, sub=20_000):
        ps = self.s1_idx.get_indexer(c.s1_id)
        pp = self.pool_idx.get_indexer(c.cand_id)
        if (ps < 0).any() or (pp < 0).any():
            raise ValueError("candidate IDs not found in normalized caches")
        A = self._with_idf(self.s1[STR_COLS].take(ps).itertuples(index=False, name=None))
        B = self._with_idf(self.pool[STR_COLS].take(pp).itertuples(index=False, name=None))
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
        X["b_name_in_s1"] = self.b_name_in_s1[pp]
        X["a_name_in_pool"] = self.a_name_in_pool[ps]
        X["idf_match_a_rank"] = X.groupby(g).idf_match_a.rank(ascending=False, method="min").astype(np.float32)
        X["comb"] = 0.5 * X.name_tset + 0.5 * X.addr_tset.clip(lower=0)
        for col in ["name_tset", "nospace_jw", "addr_tset", "comb"]:
            X[f"{col}_rank"] = X.groupby(g)[col].rank(ascending=False, method="min").astype(np.float32)
            X[f"{col}_gap"] = (X.groupby(g)[col].transform("max") - X[col]).astype(np.float32)
        return X.astype(np.float32)
