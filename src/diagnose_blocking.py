"""
src/diagnose_blocking.py
Why did true pairs miss the candidate set? Holdout only (labels used for analysis, not for blocking).

Usage: python src/diagnose_blocking.py --tag v001
"""
import argparse
import os

import numpy as np
import pandas as pd

from data_loading import CACHE_DIR, load_ground_truth, load_normalized, load_split_ids

F = ["entity_id", "country_key", "business_name", "business_address", "name_core", "addr_norm",
     "addr_nums", "addr_state", "addr_blank", "name_nonlatin"]


def overlap(x, y):
    return len(set(x.split()) & set(y.split()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v001")
    ap.add_argument("--examples", type=int, default=20)
    a = ap.parse_args()

    ids = set(load_split_ids("holdout"))
    gt = load_ground_truth()
    tp = gt[gt.source1_entity_id.isin(ids)][["source1_entity_id", "matched_ids"]].explode("matched_ids").dropna()
    tp.columns = ["s1_id", "cand_id"]
    c = pd.read_parquet(os.path.join(CACHE_DIR, f"cand_train_holdout_{a.tag}.parquet"), columns=["s1_id", "cand_id"])
    c["hit"] = True
    m = tp.merge(c, how="left")
    miss = m[m.hit.isna()][["s1_id", "cand_id"]].reset_index(drop=True)
    print(f"true pairs {len(tp)} | missed {len(miss)} ({len(miss) / len(tp):.2%})")

    s1 = load_normalized("train", 1, F).set_index("entity_id")
    pool = pd.concat([load_normalized("train", 2, F), load_normalized("train", 3, F)]).set_index("entity_id")
    A = s1.loc[miss.s1_id].reset_index(drop=True)
    B = pool.loc[miss.cand_id].reset_index(drop=True)

    d = pd.DataFrame({
        "country": A.country_key,
        "state_mismatch": (B.addr_state != "") & (B.addr_state != A.addr_state),
        "cand_blank_addr": B.addr_blank == 1,
        "cand_nonlatin_name": B.name_nonlatin == 1,
        "name_overlap": [overlap(x, y) for x, y in zip(A.name_core, B.name_core)],
        "addr_overlap": [overlap(x, y) for x, y in zip(A.addr_norm, B.addr_norm)],
        "num_overlap": [overlap(x, y) for x, y in zip(A.addr_nums, B.addr_nums)],
    })
    # exclusive categories, first match wins
    cat = np.select(
        [d.state_mismatch, d.cand_blank_addr & (d.name_overlap == 0), d.cand_nonlatin_name,
         (d.name_overlap == 0) & (d.addr_overlap == 0), d.name_overlap == 0, d.num_overlap == 0],
        ["state_mismatch", "blank_addr_no_name_overlap", "nonlatin_name", "no_token_overlap",
         "addr_only_overlap", "no_number_overlap"], default="outranked_despite_overlap")
    d["category"] = cat

    print("\nMissed pairs by category:")
    print(d.groupby(["category", "country"]).size().unstack(fill_value=0).assign(
        total=lambda x: x.sum(axis=1)).sort_values("total", ascending=False).to_string())
    print("\nnon-exclusive flags: " + ", ".join(
        f"{k}={int(d[k].sum())}" for k in ["state_mismatch", "cand_blank_addr", "cand_nonlatin_name"]))

    rng = np.random.default_rng(0)
    for cname in d.category.value_counts().index:
        idx = d.index[d.category == cname].to_numpy()
        pick = rng.choice(idx, size=min(3, len(idx)), replace=False)
        print(f"\n--- {cname} ---")
        for i in pick:
            print(f"  S1 [{A.addr_state[i]}] {A.business_name[i]!r} | {A.business_address[i]!r}")
            print(f"  -> [{B.addr_state[i]}] {B.business_name[i]!r} | {B.business_address[i]!r}")
            print(f"     core: {A.name_core[i]!r} vs {B.name_core[i]!r}")


if __name__ == "__main__":
    main()
    