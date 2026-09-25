"""
src/diagnose_country.py
Unsupervised comparison of countries on the TEST set (no labels exist there).
Finds normalization/blocking gaps for France by comparing its statistics to US/India.

Usage: python src/diagnose_country.py --tag v001 > experiments/v001/country_diag.txt
"""
import argparse
import os
from collections import Counter

import pandas as pd

from data_loading import CACHE_DIR, load_normalized

COLS = ["entity_id", "country_key", "business_address", "name_core", "addr_norm", "addr_state", "addr_blank"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v001")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()

    srcs = {n: load_normalized(a.split, n, COLS) for n in (1, 2, 3)}

    print("== state found rate (non-blank addresses) ==")
    for n, df in srcs.items():
        d = df[df.addr_blank == 0]
        print(f"  source{n}: " + ", ".join(f"{c}={v:.1%}" for c, v in
                                          (d.addr_state != "").groupby(d.country_key).mean().items()))

    print("\n== France: most common LAST address component when no state was found (S2+S3) ==")
    pool = pd.concat([srcs[2], srcs[3]])
    fr = pool[(pool.country_key == "france") & (pool.addr_state == "") & (pool.addr_blank == 0)]
    last = Counter(x.split(",")[-1].strip().lower() for x in fr.business_address)
    print("  " + ", ".join(f"{k!r}:{v}" for k, v in last.most_common(40)))
    first = Counter(x.split(",")[0].strip().lower() for x in fr.business_address)
    print("\n  FIRST component: " + ", ".join(f"{k!r}:{v}" for k, v in first.most_common(25)))

    print("\n== France: most common address tokens NOT seen in US/India (abbreviation candidates) ==")
    fr_tok = Counter(t for x in pool[pool.country_key == "france"].addr_norm for t in set(x.split()))
    short = [(t, c) for t, c in fr_tok.most_common(400) if len(t) <= 4 and not t.isdigit()]
    print("  " + ", ".join(f"{t}:{c}" for t, c in short[:50]))

    print("\n== chain-like names: share of S1 whose name_core occurs >= 5 times in S1 ==")
    s1 = srcs[1]
    freq = s1.name_core.map(s1.name_core.value_counts())
    print("  " + ", ".join(f"{c}={v:.1%}" for c, v in (freq >= 5).groupby(s1.country_key).mean().items()))

    p = os.path.join(CACHE_DIR, f"cand_{a.split}_all_{a.tag}.parquet")
    if os.path.exists(p):
        c = pd.read_parquet(p, columns=["s1_id", "score", "rank"])
        top = c[c["rank"] == 1].merge(s1[["entity_id", "country_key"]], left_on="s1_id", right_on="entity_id")
        print("\n== blocking top-1 score by country (higher = easier) ==")
        print(top.groupby("country_key").score.describe(percentiles=[.1, .25, .5]).round(3).to_string())


if __name__ == "__main__":
    main()