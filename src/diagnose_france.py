"""
src/diagnose_france.py
Unsupervised check of stage-1 probabilities on TEST (France vs India vs US), with the labelled
holdout (India/US) as a reference for what "healthy" distributions look like. No labels are used on test.

Usage: python src/diagnose_france.py --stage1 v004s1 > experiments/v004/france_diag.txt
"""
import argparse

import numpy as np
import pandas as pd

from data_loading import load_normalized
from matcher import p1_path


def stats(c, country):
    c = c.assign(country=country.reindex(c.s1_id).to_numpy())
    g = c.groupby("s1_id")
    s = pd.DataFrame({"country": g.country.first(), "top": g.p.max(),
                      "n_ge50": g.p.apply(lambda x: (x >= 0.5).sum()),
                      "n_mid": g.p.apply(lambda x: ((x >= 0.2) & (x < 0.8)).sum()),
                      "second": g.p.apply(lambda x: x.nlargest(2).iloc[-1] if len(x) > 1 else 0.0)})
    s["tie"] = (s.top >= 0.5) & (s.second >= 0.5) & ((s.top - s.second) < 0.1)
    out = s.groupby("country").agg(
        n=("top", "size"),
        top_below_0_3=("top", lambda x: (x < 0.3).mean()),
        top_0_3_to_0_8=("top", lambda x: ((x >= 0.3) & (x < 0.8)).mean()),
        top_ge_0_95=("top", lambda x: (x >= 0.95).mean()),
        avg_cands_ge_0_5=("n_ge50", "mean"),
        avg_cands_0_2_to_0_8=("n_mid", "mean"),
        share_s1_with_any_mid=("n_mid", lambda x: (x > 0).mean()))
    pairs = c.groupby("country").p.apply(lambda x: ((x >= 0.2) & (x < 0.8)).mean()).rename("pair_share_0_2_to_0_8")
    return out.join(pairs).round(4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default="v004s1")
    a = ap.parse_args()
    for split, s1set, name in [("train", "holdout", "HOLDOUT (India/US, labelled reference)"),
                               ("test", "all", "TEST (France/India/US)")]:
        c = pd.read_parquet(p1_path(split, s1set, a.stage1), columns=["s1_id", "cand_id", "p"])
        country = load_normalized(split, 1, ["entity_id", "country_key"]).set_index("entity_id").country_key
        print(f"\n=== {name} ===")
        print(stats(c, country).T.to_string())
    print("\nRead: if France shows many more S1 with top p in 0.3-0.8, or more mid-band candidates,"
          "\nthe model is uncertain on France (matcher/decision issue). Similar numbers -> errors are confident ones.")


if __name__ == "__main__":
    main()