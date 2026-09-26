"""
src/combine_by_country.py — build a submission whose rows for some countries come from another version.
Each row keeps its OWN version's candidate list and matches together (matches ⊆ candidates per row).

Example (France from v005 = unpruned candidates, India/US from v006 = pruned):
  python src/combine_by_country.py --base experiments/v006/submission --override experiments/v005/submission --countries france
Writes output/matching_results.tsv and output/candidate_pairs.tsv.
"""
import argparse
import csv
import os

import pandas as pd

from data_loading import load_normalized


def read(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--override", required=True)
    ap.add_argument("--countries", nargs="+", default=["france"])
    ap.add_argument("--out-dir", default="output")
    a = ap.parse_args()

    s1 = load_normalized("test", 1, ["entity_id", "country_key"]).set_index("entity_id").country_key
    os.makedirs(a.out_dir, exist_ok=True)
    stats = {}
    for fn, col in [("matching_results.tsv", "matched_entity_ids"), ("candidate_pairs.tsv", "candidate_entity_ids")]:
        b = read(os.path.join(a.base, fn))
        o = read(os.path.join(a.override, fn)).set_index("source1_entity_id")[col]
        if list(b.columns) != ["source1_entity_id", col]:
            raise SystemExit(f"[STOP] unexpected columns in {fn}")
        ctry = s1.reindex(b.source1_entity_id).to_numpy()
        use = pd.Series(ctry).isin(a.countries).to_numpy()
        b.loc[use, col] = o.reindex(b.source1_entity_id[use]).fillna("").to_numpy()
        n = b[col].map(lambda v: len([x for x in v.split(",") if x]))
        stats[fn] = (n, ctry)
        b.to_csv(os.path.join(a.out_dir, fn), sep="\t", index=False, quoting=csv.QUOTE_NONE, lineterminator="\n")
        print(f"{fn}: {int(use.sum())} rows taken from override ({', '.join(a.countries)}), total IDs {int(n.sum())}")

    nm, ctry = stats["matching_results.tsv"]
    nc, _ = stats["candidate_pairs.tsv"]
    df = pd.DataFrame({"country": ctry, "matches": nm.to_numpy(), "cands": nc.to_numpy()})
    print(df.groupby("country").agg(avg_matches=("matches", "mean"), empty=("matches", lambda x: (x == 0).mean()),
                                    avg_cands=("cands", "mean"), max_cands=("cands", "max")).round(3).to_string())
    print(f"overall avg candidates per S1: {nc.mean():.2f}")


if __name__ == "__main__":
    main()
    