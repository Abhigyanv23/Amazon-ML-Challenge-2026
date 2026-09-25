"""
src/make_split.py
Creates a FROZEN holdout split of train Source 1 IDs. Run ONCE.
Stratified by (country, match-count bucket). Refuses to overwrite.

Usage: python src/make_split.py --data-dir dataset
Output: experiments/splits/train_s1_ids.txt, experiments/splits/holdout_s1_ids.txt
"""
import argparse
import csv
import os

import pandas as pd


def read_tsv(p):
    return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False,
                       quoting=csv.QUOTE_NONE, encoding="utf-8-sig")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=os.path.join("experiments", "splits"))
    a = ap.parse_args()

    tr_p = os.path.join(a.out_dir, "train_s1_ids.txt")
    ho_p = os.path.join(a.out_dir, "holdout_s1_ids.txt")
    if os.path.exists(ho_p):
        raise SystemExit(f"[STOP] {ho_p} already exists. Split is frozen; not overwriting.")

    s1 = read_tsv(os.path.join(a.data_dir, "train", "train_source1.tsv")).drop_duplicates("entity_id")
    gt = read_tsv(os.path.join(a.data_dir, "train", "train_ground_truth.tsv"))
    gt["n"] = gt["matched_entity_ids"].map(lambda s: len([x for x in s.split(",") if x.strip()]))

    df = s1[["entity_id", "country"]].merge(
        gt[["source1_entity_id", "n"]], left_on="entity_id", right_on="source1_entity_id", how="left")
    df["n"] = df["n"].fillna(0).astype(int)
    df["stratum"] = df["country"].str.strip() + "|" + df["n"].clip(upper=3).astype(str)

    ho = df.groupby("stratum", group_keys=False).sample(frac=a.holdout_frac, random_state=a.seed)
    tr = df[~df["entity_id"].isin(ho["entity_id"])]

    assert set(tr.entity_id).isdisjoint(ho.entity_id), "overlap between train and holdout"
    assert len(tr) + len(ho) == len(df), "rows lost"

    os.makedirs(a.out_dir, exist_ok=True)
    tr["entity_id"].sort_values().to_csv(tr_p, index=False, header=False)
    ho["entity_id"].sort_values().to_csv(ho_p, index=False, header=False)

    print(f"train S1: {len(tr)}   holdout S1: {len(ho)}")
    print(pd.crosstab(df["stratum"], df["entity_id"].isin(ho["entity_id"]).map({True: "holdout", False: "train"})))


if __name__ == "__main__":
    main()