"""
src/check_submission.py
Local structural check of output files (rules from the challenge brief).
NOT a replacement for the organizers' utils/validate_submission.py - run both.

Usage: python src/check_submission.py [--out-dir output] [--test-dir dataset/test]
"""
import argparse
import csv
import os
import sys

import pandas as pd


def read(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)


def ids_of(v):
    return [x for x in v.split(",") if x != ""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--test-dir", default=os.path.join("dataset", "test"))
    a = ap.parse_args()
    errs = []

    s1 = set(read(os.path.join(a.test_dir, "test_source1.tsv")).entity_id)
    other = set(read(os.path.join(a.test_dir, "test_source2.tsv")).entity_id) | \
        set(read(os.path.join(a.test_dir, "test_source3.tsv")).entity_id)

    files = {"matching_results.tsv": "matched_entity_ids", "candidate_pairs.tsv": "candidate_entity_ids"}
    maps = {}
    for fn, col in files.items():
        p = os.path.join(a.out_dir, fn)
        if not os.path.exists(p):
            errs.append(f"{fn}: missing"); continue
        with open(p, encoding="utf-8") as f:
            header = f.readline().rstrip("\n")
        if header != f"source1_entity_id\t{col}":
            errs.append(f"{fn}: header {header!r} != 'source1_entity_id\\t{col}'")
        df = read(p)
        if len(df.columns) != 2:
            errs.append(f"{fn}: {len(df.columns)} columns (tab delimiter?)"); continue
        df.columns = ["s1", "ids"]
        if df.s1.duplicated().any():
            errs.append(f"{fn}: {int(df.s1.duplicated().sum())} duplicate source1 rows")
        if set(df.s1) != s1:
            errs.append(f"{fn}: S1 set mismatch (missing {len(s1 - set(df.s1))}, extra {len(set(df.s1) - s1)})")
        m, n_dup, n_pref, n_unk = {}, 0, 0, 0
        for k, v in zip(df.s1, df.ids):
            l = ids_of(v)
            n_dup += len(l) - len(set(l))
            n_pref += sum(not x.startswith(("S2-", "S3-")) for x in l)
            n_unk += sum(x not in other for x in l)
            m[k] = set(l)
        for cnt, msg in [(n_dup, "duplicate IDs in a list"), (n_pref, "IDs not S2-/S3-"), (n_unk, "IDs not in test S2/S3")]:
            if cnt:
                errs.append(f"{fn}: {cnt} {msg}")
        maps[fn] = m
        print(f"{fn}: {len(df)} rows, {sum(len(x) for x in m.values())} IDs, empty rows {sum(not x for x in m.values())}")

    if len(maps) == 2:
        bad = sum(len(v - maps["candidate_pairs.tsv"].get(k, set())) for k, v in maps["matching_results.tsv"].items())
        if bad:
            errs.append(f"{bad} matched IDs not in candidate_pairs")

    if errs:
        print("\nFAIL\n  " + "\n  ".join(errs)); sys.exit(1)
    print("\nPASS (local structural check)")


if __name__ == "__main__":
    main()