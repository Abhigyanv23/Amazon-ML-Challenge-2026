"""
src/data_inspection.py

Phase 1 EDA - Amazon ML Challenge 2026, Business Entity Resolution.
Uses ONLY the competition-provided TSVs. No network access, no external data.

Usage (from project root):
    mkdir -p experiments/v000_eda
    python src/data_inspection.py --data-dir dataset | tee experiments/v000_eda/eda_report.txt
"""
import argparse
import csv
import os
import random
import re
import unicodedata
from collections import Counter

import pandas as pd

EXPECTED_COLS = ["entity_id", "business_name", "business_address", "country"]
GT_COLS = ["source1_entity_id", "matched_entity_ids"]
PLACEHOLDERS = {"nan", "null", "none", "n/a", "na", "-", "--", "?", "0", "unknown"}


# ---------------------------------------------------------------- helpers
def hr(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def light_norm(s):
    """Diagnostic-only normalization (NOT the pipeline normalizer)."""
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = re.sub(r"[^\w\s]", " ", s)          # \w is Unicode-aware: keeps é, ç, etc.
    return re.sub(r"\s+", " ", s).strip()


def toks(s):
    return set(light_norm(s).split())


def jaccard(a, b):
    a, b = toks(a), toks(b)
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def parse_ids(s):
    return [x.strip() for x in str(s).split(",") if x.strip()]


def raw_line_count(path):
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def read_tsv(path, quote_none=True):
    """dtype=str + keep_default_na=False so nothing silently becomes NaN
    (e.g. a business literally named 'NA', or empty matched_entity_ids)."""
    bad = []
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_NONE if quote_none else csv.QUOTE_MINIMAL,
        engine="python",
        on_bad_lines=lambda line: bad.append(line) or None,
    )
    return df, bad


# ---------------------------------------------------------------- 1
def section_files(data_dir):
    hr("1. DATASET FILE STRUCTURE")
    for root, dirs, files in os.walk(data_dir):
        dirs.sort()
        for fn in sorted(files):
            p = os.path.join(root, fn)
            print(f"  {p:<60} {os.path.getsize(p) / 1024:>12.1f} KB")


# ---------------------------------------------------------------- 2-3
def load_all(data_dir):
    hr("2-3. ROW COUNTS AND COLUMNS")
    dfs = {}
    for split in ["train", "test"]:
        names = ["source1", "source2", "source3"] + (["ground_truth"] if split == "train" else [])
        for n in names:
            key = f"{split}_{n}"
            p = os.path.join(data_dir, split, f"{key}.tsv")
            if not os.path.exists(p):
                print(f"\n  [MISSING FILE] {p}")
                continue
            df, bad = read_tsv(p, quote_none=True)
            try:
                df_q, bad_q = read_tsv(p, quote_none=False)
                n_q = len(df_q)
            except Exception as e:
                n_q, bad_q = f"ERROR ({e.__class__.__name__})", []
            raw = raw_line_count(p)
            exp = GT_COLS if n == "ground_truth" else EXPECTED_COLS
            print(f"\n  {key}")
            print(f"    raw lines (incl. header)       : {raw}")
            print(f"    parsed rows, QUOTE_NONE         : {len(df)}  (bad lines skipped: {len(bad)})")
            print(f"    parsed rows, default quoting    : {n_q}  (bad lines skipped: {len(bad_q)})")
            print(f"    columns                         : {list(df.columns)}")
            if list(df.columns) != exp:
                print(f"    [WARN] expected columns {exp}")
            if raw - 1 != len(df):
                print("    [WARN] raw lines - 1 != parsed rows (blank lines / embedded newlines / bad lines?)")
            if isinstance(n_q, int) and n_q != len(df):
                print("    [WARN] quoting mode changes row count -> quote chars inside fields; pick a quoting policy")
            for b in bad[:3]:
                print(f"    bad line sample: {b}")
            dfs[key] = df
    return dfs


# ---------------------------------------------------------------- 4
def section_missing(dfs):
    hr("4. MISSING-VALUE STATISTICS  (blank = empty/whitespace; placeholder = 'nan','null','-',...)")
    for key, df in dfs.items():
        print(f"\n  {key} (n={len(df)})")
        for c in df.columns:
            s = df[c].astype(str)
            blank = int((s.str.strip() == "").sum())
            ph = int(s.str.strip().str.lower().isin(PLACEHOLDERS).sum())
            note = "   <- blank here = singleton" if c == "matched_entity_ids" else ""
            print(f"    {c:<22} blank={blank:>7} ({blank / max(len(df), 1):6.2%})  placeholder-like={ph:>6}{note}")


# ---------------------------------------------------------------- 5
def section_country(dfs):
    hr("5. COUNTRY DISTRIBUTION  (raw values via repr(), so whitespace/case variants are visible)")
    for key, df in dfs.items():
        if "country" not in df.columns:
            continue
        print(f"\n  {key}")
        for v, c in df["country"].value_counts().items():
            print(f"    {v!r:<28} {c:>8} ({c / len(df):6.2%})")

    print("\n  All-pairs size per country (S1 x (S2+S3)) -> is brute force feasible?")
    for split in ["train", "test"]:
        k1, k2, k3 = f"{split}_source1", f"{split}_source2", f"{split}_source3"
        if not all(k in dfs for k in (k1, k2, k3)):
            continue
        c1 = dfs[k1]["country"].value_counts()
        c23 = pd.concat([dfs[k2]["country"], dfs[k3]["country"]]).value_counts()
        total = 0
        for ctry in sorted(set(c1.index) | set(c23.index)):
            n = int(c1.get(ctry, 0)) * int(c23.get(ctry, 0))
            total += n
            print(f"    {split:<5} {ctry!r:<20} S1={int(c1.get(ctry, 0)):>7}  S2+S3={int(c23.get(ctry, 0)):>7}  pairs={n:>14,}")
        print(f"    {split:<5} TOTAL within-country pairs = {total:,}")


# ---------------------------------------------------------------- 6
def section_gt(dfs):
    hr("6. GROUND-TRUTH MATCH-COUNT DISTRIBUTION AND INTEGRITY")
    need = ["train_ground_truth", "train_source1", "train_source2", "train_source3"]
    if not all(k in dfs for k in need):
        print("  required train files not loaded")
        return None, None
    s1 = dfs["train_source1"].drop_duplicates("entity_id")
    other = pd.concat([dfs["train_source2"], dfs["train_source3"]]).drop_duplicates("entity_id")
    gt = dfs["train_ground_truth"].copy()
    gt["ids"] = gt["matched_entity_ids"].map(parse_ids)
    gt["n"] = gt["ids"].map(len)
    gt["n_s2"] = gt["ids"].map(lambda l: sum(i.startswith("S2-") for i in l))
    gt["n_s3"] = gt["ids"].map(lambda l: sum(i.startswith("S3-") for i in l))

    N = len(gt)
    print(f"\n  GT rows: {N}")
    print("  Matches per S1:")
    for b in range(0, 5):
        c = int((gt["n"] == b).sum())
        print(f"    {b:>2}   : {c:>7} ({c / N:6.2%})")
    c = int((gt["n"] >= 5).sum())
    print(f"    5+   : {c:>7} ({c / N:6.2%})   max={gt['n'].max()}  mean={gt['n'].mean():.3f}")
    print(f"  Singletons (0 matches): {(gt['n'] == 0).mean():.2%}   ->   all-empty submission scores this much")

    has = gt[gt["n"] > 0]
    if len(has):
        only2 = int(((has.n_s2 > 0) & (has.n_s3 == 0)).sum())
        only3 = int(((has.n_s3 > 0) & (has.n_s2 == 0)).sum())
        both = int(((has.n_s2 > 0) & (has.n_s3 > 0)).sum())
        print(f"  Among matched S1: only-S2={only2}  only-S3={only3}  both={both}")
        print(f"  Max S2 per S1={gt.n_s2.max()}  max S3 per S1={gt.n_s3.max()}  "
              f"(>1 from same source => S2/S3 are NOT deduplicated)")
        print(f"  S1 with >1 match from the same source: {int(((gt.n_s2 > 1) | (gt.n_s3 > 1)).sum())}")

    gtc = gt.merge(s1[["entity_id", "country"]], left_on="source1_entity_id", right_on="entity_id", how="left")
    print("\n  Match-count distribution by S1 country:")
    for ctry, g in gtc.groupby("country"):
        dist = g["n"].clip(upper=3).value_counts(normalize=True).sort_index()
        d = "  ".join(f"{'3+' if k == 3 else k}:{v:.1%}" for k, v in dist.items())
        print(f"    {ctry!r:<20} n={len(g):>7}   {d}")

    print("\n  Integrity checks:")
    all_ids = [i for l in gt["ids"] for i in l]
    print(f"    duplicate S1 rows in GT                 : {int(gt['source1_entity_id'].duplicated().sum())}")
    print(f"    GT S1 ids missing from train_source1    : {int((~gt['source1_entity_id'].isin(s1.entity_id)).sum())}")
    print(f"    train_source1 ids missing from GT       : {int((~s1.entity_id.isin(gt['source1_entity_id'])).sum())}")
    print(f"    duplicate ids inside a GT list          : {int(sum(len(l) - len(set(l)) for l in gt['ids']))}")
    print(f"    matched ids with non S2-/S3- prefix     : {sum(not i.startswith(('S2-', 'S3-')) for i in all_ids)}")
    print(f"    matched ids not found in S2/S3 files    : {sum(i not in set(other.entity_id) for i in all_ids)}")
    cnt = Counter(all_ids)
    multi = sum(1 for v in cnt.values() if v > 1)
    print(f"    S2/S3 ids matched to >1 S1              : {multi}   (0 => each S2/S3 record belongs to at most one S1)")
    for src in ["train_source2", "train_source3"]:
        ids = set(dfs[src]["entity_id"])
        used = len(ids & set(cnt))
        print(f"    {src}: {used}/{len(ids)} records used in any match ({used / max(len(ids), 1):.1%}); rest are distractors")

    # pair-level difficulty profile
    pairs = gt[["source1_entity_id", "ids"]].explode("ids").dropna()
    pairs.columns = ["s1", "other"]
    pairs = pairs[pairs.s1.isin(s1.entity_id) & pairs.other.isin(other.entity_id)]
    a = s1.set_index("entity_id").loc[pairs.s1].reset_index(drop=True)
    b = other.set_index("entity_id").loc[pairs.other].reset_index(drop=True)
    P = len(pairs)
    print(f"\n  True-pair profile (n={P} pairs):")
    if P:
        same_c = (a.country.values == b.country.values).mean()
        n_eq = (a.business_name.map(light_norm).values == b.business_name.map(light_norm).values).mean()
        ad_eq = (a.business_address.map(light_norm).values == b.business_address.map(light_norm).values).mean()
        nj = [jaccard(x, y) for x, y in zip(a.business_name, b.business_name)]
        aj = [jaccard(x, y) for x, y in zip(a.business_address, b.business_address)]
        nj0 = sum(v == 0 for v in nj) / P
        both0 = sum((x == 0 and y == 0) for x, y in zip(nj, aj)) / P
        print(f"    same country                          : {same_c:.2%}   (<100% => don't hard-block on country)")
        print(f"    light-normalized name identical       : {n_eq:.2%}")
        print(f"    light-normalized address identical    : {ad_eq:.2%}")
        print(f"    mean name token Jaccard               : {sum(nj) / P:.3f}")
        print(f"    mean address token Jaccard            : {sum(aj) / P:.3f}")
        print(f"    pairs sharing NO name token           : {nj0:.2%}   (token blocking on names alone misses these)")
        print(f"    pairs sharing NO name AND NO addr tok : {both0:.2%}   (need char n-gram blocking)")
    return gt, pairs


# ---------------------------------------------------------------- 7
def section_duplicates(dfs):
    hr("7. DUPLICATE STATISTICS")
    for key, df in dfs.items():
        if "entity_id" not in df.columns:
            continue
        ln = df.business_name.map(light_norm)
        la = df.business_address.map(light_norm)
        print(f"\n  {key} (n={len(df)})")
        print(f"    duplicate entity_id                     : {int(df.entity_id.duplicated().sum())}")
        print(f"    exact dup (name, address, country)      : {int(df.duplicated(['business_name', 'business_address', 'country']).sum())}")
        print(f"    light-norm dup (name, address)          : {int(pd.DataFrame({'n': ln, 'a': la}).duplicated().sum())}")
        print(f"    light-norm dup name only                : {int(ln.duplicated().sum())}   (chains/branches share names)")
        top = ln.value_counts().head(5)
        print("    most repeated names: " + ", ".join(f"{k!r}x{v}" for k, v in top.items()))

    print("\n  entity_id overlap between train and test (same source):")
    for s in ["source1", "source2", "source3"]:
        a, b = dfs.get(f"train_{s}"), dfs.get(f"test_{s}")
        if a is not None and b is not None:
            ov = len(set(a.entity_id) & set(b.entity_id))
            print(f"    {s}: {ov}   (>0 => IDs are reused across splits; never join train and test on ID)")


# ---------------------------------------------------------------- 8
def section_examples(dfs, gt, rng):
    hr("8. NAME / ADDRESS EXAMPLES")
    s1 = dfs.get("train_source1")
    if gt is not None and s1 is not None:
        other = pd.concat([dfs["train_source2"], dfs["train_source3"]]).drop_duplicates("entity_id").set_index("entity_id")
        s1i = s1.drop_duplicates("entity_id").set_index("entity_id")
        matched = gt[gt["n"] > 0]["source1_entity_id"].tolist()
        multi = gt[gt["n"] > 1]["source1_entity_id"].tolist()
        single = gt[gt["n"] == 0]["source1_entity_id"].tolist()
        picks = rng.sample(matched, min(4, len(matched))) + rng.sample(multi, min(3, len(multi)))
        for sid in picks:
            if sid not in s1i.index:
                continue
            r = s1i.loc[sid]
            print(f"\n  {sid} [{r.country}]  NAME: {r.business_name!r}")
            print(f"  {'':<10}      ADDR: {r.business_address!r}")
            for mid in gt.loc[gt.source1_entity_id == sid, "ids"].iloc[0]:
                if mid in other.index:
                    o = other.loc[mid]
                    print(f"    -> {mid} [{o.country}]  NAME: {o.business_name!r}")
                    print(f"       {'':<10}      ADDR: {o.business_address!r}")
        print("\n  Singletons (no match):")
        for sid in rng.sample(single, min(4, len(single))):
            if sid in s1i.index:
                r = s1i.loc[sid]
                print(f"    {sid} [{r.country}]  {r.business_name!r} | {r.business_address!r}")

    for key in ["test_source1", "test_source2", "test_source3"]:
        df = dfs.get(key)
        if df is None:
            continue
        print(f"\n  {key} samples per country:")
        for ctry, g in df.groupby("country"):
            for _, r in g.sample(min(3, len(g)), random_state=0).iterrows():
                print(f"    [{ctry}] {r.entity_id}  {r.business_name!r} | {r.business_address!r}")


# ---------------------------------------------------------------- 9
def section_quality(dfs):
    hr("9. DATA-QUALITY CHECKS")
    for key, df in dfs.items():
        if "entity_id" not in df.columns:
            continue
        exp_prefix = {"source1": "S1-", "source2": "S2-", "source3": "S3-"}[key.split("_")[1]]
        print(f"\n  {key}")
        print(f"    entity_id with wrong prefix            : {int((~df.entity_id.str.startswith(exp_prefix)).sum())}")
        for c in ["business_name", "business_address"]:
            s = df[c].astype(str)
            print(f"    {c}: leading/trailing ws={int((s != s.str.strip()).sum())}  "
                  f"double spaces={int(s.str.contains('  ').sum())}  "
                  f"non-ASCII={s.map(lambda x: any(ord(ch) > 127 for ch in x)).mean():.1%}  "
                  f"all-caps={s.map(lambda x: x.isupper()).mean():.1%}")
        print(f"    names <= 2 chars                        : {int((df.business_name.str.strip().str.len() <= 2).sum())}")
        print(f"    name == address                         : {int((df.business_name.str.strip() == df.business_address.str.strip()).sum())}")
        for ctry, g in df.groupby("country"):
            ad = g.business_address.astype(str)
            n6 = ad.str.contains(r"(?<!\d)\d{3}\s?\d{3}(?!\d)").mean()
            n5 = ad.str.contains(r"(?<!\d)\d{5}(?:-\d{4})?(?!\d)").mean()
            anyd = ad.str.contains(r"\d").mean()
            nl = g.business_name.str.len()
            al = ad.str.len()
            print(f"    [{ctry}] addr has any digit={anyd:.1%}  6-digit(PIN-like)={n6:.1%}  5-digit(ZIP-like)={n5:.1%}  "
                  f"name len med/p95={nl.median():.0f}/{nl.quantile(.95):.0f}  addr len med/p95={al.median():.0f}/{al.quantile(.95):.0f}")

    hr("9b. MOST FREQUENT TOKENS (document frequency) -> abbreviations / legal suffixes")
    for split in ["train", "test"]:
        parts = [dfs[f"{split}_source{i}"] for i in (1, 2, 3) if f"{split}_source{i}" in dfs]
        if not parts:
            continue
        allrec = pd.concat(parts)
        for ctry, g in allrec.groupby("country"):
            for col in ["business_name", "business_address"]:
                c = Counter()
                for v in g[col]:
                    c.update(toks(v))
                print(f"\n  {split} [{ctry}] {col} top-30:")
                print("    " + ", ".join(f"{t}:{n}" for t, n in c.most_common(30)))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    pd.set_option("display.width", 200)
    rng = random.Random(args.seed)
    print(f"pandas {pd.__version__} | data dir: {os.path.abspath(args.data_dir)} | seed {args.seed}")

    section_files(args.data_dir)
    dfs = load_all(args.data_dir)
    section_missing(dfs)
    section_country(dfs)
    gt, _ = section_gt(dfs)
    section_duplicates(dfs)
    section_examples(dfs, gt, rng)
    section_quality(dfs)
    hr("DONE")


if __name__ == "__main__":
    main()