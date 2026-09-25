"""
src/validate_local.py
Local scorer: per-Source-1 F_0.5, macro-averaged. Scores ONLY the S1 IDs in the split file.

Assumption (engineering, not official): an entity with true matches but an empty
prediction scores 0; a true singleton scores 1 only if the prediction is empty.

Usage (from repo root):
  python src/validate_local.py --selftest
  python src/validate_local.py --pred <matching_results.tsv> [--cand <candidate_pairs.tsv>] [--json out.json]
"""
import argparse
import csv
import json
import os

import pandas as pd

BETA2 = 0.25
GT_PATH = os.path.join("dataset", "train", "train_ground_truth.tsv")
S1_PATH = os.path.join("dataset", "train", "train_source1.tsv")
SPLIT_PATH = os.path.join("experiments", "splits", "holdout_s1_ids.txt")


def read_tsv(path, usecols=None):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                       quoting=csv.QUOTE_NONE, encoding="utf-8-sig", usecols=usecols)


def load_lists(path, val_col):
    df = read_tsv(path)
    exp = ["source1_entity_id", val_col]
    if list(df.columns) != exp:
        raise SystemExit(f"[FAIL] {path}: columns {list(df.columns)} != {exp}")
    dup = int(df["source1_entity_id"].duplicated().sum())
    if dup:
        raise SystemExit(f"[FAIL] {path}: {dup} duplicate source1_entity_id rows")
    out = {}
    for k, v in zip(df["source1_entity_id"], df[val_col]):
        ids = [x.strip() for x in v.split(",") if x.strip()]
        if len(ids) != len(set(ids)):
            raise SystemExit(f"[FAIL] {path}: duplicate IDs inside list for {k}")
        out[k] = set(ids)
    return out


def load_split(path=SPLIT_PATH):
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def f05(pred, true):
    if not true:
        return 1.0 if not pred else 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return (1 + BETA2) * p * r / (BETA2 * p + r)


def score(pred, gt, ids, country=None, cand=None):
    """pred/gt/cand: dict S1 -> set(ids). ids: S1 IDs to score. Returns metrics dict."""
    missing = sum(1 for i in ids if i not in pred)
    rows = []
    for i in ids:
        t, p = gt[i], pred.get(i, set())
        rows.append((i, len(t), len(p), len(p & t), f05(p, t), country.get(i, "?") if country else "?"))
    df = pd.DataFrame(rows, columns=["id", "n_true", "n_pred", "tp", "f", "country"])

    sing = df[df.n_true == 0]
    nons = df[df.n_true > 0]
    m = {
        "n_scored": len(df),
        "missing_rows_treated_empty": missing,
        "F0.5_macro": df.f.mean(),
        "micro_precision": df.tp.sum() / max(df.n_pred.sum(), 1),
        "micro_recall": df.tp.sum() / max(df.n_true.sum(), 1),
        "singleton_accuracy": (sing.n_pred == 0).mean() if len(sing) else None,
        "false_empty_rate_nonsingletons": (nons.n_pred == 0).mean() if len(nons) else None,
        "avg_pred_per_s1": df.n_pred.mean(),
        "F0.5_by_true_count": df.groupby(df.n_true.clip(upper=3).astype(str).str.replace("3", "3+"))["f"].mean().round(4).to_dict(),
        "F0.5_by_country": df.groupby("country")["f"].mean().round(4).to_dict(),
    }

    if cand is not None:
        not_subset = sum(len(pred.get(i, set()) - cand.get(i, set())) for i in ids)
        tot_true = sum(len(gt[i]) for i in ids)
        hit = sum(len(gt[i] & cand.get(i, set())) for i in ids)
        m.update({
            "candidate_recall": hit / max(tot_true, 1),
            "avg_candidates_per_s1": sum(len(cand.get(i, set())) for i in ids) / len(ids),
            "max_candidates_per_s1": max(len(cand.get(i, set())) for i in ids),
            "oracle_F0.5_ceiling": sum(f05(gt[i] & cand.get(i, set()), gt[i]) for i in ids) / len(ids),
            "pred_ids_not_in_candidates": not_subset,
        })
    return m


def print_metrics(m):
    for k, v in m.items():
        print(f"  {k:<34} {v:.4f}" if isinstance(v, float) else f"  {k:<34} {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred")
    ap.add_argument("--cand")
    ap.add_argument("--split", default=SPLIT_PATH)
    ap.add_argument("--json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    gt = load_lists(GT_PATH, "matched_entity_ids")
    ids = load_split(a.split)
    s1 = read_tsv(S1_PATH, usecols=["entity_id", "country"])
    country = dict(zip(s1.entity_id, s1.country))
    unknown = [i for i in ids if i not in gt]
    if unknown:
        raise SystemExit(f"[FAIL] {len(unknown)} split IDs not in ground truth")
    print(f"Scoring {len(ids)} S1 IDs from {a.split}")

    if a.selftest:
        print("\n[selftest] perfect prediction (expect F0.5 = 1.0000):")
        print_metrics(score({i: gt[i] for i in ids}, gt, ids, country, cand={i: gt[i] for i in ids}))
        print("\n[selftest] all-empty prediction (expect F0.5 = singleton share):")
        print_metrics(score({}, gt, ids, country))
        return

    if not a.pred:
        raise SystemExit("--pred required (or use --selftest)")
    pred = load_lists(a.pred, "matched_entity_ids")
    cand = load_lists(a.cand, "candidate_entity_ids") if a.cand else None
    m = score(pred, gt, ids, country, cand)
    print_metrics(m)
    if m.get("pred_ids_not_in_candidates"):
        print("  [WARN] predictions are not a subset of candidates -> submission would FAIL")
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2, default=float)
        print(f"  saved -> {a.json}")


if __name__ == "__main__":
    main()