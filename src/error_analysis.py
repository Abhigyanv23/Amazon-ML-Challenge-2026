"""
src/error_analysis.py  (v004 step 1)
Holdout error analysis. Labels are used ONLY to analyse; any rule change found here must be
tuned on train-fold OOF data, never on holdout.

Usage:
  $env:PYTHONIOENCODING="utf-8"
  python src/error_analysis.py --stage1 v002 --stage2 v003 > experiments/v004/error_analysis.txt
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from data_loading import CACHE_DIR, cache_path, load_ground_truth, load_normalized, load_split_ids
from matcher import decide, p1_path


def hr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def load_dec(tag):
    with open(os.path.join("experiments", tag, "decision.json"), encoding="utf-8") as f:
        return json.load(f)


def load_p2(stage2_tag, jobs):
    path = os.path.join(CACHE_DIR, f"p2_train_holdout_{stage2_tag}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    import stage2
    with Pool(jobs) as w:
        c, _, _ = stage2._predict(argparse.Namespace(tag=stage2_tag), "train", "holdout", w)
    c = c[["s1_id", "cand_id", "p"]]
    c.to_parquet(path, index=False)
    print(f"saved stage-2 holdout probabilities -> {path}")
    return c


def entity_scores(d, n_true, tp_key):
    lab = d[["s1_id", "cand_id"]].merge(tp_key, how="left")
    lab["y"] = lab["y"].fillna(0)
    g = lab.groupby("s1_id")["y"].agg(["sum", "size"])
    tp = g["sum"].reindex(n_true.index, fill_value=0).to_numpy(float)
    npred = g["size"].reindex(n_true.index, fill_value=0).to_numpy(float)
    nt = n_true.to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(npred > 0, tp / npred, 0.0)
        r = np.where(nt > 0, tp / nt, 0.0)
        f = np.where(tp > 0, 1.25 * p * r / (0.25 * p + r), 0.0)
    f = np.where(nt == 0, (npred == 0).astype(float), f)
    return pd.DataFrame({"n_true": nt, "n_pred": npred, "n_tp": tp, "f": f}, index=n_true.index)


def categorize(E):
    fp = E.n_pred - E.n_tp
    return pd.Series(np.select(
        [(E.n_true == 0) & (E.n_pred == 0), E.n_true == 0, E.n_pred == 0,
         (E.n_tp == E.n_true) & (fp == 0), (fp == 0), (E.n_tp == E.n_true), E.n_tp > 0],
        ["singleton_ok", "singleton_FP", "false_empty", "perfect", "missing_only",
         "extra_FP_only", "FP_and_missing"], default="all_wrong"), index=E.index)


def loss_table(E, by=None):
    cat = categorize(E)
    N = len(E)
    df = pd.DataFrame({"cat": cat, "loss": 1 - E.f})
    if by is not None:
        df["by"] = by
        t = df.groupby(["by", "cat"]).agg(n=("loss", "size"), loss=("loss", "sum"))
        t["loss"] = t["loss"] / N
        return t[t.n > 0].round(5).to_string()
    t = df.groupby("cat").agg(n=("loss", "size"), loss=("loss", "sum"))
    t["share"] = t.n / N
    t["F_loss"] = t["loss"] / N
    return t.drop(columns="loss").sort_values("F_loss", ascending=False).round(5).to_string()


def bucket(n):
    return np.where(n >= 3, "3+", n.astype(int).astype(str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default="v002")
    ap.add_argument("--stage2", default="v003")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    os.makedirs(os.path.join("experiments", "v004"), exist_ok=True)
    out = {}

    ids = load_split_ids("holdout")
    idset = set(ids)
    gt = load_ground_truth()
    owner = gt[["source1_entity_id", "matched_ids"]].explode("matched_ids").dropna()
    owner = owner.set_index("matched_ids")["source1_entity_id"]
    gth = gt[gt.source1_entity_id.isin(idset)]
    n_true = gth.set_index("source1_entity_id").matched_ids.map(len).reindex(ids)
    tp = gth[["source1_entity_id", "matched_ids"]].explode("matched_ids").dropna()
    tp.columns = ["s1_id", "cand_id"]
    tp_key = tp.assign(y=1)

    dec1, dec2 = load_dec(a.stage1), load_dec(a.stage2)
    c1 = pd.read_parquet(p1_path("train", "holdout", a.stage1))[["s1_id", "cand_id", "p"]]
    c2 = load_p2(a.stage2, a.jobs)
    d1 = decide(c1, dec1["t"], dec1["t_top"])
    d2 = decide(c2, dec2["t"], dec2["t_top"])
    E1, E2 = entity_scores(d1, n_true, tp_key), entity_scores(d2, n_true, tp_key)

    s1 = load_normalized("train", 1, ["entity_id", "country_key"]).set_index("entity_id")
    country = s1.country_key.reindex(E1.index).to_numpy()
    nb = bucket(E1.n_true.to_numpy())

    hr("1. HOLDOUT F0.5 AND BOOTSTRAP 95% CI (resampling S1 entities)")
    rng = np.random.default_rng(0)
    f1, f2 = E1.f.to_numpy(), E2.f.to_numpy()
    bs = np.array([(f1[i].mean(), f2[i].mean()) for i in
                   (rng.integers(0, len(f1), len(f1)) for _ in range(a.boot))])
    diff = bs[:, 1] - bs[:, 0]
    for name, v, b in [(f"stage1 {a.stage1}", f1.mean(), bs[:, 0]), (f"stage2 {a.stage2}", f2.mean(), bs[:, 1]),
                       ("difference", f2.mean() - f1.mean(), diff)]:
        lo, hi = np.percentile(b, [2.5, 97.5])
        print(f"  {name:<16} {v:.4f}   95% CI [{lo:.4f}, {hi:.4f}]   std {b.std():.4f}")
    out["bootstrap"] = {"f1": float(f1.mean()), "f2": float(f2.mean()), "diff_ci": list(np.percentile(diff, [2.5, 97.5]))}

    hr(f"2. WHERE THE F0.5 LOSS IS (stage2 {a.stage2}): per-entity outcome, F_loss = share of 1-F0.5")
    print(loss_table(E2))
    print("\n  by country:")
    print(loss_table(E2, country))
    print("\n  by true match count:")
    print(loss_table(E2, nb))

    hr("3. PAIR-LEVEL ERRORS (stage2)")
    P = c2.merge(c1.rename(columns={"p": "p1"}), how="left").merge(tp_key, how="left")
    P["y"] = P["y"].fillna(0).astype(int)
    P = P.merge(d1[["s1_id", "cand_id"]].assign(pred1=1), how="left").merge(
        d2[["s1_id", "cand_id"]].assign(pred2=1), how="left")
    P[["pred1", "pred2"]] = P[["pred1", "pred2"]].fillna(0).astype(int)

    cand = pd.read_parquet(os.path.join(CACHE_DIR, f"cand_train_holdout_{a.stage1}.parquet"),
                           columns=["s1_id", "cand_id", "rank", "blk_tri_rank", "blk_nl_rank"])
    P = P.merge(cand, how="left")
    P["channel"] = np.select([P["rank"] <= 20, P["blk_tri_rank"] <= 10], ["word", "trigram_only"], "nonlatin_only")
    pool_attr = pd.concat([pd.read_parquet(cache_path("train", n), columns=[
        "entity_id", "name_nonlatin", "addr_blank", "addr_state_src"]) for n in (2, 3)]).set_index("entity_id")
    at = pool_attr.reindex(P.cand_id)
    P["cand_nonlatin"] = at.name_nonlatin.to_numpy()
    P["cand_blank_addr"] = at.addr_blank.to_numpy()
    P["cand_state_inferred"] = (at.addr_state_src.to_numpy() == 2).astype(int)
    P["is_s3"] = P.cand_id.str.startswith("S3-").astype(int)
    P["country"] = s1.country_key.reindex(P.s1_id).to_numpy()
    P["n_true"] = bucket(n_true.reindex(P.s1_id).to_numpy())

    t2, tt2 = dec2["t"], dec2["t_top"]
    in_c = tp.merge(P[["s1_id", "cand_id"]].assign(inc=1), how="left")["inc"].notna()
    pos = P[P.y == 1]
    taken = set(d2.cand_id)
    fn = pos[pos.pred2 == 0]
    fn_reason = np.select([fn.cand_id.isin(taken), fn.p >= t2 - 0.2], ["lost_to_other_S1", "near_miss_p>=t-0.2"],
                          "low_p")
    print(f"  true pairs {len(tp)} | blocking miss {int((~in_c).sum())} | in candidates {int(in_c.sum())}")
    print(f"  TP {int(pos.pred2.sum())} | FN {len(fn)}: " +
          ", ".join(f"{k}={v}" for k, v in pd.Series(fn_reason).value_counts().items()))
    fp = P[(P.y == 0) & (P.pred2 == 1)]
    own = owner.reindex(fp.cand_id)
    fp_type = np.select([own.isna().to_numpy(), own.isin(idset).to_numpy()],
                        ["distractor(no owner)", "belongs_other_holdout_S1"], "belongs_train_fold_S1")
    print(f"  FP {len(fp)}: " + ", ".join(f"{k}={v}" for k, v in pd.Series(fp_type).value_counts().items()))
    out["pairs"] = {"blocking_miss": int((~in_c).sum()), "FN": len(fn), "FP": len(fp),
                    "fn_reason": pd.Series(fn_reason).value_counts().to_dict(),
                    "fp_type": pd.Series(fp_type).value_counts().to_dict()}

    print("\n  by slice: recall among candidates, FP share of predictions")
    for col in ["country", "n_true", "channel", "cand_nonlatin", "cand_blank_addr", "cand_state_inferred", "is_s3"]:
        g = P.groupby(col).apply(lambda x: pd.Series({
            "pos": int(x.y.sum()), "recall_in_cands": x[x.y == 1].pred2.mean() if x.y.sum() else np.nan,
            "pred": int(x.pred2.sum()),
            "fp_share": ((x.y == 0) & (x.pred2 == 1)).sum() / max(x.pred2.sum(), 1)}), include_groups=False)
        print(f"\n  [{col}]\n" + g.round(4).to_string())

    hr(f"4. STAGE 1 ({a.stage1}) -> STAGE 2 ({a.stage2}) TRANSITIONS (pairs)")
    P["trans"] = np.select(
        [(P.y == 1) & (P.pred1 == 0) & (P.pred2 == 1), (P.y == 1) & (P.pred1 == 1) & (P.pred2 == 0),
         (P.y == 0) & (P.pred1 == 1) & (P.pred2 == 0), (P.y == 0) & (P.pred1 == 0) & (P.pred2 == 1)],
        ["TP_recovered", "TP_lost", "FP_removed", "FP_new"], "unchanged")
    for col in [None, "n_true", "country", "cand_nonlatin", "channel"]:
        g = P[P.trans != "unchanged"]
        t = g.trans.value_counts() if col is None else g.groupby([col, "trans"]).size().unstack(fill_value=0)
        print(f"\n  [{col or 'all'}]\n" + t.to_string())
    out["transitions"] = P.trans.value_counts().to_dict()

    hr("5. 1-MATCH S1 (the v003 regression)")
    one = E1.n_true == 1
    print(f"  n={int(one.sum())}  F stage1 {E1.f[one].mean():.4f} -> stage2 {E2.f[one].mean():.4f}")
    print(pd.DataFrame({"stage1": categorize(E1[one]).value_counts(),
                        "stage2": categorize(E2[one]).value_counts()}).fillna(0).astype(int).to_string())
    o = P[P.n_true == "1"]
    print("\n  1-match true pairs: p1 vs p2 (quantiles)")
    print(o[o.y == 1][["p1", "p"]].rename(columns={"p": "p2"}).describe(percentiles=[.1, .25, .5]).round(3).to_string())

    hr("6. ONE-S1-PER-CANDIDATE RULE: A/B on stage2 probabilities (diagnosis only)")
    top = c2.groupby("s1_id").p.transform("max")
    base = c2[(c2.p >= t2) & (top >= tt2)]
    win = base.sort_values("p", ascending=False, kind="stable").drop_duplicates("cand_id")
    strong = set(win[win.p >= 0.9].cand_id)
    variants = {"A_no_uniqueness": base,
                "B_current_uniqueness": d2,
                "C_unique_only_if_winner_p>=0.9": pd.concat([win[win.cand_id.isin(strong)],
                                                            base[~base.cand_id.isin(strong)]])}
    for k, dv in variants.items():
        f = entity_scores(dv, n_true, tp_key).f.mean()
        print(f"  {k:<34} F0.5 {f:.4f}   predicted pairs {len(dv)}")
        out.setdefault("uniqueness_ab", {})[k] = float(f)
    print(f"  S2/S3 records matched to >1 S1 in full ground truth: {int(owner.index.duplicated().sum())}")

    hr("7. EXAMPLES (raw text)")
    ex_fn = fn.assign(r=fn_reason)
    ex_fn = ex_fn[ex_fn.r != "lost_to_other_S1"].sample(n=min(8, len(ex_fn)), random_state=0)
    ex_fp = fp.assign(r=fp_type).sample(n=min(8, len(fp)), random_state=0)
    need_s1 = set(ex_fn.s1_id) | set(ex_fp.s1_id)
    need_c = set(ex_fn.cand_id) | set(ex_fp.cand_id)
    raw_cols = ["entity_id", "business_name", "business_address"]
    rs1 = pd.read_parquet(cache_path("train", 1), columns=raw_cols,
                          filters=[("entity_id", "in", list(need_s1))]).set_index("entity_id")
    rp = pd.concat([pd.read_parquet(cache_path("train", n), columns=raw_cols,
                                    filters=[("entity_id", "in", list(need_c))]) for n in (2, 3)]).set_index("entity_id")
    for title, ex in [("FALSE NEGATIVES (rejected true matches)", ex_fn), ("FALSE POSITIVES", ex_fp)]:
        print(f"\n--- {title} ---")
        for _, r in ex.iterrows():
            print(f"  [{r.country}, n_true={r.n_true}, {r.r}] p1={r.p1:.2f} p2={r.p:.2f}")
            print(f"    S1 {rs1.business_name.get(r.s1_id)!r} | {rs1.business_address.get(r.s1_id)!r}")
            print(f"    -> {rp.business_name.get(r.cand_id)!r} | {rp.business_address.get(r.cand_id)!r}")

    with open(os.path.join("experiments", "v004", "error_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    print("\nsaved experiments/v004/error_analysis.json")


if __name__ == "__main__":
    main()