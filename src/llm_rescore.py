"""
src/llm_rescore.py  (v008 option)
GPT-2 cross-encoder (Ditto-style) for stage 2. Each uncertain (S1, candidate) pair is serialized as
text from the RAW business name/address and scored by a fine-tuned GPT2ForSequenceClassification.
The probability p_llm becomes a stage-2 feature (stage2.py --llm <tag>); pairs that are not
rescored get NaN (LightGBM treats it as missing).

Only pairs inside a stage-1 band (lo <= p1 <= hi, top-N per S1 by p1) are scored: confident pairs are
already right, and 30M+ test pairs are far too many for a 4 GB laptop GPU.

  python src/llm_rescore.py count   --stage1 v005ws1                 # band sizes only, no GPU
  python src/llm_rescore.py train   --stage1 v005ws1 --tag g001     # K-fold by S1 -> OOF p_llm
  python src/llm_rescore.py holdout --stage1 v005ws1 --tag g001     # average of fold models
  python src/llm_rescore.py test    --stage1 v005ws1 --tag g001
then
  python src/stage2.py train|holdout|test --stage1 v005ws1 --tag v008 --llm g001

Train mode uses train-fold S1 only (GroupKFold by S1, OOF predictions); holdout labels are never used.
Model: GPT-2 small (124M, MIT licence) via Hugging Face transformers (Apache 2.0) + PyTorch (BSD).
Any other sequence-classification checkpoint also works, e.g. --model xlm-roberta-base (MIT, 278M,
multilingual: the pair is fed as a text pair with the tokenizer's own special tokens); --bf16 recommended.
Download once (`--model gpt2`, or a local directory for offline runs); no network calls afterwards.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from data_loading import CACHE_DIR, load_normalized, load_split_ids
from matcher import p1_path

REC_COLS = ["entity_id", "business_name", "business_address"]


def llm_path(split, s1set, tag):
    return os.path.join(CACHE_DIR, f"llm_{split}_{s1set}_{tag}.parquet")


def select_band(c, lo, hi, top):
    """Pairs with lo <= p1 <= hi, at most `top` per S1 (highest p1 first)."""
    c = c[(c.p >= lo) & (c.p <= hi)]
    c = c.sort_values(["s1_id", "p"], ascending=[True, False], kind="stable")
    return c[c.groupby("s1_id").cumcount() < top].reset_index(drop=True)


class Texts:
    """Raw name/address lookup for S1 and the S2+S3 pool of one split."""

    def __init__(self, split):
        s1 = load_normalized(split, 1, REC_COLS)
        pool = pd.concat([load_normalized(split, 2, REC_COLS), load_normalized(split, 3, REC_COLS)],
                         ignore_index=True)
        self.s1 = s1.set_index("entity_id")
        self.pool = pool.set_index("entity_id")

    def pairs(self, c):
        a = self.s1.reindex(c.s1_id)
        b = self.pool.reindex(c.cand_id)
        if a.business_name.isna().any() or b.business_name.isna().any():
            raise ValueError("candidate IDs not in caches")
        return (list(zip(a.business_name, a.business_address)), list(zip(b.business_name, b.business_address)))


class Encoder:
    """Tokenizes 'name: .. addr: ..' for both sides; each side truncated to half the budget.
    GPT-2: a <eos> b. Other models: the tokenizer's own pair format (e.g. <s> a </s></s> b </s>)."""

    def __init__(self, tok, maxlen):
        self.tok, self.maxlen, self.half = tok, maxlen, maxlen // 2
        self.gpt = is_gpt2(tok)

    def side(self, name, addr):
        ids = self.tok.encode(f" name: {name} addr: {addr}", add_special_tokens=False)
        return ids[:self.half]

    def __call__(self, A, B):
        if self.gpt:
            eos = self.tok.eos_token_id
            return [self.side(*a) + [eos] + self.side(*b) for a, b in zip(A, B)]
        text = lambda r: f"name: {r[0]} addr: {r[1]}"
        out = []
        for i in range(0, len(A), 50_000):       # chunked: one call on millions of pairs needs >20 GB RAM
            enc = self.tok([text(a) for a in A[i:i + 50_000]], [text(b) for b in B[i:i + 50_000]],
                           truncation="longest_first", max_length=self.maxlen, return_attention_mask=False)
            out.extend(np.asarray(x, dtype=np.int32) for x in enc["input_ids"])
        return out


def is_gpt2(tok):
    return tok.pad_token_id == tok.eos_token_id and tok.eos_token == "<|endoftext|>"


def batches(seqs, bs, pad_id, order):
    import torch
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        L = max(len(seqs[j]) for j in idx)
        ids = torch.full((len(idx), L), pad_id, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for r, j in enumerate(idx):
            s = seqs[j]
            ids[r, :len(s)] = torch.tensor(s)
            att[r, :len(s)] = 1
        yield idx, ids, att


def load_model(name):
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if AutoConfig.from_pretrained(name).model_type == "gpt2":
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1)
    model.config.pad_token_id = tok.pad_token_id
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return tok, model.to(dev), dev


def last_logit(model, ids, att):
    """GPT-2: logit at the last real token (right padding; pad == eos, so do not rely on pad_token_id).
    Other models: their own classification head."""
    import torch
    if model.config.model_type != "gpt2":
        return model(input_ids=ids, attention_mask=att).logits.squeeze(-1)
    h = model.transformer(input_ids=ids, attention_mask=att).last_hidden_state
    last = att.sum(1) - 1
    return model.score(h[torch.arange(len(ids), device=ids.device), last]).squeeze(-1)


def amp_dtype(dev, bf16):
    import torch
    return torch.bfloat16 if (bf16 or dev != "cuda") else torch.float16


def predict(model, dev, seqs, pad_id, bs, bf16=False):
    import torch
    model.eval()
    order = np.argsort([len(s) for s in seqs], kind="stable")      # length-sorted = less padding
    out = np.zeros(len(seqs), np.float32)
    t, done = time.time(), 0
    with torch.no_grad(), torch.autocast(dev, dtype=amp_dtype(dev, bf16)):
        for idx, ids, att in batches(seqs, bs, pad_id, order):
            out[idx] = last_logit(model, ids.to(dev), att.to(dev)).float().cpu().numpy()
            done += len(idx)
            if done % (bs * 2000) < bs:
                print(f"    predicted {done}/{len(seqs)} ({done / (time.time() - t):.0f} pairs/s)")
    return out


def train_one(a, seqs, y, pad_id, out_dir):
    import torch
    tok, model, dev = load_model(a.model)
    torch.manual_seed(a.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    rng = np.random.default_rng(a.seed)
    steps = a.epochs * ((len(seqs) + a.bs - 1) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.06,
                                                anneal_strategy="linear")
    scaler = torch.amp.GradScaler(enabled=dev == "cuda" and not a.bf16)
    pos_w = torch.tensor(a.pos_weight, device=dev)
    model.train()
    step, t = 0, time.time()
    for ep in range(a.epochs):
        # bucket by length inside shuffled chunks: fast batches, still random order
        perm = rng.permutation(len(seqs))
        chunks = [perm[i:i + a.bs * 100] for i in range(0, len(perm), a.bs * 100)]
        order = np.concatenate([ch[np.argsort([len(seqs[j]) for j in ch], kind="stable")] for ch in chunks])
        blocks = [order[i:i + a.bs] for i in range(0, len(order), a.bs)]
        rng.shuffle(blocks)
        loss_sum = 0.0
        for idx, ids, att in batches(seqs, a.bs, pad_id, np.concatenate(blocks)):
            yt = torch.tensor(y[idx], dtype=torch.float32, device=dev)
            with torch.autocast(dev, dtype=amp_dtype(dev, a.bf16)):
                logit = last_logit(model, ids.to(dev), att.to(dev)).float()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, yt, pos_weight=pos_w)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            loss_sum += loss.item()
            if step % 500 == 0:
                print(f"    ep {ep} step {step}/{steps} loss {loss_sum / 500:.4f} "
                      f"({step * a.bs / (time.time() - t):.0f} pairs/s)")
                loss_sum = 0.0
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return model, dev


def report(p1, p_llm, y):
    from sklearn.metrics import log_loss, roc_auc_score
    q = 1 / (1 + np.exp(-p_llm))
    print(f"  band pairs {len(y)} | positive {y.mean():.3f}")
    print(f"  AUC  stage-1 {roc_auc_score(y, p1):.4f} | gpt2 {roc_auc_score(y, q):.4f} | "
          f"mean {roc_auc_score(y, (p1 + q) / 2):.4f}")
    print(f"  logloss stage-1 {log_loss(y, np.clip(p1, 1e-6, 1 - 1e-6)):.4f} | gpt2 {log_loss(y, np.clip(q, 1e-6, 1 - 1e-6)):.4f}")


def save(c, logit, split, s1set, tag):
    out = pd.DataFrame({"s1_id": c.s1_id.to_numpy(), "cand_id": c.cand_id.to_numpy(),
                        "p_llm": (1 / (1 + np.exp(-logit))).astype(np.float32)})
    out.to_parquet(llm_path(split, s1set, tag), index=False)
    print(f"  saved -> {llm_path(split, s1set, tag)}")


def run_count(a):
    for split, s1set in [("train", "train"), ("train", "holdout"), ("test", "all")]:
        path = p1_path(split, s1set, a.stage1)
        if not os.path.exists(path):
            print(f"{path}: missing")
            continue
        c = pd.read_parquet(path)
        b = select_band(c, a.lo, a.hi, a.top)
        extra = f" | positive {b.label.mean():.3f}" if "label" in b.columns else ""
        print(f"{split}/{s1set}: {len(c)} pairs -> band {len(b)} ({len(b) / len(c):.2%}), "
              f"{b.s1_id.nunique()} S1{extra}")


def run_train(a):
    c = select_band(pd.read_parquet(p1_path("train", "train", a.stage1)), a.lo, a.hi, a.top)
    if "label" not in c.columns:
        raise SystemExit("train p1 file has no labels - re-run matcher.py train with the latest matcher.py")
    if not a.final_fit and set(c.s1_id) & set(load_split_ids("holdout")):
        raise SystemExit("[STOP] holdout IDs found in training data")
    y = c.label.to_numpy(np.float32)
    print(f"gpt2 train: {len(c)} band pairs, {c.s1_id.nunique()} S1, positive {y.mean():.3f}")
    tok, _, _ = load_model(a.model)
    seqs = Encoder(tok, a.maxlen)(*Texts("train").pairs(c))
    oof = np.zeros(len(c), np.float32)
    rng = np.random.default_rng(a.seed)
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=a.folds).split(c, y, c.s1_id.to_numpy())):
        d = os.path.join("models", a.tag, f"gpt2_fold{fold}")
        t = time.time()
        if a.max_train and len(tr) > a.max_train:
            tr = np.sort(rng.choice(tr, a.max_train, replace=False))
        print(f"  fold {fold}: train {len(tr)} / predict {len(va)}")
        if os.path.exists(os.path.join(d, "config.json")) and not a.retrain:
            print(f"  fold {fold}: reusing {d}")
            _, model, dev = load_model(d)
        else:
            model, dev = train_one(a, [seqs[i] for i in tr], y[tr], tok.pad_token_id, d)
        oof[va] = predict(model, dev, [seqs[i] for i in va], tok.pad_token_id, a.bs * 4, a.bf16)
        print(f"  fold {fold} done ({time.time() - t:.0f}s)")
        del model
    report(c.p.to_numpy(), oof, y)
    save(c, oof, "train", "train", a.tag)
    os.makedirs(os.path.join("experiments", a.tag), exist_ok=True)
    with open(os.path.join("experiments", a.tag, "llm.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in vars(a).items() if k != "mode"}, f, indent=2)


def run_predict(a, split, s1set):
    with open(os.path.join("experiments", a.tag, "llm.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    c = select_band(pd.read_parquet(p1_path(split, s1set, cfg["stage1"])), cfg["lo"], cfg["hi"], cfg["top"])
    print(f"gpt2 {split}/{s1set}: {len(c)} band pairs, {c.s1_id.nunique()} S1")
    texts = Texts(split).pairs(c)
    logits = []
    for fold in range(cfg["folds"]):
        tok, model, dev = load_model(os.path.join("models", a.tag, f"gpt2_fold{fold}"))
        seqs = Encoder(tok, cfg["maxlen"])(*texts)
        logits.append(predict(model, dev, seqs, tok.pad_token_id, cfg["bs"] * 4, cfg.get("bf16", False)))
        del model
    save(c, np.mean(logits, axis=0), split, s1set, a.tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["count", "train", "holdout", "test"])
    ap.add_argument("--stage1", default="v005ws1")
    ap.add_argument("--tag", default="g001")
    ap.add_argument("--model", default="gpt2", help="HF id or local dir (gpt2, gpt2-medium, ...)")
    ap.add_argument("--lo", type=float, default=0.02)
    ap.add_argument("--hi", type=float, default=0.98)
    ap.add_argument("--top", type=int, default=8, help="max band pairs per S1")
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--max-train", type=int, default=400_000, help="cap training pairs per fold (0 = all)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--maxlen", type=int, default=96)
    ap.add_argument("--pos-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bf16", action="store_true", help="bfloat16 autocast instead of fp16 + loss scaling")
    ap.add_argument("--retrain", action="store_true", help="ignore saved fold models")
    ap.add_argument("--final-fit", action="store_true", help="stage-1 OOF came from a final-fit run (includes holdout)")
    a = ap.parse_args()
    t = time.time()
    if a.mode == "count":
        run_count(a)
    elif a.mode == "train":
        run_train(a)
    else:
        run_predict(a, *{"holdout": ("train", "holdout"), "test": ("test", "all")}[a.mode])
    print(f"done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
