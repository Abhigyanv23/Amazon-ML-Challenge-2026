"""
src/data_loading.py
Single place for reading competition files, split IDs and normalized caches.
"""
import csv
import os

import pandas as pd

DATA_DIR = "dataset"
SPLIT_DIR = os.path.join("experiments", "splits")
CACHE_DIR = os.path.join("experiments", "cache")


def read_tsv(path, usecols=None):
    # Everything as str; empty cells stay "" (never NaN). Always tab-separated.
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                       quoting=csv.QUOTE_NONE, encoding="utf-8-sig", usecols=usecols)


def source_path(split, n, data_dir=DATA_DIR):
    return os.path.join(data_dir, split, f"{split}_source{n}.tsv")


def load_source(split, n, data_dir=DATA_DIR, usecols=None):
    return read_tsv(source_path(split, n, data_dir), usecols=usecols)


def load_ground_truth(data_dir=DATA_DIR):
    gt = read_tsv(os.path.join(data_dir, "train", "train_ground_truth.tsv"))
    gt["matched_ids"] = gt["matched_entity_ids"].map(lambda v: [x.strip() for x in v.split(",") if x.strip()])
    return gt


def load_split_ids(name):
    """name: 'train' or 'holdout'"""
    with open(os.path.join(SPLIT_DIR, f"{name}_s1_ids.txt"), encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def cache_path(split, n):
    return os.path.join(CACHE_DIR, f"{split}_source{n}.parquet")


def load_normalized(split, n, columns=None):
    return pd.read_parquet(cache_path(split, n), columns=columns)