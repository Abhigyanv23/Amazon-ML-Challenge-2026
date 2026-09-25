"""
src/normalization.py
Rule-based normalization (no learned parameters, no external data).
Maps (US states, Indian states incl. Hindi/Tamil names, French regions/departments,
street abbreviations) come from general language knowledge, not business data.

Adds columns (originals kept):
  country_key, name_norm, name_core, name_legal, name_nospace, name_nonlatin,
  addr_norm, addr_nums, addr_state, addr_blank

Usage (repo root):
  python src/normalization.py --demo
  python src/normalization.py --eval            # train-fold true pairs vs random same-country pairs
  python src/normalization.py --split train     # -> experiments/cache/train_source{1,2,3}.parquet
  python src/normalization.py --split test
"""
import argparse
import os
import re
import time
import unicodedata
from multiprocessing import Pool

import numpy as np
import pandas as pd

from data_loading import (CACHE_DIR, cache_path, load_ground_truth, load_source,
                          load_split_ids)

# ------------------------------------------------------------------ unicode / basic
_EXTRA = str.maketrans({"ø": "o", "Ø": "o", "ł": "l", "Ł": "l", "ß": "ss", "æ": "ae", "Æ": "ae",
                        "œ": "oe", "Œ": "oe", "đ": "d", "ı": "i"})
_PUNCT = re.compile(r"[!-/:-@\[-`{-~\u00A0-\u00BF\u00D7\u00F7\u2000-\u206F\u20A0-\u20CF"
                    r"\u2100-\u214F\u0964\u0965\u3000-\u303F]")
_SPACE = re.compile(r"\s+")


def squash(s):
    return _SPACE.sub(" ", s).strip()


def _fold_latin_accents(s):
    # Drop combining marks only after ASCII letters (é->e) so Devanagari/Tamil vowel signs survive.
    out = []
    for ch in unicodedata.normalize("NFKD", s):
        if unicodedata.combining(ch) and out and out[-1].isascii():
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def base_clean(s):
    s = unicodedata.normalize("NFKC", str(s))
    if not s.isascii():
        s = _fold_latin_accents(s.translate(_EXTRA))
    return s.lower()


def has_nonlatin(s):
    return (not s.isascii()) and any(ch.isalpha() and ord(ch) > 0x24F for ch in s)


def light_norm(s):
    """EDA-level baseline, used only for before/after comparison."""
    s = unicodedata.normalize("NFKC", str(s)).lower()
    return squash(re.sub(r"[^\w\s]", " ", s))


def _key(s):
    return squash(_PUNCT.sub(" ", base_clean(s)))


def _parse_map(raw):
    m = {}
    for item in raw.split(","):
        k, v = item.rsplit(":", 1)
        m[_key(k)] = v.strip()
    for v in set(m.values()):
        m.setdefault(v, v)
    return m


# ------------------------------------------------------------------ country
_COUNTRY_KEY = {"us": "us", "usa": "us", "united states": "us", "india": "india", "in": "india",
                "france": "france", "fr": "france"}


def normalize_country(c):
    return str(c).strip()          # open set: unknown countries pass through unchanged


def country_key(c):
    c = str(c).strip().lower()
    return _COUNTRY_KEY.get(c, c)


# ------------------------------------------------------------------ names
NAME_ABBR = {"pvt": "private", "pte": "private", "ltd": "limited", "ltda": "limited",
             "corp": "corporation", "inc": "incorporated", "incorp": "incorporated", "co": "company",
             "intl": "international", "bros": "brothers", "svcs": "services", "svc": "service",
             "mfg": "manufacturing", "assoc": "associates", "assocs": "associates", "ctr": "center",
             "centre": "center", "mgmt": "management", "natl": "national", "univ": "university",
             "hosp": "hospital", "grp": "group", "cie": "compagnie", "ets": "etablissements"}
LEGAL_ANY = {"private", "limited", "llp", "llc", "incorporated", "corporation", "pllc", "plc", "lp",
             "sarl", "sas", "sasu", "eurl", "sci", "snc", "selarl", "gmbh"}
LEGAL_TAIL = {"company", "pc", "pa", "sa", "public"}          # only stripped at the end
TAIL_JUNK = {"and", "of", "the"}
LEAD_DROP = {"the"}

_DOMAIN = re.compile(r"\b(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:co\.in|com|net|org|biz|info|in|fr|us|io)\b")
_HASHNUM = re.compile(r"#\s*\d+")
_POSS = re.compile(r"['\u2019`]s\b")


def normalize_name(raw):
    """-> (name_norm, name_core, name_legal, name_nospace, name_nonlatin)"""
    s = base_clean(raw)
    nonlatin = int(has_nonlatin(s))
    s = _DOMAIN.sub(r" \1 ", s)
    s = _HASHNUM.sub(" ", s)
    s = _POSS.sub("", s)
    s = s.replace("&", " and ").replace(".", "")
    s = _PUNCT.sub(" ", s)
    toks = [NAME_ABBR.get(t, t) for t in s.split()]
    norm = " ".join(toks)

    legal = {t for t in toks if t in LEGAL_ANY}
    core = [t for t in toks if t not in LEGAL_ANY]
    while core and (core[-1] in LEGAL_TAIL or core[-1] in TAIL_JUNK):
        t = core.pop()
        if t in LEGAL_TAIL:
            legal.add(t)
    while core and core[0] in LEAD_DROP:
        core.pop(0)
    if not core:                      # never let a name collapse to nothing
        core = toks
    core_s = " ".join(core)
    return norm, core_s, " ".join(sorted(legal)), core_s.replace(" ", ""), nonlatin


# ------------------------------------------------------------------ addresses
ADDR_COMMON = {"apartment": "apt", "building": "bldg", "floor": "fl", "flr": "fl", "number": "no", "num": "no"}
ADDR_US = {"street": "st", "road": "rd", "drive": "dr", "avenue": "ave", "av": "ave", "boulevard": "blvd",
           "lane": "ln", "terrace": "ter", "place": "pl", "court": "ct", "highway": "hwy", "parkway": "pkwy",
           "circle": "cir", "square": "sq", "trail": "trl", "suite": "ste", "north": "n", "south": "s",
           "east": "e", "west": "w", "northeast": "ne", "northwest": "nw", "southeast": "se",
           "southwest": "sw", "mount": "mt", "fort": "ft", "saint": "st", "expressway": "expy",
           "freeway": "fwy", "center": "ctr", "centre": "ctr"}
ADDR_IN = {"road": "rd", "street": "st", "sector": "sec", "sect": "sec", "nr": "near", "opposite": "opp",
           "extension": "extn", "ext": "extn", "phase": "ph", "block": "blk", "lane": "ln",
           "avenue": "ave", "industrial": "indl", "cross": "crs"}
ADDR_FR = {"r": "rue", "bd": "bd", "bld": "bd", "blvd": "bd", "boulevard": "bd", "av": "av", "ave": "av",
           "avenue": "av", "pl": "pl", "place": "pl", "imp": "imp", "impasse": "imp", "ch": "chemin",
           "che": "chemin", "chem": "chemin", "rte": "rte", "route": "rte", "all": "allee",
           "fbg": "fbg", "faubourg": "fbg", "qu": "quai", "sq": "sq", "square": "sq", "crs": "cours",
           "st": "saint", "ste": "sainte", "pass": "passage", "res": "residence"}
ADDR_MAPS = {"us": {**ADDR_COMMON, **ADDR_US}, "india": {**ADDR_COMMON, **ADDR_IN},
             "france": {**ADDR_COMMON, **ADDR_FR}}

US_STATES = _parse_map(
    "alabama:al,alaska:ak,arizona:az,arkansas:ar,california:ca,colorado:co,connecticut:ct,delaware:de,"
    "district of columbia:dc,florida:fl,georgia:ga,hawaii:hi,idaho:id,illinois:il,indiana:in,iowa:ia,"
    "kansas:ks,kentucky:ky,louisiana:la,maine:me,maryland:md,massachusetts:ma,michigan:mi,minnesota:mn,"
    "mississippi:ms,missouri:mo,montana:mt,nebraska:ne,nevada:nv,new hampshire:nh,new jersey:nj,"
    "new mexico:nm,new york:ny,north carolina:nc,north dakota:nd,ohio:oh,oklahoma:ok,oregon:or,"
    "pennsylvania:pa,rhode island:ri,south carolina:sc,south dakota:sd,tennessee:tn,texas:tx,utah:ut,"
    "vermont:vt,virginia:va,washington:wa,west virginia:wv,wisconsin:wi,wyoming:wy,puerto rico:pr")
IN_STATES = _parse_map(
    "andhra pradesh:ap,arunachal pradesh:ar,assam:as,bihar:br,chhattisgarh:cg,chattisgarh:cg,ct:cg,goa:ga,"
    "gujarat:gj,haryana:hr,himachal pradesh:hp,jharkhand:jh,karnataka:ka,kerala:kl,madhya pradesh:mp,"
    "maharashtra:mh,manipur:mn,meghalaya:ml,mizoram:mz,nagaland:nl,odisha:od,orissa:od,or:od,punjab:pb,"
    "rajasthan:rj,sikkim:sk,tamil nadu:tn,tamilnadu:tn,telangana:tg,ts:tg,tripura:tr,uttar pradesh:up,"
    "uttarakhand:uk,uttaranchal:uk,ut:uk,west bengal:wb,delhi:dl,nct of delhi:dl,jammu and kashmir:jk,"
    "jammu kashmir:jk,ladakh:la,puducherry:py,pondicherry:py,chandigarh:ch,"
    "andaman and nicobar islands:an,lakshadweep:ld,"
    "उत्तर प्रदेश:up,दिल्ली:dl,महाराष्ट्र:mh,मध्य प्रदेश:mp,राजस्थान:rj,बिहार:br,गुजरात:gj,हरियाणा:hr,"
    "पंजाब:pb,कर्नाटक:ka,तमिलनाडु:tn,तमिल नाडु:tn,केरल:kl,पश्चिम बंगाल:wb,तेलंगाना:tg,आंध्र प्रदेश:ap,"
    "ओडिशा:od,झारखंड:jh,झारखण्ड:jh,छत्तीसगढ़:cg,उत्तराखंड:uk,उत्तराखण्ड:uk,असम:as,गोवा:ga,"
    "हिमाचल प्रदेश:hp,चंडीगढ़:ch,தமிழ்நாடு:tn")
FR_REGIONS = _parse_map(
    "ile de france:idf,hauts de france:hdf,nouvelle aquitaine:naq,pays de la loire:pdl,bretagne:bre,"
    "normandie:nor,grand est:ges,bourgogne franche comte:bfc,centre val de loire:cvl,"
    "auvergne rhone alpes:ara,provence alpes cote d azur:pac,paca:pac,occitanie:occ,corse:cor,"
    "nord:hdf,pas de calais:hdf,somme:hdf,aisne:hdf,oise:hdf,gironde:naq,landes:naq,dordogne:naq,"
    "lot et garonne:naq,pyrenees atlantiques:naq,loire atlantique:pdl,vendee:pdl,maine et loire:pdl,"
    "sarthe:pdl,mayenne:pdl")
STATE_MAPS = {"us": US_STATES, "india": IN_STATES, "france": FR_REGIONS}

_NUM = re.compile(r"\d+")
_ALNUM_SPLIT = re.compile(r"(?<=\d)(?=[a-z]{3,})")      # 15south -> 15 south (keeps 3906a, 33rd)


def normalize_address(raw, country):
    """-> (addr_norm, addr_nums, addr_state, addr_blank)"""
    if not str(raw).strip():
        return "", "", "", 1
    ck = country_key(country)
    amap = ADDR_MAPS.get(ck, ADDR_COMMON)
    smap = STATE_MAPS.get(ck, {})
    s = base_clean(raw).replace("&", " and ")

    nums = []
    for x in _NUM.findall(s):
        x = x.lstrip("0") or "0"
        if x not in nums:
            nums.append(x)

    state, comps = "", []
    for c in s.split(","):
        c = squash(_PUNCT.sub(" ", c))
        if not c:
            continue
        code = smap.get(c)
        if code:                       # a whole comma-component that is a state/region
            state = state or code
            comps.append(code)
            continue
        c = _ALNUM_SPLIT.sub(" ", c)
        comps.append(" ".join(amap.get(t, t) for t in c.split()))
    return " ".join(comps), " ".join(nums), state, 0


# ------------------------------------------------------------------ frames
COLS = ["name_norm", "name_core", "name_legal", "name_nospace", "name_nonlatin",
        "addr_norm", "addr_nums", "addr_state", "addr_blank"]


def _norm_chunk(rows):
    return [normalize_name(n) + normalize_address(a, c) for n, a, c in rows]


def normalize_frame(df, n_jobs=None, chunk=50_000):
    rows = list(zip(df["business_name"], df["business_address"], df["country"]))
    chunks = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]
    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 1)
    if n_jobs == 1 or len(chunks) <= 1:
        res = [_norm_chunk(c) for c in chunks]
    else:
        with Pool(n_jobs) as p:
            res = p.map(_norm_chunk, chunks)
    out = pd.DataFrame([r for part in res for r in part], columns=COLS, index=df.index)
    out["name_nonlatin"] = out["name_nonlatin"].astype("int8")
    out["addr_blank"] = out["addr_blank"].astype("int8")
    df = df.copy()
    df["country"] = df["country"].map(normalize_country)
    df["country_key"] = df["country"].map(country_key)
    return pd.concat([df, out], axis=1)


# ------------------------------------------------------------------ demo / eval
DEMO = [
    ("Yazzie Empire Enterprises LLC", "24800 Euclid Avenue, Euclid, OH", "US"),
    ("Yazzie-Empire Enterprises", "24800 Euclid Ave, Euclid, Ohio", "US"),
    ("tuckermetrotransalta.com", "534 Stinchomb Dr, Fsotoria, Ohio", "US"),
    ("surgical care associates of topeka inc #35740", "3906a 33rd Ter, Topeka, Kansas", "US"),
    ("Electronic Co &", "SR 58 PART BLDG 3/C WING FL13 JAGDISH NAGAR, AUNDH, PUNE, Maharashtra", "India"),
    ("Silver Fóundation Private", "F-25, Lucknow, UP", "India"),
    ("Silver Foundation Private Limited", "F-25, Asha Complex, Sec- 18, Lucknow, Uttar Pradesh", "India"),
    ("LIMITED SUNBEAM EAE", "H 15SOUTH EXTN PART I, SOUTH DELHI, NEW DELHI, दिल्ली", "India"),
    ("सिल्वर फाउंडेशन प्राइवेट लिमिटेड", "F-25, LUCKNOW HQ REGION, Uttar Pradesh", "India"),
    ("Chantons Tgavsax SA", "N°23 R. D'arras, Lille, Nord", "France"),
    ("Fédération du Musical", "3 R MIMEREL, ROUBAIX, Hauts-de-France", "France"),
    ("Rivas's Fisheries", "", "US"),
]


def run_demo():
    for n, a, c in DEMO:
        nn = normalize_name(n)
        aa = normalize_address(a, c)
        print(f"\n[{c}] {n!r} | {a!r}")
        print(f"   name_core={nn[1]!r}  legal={nn[2]!r}  nospace={nn[3]!r}  nonlatin={nn[4]}")
        print(f"   addr_norm={aa[0]!r}  nums={aa[1]!r}  state={aa[2]!r}  blank={aa[3]}")


def _jac(x, y):
    a, b = set(x.split()), set(y.split())
    u = a | b
    return 1.0 if not u else len(a & b) / len(u)


def _eq(x, y):
    return float(np.mean(x.to_numpy(dtype=object) == y.to_numpy(dtype=object)))


def _pair_stats(a, b):
    r = {}
    ln_a = [light_norm(x) for x in a.business_name]
    ln_b = [light_norm(x) for x in b.business_name]
    r["name exact (EDA light_norm)"] = float(np.mean([x == y for x, y in zip(ln_a, ln_b)]))
    r["name_norm exact"] = _eq(a.name_norm, b.name_norm)
    r["name_core exact"] = _eq(a.name_core, b.name_core)
    r["name_nospace exact"] = _eq(a.name_nospace, b.name_nospace)
    r["name jaccard (EDA light_norm)"] = float(np.mean([_jac(x, y) for x, y in zip(ln_a, ln_b)]))
    r["name_core jaccard"] = float(np.mean([_jac(x, y) for x, y in zip(a.name_core, b.name_core)]))
    r["addr jaccard (EDA light_norm)"] = float(np.mean(
        [_jac(light_norm(x), light_norm(y)) for x, y in zip(a.business_address, b.business_address)]))
    r["addr_norm jaccard"] = float(np.mean([_jac(x, y) for x, y in zip(a.addr_norm, b.addr_norm)]))
    both = [(x, y) for x, y in zip(a.addr_nums, b.addr_nums) if x and y]
    r["share >=1 number (both have nums)"] = float(np.mean([bool(set(x.split()) & set(y.split())) for x, y in both])) if both else float("nan")
    both = [(x, y) for x, y in zip(a.addr_state, b.addr_state) if x and y]
    r["state agrees (both have state)"] = float(np.mean([x == y for x, y in both])) if both else float("nan")
    r["no shared token name_core+addr_norm"] = float(np.mean(
        [not (set(w.split()) & set(x.split())) and not (set(y.split()) & set(z.split()))
         for w, x, y, z in zip(a.name_core, b.name_core, a.addr_norm, b.addr_norm)]))
    return r


def run_eval(n_pairs=200_000, seed=0):
    t = time.time()
    gt = load_ground_truth()
    gt = gt[gt.source1_entity_id.isin(set(load_split_ids("train")))]      # train fold only
    pairs = gt[["source1_entity_id", "matched_ids"]].explode("matched_ids").dropna()
    pairs = pairs.sample(n=min(n_pairs, len(pairs)), random_state=seed).reset_index(drop=True)

    s1 = load_source("train", 1)
    s1 = s1[s1.entity_id.isin(set(pairs.source1_entity_id))]
    need = set(pairs.matched_ids)
    oth = pd.concat([load_source("train", 2), load_source("train", 3)])
    oth = oth[oth.entity_id.isin(need)]
    print(f"loaded in {time.time() - t:.0f}s; normalizing {len(s1)} S1 + {len(oth)} S2/S3 records")

    a = normalize_frame(s1).set_index("entity_id").loc[pairs.source1_entity_id].reset_index()
    b = normalize_frame(oth).set_index("entity_id").loc[pairs.matched_ids].reset_index()

    rng = np.random.default_rng(seed)
    perm = np.arange(len(b))
    for _, idx in b.groupby("country_key").indices.items():
        perm[idx] = rng.permutation(idx)
    bn = b.iloc[perm].reset_index(drop=True)          # random same-country pairs (negatives)

    pos, neg = _pair_stats(a, b), _pair_stats(a, bn)
    print(f"\n{'metric':<40}{'TRUE pairs':>12}{'RANDOM pairs':>14}")
    for k in pos:
        print(f"{k:<40}{pos[k]:>12.4f}{neg[k]:>14.4f}")
    print("\nWant: TRUE up vs EDA baseline, RANDOM stays near 0 for 'exact' rows.")


def run_split(split):
    os.makedirs(CACHE_DIR, exist_ok=True)
    for n in (1, 2, 3):
        t = time.time()
        df = normalize_frame(load_source(split, n))
        df.to_parquet(cache_path(split, n), index=False)
        print(f"{split}_source{n}: {len(df)} rows in {time.time() - t:.0f}s | "
              f"nonlatin names {df.name_nonlatin.mean():.1%} | state found {(df.addr_state != '').mean():.1%} | "
              f"blank addr {df.addr_blank.mean():.1%} -> {cache_path(split, n)}")
        del df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--split", choices=["train", "test"])
    a = ap.parse_args()
    if a.demo:
        run_demo()
    if a.eval:
        run_eval()
    if a.split:
        run_split(a.split)
    if not (a.demo or a.eval or a.split):
        ap.print_help()


if __name__ == "__main__":
    main()