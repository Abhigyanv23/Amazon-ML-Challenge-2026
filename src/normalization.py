"""
src/normalization.py
Rule-based normalization (no learned parameters, no external data).
Maps (US states, Indian states incl. Hindi/Tamil names, French regions/departments,
street abbreviations) come from general language knowledge, not business data.

Adds columns (originals kept):
  country_key, name_norm, name_core, name_legal, name_nospace, name_nonlatin,
  addr_norm, addr_nums, addr_zip, addr_state, addr_blank, addr_state_src, name_skel, addr_hn

v2 changes: postcodes split into addr_zip (not house numbers); missing states filled from an
address-component -> state map LEARNED FROM THE SAME SPLIT'S S1 (unlabeled text only); FR abbreviations.
v3 changes:
  - state choice: last 2-letter code component wins, else FIRST spelled-out state
    (fixes "OH, Delaware, ..." and "DC, ..., Washington")
  - names: digits inside words mapped to letters (co1onial -> colonial); digit runs of 6+ removed
  - addresses: letter->digit split (fl13 -> fl 13); composite house number addr_hn ("0071/1" -> "71/1")
  - name_skel: consonant skeleton for cross-script matching. Indic scripts (Devanagari, Bengali,
    Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam) are mapped through the shared
    Unicode block layout onto one Devanagari consonant table; Latin names use the same classes.
    General script knowledge only (Unicode code charts), no external data.

Usage (repo root):
  python src/normalization.py --demo
  python src/normalization.py --eval            # train-fold true pairs vs random same-country pairs
  python src/normalization.py --split train     # -> experiments/cache/train_source{1,2,3}.parquet
  python src/normalization.py --split test
"""
import argparse
import os
from collections import Counter
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
_LEGAL_WORDS = ["private", "limited", "pvt", "ltd", "llp", "llc", "incorporated", "inc", "corporation",
                "corp", "company", "co", "public", "sarl", "sas", "sasu", "eurl", "gmbh", "plc", "pllc"]

_DOMAIN = re.compile(r"\b(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:co\.in|com|net|org|biz|info|in|fr|us|io)\b")
_HASHNUM = re.compile(r"#\s*\d+")
_POSS = re.compile(r"['\u2019`]s\b")
_DIGIT_IN_WORD = re.compile(r"(?<=[a-z])[0134578](?=[a-z])")
_DIGIT_MAP = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"}
_LONG_DIGITS = re.compile(r"(?<![a-z0-9])\d{6,}(?![a-z0-9])")


def normalize_name(raw):
    """-> (name_norm, name_core, name_legal, name_nospace, name_nonlatin, name_skel)"""
    s = base_clean(raw)
    nonlatin = int(has_nonlatin(s))
    s = _DOMAIN.sub(r" \1 ", s)
    s = _HASHNUM.sub(" ", s)
    s = _LONG_DIGITS.sub(" ", s)
    s = _DIGIT_IN_WORD.sub(lambda m: _DIGIT_MAP[m.group(0)], s)
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
    return norm, core_s, " ".join(sorted(legal)), core_s.replace(" ", ""), nonlatin, skeleton(core_s)


# ------------------------------------------------------------------ cross-script skeleton
# Devanagari offsets (code point - 0x0900) -> consonant class. Voicing collapsed (g->k, j->c, d->t, b->p)
_DEV = {0x01: "n", 0x02: "n", 0x15: "k", 0x16: "k", 0x17: "k", 0x18: "k", 0x19: "n", 0x1A: "c", 0x1B: "c",
        0x1C: "c", 0x1D: "c", 0x1E: "n", 0x1F: "t", 0x20: "t", 0x21: "t", 0x22: "t", 0x23: "n", 0x24: "t",
        0x25: "t", 0x26: "t", 0x27: "t", 0x28: "n", 0x29: "n", 0x2A: "p", 0x2B: "f", 0x2C: "p", 0x2D: "p",
        0x2E: "m", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l", 0x35: "v", 0x36: "s", 0x37: "s",
        0x38: "s", 0x39: "h", 0x58: "k", 0x59: "k", 0x5A: "k", 0x5B: "c", 0x5C: "t", 0x5D: "t", 0x5E: "f"}
_LAT_DIGRAPHS = [("mp", "np"), ("mb", "np"), ("ch", "c"), ("sh", "s"), ("th", "t"), ("ph", "f"), ("kh", "k"),
                 ("gh", "k"), ("bh", "p"), ("dh", "t"), ("jh", "c"), ("ck", "k")]
_LAT = {"b": "p", "c": "k", "d": "t", "f": "f", "g": "k", "h": "h", "j": "c", "k": "k", "l": "l", "m": "m",
        "n": "n", "p": "p", "q": "k", "r": "r", "s": "s", "t": "t", "v": "v", "w": "v", "x": "ks", "z": "c"}
_REPEAT = re.compile(r"(.)\1+")


def _skel_word(w):
    if w.isascii():
        for a, b in _LAT_DIGRAPHS:
            w = w.replace(a, b)
        x = "".join(_LAT.get(ch, "") for ch in w)
    else:
        x = "".join(_DEV.get((ord(ch) - 0x0900) % 0x80, "") if 0x0900 <= ord(ch) < 0x0D80 else ""
                    for ch in w)
    return _REPEAT.sub(r"\1", x)


_LEGAL_SKEL = None


def _init_legal_skel():
    global _LEGAL_SKEL
    _LEGAL_SKEL = set()
    _LEGAL_SKEL = {_skel_word(w) for w in _LEGAL_WORDS} - {""}


def skeleton(name):
    """Consonant skeleton of a name (Latin or Indic script), legal words removed."""
    if _LEGAL_SKEL is None:
        _init_legal_skel()
    out = [k for k in (_skel_word(w) for w in name.split()) if k and k not in _LEGAL_SKEL]
    return " ".join(out)


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
           "st": "saint", "ste": "sainte", "pass": "passage", "res": "residence", "q": "quai",
           "appt": "apt", "n": "no"}
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
_ALPHA_DIGIT = re.compile(r"(?<=[a-z]{2})(?=\d)")        # fl13 -> fl 13 (keeps d1, b123)
_HN_PART = r"(?:[a-z]{1,2}|[a-z]?\d+[a-z]?)"
_HN_COMP = re.compile(r"(?<![a-z0-9])(" + _HN_PART + r"(?:\s?[/-]\s?" + _HN_PART + r")+)(?![a-z0-9])")


def _composite_hn(s):
    """First composite house number like 71/1, 3-30-28/b/1, d-21 -> canonical 'd/21'."""
    for m in _HN_COMP.finditer(s):
        parts = re.split(r"\s?[/-]\s?", m.group(1))
        if not any(ch.isdigit() for ch in m.group(1)):
            continue
        if parts[0].isdigit() and len(parts[0]) >= 5:      # zip+4 or PIN-like, not a house number
            continue
        return "/".join((p.lstrip("0") or "0") if p[:1].isdigit() else p for p in parts)
    return ""
_IN_PIN_SPLIT = re.compile(r"(?<!\d)([1-8]\d{2}) (\d{3})(?!\d)")   # "600 042" -> "600042"
_US_ZIP4 = re.compile(r"(?<!\d)(\d{5}) (\d{4})$")                  # "75001 1234" -> "75001"
_RE_ZIP = {"us": re.compile(r"\d{5}"), "france": re.compile(r"\d{5}"), "india": re.compile(r"[1-8]\d{5}")}


def _is_zip(t, j, n, ck):
    rx = _RE_ZIP.get(ck)
    if rx is None or not rx.fullmatch(t):
        return False
    if ck == "us":                     # US house numbers can be 5 digits: ZIP only as last token, not first
        return j == n - 1 and (n == 1 or j > 0)
    return True


def _clean_component(c, ck):
    c = squash(_PUNCT.sub(" ", c))
    if ck == "india":
        c = _IN_PIN_SPLIT.sub(r"\1\2", c)
    elif ck == "us":
        c = _US_ZIP4.sub(r"\1", c)
    return c


def normalize_address(raw, country):
    """-> (addr_norm, addr_nums, addr_zip, addr_state, addr_blank, addr_hn)"""
    if not str(raw).strip():
        return "", "", "", "", 1, ""
    ck = country_key(country)
    amap = ADDR_MAPS.get(ck, ADDR_COMMON)
    smap = STATE_MAPS.get(ck, {})
    s = base_clean(raw).replace("&", " and ")
    hn = _composite_hn(s)

    nums, zips, comps = [], [], []
    st_codes, st_names = [], []
    for c in s.split(","):
        c = _clean_component(c, ck)
        if not c:
            continue
        toks = c.split()
        keep = []
        for j, t in enumerate(toks):
            if _is_zip(t, j, len(toks), ck):
                if t not in zips:
                    zips.append(t)
            else:
                keep.append(t)
        if not keep:
            continue
        c = " ".join(keep)
        code = smap.get(c)
        if code:                       # whole component is a state/region
            (st_codes if c == code else st_names).append(code)
            comps.append(code)
            continue
        for x in _NUM.findall(c):
            x = x.lstrip("0") or "0"
            if x not in nums:
                nums.append(x)
        c = _ALPHA_DIGIT.sub(" ", _ALNUM_SPLIT.sub(" ", c))
        comps.append(" ".join(amap.get(t, t) for t in c.split()))
    state = st_codes[-1] if st_codes else (st_names[0] if st_names else "")
    return " ".join(comps), " ".join(nums), " ".join(zips), state, 0, hn


# ------------------------------------------------------------------ learned state fill (data-derived)
def _comp_keys(raw, ck):
    """Address components as lookup keys: cleaned, abbreviations mapped, digits/postcodes removed."""
    amap = ADDR_MAPS.get(ck, ADDR_COMMON)
    keys = []
    for c in base_clean(raw).replace("&", " and ").split(","):
        toks = [amap.get(t, t) for t in _clean_component(c, ck).split() if not any(ch.isdigit() for ch in t)]
        if toks:
            keys.append(" ".join(toks))
    return keys


def build_state_map(s1n, min_count=20, purity=0.98):
    """(country, component) -> state, learned from S1 records of the same split (which all have a state)."""
    cnt, tot = Counter(), Counter()
    for addr, ck, st in zip(s1n.business_address, s1n.country_key, s1n.addr_state):
        if not st:
            continue
        smap = STATE_MAPS.get(ck, {})
        for k in set(_comp_keys(addr, ck)):
            if k in smap:
                continue
            cnt[(ck, k, st)] += 1
            tot[(ck, k)] += 1
    return {(ck, k): st for (ck, k, st), n in cnt.items()
            if n >= min_count and n / tot[(ck, k)] >= purity}


def fill_state(df, smap_learned):
    """Fill empty addr_state from learned components (majority vote). Sets addr_state_src=2."""
    miss = np.where((df.addr_state.to_numpy(dtype=object) == "") & (df.addr_blank.to_numpy() == 0))[0]
    addrs = df.business_address.to_numpy(dtype=object)[miss]
    cks = df.country_key.to_numpy(dtype=object)[miss]
    rows, vals = [], []
    for i, a, ck in zip(miss, addrs, cks):
        votes = Counter(smap_learned[(ck, k)] for k in _comp_keys(a, ck) if (ck, k) in smap_learned)
        if votes:
            top = votes.most_common(2)
            if len(top) == 1 or top[0][1] > top[1][1]:
                rows.append(i); vals.append(top[0][0])
    if rows:
        st = df.addr_state.to_numpy(dtype=object).copy()
        src = df.addr_state_src.to_numpy().copy()
        st[rows] = vals
        src[rows] = 2
        df["addr_state"] = st
        df["addr_state_src"] = src.astype("int8")
    return df, len(rows)


# ------------------------------------------------------------------ frames
COLS = ["name_norm", "name_core", "name_legal", "name_nospace", "name_nonlatin", "name_skel",
        "addr_norm", "addr_nums", "addr_zip", "addr_state", "addr_blank", "addr_hn"]


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
    df = pd.concat([df, out], axis=1)
    df["addr_state_src"] = (df["addr_state"] != "").astype("int8")     # 1 explicit, 0 none, 2 learned
    return df


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
    ("Innovative Entergy", "19 Johnson Lane, Unit Apartment 2, Washington, NY", "US"),
    ("Blue Program", "261 Jo Mar Road, Ardmore, AL 36049-1234", "US"),
    ("Amicale", "3 R Mimerel, 59100 Roubaix", "France"),
    ("Kumar Stores", "Old No#75, Chennai 600 042., TN", "India"),
    ("Heartland Holding Company", "OH, Delaware, 100 Georgetowne Drive, Unit 105", "US"),
    ("Dental Precision Partners", "District of Columbia, 1001 4th Street, Washington, # 502", "US"),
    ("Co1onial Preparatory Aachemdd", "11 JESSRMINE AVENUE, SAINT PAUL, MN", "US"),
    ("UPTOWN PALNITSE INC - 1566784888", "", "US"),
    ("Balaji Industries", "H No 71/1, Bhawani Nagar, Meerut, Uttar Pradesh", "India"),
    ("बालाजी इंडस्ट्रीज", "0071/1, Meerut, UP", "India"),
    ("Lotus Consulting Private Limited", "Flat No. 601, Mumbai, Floor -6, Maharashtra", "India"),
    ("लोटस कंसल्टिंग प्राइवेट लिमिटेड", "DOOR NO 601/8, MUMBAI, Maharashtra", "India"),
    ("White Software Private Limited", "D-1 Arnav Appartment - 2Opp Arjav Appartment, Surat, Gujarat", "India"),
    ("વ્હાઇટ સોફ્ટવેર પ્રાઇવેટ લિમિટેડ", "D-1 ARNAV APPARTMENT - 2OPP ARJAV APPARTEMNT, SURAT, Gujarat", "India"),
    ("Modern Alpha Media Private Limited", "A-34, 1St Floor Sector-49, Noida, Uttar Pradesh", "India"),
    ("मॉडर्न अल्फा मीडिया प्राइवेट लिमिटेड", "Noida, Door No ##435 A-34, UP, 1St Floor Sector-49", "India"),
]


def run_demo():
    for n, a, c in DEMO:
        nn = normalize_name(n)
        aa = normalize_address(a, c)
        print(f"\n[{c}] {n!r} | {a!r}")
        print(f"   name_core={nn[1]!r}  legal={nn[2]!r}  nonlatin={nn[4]}  skel={nn[5]!r}")
        print(f"   addr_norm={aa[0]!r}  nums={aa[1]!r}  zip={aa[2]!r}  state={aa[3]!r}  hn={aa[5]!r}  blank={aa[4]}")


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

    smap = build_state_map(normalize_frame(load_source("train", 1)))
    print(f"learned state map: {len(smap)} components")
    a = normalize_frame(s1).set_index("entity_id").loc[pairs.source1_entity_id].reset_index()
    b, n_fill = fill_state(normalize_frame(oth), smap)
    print(f"filled state for {n_fill} S2/S3 records")
    b = b.set_index("entity_id").loc[pairs.matched_ids].reset_index()

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
    smap = None
    for n in (1, 2, 3):
        t = time.time()
        df = normalize_frame(load_source(split, n))
        if n == 1:
            smap = build_state_map(df)
            print(f"learned state map from {split}_source1: {len(smap)} components")
            filled = 0
        else:
            df, filled = fill_state(df, smap)
        df.to_parquet(cache_path(split, n), index=False)
        found = (df.addr_state != "").groupby(df.country_key).mean()
        print(f"{split}_source{n}: {len(df)} rows in {time.time() - t:.0f}s | state filled {filled} | "
              f"state found " + ", ".join(f"{k}={v:.1%}" for k, v in found.items()) +
              f" | zip present {(df.addr_zip != '').mean():.1%} -> {cache_path(split, n)}")
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