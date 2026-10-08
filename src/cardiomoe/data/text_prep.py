"""Client B text preparation: language detection, translation, diagnostic-term masking (Section 5.5).

    python -m cardiomoe.data.text_prep --prepare [--debug] [--mask_scope nonrhythm|diagnostic|all]

Writes to data/processed/ (or .../debug/):
    ptbxl_reports_en.parquet   translation cache (ecg_id, lang, report, report_en); computed once,
                               re-runs only translate reports that are not cached yet
    ptbxl_text.parquet         ecg_id, lang, report_raw (oracle / upper bound), report_masked (default)
    mask_terms.txt             the exact leakage-term list used for report_masked

Mask scope for the scp_statements.csv descriptions and codes (items a and b of 5.5):
    nonrhythm (default)  diagnostic + form statements. Rhythm statements are kept because the plan
                         asks to keep rhythm content and labels come only from diagnostic statements.
    diagnostic           diagnostic statements only
    all                  every statement, including rhythm
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from cardiomoe.data.ptbxl import load_database, processed_dir
from cardiomoe.utils import get_device, load_config, repo_path, set_seed

log = logging.getLogger(__name__)

LANGS = ("de", "en", "sv")
MT_MODELS = {"de": "Helsinki-NLP/opus-mt-de-en", "sv": "Helsinki-NLP/opus-mt-sv-en"}
MASK = "[MASK]"

# Item (c) of 5.5: English and German stems/phrases. A trailing * means "word starting with".
# "ischaem*" and "normal ekg" are spelling variants of listed terms seen in the reports and
# in the MT output ("ekg" is how opus-mt renders German "EKG").
STEMS = [
    "infarct*", "ischem*", "ischäm*", "ischaem*", "hypertroph*", "block*", "bundle*", "myocard*",
    "injury", "st elevation", "st depression", "t wave", "t-wave", "q wave", "normal ecg",
    "normal ekg", "abnormal", "pathologic*", "lvh", "rvh", "mi",
]


# --------------------------------------------------------------------------- language + translation
def detect_language(text: str) -> str:
    """Most probable of de/en/sv according to langdetect; German when undecidable (the PTB-XL majority)."""
    from langdetect import DetectorFactory, detect_langs
    from langdetect.lang_detect_exception import LangDetectException

    DetectorFactory.seed = 0
    try:
        for cand in detect_langs(text):
            if cand.lang in LANGS:
                return cand.lang
    except LangDetectException:
        pass
    return "de"


def translate(texts: list[str], lang: str, device, batch_size: int = 32) -> list[str]:
    from transformers import MarianMTModel, MarianTokenizer
    import torch

    name = MT_MODELS[lang]
    tok = MarianTokenizer.from_pretrained(name)
    model = MarianMTModel.from_pretrained(name, use_safetensors=False).to(device).eval()
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # length-bucketed batches
    out = [""] * len(texts)
    for s in tqdm(range(0, len(order), batch_size), desc=f"translate {lang}->en"):
        idx = order[s : s + batch_size]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, num_beams=4, max_new_tokens=512)
        for i, t in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
            out[i] = t
    return out


def build_translation_cache(reports: pd.Series, cache_path: Path, device) -> pd.DataFrame:
    cache = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame(
        columns=["lang", "report", "report_en"], index=pd.Index([], name="ecg_id"))
    todo = reports[~reports.index.isin(cache.index)]
    if len(todo):
        # Sources are lowercased first: some Swedish reports have stray uppercase umlauts
        # ("fÖrÄndring") that derail both langdetect and the MT model.
        new = pd.DataFrame({"report": todo.astype(str).str.lower()})
        new["lang"] = [detect_language(t) for t in tqdm(new.report, desc="langdetect")]
        new["report_en"] = new.report
        for lang in MT_MODELS:
            sel = new.lang == lang
            if sel.any():
                new.loc[sel, "report_en"] = translate(new.loc[sel, "report"].tolist(), lang, device)
        cache = pd.concat([cache, new[["lang", "report", "report_en"]]]).sort_index()
        cache.index.name = "ecg_id"
        cache.to_parquet(cache_path)
    return cache


# --------------------------------------------------------------------------- masking
def normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", str(t).lower()).strip()


def build_mask_terms(scp: pd.DataFrame, scope: str = "nonrhythm") -> list[str]:
    if scope == "diagnostic":
        rows = scp[scp.diagnostic == 1]
    elif scope == "nonrhythm":
        rows = scp[scp.rhythm != 1]
    elif scope == "all":
        rows = scp
    else:
        raise ValueError(scope)
    terms = {normalize_text(d) for d in rows.description.dropna()}  # (a) descriptions
    terms |= {normalize_text(c) for c in rows.index}                # (b) SCP codes
    terms |= set(STEMS)                                             # (c) stems
    return sorted(t for t in terms if t)


def compile_mask(terms: list[str]) -> re.Pattern:
    """One alternation, longest terms first, matched on token boundaries."""
    parts = []
    for t in sorted(terms, key=len, reverse=True):
        stem = t.endswith("*")
        body = re.escape(t.rstrip("*")).replace(r"\ ", r"[\s\-]+")
        plural = r"s?" if " " in t or "-" in t else ""  # "t wave" also matches "t waves"
        parts.append(body + (r"\w*" if stem else plural))
    return re.compile(r"(?<![\w])(?:" + "|".join(parts) + r")(?![\w])")


def mask_text(text: str, pattern: re.Pattern) -> str:
    t = pattern.sub(MASK, text)
    t = re.sub(r"\[MASK\](?:[\s,;:/\-]*\[MASK\])+", MASK, t)  # collapse runs so length can't leak
    return re.sub(r"\s+", " ", t).strip()


# --------------------------------------------------------------------------- main
def load_text(cfg: dict, debug: bool = False) -> pd.DataFrame:
    return pd.read_parquet(processed_dir(cfg, debug) / "ptbxl_text.parquet")


def prepare(cfg: dict, debug: bool, scope: str) -> pd.DataFrame:
    out = processed_dir(cfg, debug)
    db, scp = load_database(repo_path(cfg["paths"]["ptbxl"]))
    reports = db.report
    if debug:
        reports = reports.sample(n=cfg["debug_n_samples"], random_state=cfg["seed"]).sort_index()
    device = get_device(cfg.get("device", "auto"), debug)
    cache = build_translation_cache(reports, out / "ptbxl_reports_en.parquet", device)
    cache = cache.loc[reports.index]
    coverage = cache.report_en.notna().mean()
    print(f"translation cache covers {coverage:.1%} of {len(reports)} reports; languages: "
          + ", ".join(f"{k}={v}" for k, v in cache.lang.value_counts().items()))

    terms = build_mask_terms(scp, scope)
    (out / "mask_terms.txt").write_text(f"# scope={scope}, {len(terms)} terms\n" + "\n".join(terms) + "\n")
    pat = compile_mask(terms)
    text = pd.DataFrame({"lang": cache.lang, "report_raw": cache.report_en.map(normalize_text)})
    text["report_masked"] = text.report_raw.map(lambda t: mask_text(t, pat))
    text.to_parquet(out / "ptbxl_text.parquet")

    frac = text.report_masked.str.contains(re.escape(MASK)).mean()
    print(f"{len(terms)} mask terms (scope={scope}); {frac:.1%} of reports contain at least one {MASK}")
    for _, r in text.sample(5, random_state=cfg["seed"]).iterrows():
        print(f"  raw:    {r.report_raw}\n  masked: {r.report_masked}")
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--mask_scope", default="nonrhythm", choices=["nonrhythm", "diagnostic", "all"])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    set_seed(cfg["seed"])
    if args.prepare:
        prepare(cfg, args.debug, args.mask_scope)


if __name__ == "__main__":
    main()
