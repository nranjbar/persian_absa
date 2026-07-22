# -*- coding: utf-8 -*-
"""
Kaggle-ready Persian ABSA rerun pipeline — v3 all-human + Word-aspect-column support.

Modes:
1) prepare_annotation:
   Attach raw Persian review datasets in Kaggle using "Add Input".
   Exports three annotator CSV files for independent human annotation.

2) build_human_gold:
   Upload completed annotator_*.csv files as a Kaggle dataset.
   Exports human_only_test.csv from annotator consensus. All annotator_*.csv files are treated as human annotations.

3) train_evaluate:
   Runs LLaMA3 zero-shot, LLaMA3 fine-tuned variants, optional mT5 baseline, and exports tables.

Before training, run the install cell from the accompanying notebook.
"""

from __future__ import annotations

import ast
import gc
import json
import math
import os
import random
import re
import time
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None

import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split


# =========================
# 1. Configuration
# =========================

@dataclass
class Config:
    # Paths
    data_dir: str = "/kaggle/input"
    output_root: str = "/kaggle/working/absa_rerun_outputs"

    gpt_train_path: str = "train_tagged_ABSA_persian.csv"
    gpt_test_path: str = "test_tagged_ABSA_persian.csv"
    parsinlu_train_path: str = "ABSA_Dataset_train.csv"
    parsinlu_food_test_path: str = "ABSA_Dataset_food_test.csv"
    parsinlu_movie_test_path: str = "ABSA_Dataset_movie_test.csv"
    pars_absa_test_path: str = "test_Pars_ABSA.csv"
    human_only_test_path: str = "human_only_test.csv"

    # Model
    llama_model_name: str = "unsloth/llama-3-8b-bnb-4bit"
    mt5_model_name: str = "google/mt5-base"

    # Prompt and generation
    max_prompt_tokens: int = 256
    max_output_tokens: int = 128
    max_seq_length: int = 512
    generation_batch_size: int = 8   # Use 4 on T4 if OOM; 8 is faster on 2xT4/P100/A100.
    reuse_cached_predictions: bool = True
    run_zero_shot_once: bool = True
    train_batch_size: int = 2
    gradient_accumulation_steps: int = 4

    # Training
    run_seeds: Tuple[int, ...] = (3407,)  # For the paper use: (3407, 42, 1234, 2025, 2026)
    num_train_epochs: float = 1.0
    max_steps: int = -1               # -1 means use num_train_epochs. Use 80 only for a smoke test, not for final paper.
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    logging_steps: int = 10
    save_steps: int = 200
    eval_holdout_ratio: float = 0.0   # Set 0.1 if you want dev monitoring.

    # QLoRA / LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # These modules mean "LoRA adapters are inserted in all transformer layers for these projections".
    # If you keep these 7 modules, report them exactly in the paper.
    target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

    # Evaluation
    fuzzy_threshold: int = 50
    category_fuzzy_threshold: int = 80
    run_llama_training: bool = True
    run_llama_zero_shot: bool = True
    run_mt5_baseline: bool = False     # Turn on for Reviewer 1's "other generative models" comment.
    run_smoke_test_only: bool = False  # True: train/evaluate on tiny subsets just to test code.

    # Kaggle workflow mode:
    # "prepare_annotation": scan attached Kaggle datasets and export 150-300 raw reviews for humans.
    # "build_human_gold": merge annotator CSVs into human_only_test.csv.
    # "train_evaluate": run the full LLaMA3/mT5 evaluation.
    pipeline_mode: str = "train_evaluate"

    # Human-only annotation set creation
    input_root: str = "/kaggle/input"
    annotation_candidate_total: int = 240
    annotation_min_words: int = 5
    annotation_max_words: int = 60
    annotator_files_glob: str = "/kaggle/input/**/annotator_*.csv"
    raw_review_csv_paths: Tuple[str, ...] = ()  # optional manual list; otherwise auto-discover CSV files.


CFG = Config()
Path(CFG.output_root).mkdir(parents=True, exist_ok=True)


# =========================
# 2. Reproducibility and text normalization
# =========================

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


_ARABIC_TO_PERSIAN = str.maketrans({
    "ي": "ی", "ى": "ی", "ك": "ک", "ۀ": "ه", "ة": "ه", "ؤ": "و", "إ": "ا", "أ": "ا", "آ": "ا",
    "ٱ": "ا", "ء": "", "ئ": "ی",
})
_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")
_PUNCT = re.compile(r"[^\w\s\u0600-\u06FF]")


def normalize_persian(text: Any, *, keep_punct: bool = False) -> str:
    """Lightweight Persian normalization used consistently for leakage checking and fuzzy matching."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_ARABIC_TO_PERSIAN)
    text = _DIACRITICS.sub("", text)
    text = text.replace("\u200c", " ")  # half-space -> space for matching
    text = text.replace("ـ", "")
    if not keep_punct:
        text = _PUNCT.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


_SENTIMENT_MAP = {
    "positive": "positive", "pos": "positive", "1": "positive", "2": "positive", "+1": "positive", "+2": "positive",
    "very positive": "positive", "مثبت": "positive", "خیلی مثبت": "positive",
    "negative": "negative", "neg": "negative", "-1": "negative", "-2": "negative",
    "very negative": "negative", "منفی": "negative", "خیلی منفی": "negative",
    "neutral": "neutral", "neu": "neutral", "0": "neutral", "خنثی": "neutral", "خنثي": "neutral",
}


def normalize_sentiment(label: Any) -> str:
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return ""
    if isinstance(label, (int, np.integer)):
        return _SENTIMENT_MAP.get(str(int(label)), "")
    if isinstance(label, (float, np.floating)) and float(label).is_integer():
        return _SENTIMENT_MAP.get(str(int(label)), "")
    s = normalize_persian(str(label), keep_punct=True)
    s = s.replace("_", " ").strip()
    return _SENTIMENT_MAP.get(s, s)


def clean_value(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip()


# =========================
# 3. Data loading
# =========================

TEXT_COLS = ["text", "review", "sentence", "Sentence", "comment", "Input", "Review"]
ID_COLS = ["id", "review_id", "sentence_id", "SentenceId", "comment_id", "ID"]
ASPECT_COLS = ["aspect_term", "aspect", "Aspect", "Word", "word", "aspectTerm", "target", "term"]
CATEGORY_COLS = ["aspect_category", "category", "Category", "aspectCategory"]
OPINION_COLS = ["opinion", "Opinion", "opinion_term", "opinionTerm"]
SENTIMENT_COLS = ["sentiment", "label", "polarity", "Sentiment", "Label", "polarity_label"]


def find_col(df: pd.DataFrame, candidates: Iterable[str], required: bool = False, label: str = "column") -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    if required:
        raise ValueError(f"Could not find required {label}. Available columns: {list(df.columns)}")
    return None


def row_to_target(row: pd.Series,
                  aspect_col: Optional[str],
                  sentiment_col: Optional[str],
                  category_col: Optional[str],
                  opinion_col: Optional[str]) -> Optional[Dict[str, Any]]:
    aspect = clean_value(row[aspect_col]) if aspect_col else ""
    if not aspect:
        return None
    target = {
        "aspect": aspect,
        "category": clean_value(row[category_col]) if category_col else None,
        "opinion": clean_value(row[opinion_col]) if opinion_col else None,
        "sentiment": normalize_sentiment(row[sentiment_col]) if sentiment_col else "",
    }
    return target


def dedupe_targets(targets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for t in targets:
        key = (
            normalize_persian(t.get("aspect", "")),
            normalize_persian(t.get("category", "")),
            normalize_persian(t.get("opinion", "")),
            normalize_sentiment(t.get("sentiment", "")),
        )
        if key[0] and key not in seen:
            seen.add(key)
            out.append(t)
    return out


def load_row_level_absa(path: str, source: str) -> List[Dict[str, Any]]:
    """Load row-level ABSA files where each row is one aspect annotation."""
    if not path or not Path(path).exists():
        print(f"[SKIP] {source}: file not found -> {path}")
        return []

    df = pd.read_csv(path)
    text_col = find_col(df, TEXT_COLS, required=True, label=f"text column for {source}")
    id_col = find_col(df, ID_COLS, required=False)
    aspect_col = find_col(df, ASPECT_COLS, required=True, label=f"aspect column for {source}")
    sentiment_col = find_col(df, SENTIMENT_COLS, required=False)
    category_col = find_col(df, CATEGORY_COLS, required=False)
    opinion_col = find_col(df, OPINION_COLS, required=False)

    # Prefer stable ID; if absent group by normalized text.
    group_key = id_col if id_col else text_col
    records = []
    for gid, g in df.groupby(group_key, sort=False):
        text = clean_value(g[text_col].iloc[0])
        targets = []
        for _, row in g.iterrows():
            t = row_to_target(row, aspect_col, sentiment_col, category_col, opinion_col)
            if t is not None:
                targets.append(t)
        records.append({
            "id": str(gid),
            "text": text,
            "source": source,
            "targets": dedupe_targets(targets),
        })
    print(f"[LOAD] {source}: {len(records)} unique reviews from {len(df)} rows. Columns: text={text_col}, id={id_col}, aspect={aspect_col}, sentiment={sentiment_col}, category={category_col}, opinion={opinion_col}")
    return records


def records_to_frame(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in records:
        sentiments = [normalize_sentiment(t.get("sentiment")) for t in r["targets"] if normalize_sentiment(t.get("sentiment"))]
        rows.append({
            "source": r["source"],
            "id": r["id"],
            "text": r["text"],
            "n_aspects": len(r["targets"]),
            "n_words": len(normalize_persian(r["text"]).split()),
            "positive": sentiments.count("positive"),
            "negative": sentiments.count("negative"),
            "neutral": sentiments.count("neutral"),
        })
    return pd.DataFrame(rows)


def dataset_stats(records_by_name: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    out = []
    for name, records in records_by_name.items():
        if not records:
            continue
        f = records_to_frame(records)
        n_sent = len(f)
        total_sentiments = f[["positive", "negative", "neutral"]].sum().sum()
        out.append({
            "dataset": name,
            "unique_sentences": n_sent,
            "avg_aspects_per_sentence": f["n_aspects"].mean(),
            "avg_words_per_sentence": f["n_words"].mean(),
            "positive_pct": 100 * f["positive"].sum() / total_sentiments if total_sentiments else np.nan,
            "negative_pct": 100 * f["negative"].sum() / total_sentiments if total_sentiments else np.nan,
            "neutral_pct": 100 * f["neutral"].sum() / total_sentiments if total_sentiments else np.nan,
        })
    return pd.DataFrame(out)


def remove_text_leakage(train_records: List[Dict[str, Any]],
                        test_records: List[Dict[str, Any]],
                        split_name: str,
                        output_dir: Path) -> List[Dict[str, Any]]:
    train_texts = {normalize_persian(r["text"]) for r in train_records}
    kept, leaked = [], []
    for r in test_records:
        if normalize_persian(r["text"]) in train_texts:
            leaked.append(r)
        else:
            kept.append(r)
    if leaked:
        leak_path = output_dir / f"leakage_removed_{split_name}.json"
        with open(leak_path, "w", encoding="utf-8") as f:
            json.dump(leaked, f, ensure_ascii=False, indent=2)
        print(f"[LEAKAGE] Removed {len(leaked)} leaked test reviews from {split_name}. Saved: {leak_path}")
    else:
        print(f"[LEAKAGE] No train/test text overlap for {split_name}.")
    return kept



# =========================
# 3.5. Kaggle path resolver
# =========================

def resolve_kaggle_file(path_or_name: str) -> str:
    """Resolve simple filenames under /kaggle/input recursively. Keeps absolute paths unchanged."""
    if not path_or_name:
        return path_or_name
    p = Path(path_or_name)
    if p.is_absolute() and p.exists():
        return str(p)
    if p.exists():
        return str(p)
    for q in Path(CFG.input_root).rglob(p.name):
        if q.is_file():
            print(f"[PATH] {p.name} -> {q}")
            return str(q)
    return path_or_name


def resolve_all_config_paths() -> None:
    for attr in [
        "gpt_train_path", "gpt_test_path", "parsinlu_train_path",
        "parsinlu_food_test_path", "parsinlu_movie_test_path",
        "pars_absa_test_path", "human_only_test_path",
    ]:
        setattr(CFG, attr, resolve_kaggle_file(getattr(CFG, attr)))



def load_all_splits() -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    """Returns train_splits and test_splits dictionaries."""
    train_splits = {
        "gpt_assisted_human_validated_train": load_row_level_absa(CFG.gpt_train_path, "gpt_assisted_human_validated_train"),
        "parsinlu_train": load_row_level_absa(CFG.parsinlu_train_path, "parsinlu_train"),
    }
    test_splits = {
        "gpt_assisted_human_validated_test": load_row_level_absa(CFG.gpt_test_path, "gpt_assisted_human_validated_test"),
        "pars_absa_test": load_row_level_absa(CFG.pars_absa_test_path, "pars_absa_test"),
        "parsinlu_food_test": load_row_level_absa(CFG.parsinlu_food_test_path, "parsinlu_food_test"),
        "parsinlu_movie_test": load_row_level_absa(CFG.parsinlu_movie_test_path, "parsinlu_movie_test"),
        "human_only_test": load_row_level_absa(CFG.human_only_test_path, "human_only_test"),
    }
    test_splits = {k: v for k, v in test_splits.items() if len(v) > 0}
    if CFG.run_smoke_test_only:
        train_splits = {k: v[:16] for k, v in train_splits.items()}
        test_splits = {k: v[:8] for k, v in test_splits.items()}
    stats = dataset_stats({**train_splits, **test_splits})
    stats.to_csv(Path(CFG.output_root) / "dataset_statistics.csv", index=False)
    print("\n[DATASET STATISTICS]")
    print(stats.to_string(index=False))
    return train_splits, test_splits




# =========================
# 3.6. Human-only test-set preparation
# =========================

RAW_TEXT_COLS = [
    "comment", "comments", "comment_text", "commentText", "body", "content",
    "text", "review", "Review", "sentence", "description", "title"
]
RAW_LABEL_COLS = ["label", "sentiment", "rate", "rating", "score", "stars", "recommend"]


def persian_char_ratio(text: str) -> float:
    text = clean_value(text)
    if not text:
        return 0.0
    letters = re.findall(r"[A-Za-z\u0600-\u06FF]", text)
    if not letters:
        return 0.0
    persian = re.findall(r"[\u0600-\u06FF]", text)
    return len(persian) / len(letters)


def find_raw_text_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    lower_map = {c.lower(): c for c in df.columns}
    for cand in RAW_TEXT_COLS:
        if cand in df.columns:
            cols.append(cand)
        elif cand.lower() in lower_map:
            cols.append(lower_map[cand.lower()])
    for c in df.columns:
        if c in cols:
            continue
        if df[c].dtype == object:
            sample = df[c].dropna().astype(str).head(200)
            if len(sample) and sample.map(persian_char_ratio).mean() > 0.45:
                cols.append(c)
    return list(dict.fromkeys(cols))


def find_raw_label_column(df: pd.DataFrame) -> Optional[str]:
    return find_col(df, RAW_LABEL_COLS, required=False)


def is_known_absa_file(path: Path) -> bool:
    known = {
        "train_tagged_ABSA_persian.csv", "test_tagged_ABSA_persian.csv",
        "ABSA_Dataset_train.csv", "ABSA_Dataset_food_test.csv",
        "ABSA_Dataset_movie_test.csv", "test_Pars_ABSA.csv",
        "human_only_test.csv",
    }
    return path.name in known or path.name.startswith("annotator_")


def discover_raw_review_csvs() -> List[Path]:
    if CFG.raw_review_csv_paths:
        return [Path(resolve_kaggle_file(p)) for p in CFG.raw_review_csv_paths]
    out = []
    for p in Path(CFG.input_root).rglob("*.csv"):
        if is_known_absa_file(p):
            continue
        try:
            df = pd.read_csv(p, nrows=500)
            text_cols = find_raw_text_columns(df)
            if text_cols:
                out.append(p)
        except Exception:
            continue
    print("[RAW CSV DISCOVERY]")
    for p in out[:30]:
        print(" -", p)
    if len(out) > 30:
        print(f" ... {len(out)-30} more")
    return out


def load_existing_absa_texts_for_exclusion() -> set:
    """Exclude texts already used in model train/test files, so the human-only set is independent."""
    texts = set()
    for path in [
        CFG.gpt_train_path, CFG.gpt_test_path, CFG.parsinlu_train_path,
        CFG.parsinlu_food_test_path, CFG.parsinlu_movie_test_path,
        CFG.pars_absa_test_path, CFG.human_only_test_path,
    ]:
        rp = resolve_kaggle_file(path)
        if not rp or not Path(rp).exists():
            continue
        try:
            df = pd.read_csv(rp)
            tcol = find_col(df, TEXT_COLS, required=False)
            if tcol:
                texts.update(normalize_persian(x) for x in df[tcol].dropna().astype(str))
        except Exception:
            pass
    texts.discard("")
    return texts


def collect_raw_reviews_for_annotation() -> pd.DataFrame:
    """Create a pool of raw, non-ABSA Persian reviews from attached Kaggle input datasets."""
    resolve_all_config_paths()
    exclude_texts = load_existing_absa_texts_for_exclusion()
    rows = []
    for p in discover_raw_review_csvs():
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[SKIP RAW] {p}: {e}")
            continue
        text_cols = find_raw_text_columns(df)
        if not text_cols:
            continue
        label_col = find_raw_label_column(df)
        for i, row in df.iterrows():
            parts = []
            for c in text_cols[:2]:
                val = clean_value(row.get(c, ""))
                if val and val.lower() not in {"nan", "none"}:
                    parts.append(val)
            text = " ".join(parts).strip()
            norm = normalize_persian(text)
            n_words = len(norm.split())
            if not norm or norm in exclude_texts:
                continue
            if n_words < CFG.annotation_min_words or n_words > CFG.annotation_max_words:
                continue
            if persian_char_ratio(text) < 0.55:
                continue
            rows.append({
                "source_file": str(p.relative_to(Path(CFG.input_root))) if str(p).startswith(CFG.input_root) else str(p),
                "source_row": i,
                "raw_label": clean_value(row.get(label_col, "")) if label_col else "",
                "text": text,
                "normalized_text": norm,
                "n_words": n_words,
            })
    pool = pd.DataFrame(rows)
    if pool.empty:
        raise RuntimeError(
            "No raw review candidates found. In Kaggle, click Add Input and attach raw Persian review datasets "
            "such as Snappfood and Digikala comments, or set CFG.raw_review_csv_paths manually."
        )
    pool = pool.drop_duplicates("normalized_text").reset_index(drop=True)
    print(f"[RAW POOL] {len(pool)} candidate reviews after filtering and duplicate removal.")
    return pool


def sample_annotation_candidates() -> pd.DataFrame:
    """Balanced sample for a human-only ABSA test set."""
    pool = collect_raw_reviews_for_annotation()
    n = min(CFG.annotation_candidate_total, len(pool))
    groups = list(pool.groupby("source_file"))
    per_group = max(1, math.ceil(n / max(1, len(groups))))
    sampled = []
    for src, g in groups:
        take = min(per_group, len(g))
        sampled.append(g.sample(n=take, random_state=3407))
    cand = pd.concat(sampled, ignore_index=True)
    if len(cand) > n:
        cand = cand.sample(n=n, random_state=3407)
    elif len(cand) < n:
        rest = pool[~pool["normalized_text"].isin(cand["normalized_text"])]
        if len(rest):
            cand = pd.concat([cand, rest.sample(n=min(n-len(cand), len(rest)), random_state=3407)], ignore_index=True)
    cand = cand.sample(frac=1, random_state=3407).reset_index(drop=True)
    cand.insert(0, "review_id", [f"HUMAN_{i+1:04d}" for i in range(len(cand))])
    cand["annotations_json"] = ""
    cand["annotator_notes"] = ""
    out_dir = Path(CFG.output_root) / "human_annotation"
    out_dir.mkdir(parents=True, exist_ok=True)
    master_cols = ["review_id", "source_file", "source_row", "raw_label", "n_words", "text", "annotations_json", "annotator_notes"]
    master = cand[master_cols]
    master.to_csv(out_dir / "human_annotation_candidates_master.csv", index=False)
    for k in [1, 2, 3]:
        assign = cand[["review_id", "text", "annotations_json", "annotator_notes"]].copy()
        assign.to_csv(out_dir / f"annotator_{k}_assignment.csv", index=False)
    print(f"[ANNOTATION FILES SAVED] {out_dir}")
    print("Give annotator_1_assignment.csv, annotator_2_assignment.csv, annotator_3_assignment.csv to three native Persian annotators.")
    return master


def parse_annotation_json(x: Any) -> List[Dict[str, Any]]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    s = str(x).strip()
    if not s:
        return []
    try:
        obj = json.loads(s)
    except Exception:
        try:
            obj = ast.literal_eval(s)
        except Exception:
            return []
    if isinstance(obj, dict):
        obj = [obj]
    out = []
    if isinstance(obj, list):
        for item in obj:
            t = coerce_target_item(item)
            if t:
                out.append(t)
    return dedupe_targets(out)


def build_human_only_gold_from_annotators() -> pd.DataFrame:
    """Merge 3 annotator files. Keeps aspect items supported by at least two annotators."""
    files = sorted(Path("/kaggle/input").glob("**/annotator_*.csv")) + sorted((Path(CFG.output_root) / "human_annotation").glob("annotator_*_assignment.csv"))
    files = [p for p in files if p.exists()]
    if not files:
        raise RuntimeError("No annotator_*.csv files found. Upload completed annotator files as a Kaggle dataset or keep them in /kaggle/working/absa_rerun_outputs/human_annotation.")
    ann_rows = []
    for p in files:
        df = pd.read_csv(p)
        if "review_id" not in df.columns or "text" not in df.columns or "annotations_json" not in df.columns:
            print(f"[SKIP ANNOTATOR FILE] {p}: required columns missing.")
            continue
        annotator_id = re.sub(r"\W+", "_", p.stem)
        for _, row in df.iterrows():
            items = parse_annotation_json(row["annotations_json"])
            for item in items:
                ann_rows.append({
                    "review_id": row["review_id"],
                    "text": row["text"],
                    "annotator": annotator_id,
                    **item,
                })
    ann = pd.DataFrame(ann_rows)
    if ann.empty:
        raise RuntimeError("Annotator files were found, but no valid annotations_json entries were parsed.")
    gold_rows = []
    agreement_rows = []
    for rid, g in ann.groupby("review_id", sort=False):
        text = clean_value(g["text"].iloc[0])
        items = g.to_dict("records")
        used = set()
        clusters = []
        for i, item in enumerate(items):
            if i in used:
                continue
            cluster = [i]
            used.add(i)
            for j, other in enumerate(items):
                if j in used:
                    continue
                if similarity(item.get("aspect", ""), other.get("aspect", "")) >= 70:
                    cluster.append(j)
                    used.add(j)
            clusters.append(cluster)
        kept = 0
        for cluster in clusters:
            citems = [items[i] for i in cluster]
            annotators = sorted(set(x["annotator"] for x in citems))
            if len(annotators) < 2:
                continue
            kept += 1
            aspects = [clean_value(x.get("aspect")) for x in citems if clean_value(x.get("aspect"))]
            aspect = sorted(aspects, key=lambda z: (len(normalize_persian(z).split()), len(z)))[0] if aspects else ""
            sentiments = [normalize_sentiment(x.get("sentiment")) for x in citems if normalize_sentiment(x.get("sentiment"))]
            sentiment = pd.Series(sentiments).mode().iloc[0] if sentiments else ""
            cats = [clean_value(x.get("category")) for x in citems if clean_value(x.get("category"))]
            ops = [clean_value(x.get("opinion")) for x in citems if clean_value(x.get("opinion"))]
            category = pd.Series(cats).mode().iloc[0] if cats else ""
            opinion = sorted(ops, key=lambda z: (len(normalize_persian(z).split()), len(z)))[0] if ops else ""
            gold_rows.append({
                "id": rid,
                "text": text,
                "aspect_term": aspect,
                "aspect_category": category,
                "opinion": opinion,
                "sentiment": sentiment,
                "supporting_annotators": ";".join(annotators),
            })
        agreement_rows.append({
            "review_id": rid,
            "n_annotated_items_total": len(items),
            "n_consensus_aspects": kept,
            "n_annotators_with_any_item": len(set(x["annotator"] for x in items)),
        })
    out_dir = Path(CFG.output_root) / "human_annotation"
    out_dir.mkdir(parents=True, exist_ok=True)
    gold = pd.DataFrame(gold_rows)
    gold.to_csv(out_dir / "human_only_test.csv", index=False)
    pd.DataFrame(agreement_rows).to_csv(out_dir / "human_annotation_agreement_summary.csv", index=False)
    print(f"[HUMAN GOLD SAVED] {out_dir / 'human_only_test.csv'}")
    print("Upload or copy this CSV to the same Kaggle dataset as your other ABSA CSVs before final train_evaluate.")
    return gold


# =========================
# 4. Prompt and target formatting
# =========================

def canonical_targets_json(targets: List[Dict[str, Any]]) -> str:
    """A stable non-ChatGPT-specific schema. This avoids natural-language style imitation."""
    arr = []
    for t in targets:
        aspect = clean_value(t.get("aspect"))
        if not aspect:
            continue
        arr.append({
            "aspect": aspect,
            "category": clean_value(t.get("category")) if t.get("category") is not None else None,
            "opinion": clean_value(t.get("opinion")) if t.get("opinion") is not None else None,
            "sentiment": normalize_sentiment(t.get("sentiment")),
        })
    return json.dumps(arr, ensure_ascii=False, separators=(",", ":"))


def build_prompt(review_text: str) -> str:
    return (
        "You are an aspect-based sentiment analysis system for Persian reviews.\n"
        "Extract every aspect mentioned in the review. Return ONLY a valid JSON array.\n"
        "Each item must have exactly these keys: aspect, category, opinion, sentiment.\n"
        "Use null when category or opinion is unavailable. Sentiment must be one of: positive, negative, neutral.\n"
        "Do not write explanations, markdown, or extra text.\n\n"
        f"### Review:\n{clean_value(review_text)}\n\n"
        "### Output:\n"
    )


def build_prompt_and_target(record: Dict[str, Any]) -> Dict[str, str]:
    return {
        "prompt": build_prompt(record["text"]),
        "target": canonical_targets_json(record["targets"]),
        "text": record["text"],
        "source": record["source"],
        "id": record["id"],
    }


def records_to_prompt_dataset(records: List[Dict[str, Any]]) -> Dataset:
    return Dataset.from_list([build_prompt_and_target(r) for r in records])


# =========================
# 5. Output parsing
# =========================

def extract_json_substring(text: str) -> str:
    text = clean_value(text)
    # Remove prompt if present.
    if "### Output:" in text:
        text = text.split("### Output:", 1)[-1].strip()
    # Strip code fences.
    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.I).strip()
    text = re.sub(r"```$", "", text.strip()).strip()

    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        return text[start:end + 1]

    # Sometimes model returns one object.
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return "[" + text[start:end + 1] + "]"

    return text


def coerce_target_item(x: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(x, dict):
        return None
    # Flexible key aliases.
    aliases = {
        "aspect": ["aspect", "aspect_term", "term", "target"],
        "category": ["category", "aspect_category", "aspectCategory"],
        "opinion": ["opinion", "opinion_term", "opinionTerm"],
        "sentiment": ["sentiment", "polarity", "label"],
    }
    out = {}
    for canonical, keys in aliases.items():
        val = None
        for k in keys:
            if k in x:
                val = x[k]
                break
        out[canonical] = clean_value(val) if val is not None else None
    out["sentiment"] = normalize_sentiment(out.get("sentiment"))
    if not out.get("aspect"):
        return None
    return out


def parse_model_output(text: str) -> List[Dict[str, Any]]:
    """Parse canonical JSON; also supports the old 'Aspect: ..., Sentiment: ...' format as fallback."""
    raw = extract_json_substring(text)
    parsed = None

    for candidate in [raw, raw.replace("'", '"')]:
        try:
            parsed = json.loads(candidate)
            break
        except Exception:
            pass

    if parsed is None:
        try:
            parsed = ast.literal_eval(raw)
        except Exception:
            parsed = None

    items: List[Dict[str, Any]] = []
    if isinstance(parsed, dict):
        parsed = [parsed]
    if isinstance(parsed, list):
        for obj in parsed:
            item = coerce_target_item(obj)
            if item:
                items.append(item)

    # Fallback for previous notebook outputs: Aspect: term, Category: cat, Opinion: op, Sentiment: label;
    if not items:
        chunks = re.split(r";|\n", clean_value(text))
        for ch in chunks:
            if "Aspect" not in ch and "aspect" not in ch:
                continue
            m_aspect = re.search(r"Aspect\s*:\s*([^,;]+)", ch, flags=re.I)
            m_cat = re.search(r"Category\s*:\s*([^,;]+)", ch, flags=re.I)
            m_op = re.search(r"Opinion\s*:\s*([^,;]+)", ch, flags=re.I)
            m_sent = re.search(r"Sentiment\s*:\s*([^,;]+)", ch, flags=re.I)
            if m_aspect:
                items.append({
                    "aspect": clean_value(m_aspect.group(1)),
                    "category": clean_value(m_cat.group(1)) if m_cat else None,
                    "opinion": clean_value(m_op.group(1)) if m_op else None,
                    "sentiment": normalize_sentiment(m_sent.group(1)) if m_sent else "",
                })

    return dedupe_targets(items)


# =========================
# 6. Evaluation
# =========================

def similarity(a: str, b: str) -> float:
    a_norm = normalize_persian(a)
    b_norm = normalize_persian(b)
    if not a_norm or not b_norm:
        return 0.0
    if a_norm == b_norm:
        return 100.0
    if a_norm in b_norm or b_norm in a_norm:
        return 90.0
    if fuzz is not None:
        return float(max(fuzz.ratio(a_norm, b_norm), fuzz.token_set_ratio(a_norm, b_norm)))
    # Fallback without rapidfuzz.
    from difflib import SequenceMatcher
    return 100.0 * SequenceMatcher(None, a_norm, b_norm).ratio()


def match_aspects(gold: List[Dict[str, Any]],
                  pred: List[Dict[str, Any]],
                  threshold: int) -> List[Tuple[int, int, float]]:
    candidates = []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            score = similarity(g.get("aspect", ""), p.get("aspect", ""))
            if score >= threshold:
                candidates.append((score, gi, pi))
    candidates.sort(reverse=True)
    used_g, used_p = set(), set()
    matches = []
    for score, gi, pi in candidates:
        if gi not in used_g and pi not in used_p:
            used_g.add(gi)
            used_p.add(pi)
            matches.append((gi, pi, score))
    return matches


def f1(p: float, r: float) -> float:
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def evaluate_prediction_lists(records: List[Dict[str, Any]],
                              predictions: List[List[Dict[str, Any]]],
                              dataset_name: str,
                              model_name: str,
                              seed: int,
                              output_dir: Path) -> Dict[str, Any]:
    assert len(records) == len(predictions), f"records/predictions length mismatch for {dataset_name}"

    total_gold = total_pred = correct_aspect = correct_pair = 0
    matched_with_sentiment = matched_sentiment_correct = 0
    category_evaluable = category_correct = 0
    opinion_evaluable = opinion_correct = 0
    error_rows = []

    for r, pred_items in zip(records, predictions):
        gold_items = dedupe_targets(r["targets"])
        pred_items = dedupe_targets(pred_items)
        total_gold += len(gold_items)
        total_pred += len(pred_items)

        matches = match_aspects(gold_items, pred_items, CFG.fuzzy_threshold)
        correct_aspect += len(matches)

        matched_gold = {gi for gi, _, _ in matches}
        matched_pred = {pi for _, pi, _ in matches}

        for gi, pi, score in matches:
            g = gold_items[gi]
            p = pred_items[pi]

            g_sent = normalize_sentiment(g.get("sentiment"))
            p_sent = normalize_sentiment(p.get("sentiment"))
            if g_sent:
                matched_with_sentiment += 1
                if g_sent == p_sent:
                    matched_sentiment_correct += 1
                    correct_pair += 1

            g_cat = clean_value(g.get("category"))
            if g_cat:
                category_evaluable += 1
                if similarity(g_cat, clean_value(p.get("category"))) >= CFG.category_fuzzy_threshold:
                    category_correct += 1

            g_op = clean_value(g.get("opinion"))
            if g_op:
                opinion_evaluable += 1
                if similarity(g_op, clean_value(p.get("opinion"))) >= CFG.category_fuzzy_threshold:
                    opinion_correct += 1

        # Error case storage
        missed = [gold_items[i] for i in range(len(gold_items)) if i not in matched_gold]
        false_pos = [pred_items[i] for i in range(len(pred_items)) if i not in matched_pred]
        wrong_sent = []
        for gi, pi, score in matches:
            g_sent = normalize_sentiment(gold_items[gi].get("sentiment"))
            p_sent = normalize_sentiment(pred_items[pi].get("sentiment"))
            if g_sent and p_sent and g_sent != p_sent:
                wrong_sent.append({"gold": gold_items[gi], "pred": pred_items[pi], "match_score": score})
        if missed or false_pos or wrong_sent:
            error_rows.append({
                "dataset": dataset_name,
                "id": r["id"],
                "text": r["text"],
                "gold": json.dumps(gold_items, ensure_ascii=False),
                "pred": json.dumps(pred_items, ensure_ascii=False),
                "missed_gold": json.dumps(missed, ensure_ascii=False),
                "false_positive_pred": json.dumps(false_pos, ensure_ascii=False),
                "wrong_sentiment": json.dumps(wrong_sent, ensure_ascii=False),
            })

    aspect_precision = correct_aspect / total_pred if total_pred else 0.0
    aspect_recall = correct_aspect / total_gold if total_gold else 0.0

    pair_precision = correct_pair / total_pred if total_pred else 0.0
    pair_recall = correct_pair / total_gold if total_gold else 0.0

    result = {
        "model": model_name,
        "seed": seed,
        "dataset": dataset_name,
        "n_reviews": len(records),
        "n_gold_aspects": total_gold,
        "n_pred_aspects": total_pred,
        "aspect_precision": aspect_precision,
        "aspect_recall": aspect_recall,
        "aspect_f1": f1(aspect_precision, aspect_recall),
        "aspect_sentiment_precision": pair_precision,
        "aspect_sentiment_recall": pair_recall,
        "aspect_sentiment_f1": f1(pair_precision, pair_recall),
        "matched_sentiment_accuracy": matched_sentiment_correct / matched_with_sentiment if matched_with_sentiment else np.nan,
        "category_accuracy_on_matched": category_correct / category_evaluable if category_evaluable else np.nan,
        "opinion_accuracy_on_matched": opinion_correct / opinion_evaluable if opinion_evaluable else np.nan,
    }

    if error_rows:
        pd.DataFrame(error_rows).to_csv(output_dir / f"errors_{model_name}_seed{seed}_{dataset_name}.csv", index=False)
    return result


def format_mean_std(mean: float, std: float) -> str:
    if pd.isna(std) or std == 0:
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {std:.3f}"


def save_aggregate_tables(results: List[Dict[str, Any]], output_dir: Path) -> None:
    if not results:
        return
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "all_seed_results_long.csv", index=False)

    metrics = [
        "aspect_precision", "aspect_recall", "aspect_f1",
        "aspect_sentiment_precision", "aspect_sentiment_recall", "aspect_sentiment_f1",
        "matched_sentiment_accuracy", "category_accuracy_on_matched", "opinion_accuracy_on_matched",
    ]
    agg = df.groupby(["model", "dataset"])[metrics].agg(["mean", "std"]).reset_index()
    agg.to_csv(output_dir / "all_seed_results_mean_std_raw.csv", index=False)

    # Publication-style Table 4: aspect-finding metrics
    rows = []
    for (model, dataset), g in df.groupby(["model", "dataset"]):
        row = {"Model": model, "Dataset": dataset}
        for m in ["aspect_precision", "aspect_recall", "aspect_f1"]:
            row[m] = format_mean_std(g[m].mean(), g[m].std(ddof=1) if len(g) > 1 else np.nan)
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "table_aspect_finding_mean_std.csv", index=False)

    # Publication-style Table 5: aspect+sentiment metrics
    rows = []
    for (model, dataset), g in df.groupby(["model", "dataset"]):
        row = {"Model": model, "Dataset": dataset}
        for m in ["aspect_sentiment_precision", "aspect_sentiment_recall", "aspect_sentiment_f1"]:
            row[m] = format_mean_std(g[m].mean(), g[m].std(ddof=1) if len(g) > 1 else np.nan)
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "table_aspect_sentiment_mean_std.csv", index=False)

    print("\n[SAVED TABLES]")
    print(output_dir / "table_aspect_finding_mean_std.csv")
    print(output_dir / "table_aspect_sentiment_mean_std.csv")


# =========================
# 7. LLaMA3 / Unsloth training and inference
# =========================

def load_llama_for_training(seed: int):
    from unsloth import FastLanguageModel

    set_global_seed(seed)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CFG.llama_model_name,
        max_seq_length=CFG.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=CFG.lora_r,
        target_modules=list(CFG.target_modules),
        lora_alpha=CFG.lora_alpha,
        lora_dropout=CFG.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
        use_rslora=False,
        loftq_config=None,
    )
    return model, tokenizer


def tokenize_causal_dataset(dataset: Dataset, tokenizer) -> Dataset:
    def _tok(ex):
        prompt_ids = tokenizer(
            ex["prompt"],
            add_special_tokens=False,
            truncation=True,
            max_length=CFG.max_prompt_tokens,
        )["input_ids"]
        target_text = ex["target"] + (tokenizer.eos_token or "")
        target_ids = tokenizer(
            target_text,
            add_special_tokens=False,
            truncation=True,
            max_length=CFG.max_output_tokens,
        )["input_ids"]
        # Ensure room for at least one target token.
        if len(prompt_ids) >= CFG.max_seq_length:
            prompt_ids = prompt_ids[: CFG.max_seq_length - 1]
        remaining = CFG.max_seq_length - len(prompt_ids)
        target_ids = target_ids[:remaining]
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }
    cols_to_remove = list(dataset.column_names)
    return dataset.map(_tok, remove_columns=cols_to_remove)


def print_trainable_parameters(model) -> int:
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    pct = 100 * trainable / total if total else 0.0
    print(f"[PARAMETERS] trainable={trainable:,} total={total:,} trainable%={pct:.4f}")
    return trainable


def train_llama(train_records: List[Dict[str, Any]], exp_name: str, seed: int, output_dir: Path):
    from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
    from unsloth import is_bfloat16_supported

    model, tokenizer = load_llama_for_training(seed)
    print_trainable_parameters(model)

    ds = records_to_prompt_dataset(train_records)
    if CFG.eval_holdout_ratio and len(ds) > 20:
        idx_train, idx_eval = train_test_split(list(range(len(ds))), test_size=CFG.eval_holdout_ratio, random_state=seed)
        train_ds = ds.select(idx_train)
        eval_ds = ds.select(idx_eval)
    else:
        train_ds = ds
        eval_ds = None

    tokenized_train = tokenize_causal_dataset(train_ds, tokenizer)
    tokenized_eval = tokenize_causal_dataset(eval_ds, tokenizer) if eval_ds is not None else None

    run_dir = output_dir / exp_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(run_dir),
        per_device_train_batch_size=CFG.train_batch_size,
        gradient_accumulation_steps=CFG.gradient_accumulation_steps,
        num_train_epochs=CFG.num_train_epochs,
        max_steps=CFG.max_steps,
        learning_rate=CFG.learning_rate,
        warmup_ratio=CFG.warmup_ratio,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=CFG.logging_steps,
        save_steps=CFG.save_steps,
        optim="adamw_8bit",
        weight_decay=CFG.weight_decay,
        lr_scheduler_type="linear",
        seed=seed,
        report_to="none",
        save_total_limit=1,
        evaluation_strategy="no" if tokenized_eval is None else "steps",
        eval_steps=CFG.save_steps if tokenized_eval is not None else None,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    t0 = time.time()
    trainer_stats = trainer.train()
    runtime = time.time() - t0
    print(f"[TRAINING DONE] {exp_name} seed={seed} runtime_sec={runtime:.1f}")

    model.save_pretrained(str(run_dir / "lora_adapter"))
    tokenizer.save_pretrained(str(run_dir / "lora_adapter"))

    with open(run_dir / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"runtime_sec": runtime, "trainer_metrics": trainer_stats.metrics}, f, indent=2)

    return model, tokenizer, run_dir


@torch.inference_mode()
def generate_llama(model, tokenizer, records: List[Dict[str, Any]], model_name: str, dataset_name: str, seed: int, output_dir: Path) -> List[List[Dict[str, Any]]]:
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pred_path = output_dir / f"predictions_{model_name}_seed{seed}_{dataset_name}.csv"
    if CFG.reuse_cached_predictions and pred_path.exists():
        try:
            cached = pd.read_csv(pred_path)
            if len(cached) == len(records) and "parsed_prediction" in cached.columns:
                print(f"[CACHE] Reusing predictions: {pred_path}")
                return [json.loads(x) if isinstance(x, str) and x.strip() else [] for x in cached["parsed_prediction"]]
        except Exception as e:
            print(f"[CACHE SKIP] Could not reuse {pred_path}: {e}")

    raw_rows = []
    parsed_predictions = []
    prompts = [build_prompt(r["text"]) for r in records]

    for start in range(0, len(prompts), CFG.generation_batch_size):
        batch_prompts = prompts[start:start + CFG.generation_batch_size]
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=CFG.max_prompt_tokens,
        ).to("cuda")
        out = model.generate(
            **enc,
            max_new_tokens=CFG.max_output_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        for j, full_text in enumerate(decoded):
            i = start + j
            pred_text = full_text.split("### Output:", 1)[-1].strip() if "### Output:" in full_text else full_text
            parsed = parse_model_output(pred_text)
            parsed_predictions.append(parsed)
            raw_rows.append({
                "model": model_name,
                "seed": seed,
                "dataset": dataset_name,
                "id": records[i]["id"],
                "review": records[i]["text"],
                "raw_generation": pred_text,
                "parsed_prediction": json.dumps(parsed, ensure_ascii=False),
                "gold": json.dumps(records[i]["targets"], ensure_ascii=False),
            })
        del enc, out
        torch.cuda.empty_cache()

    pd.DataFrame(raw_rows).to_csv(pred_path, index=False)
    return parsed_predictions


def run_llama_experiment(train_records: List[Dict[str, Any]],
                         exp_name: str,
                         seed: int,
                         test_splits: Dict[str, List[Dict[str, Any]]],
                         output_dir: Path) -> List[Dict[str, Any]]:
    full_train = train_records
    cleaned_tests = {name: remove_text_leakage(full_train, recs, name, output_dir) for name, recs in test_splits.items()}

    if CFG.run_llama_training:
        model, tokenizer, run_dir = train_llama(full_train, exp_name, seed, output_dir)
    else:
        raise ValueError("CFG.run_llama_training=False requires you to implement adapter loading path.")

    results = []
    for dataset_name, records in cleaned_tests.items():
        if not records:
            continue
        preds = generate_llama(model, tokenizer, records, exp_name, dataset_name, seed, output_dir)
        res = evaluate_prediction_lists(records, preds, dataset_name, exp_name, seed, output_dir)
        results.append(res)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return results


def run_llama_zero_shot(test_splits: Dict[str, List[Dict[str, Any]]], seed: int, output_dir: Path) -> List[Dict[str, Any]]:
    if not CFG.run_llama_zero_shot:
        return []

    from unsloth import FastLanguageModel
    set_global_seed(seed)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CFG.llama_model_name,
        max_seq_length=CFG.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_name = "LLaMA3_zero_shot"
    results = []
    for dataset_name, records in test_splits.items():
        preds = generate_llama(model, tokenizer, records, model_name, dataset_name, seed, output_dir)
        res = evaluate_prediction_lists(records, preds, dataset_name, model_name, seed, output_dir)
        results.append(res)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return results


# =========================
# 8. Optional mT5 generative baseline
# =========================

def run_mt5_baseline(train_records: List[Dict[str, Any]],
                     exp_name: str,
                     seed: int,
                     test_splits: Dict[str, List[Dict[str, Any]]],
                     output_dir: Path) -> List[Dict[str, Any]]:
    if not CFG.run_mt5_baseline:
        return []
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    set_global_seed(seed)
    model_name = f"mT5_base_{exp_name}"
    run_dir = output_dir / model_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(CFG.mt5_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(CFG.mt5_model_name)

    ds = records_to_prompt_dataset(train_records)

    def tok(ex):
        model_inputs = tokenizer(ex["prompt"], max_length=CFG.max_prompt_tokens, truncation=True)
        labels = tokenizer(text_target=ex["target"], max_length=CFG.max_output_tokens, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train = ds.map(tok, remove_columns=list(ds.column_names))
    args = Seq2SeqTrainingArguments(
        output_dir=str(run_dir),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        num_train_epochs=3,
        predict_with_generate=True,
        generation_max_length=CFG.max_output_tokens,
        logging_steps=20,
        save_steps=250,
        save_total_limit=1,
        report_to="none",
        seed=seed,
        fp16=torch.cuda.is_available(),
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    )
    trainer.train()
    trainer.save_model(str(run_dir / "model"))
    tokenizer.save_pretrained(str(run_dir / "model"))

    results = []
    for dataset_name, records in test_splits.items():
        raw_rows, parsed = [], []
        prompts = [build_prompt(r["text"]) for r in records]
        for start in range(0, len(prompts), CFG.generation_batch_size):
            batch_prompts = prompts[start:start + CFG.generation_batch_size]
            enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=CFG.max_prompt_tokens).to(model.device)
            out = model.generate(**enc, max_new_tokens=CFG.max_output_tokens, do_sample=False)
            decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
            for j, pred_text in enumerate(decoded):
                i = start + j
                items = parse_model_output(pred_text)
                parsed.append(items)
                raw_rows.append({
                    "model": model_name,
                    "seed": seed,
                    "dataset": dataset_name,
                    "id": records[i]["id"],
                    "review": records[i]["text"],
                    "raw_generation": pred_text,
                    "parsed_prediction": json.dumps(items, ensure_ascii=False),
                    "gold": json.dumps(records[i]["targets"], ensure_ascii=False),
                })
        pd.DataFrame(raw_rows).to_csv(pred_path, index=False)
        results.append(evaluate_prediction_lists(records, parsed, dataset_name, model_name, seed, output_dir))

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return results


# =========================
# 9. Main runner
# =========================

def main():
    output_dir = Path(CFG.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolve_all_config_paths()
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(CFG), f, ensure_ascii=False, indent=2)

    if CFG.pipeline_mode == "prepare_annotation":
        sample_annotation_candidates()
        return
    if CFG.pipeline_mode == "build_human_gold":
        build_human_only_gold_from_annotators()
        return

    train_splits, test_splits = load_all_splits()

    gpt_train = train_splits.get("gpt_assisted_human_validated_train", [])
    parsinlu_train = train_splits.get("parsinlu_train", [])

    if not gpt_train:
        raise RuntimeError("GPT-assisted human-validated train set is empty. Check CFG.gpt_train_path.")
    if not test_splits:
        raise RuntimeError("No test datasets were loaded. Check CFG.*_test_path.")

    # Two main experiments requested by the paper.
    experiments = {
        "LLaMA3_GPTval": gpt_train,
        "LLaMA3_GPTval_plus_ParsiNLU": gpt_train + parsinlu_train,
    }

    all_results: List[Dict[str, Any]] = []

    first_seed = CFG.run_seeds[0] if len(CFG.run_seeds) else 3407
    for seed in CFG.run_seeds:
        print(f"\n========== SEED {seed} ==========")

        # Zero-shot is deterministic and expensive; run it once unless explicitly disabled.
        if (not CFG.run_zero_shot_once) or seed == first_seed:
            all_results.extend(run_llama_zero_shot(test_splits, seed, output_dir))
        else:
            print(f"[SKIP] Zero-shot already run for seed {first_seed}; skipping for seed {seed}.")

        for exp_name, train_records in experiments.items():
            print(f"\n[EXPERIMENT] {exp_name}: {len(train_records)} training reviews")
            all_results.extend(run_llama_experiment(train_records, exp_name, seed, test_splits, output_dir))

            # Optional mT5 generative baseline with the same output schema.
            all_results.extend(run_mt5_baseline(train_records, exp_name, seed, test_splits, output_dir))

    save_aggregate_tables(all_results, output_dir)

    print("\n[DONE]")
    print(f"All outputs saved under: {output_dir}")


if __name__ == "__main__":
    main()
