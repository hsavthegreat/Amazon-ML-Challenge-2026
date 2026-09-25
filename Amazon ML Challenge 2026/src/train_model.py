#!/usr/bin/env python3
"""Train a precision-oriented classifier on blocked entity pairs.

The candidate-pairs file must contain at least:

    source1_entity_id, candidate_entity_id, candidate_source,
    cosine_similarity

The ground-truth file may be either a pair table with
``source1_entity_id`` and ``candidate_entity_id`` columns, or a wide table
with ``source1_entity_id``, ``source2_entity_id``, and/or
``source3_entity_id`` columns.

Example:
    python3 src/train_model.py \
        --candidate-pairs candidate_pairs_train.tsv \
        --source1 data/source1.tsv \
        --source2 data/source2.tsv \
        --source3 data/source3.tsv \
        --ground-truth data/ground_truth.tsv \
        --model-out matching_model.joblib
"""

from __future__ import annotations

import argparse
import csv
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split


ABBREVIATIONS: Dict[str, str] = {
    "corp": "corporation",
    "co": "company",
    "inc": "incorporated",
    "ltd": "limited",
    "llc": "limited liability company",
    "rd": "road",
    "st": "street",
    "ste": "suite",
    "ave": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "pvt": "private",
    "plc": "public limited company",
}
TOKEN_RE = re.compile(r"[^a-z0-9\s]")
SPACE_RE = re.compile(r"\s+")

FEATURE_COLUMNS = [
    "name_levenshtein_distance",
    "name_levenshtein_ratio",
    "address_levenshtein_distance",
    "address_levenshtein_ratio",
    "name_jaccard_similarity",
    "address_jaccard_similarity",
    "tfidf_cosine_similarity",
    "same_country",
]


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = TOKEN_RE.sub(" ", str(value).lower())
    expanded: List[str] = []
    for token in text.split():
        expanded.extend(ABBREVIATIONS.get(token, token).split())
    return SPACE_RE.sub(" ", " ".join(expanded)).strip()


def levenshtein_distance(left: str, right: str) -> int:
    """Compute edit distance using two rows of dynamic programming."""

    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    if len(left) > len(right):
        left, right = right, left

    previous = list(range(len(left) + 1))
    for right_index, right_char in enumerate(right, start=1):
        current = [right_index]
        for left_index, left_char in enumerate(left, start=1):
            insert_cost = current[left_index - 1] + 1
            delete_cost = previous[left_index] + 1
            replace_cost = previous[left_index - 1] + (left_char != right_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def levenshtein_ratio(left: str, right: str, distance: int | None = None) -> float:
    max_length = max(len(left), len(right))
    if max_length == 0:
        return 1.0
    if distance is None:
        distance = levenshtein_distance(left, right)
    return 1.0 - (distance / max_length)


def jaccard_similarity(left: str, right: str) -> float:
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens and not right_tokens:
        return 1.0
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def load_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def source_id_column(frame: pd.DataFrame, source_name: str) -> str:
    candidates = {
        "source1": ["source1_entity_id", "source_1_entity_id", "entity_id"],
        "source2": ["source2_entity_id", "source_2_entity_id", "entity_id"],
        "source3": ["source3_entity_id", "source_3_entity_id", "entity_id"],
    }[source_name]
    for column in candidates:
        if column in frame.columns:
            return column
    raise ValueError(f"Could not find an entity ID column for {source_name}")


def load_ground_truth(path: Path) -> Set[Tuple[str, str, str]]:
    """Normalize supported ground-truth layouts to (source1, source, candidate)."""

    frame = load_tsv(path)
    pairs: Set[Tuple[str, str, str]] = set()

    source1_column = next(
        (column for column in ("source1_entity_id", "source_1_entity_id") if column in frame),
        None,
    )
    if source1_column is None:
        raise ValueError("Ground truth must contain source1_entity_id")

    if "candidate_entity_id" in frame.columns:
        source_column = "candidate_source" if "candidate_source" in frame.columns else None
        if source_column:
            for _, row in frame.iterrows():
                pairs.add((str(row[source1_column]), str(row[source_column]), str(row["candidate_entity_id"])))
        else:
            # If source is absent, pair IDs are still usable when candidate IDs
            # are globally unique; the source is resolved during labeling.
            for _, row in frame.iterrows():
                pairs.add((str(row[source1_column]), "", str(row["candidate_entity_id"])))
        return pairs

    for source_name, columns in (
        ("source2", ("source2_entity_id", "source_2_entity_id")),
        ("source3", ("source3_entity_id", "source_3_entity_id")),
    ):
        source_column = next((column for column in columns if column in frame.columns), None)
        if source_column is None:
            continue
        for _, row in frame.iterrows():
            candidate_id = str(row[source_column]).strip()
            if candidate_id and candidate_id.lower() not in {"nan", "none", "null"}:
                pairs.add((str(row[source1_column]), source_name, candidate_id))

    if not pairs:
        raise ValueError(
            "Ground truth must contain candidate_entity_id or source2_entity_id/source3_entity_id"
        )
    return pairs


def make_features(
    candidate_pairs: pd.DataFrame,
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
) -> pd.DataFrame:
    """Join records and calculate the requested pairwise features."""

    required_pair_columns = {"source1_entity_id", "candidate_entity_id"}
    missing = required_pair_columns.difference(candidate_pairs.columns)
    if missing:
        raise ValueError("Candidate pairs missing columns: " + ", ".join(sorted(missing)))

    s1 = source1.copy()
    s2 = source2.copy()
    s3 = source3.copy()
    for frame in (s1, s2, s3):
        for column in ("entity_id", "business_name", "business_address", "country"):
            if column not in frame.columns:
                raise ValueError(f"Source file is missing required column: {column}")

    s1 = s1.rename(
        columns={
            "entity_id": "source1_entity_id",
            "business_name": "source1_business_name",
            "business_address": "source1_business_address",
            "country": "source1_country",
        }
    )
    s2 = s2.rename(
        columns={
            "entity_id": "candidate_entity_id",
            "business_name": "candidate_business_name",
            "business_address": "candidate_business_address",
            "country": "candidate_country",
        }
    )
    s3 = s3.rename(columns=s2.columns.to_dict())

    pairs = candidate_pairs.copy()
    pairs["source1_entity_id"] = pairs["source1_entity_id"].astype(str)
    pairs["candidate_entity_id"] = pairs["candidate_entity_id"].astype(str)
    s1["source1_entity_id"] = s1["source1_entity_id"].astype(str)
    s2["candidate_entity_id"] = s2["candidate_entity_id"].astype(str)
    s3["candidate_entity_id"] = s3["candidate_entity_id"].astype(str)

    if "candidate_source" not in pairs.columns:
        pairs["candidate_source"] = ""
    pairs["candidate_source"] = pairs["candidate_source"].astype(str).str.lower()
    pairs = pairs.merge(s1, on="source1_entity_id", how="left", validate="many_to_one")

    s2["_candidate_source"] = "source2"
    s3["_candidate_source"] = "source3"
    candidate_records = pd.concat([s2, s3], ignore_index=True)
    pairs = pairs.merge(
        candidate_records,
        left_on=["candidate_entity_id", "candidate_source"],
        right_on=["candidate_entity_id", "_candidate_source"],
        how="left",
        validate="many_to_one",
    )

    # Handle candidate-pairs files without candidate_source when IDs are unique.
    missing_candidate = pairs["candidate_business_name"].isna()
    if missing_candidate.any() and (pairs["candidate_source"] == "").any():
        fallback = pairs.loc[missing_candidate & (pairs["candidate_source"] == "")].drop(
            columns=[
                "candidate_business_name",
                "candidate_business_address",
                "candidate_country",
                "_candidate_source",
            ],
            errors="ignore",
        )
        fallback = fallback.merge(
            candidate_records.drop_duplicates("candidate_entity_id"),
            on="candidate_entity_id",
            how="left",
            suffixes=("", "_fallback"),
        )
        for column in ("candidate_business_name", "candidate_business_address", "candidate_country"):
            fallback_column = column + "_fallback"
            if fallback_column in fallback:
                pairs.loc[fallback.index, column] = fallback[fallback_column]

    for column in (
        "source1_business_name",
        "source1_business_address",
        "source1_country",
        "candidate_business_name",
        "candidate_business_address",
        "candidate_country",
    ):
        if column not in pairs:
            pairs[column] = ""
        pairs[column] = pairs[column].fillna("").map(clean_text)

    feature_rows: List[Dict[str, float]] = []
    for _, row in pairs.iterrows():
        name_left, name_right = row["source1_business_name"], row["candidate_business_name"]
        address_left, address_right = row["source1_business_address"], row["candidate_business_address"]
        name_distance = levenshtein_distance(name_left, name_right)
        address_distance = levenshtein_distance(address_left, address_right)
        cosine_value = row.get("cosine_similarity", row.get("tfidf_cosine_similarity", 0.0))
        feature_rows.append(
            {
                "name_levenshtein_distance": float(name_distance),
                "name_levenshtein_ratio": levenshtein_ratio(name_left, name_right, name_distance),
                "address_levenshtein_distance": float(address_distance),
                "address_levenshtein_ratio": levenshtein_ratio(
                    address_left, address_right, address_distance
                ),
                "name_jaccard_similarity": jaccard_similarity(name_left, name_right),
                "address_jaccard_similarity": jaccard_similarity(address_left, address_right),
                "tfidf_cosine_similarity": float(pd.to_numeric(cosine_value, errors="coerce") or 0.0),
                "same_country": float(
                    bool(row["source1_country"])
                    and row["source1_country"] == row["candidate_country"]
                ),
            }
        )

    features = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)
    return pd.concat([pairs.reset_index(drop=True), features], axis=1)


def add_labels(frame: pd.DataFrame, ground_truth: Set[Tuple[str, str, str]]) -> pd.DataFrame:
    labels = []
    for _, row in frame.iterrows():
        source1_id = str(row["source1_entity_id"])
        candidate_source = str(row.get("candidate_source", "")).lower()
        candidate_id = str(row["candidate_entity_id"])
        exact = (source1_id, candidate_source, candidate_id) in ground_truth
        if not exact and (source1_id, "", candidate_id) in ground_truth:
            exact = True
        labels.append(int(exact))
    result = frame.copy()
    result["label"] = labels
    return result


def f_beta_score(y_true: Sequence[int], probabilities: np.ndarray, threshold: float, beta: float = 0.5) -> float:
    predictions = probabilities >= threshold
    true_positive = int(np.sum(predictions & (np.asarray(y_true) == 1)))
    false_positive = int(np.sum(predictions & (np.asarray(y_true) == 0)))
    false_negative = int(np.sum(~predictions & (np.asarray(y_true) == 1)))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    denominator = beta * beta * precision + recall
    return (1 + beta * beta) * precision * recall / denominator if denominator else 0.0


def select_threshold(y_true: Sequence[int], probabilities: np.ndarray) -> Tuple[float, float]:
    thresholds = np.unique(np.r_[0.0, np.clip(probabilities, 0.0, 1.0), 1.0])
    scores = [f_beta_score(y_true, probabilities, threshold) for threshold in thresholds]
    best_index = int(np.argmax(scores))
    return float(thresholds[best_index]), float(scores[best_index])


def train_classifier(x_train: pd.DataFrame, y_train: pd.Series):
    """Prefer LightGBM, then XGBoost, then the always-available sklearn fallback."""

    try:
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            objective="binary",
            n_estimators=300,
            learning_rate=0.04,
            num_leaves=31,
            class_weight="balanced",
            random_state=42,
            verbosity=-1,
        )
        backend = "lightgbm"
    except ImportError:
        try:
            from xgboost import XGBClassifier

            model = XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.04,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=42,
            )
            backend = "xgboost"
        except ImportError:
            model = RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            )
            backend = "random_forest"

    model.fit(x_train, y_train)
    return model, backend


def save_bundle(bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import joblib

        joblib.dump(bundle, path)
    except ImportError:
        with path.open("wb") as handle:
            pickle.dump(bundle, handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pairs", type=Path, required=True)
    parser.add_argument("--source1", type=Path, required=True)
    parser.add_argument("--source2", type=Path, required=True)
    parser.add_argument("--source3", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, default=Path("matching_model.joblib"))
    parser.add_argument("--validation-size", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        candidate_pairs = load_tsv(args.candidate_pairs)
        source1 = load_tsv(args.source1)
        source2 = load_tsv(args.source2)
        source3 = load_tsv(args.source3)
        ground_truth = load_ground_truth(args.ground_truth)
        data = add_labels(make_features(candidate_pairs, source1, source2, source3), ground_truth)

        if data.empty:
            raise ValueError("Candidate-pairs file is empty")
        if data["label"].nunique() < 2:
            raise ValueError("Training candidates must contain both positive and negative labels")

        x = data[FEATURE_COLUMNS].astype(float)
        y = data["label"].astype(int)
        x_train, x_valid, y_train, y_valid = train_test_split(
            x,
            y,
            test_size=args.validation_size,
            random_state=42,
            stratify=y,
        )
        model, backend = train_classifier(x_train, y_train)
        probabilities = model.predict_proba(x_valid)[:, 1]
        threshold, validation_f05 = select_threshold(y_valid.to_numpy(), probabilities)

        bundle = {
            "model": model,
            "threshold": threshold,
            "feature_columns": FEATURE_COLUMNS,
            "backend": backend,
            "metric": "F0.5",
            "validation_f05": validation_f05,
        }
        save_bundle(bundle, args.model_out)

        print(f"Backend: {backend}")
        print(f"Rows: {len(data):,}; positives: {int(y.sum()):,}")
        print(f"Optimal validation F0.5 threshold: {threshold:.6f}")
        print(f"Validation F0.5: {validation_f05:.6f}")
        print(f"Saved model bundle to {args.model_out}")
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
