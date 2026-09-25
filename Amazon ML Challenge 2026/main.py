#!/usr/bin/env python3
"""Run the complete business entity-resolution pipeline."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.blocking import load_split, paths_from_directory, top_candidates, validate_paths, write_tsv
from src.predict import aggregate_candidates, aggregate_matches, load_model_bundle, write_submission
from src.train_model import (
    FEATURE_COLUMNS,
    add_labels,
    load_ground_truth,
    load_tsv,
    make_features,
    save_bundle,
    select_threshold,
    train_classifier,
)


ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "data" / "train"
TEST_DIR = ROOT / "data" / "test"
GROUND_TRUTH = ROOT / "data" / "ground_truth.tsv"
OUTPUT_DIR = ROOT / "output"


def train_model_from_candidates(
    candidate_pairs: pd.DataFrame,
    source1_path: Path,
    source2_path: Path,
    source3_path: Path,
    ground_truth_path: Path,
    model_path: Path,
) -> dict:
    """Create features, train the classifier, select F0.5 threshold, and save it."""

    source1 = load_tsv(source1_path)
    source2 = load_tsv(source2_path)
    source3 = load_tsv(source3_path)
    features = make_features(candidate_pairs, source1, source2, source3)
    labeled = add_labels(features, load_ground_truth(ground_truth_path))

    if labeled.empty:
        raise ValueError("Training candidate pairs are empty")
    if labeled["label"].nunique() < 2:
        raise ValueError("Training data must contain both positive and negative labels")

    x = labeled[FEATURE_COLUMNS].astype(float)
    y = labeled["label"].astype(int)
    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model, backend = train_classifier(x_train, y_train)
    validation_probabilities = model.predict_proba(x_valid)[:, 1]
    threshold, validation_f05 = select_threshold(y_valid.to_numpy(), validation_probabilities)

    bundle = {
        "model": model,
        "threshold": threshold,
        "feature_columns": FEATURE_COLUMNS,
        "backend": backend,
        "metric": "F0.5",
        "validation_f05": validation_f05,
    }
    save_bundle(bundle, model_path)
    print(f"Trained {backend} model; validation F0.5={validation_f05:.6f}")
    print(f"Saved model to {model_path}")
    return bundle


def run_pipeline() -> None:
    train_paths = paths_from_directory(TRAIN_DIR)
    test_paths = paths_from_directory(TEST_DIR)
    validate_paths(train_paths.values())
    validate_paths(test_paths.values())
    validate_paths([GROUND_TRUTH])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 2: generate training candidates.
    train_source1, train_candidates = load_split(train_paths)
    train_candidate_pairs = top_candidates(train_source1, train_candidates, top_k=50)
    train_candidate_path = OUTPUT_DIR / "candidate_pairs_train.tsv"
    write_tsv(train_candidate_pairs, train_candidate_path)
    print(f"Generated {len(train_candidate_pairs):,} training candidates")

    # Step 3: train the matching classifier and save its threshold.
    model_path = OUTPUT_DIR / "matching_model.joblib"
    bundle = train_model_from_candidates(
        train_candidate_pairs,
        train_paths["source1"],
        train_paths["source2"],
        train_paths["source3"],
        GROUND_TRUTH,
        model_path,
    )

    # Step 4: block and score test candidates.
    test_source1, test_candidates = load_split(test_paths)
    test_candidate_pairs = top_candidates(test_source1, test_candidates, top_k=50)
    candidate_output = aggregate_candidates(test_candidate_pairs, test_source1)
    write_submission(candidate_output, OUTPUT_DIR / "candidate_pairs.tsv")

    test_source1_raw = load_tsv(test_paths["source1"])
    test_source2_raw = load_tsv(test_paths["source2"])
    test_source3_raw = load_tsv(test_paths["source3"])
    test_features = make_features(
        test_candidate_pairs,
        test_source1_raw,
        test_source2_raw,
        test_source3_raw,
    )
    probabilities = bundle["model"].predict_proba(test_features[FEATURE_COLUMNS].astype(float))[:, 1]
    scored_pairs = test_candidate_pairs.copy()
    scored_pairs["match_probability"] = probabilities
    matching_output = aggregate_matches(
        scored_pairs,
        test_source1_raw,
        float(bundle["threshold"]),
    )
    write_submission(matching_output, OUTPUT_DIR / "matching_results.tsv")

    print(f"Wrote {len(test_candidate_pairs):,} test candidates")
    print(f"Wrote final files to {OUTPUT_DIR}")
    # Before submitting, run: python3 utils/validate_submission.py


if __name__ == "__main__":
    try:
        run_pipeline()
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print(f"Pipeline failed: {error}", file=sys.stderr)
        raise SystemExit(1)
