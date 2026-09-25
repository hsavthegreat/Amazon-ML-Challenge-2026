#!/usr/bin/env python3
"""Run blocking and model inference, then write the competition submissions.

Example:
    python3 src/predict.py \
        --test-dir data/test \
        --model matching_model.joblib \
        --matching-output matching_results.tsv \
        --candidate-output candidate_pairs.tsv

The test directory must contain ``source1.tsv``, ``source2.tsv``, and
``source3.tsv`` with the standard entity-resolution columns.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from blocking import load_split, paths_from_directory, top_candidates, validate_paths
from train_model import FEATURE_COLUMNS, load_tsv, make_features


def load_model_bundle(path: Path) -> dict:
    """Load a joblib bundle, falling back to pickle."""

    try:
        import joblib

        bundle = joblib.load(path)
    except ImportError:
        with path.open("rb") as handle:
            bundle = pickle.load(handle)

    if not isinstance(bundle, dict) or "model" not in bundle or "threshold" not in bundle:
        raise ValueError("Model file must contain a dictionary with 'model' and 'threshold'")
    return bundle


def unique_ids(values: Iterable[object]) -> str:
    """Return non-empty IDs once each, preserving their input order."""

    seen = set()
    result: List[str] = []
    for value in values:
        value_text = str(value)
        if not value_text or value_text.lower() in {"nan", "none", "null"}:
            continue
        if value_text not in seen:
            seen.add(value_text)
            result.append(value_text)
    return ",".join(result)


def aggregate_candidates(candidate_pairs: pd.DataFrame, source1: pd.DataFrame) -> pd.DataFrame:
    """Create one candidate_entity_ids row for every Source 1 entity."""

    candidate_lists = (
        candidate_pairs.groupby("source1_entity_id", sort=False)["candidate_entity_id"]
        .apply(unique_ids)
        .to_dict()
    )
    return pd.DataFrame(
        {
            "source1_entity_id": source1["entity_id"].astype(str),
            "candidate_entity_ids": [
                candidate_lists.get(str(entity_id), "") for entity_id in source1["entity_id"]
            ],
        }
    )


def aggregate_matches(
    scored_pairs: pd.DataFrame,
    source1: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Create one matching_results row for every Source 1 entity."""

    matched = scored_pairs[scored_pairs["match_probability"] >= threshold]
    match_lists = (
        matched.groupby("source1_entity_id", sort=False)["candidate_entity_id"]
        .apply(unique_ids)
        .to_dict()
    )
    return pd.DataFrame(
        {
            "source1_entity_id": source1["entity_id"].astype(str),
            "matched_entity_ids": [
                match_lists.get(str(entity_id), "") for entity_id in source1["entity_id"]
            ],
        }
    )


def write_submission(frame: pd.DataFrame, path: Path) -> None:
    """Write exactly two quote-free tab-separated columns."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        sep="\t",
        index=False,
        columns=list(frame.columns),
        quoting=csv.QUOTE_NONE,
        escapechar="\\",
        lineterminator="\n",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, help="Directory containing the three test TSV files")
    parser.add_argument("--source1", type=Path, help="Test Source 1 TSV")
    parser.add_argument("--source2", type=Path, help="Test Source 2 TSV")
    parser.add_argument("--source3", type=Path, help="Test Source 3 TSV")
    parser.add_argument("--model", type=Path, required=True, help="Model bundle from train_model.py")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--matching-output", type=Path, default=Path("matching_results.tsv"))
    parser.add_argument("--candidate-output", type=Path, default=Path("candidate_pairs.tsv"))
    args = parser.parse_args()

    explicit = [args.source1, args.source2, args.source3]
    if args.test_dir and any(path is not None for path in explicit):
        parser.error("Use either --test-dir or --source1/--source2/--source3")
    if not args.test_dir and not all(path is not None for path in explicit):
        parser.error("Supply --test-dir or all of --source1, --source2, and --source3")
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    return args


def main() -> int:
    args = parse_args()
    try:
        if args.test_dir:
            source_paths = paths_from_directory(args.test_dir)
        else:
            source_paths = {
                "source1": args.source1,
                "source2": args.source2,
                "source3": args.source3,
            }
        validate_paths(source_paths.values())
        validate_paths([args.model])

        source1, candidates = load_split(source_paths)
        candidate_pairs = top_candidates(source1, candidates, args.top_k)
        candidate_output = aggregate_candidates(candidate_pairs, source1)
        write_submission(candidate_output, args.candidate_output)

        source1_raw = load_tsv(source_paths["source1"])
        source2_raw = load_tsv(source_paths["source2"])
        source3_raw = load_tsv(source_paths["source3"])
        feature_frame = make_features(candidate_pairs, source1_raw, source2_raw, source3_raw)

        bundle = load_model_bundle(args.model)
        model = bundle["model"]
        threshold = float(bundle["threshold"])
        feature_columns = bundle.get("feature_columns", FEATURE_COLUMNS)
        if list(feature_columns) != list(FEATURE_COLUMNS):
            raise ValueError("Saved model feature columns do not match train_model.py")

        probabilities = model.predict_proba(feature_frame[feature_columns].astype(float))[:, 1]
        scored_pairs = candidate_pairs.copy()
        scored_pairs["match_probability"] = probabilities
        matching_output = aggregate_matches(scored_pairs, source1_raw, threshold)
        write_submission(matching_output, args.matching_output)

        print(f"Wrote {len(candidate_pairs):,} candidate rows to {args.candidate_output}")
        print(f"Wrote {len(matching_output):,} Source 1 rows to {args.matching_output}")
        print(f"Applied probability threshold: {threshold:.6f}")
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
