#!/usr/bin/env python3
"""Generate high-recall candidate pairs for business-entity resolution.

The script reads three tab-separated files, treats Source 1 as the query set,
and retrieves the top ``k`` entities from the union of Sources 2 and 3 using
TF-IDF cosine similarity over normalized business name and address text.

Examples
--------
Single split (defaults):
    python src/blocking.py

Single split with explicit files:
    python src/blocking.py \
        --source1 data/source1.tsv \
        --source2 data/source2.tsv \
        --source3 data/source3.tsv \
        --output candidate_pairs.tsv

Train and test directory layout:
    data/train/source1.tsv, data/train/source2.tsv, data/train/source3.tsv
    data/test/source1.tsv,  data/test/source2.tsv,  data/test/source3.tsv

    python src/blocking.py --train-dir data/train --test-dir data/test
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


REQUIRED_COLUMNS = {
    "entity_id",
    "business_name",
    "business_address",
    "country",
}

# Replacements are token-based so that, for example, "st" is not changed
# inside an unrelated word. Add challenge-specific abbreviations here as
# further data inspection suggests.
ABBREVIATIONS: Dict[str, str] = {
    "corp": "corporation",
    "corporation": "corporation",
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


def clean_text(value: object) -> str:
    """Lowercase, remove punctuation, expand abbreviations, and normalize spaces."""

    if pd.isna(value):
        return ""

    text = str(value).lower()
    text = TOKEN_RE.sub(" ", text)
    tokens = []
    for token in text.split():
        tokens.extend(ABBREVIATIONS.get(token, token).split())
    return SPACE_RE.sub(" ", " ".join(tokens)).strip()


def load_source(path: Path, source_name: str) -> pd.DataFrame:
    """Load and validate one source TSV file."""

    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{path} is missing required columns: {missing_text}")

    frame = frame.copy()
    frame["_source"] = source_name
    frame["_row_number"] = np.arange(len(frame), dtype=np.int64)
    frame["_combined_text"] = (
        frame["business_name"].map(clean_text)
        + " "
        + frame["business_address"].map(clean_text)
    ).str.strip()
    return frame


def load_split(source_paths: Dict[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load Source 1 and the concatenated Source 2/3 candidate pool."""

    source1 = load_source(source_paths["source1"], "source1")
    source2 = load_source(source_paths["source2"], "source2")
    source3 = load_source(source_paths["source3"], "source3")
    candidates = pd.concat([source2, source3], ignore_index=True)
    return source1, candidates


def top_candidates(
    source1: pd.DataFrame,
    candidates: pd.DataFrame,
    top_k: int,
    batch_size: int = 2048,
) -> pd.DataFrame:
    """Return up to ``top_k`` highest-scoring candidates for every Source 1 row."""

    if candidates.empty or source1.empty:
        return pd.DataFrame(
            columns=[
                "source1_entity_id",
                "candidate_entity_id",
                "candidate_source",
                "cosine_similarity",
            ]
        )

    # Character n-grams improve recall for typos and small formatting changes.
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=1,
        sublinear_tf=True,
        dtype=np.float32,
    )
    candidate_matrix = vectorizer.fit_transform(candidates["_combined_text"])
    source1_matrix = vectorizer.transform(source1["_combined_text"])

    rows: List[Dict[str, object]] = []
    limit = min(top_k, len(candidates))

    for start in range(0, len(source1), batch_size):
        stop = min(start + batch_size, len(source1))
        similarities = cosine_similarity(source1_matrix[start:stop], candidate_matrix)

        for local_index, scores in enumerate(similarities):
            query_index = start + local_index
            # argpartition avoids sorting the entire candidate list. A final
            # stable sort makes output deterministic for equal similarities.
            if limit < len(scores):
                candidate_indices = np.argpartition(scores, -limit)[-limit:]
            else:
                candidate_indices = np.arange(len(scores))

            candidate_indices = sorted(
                candidate_indices,
                key=lambda index: (-float(scores[index]), int(index)),
            )
            query_id = source1.iloc[query_index]["entity_id"]

            for candidate_index in candidate_indices:
                candidate_row = candidates.iloc[candidate_index]
                rows.append(
                    {
                        "source1_entity_id": query_id,
                        "candidate_entity_id": candidate_row["entity_id"],
                        "candidate_source": candidate_row["_source"],
                        "cosine_similarity": float(scores[candidate_index]),
                    }
                )

    return pd.DataFrame(rows)


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    """Write a quote-free TSV, creating its parent directory if needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        sep="\t",
        index=False,
        quoting=csv.QUOTE_NONE,
        escapechar="\\",
        lineterminator="\n",
    )


def paths_from_directory(directory: Path) -> Dict[str, Path]:
    return {
        "source1": directory / "source1.tsv",
        "source2": directory / "source2.tsv",
        "source3": directory / "source3.tsv",
    }


def validate_paths(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Input TSV file(s) not found: " + ", ".join(missing))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source1", type=Path, help="Source 1 TSV for a single split")
    parser.add_argument("--source2", type=Path, help="Source 2 TSV for a single split")
    parser.add_argument("--source3", type=Path, help="Source 3 TSV for a single split")
    parser.add_argument(
        "--train-dir",
        type=Path,
        help="Directory containing train/source1.tsv, source2.tsv, and source3.tsv",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        help="Directory containing test/source1.tsv, source2.tsv, and source3.tsv",
    )
    parser.add_argument("--top-k", type=int, default=50, help="Candidates per Source 1 row")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("candidate_pairs.tsv"),
        help="Output TSV for a single split (default: candidate_pairs.tsv)",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path("candidate_pairs_train.tsv"),
        help="Output TSV for the train split",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=Path("candidate_pairs_test.tsv"),
        help="Output TSV for the test split",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    explicit_sources = [args.source1, args.source2, args.source3]
    has_explicit_sources = any(path is not None for path in explicit_sources)
    if has_explicit_sources and not all(
        path is not None for path in explicit_sources
    ):
        parser.error("--source1, --source2, and --source3 must be supplied together")

    if has_explicit_sources and (args.train_dir or args.test_dir):
        parser.error("Use either --source1/--source2/--source3 or --train-dir/--test-dir")

    if not has_explicit_sources and not args.train_dir and not args.test_dir:
        args.source1 = Path("data/source1.tsv")
        args.source2 = Path("data/source2.tsv")
        args.source3 = Path("data/source3.tsv")

    return args


def main() -> int:
    args = parse_args()

    try:
        if args.source1:
            paths = {"source1": args.source1, "source2": args.source2, "source3": args.source3}
            validate_paths(paths.values())
            source1, candidates = load_split(paths)
            result = top_candidates(source1, candidates, args.top_k)
            write_tsv(result, args.output)
            print(f"Wrote {len(result):,} candidate pairs to {args.output}")
            return 0

        for split_name, split_dir, output_path in (
            ("train", args.train_dir, args.train_output),
            ("test", args.test_dir, args.test_output),
        ):
            if split_dir is None:
                continue
            paths = paths_from_directory(split_dir)
            validate_paths(paths.values())
            source1, candidates = load_split(paths)
            result = top_candidates(source1, candidates, args.top_k)
            result.insert(0, "split", split_name)
            write_tsv(result, output_path)
            print(f"Wrote {len(result):,} {split_name} candidate pairs to {output_path}")
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
