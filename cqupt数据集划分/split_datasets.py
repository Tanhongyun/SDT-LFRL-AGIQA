#!/usr/bin/env python3
"""Create five-fold cross-validation splits for indexed AGIQA CSV files.

Default behavior:
- 1k: stratified by generation-model coverage. Each fold uses 18 single-model
  prompt groups and 18 two-model prompt groups as test groups.
- 3k and 2023: folded directly by prompt-group index.

The script writes train/test CSVs plus one assignment CSV per dataset per fold.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    folder: str
    input_path: Path
    stratify_1k_models: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create reproducible five-fold train/test splits for indexed AGIQA datasets."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Dataset root directory. Defaults to this script's directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <base-dir>/cv_folds.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of cross-validation folds. Default: 5.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"prompt", "image_name", "index"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return df


def generation_model(image_name: object) -> str:
    text = str(image_name)
    if text.startswith("deepai"):
        return "deepai"
    if text.startswith("dreamStudio"):
        return "dreamStudio"
    return "other"


def assign_folds(group_ids: pd.Series | list[int], folds: int, rng: np.random.Generator) -> dict[int, int]:
    ids = np.array(sorted(group_ids), dtype=int)
    if len(ids) < folds:
        raise ValueError(f"Need at least {folds} groups, got {len(ids)}")
    if len(ids) % folds != 0:
        raise ValueError(f"Group count must be divisible by folds: groups={len(ids)}, folds={folds}")
    shuffled = rng.permutation(ids)
    return {int(group_id): int(pos % folds) for pos, group_id in enumerate(shuffled)}


def regular_assignments(df: pd.DataFrame, folds: int, rng: np.random.Generator) -> pd.DataFrame:
    group_ids = sorted(int(x) for x in df["index"].drop_duplicates())
    fold_by_index = assign_folds(group_ids, folds, rng)
    return pd.DataFrame({"index": group_ids, "fold": [fold_by_index[idx] for idx in group_ids]})


def balanced_1k_assignments(df: pd.DataFrame, folds: int, rng: np.random.Generator) -> pd.DataFrame:
    meta = (
        df.assign(_model=df["image_name"].map(generation_model))
        .groupby("index", as_index=False)
        .agg(
            rows=("image_name", "size"),
            model_count=("_model", "nunique"),
            models=("_model", lambda values: "+".join(sorted(set(values)))),
        )
    )

    model_counts = meta["model_count"].value_counts().sort_index().to_dict()
    if model_counts != {1: 90, 2: 90}:
        raise ValueError(
            "1k expected exactly 90 single-model and 90 two-model prompt groups; "
            f"got {model_counts}"
        )
    if 90 % folds != 0:
        raise ValueError(f"1k strata contain 90 groups each, which is not divisible by folds={folds}")

    assignment_parts = []
    for model_count in [1, 2]:
        stratum = meta.loc[meta["model_count"] == model_count].copy()
        fold_by_index = assign_folds(stratum["index"], folds, rng)
        stratum["fold"] = stratum["index"].map(lambda idx: fold_by_index[int(idx)])
        stratum["stratum"] = stratum["model_count"].map({1: "single_model", 2: "two_model"})
        assignment_parts.append(stratum[["index", "fold", "stratum", "rows", "model_count", "models"]])

    assignments = pd.concat(assignment_parts, ignore_index=True).sort_values("index", kind="stable")
    return assignments


def validate_split(df: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, name: str) -> None:
    if len(train) + len(test) != len(df):
        raise AssertionError(f"{name}: train/test row counts do not sum to original")
    train_ids = set(int(x) for x in train["index"].unique())
    test_ids = set(int(x) for x in test["index"].unique())
    if train_ids & test_ids:
        raise AssertionError(f"{name}: prompt-group leakage between train and test: {sorted(train_ids & test_ids)[:10]}")
    if train_ids | test_ids != set(int(x) for x in df["index"].unique()):
        raise AssertionError(f"{name}: split does not cover all prompt groups")

    original_prompt_per_index = df.groupby("index")["prompt"].nunique(dropna=False).max()
    train_prompt_per_index = train.groupby("index")["prompt"].nunique(dropna=False).max()
    test_prompt_per_index = test.groupby("index")["prompt"].nunique(dropna=False).max()
    if max(original_prompt_per_index, train_prompt_per_index, test_prompt_per_index) < 1:
        raise AssertionError(f"{name}: invalid prompt grouping")


def write_dataset_split(
    spec: DatasetSpec,
    output_dir: Path,
    folds: int,
    seed: int,
) -> list[dict[str, int | str]]:
    df = read_csv(spec.input_path)
    rng = np.random.default_rng(seed)

    if spec.stratify_1k_models:
        assignments = balanced_1k_assignments(df, folds, rng)
    else:
        assignments = regular_assignments(df, folds, rng)

    rows: list[dict[str, int | str]] = []
    for fold in range(folds):
        test_ids = set(int(x) for x in assignments.loc[assignments["fold"] == fold, "index"])
        train_ids = set(int(x) for x in assignments.loc[assignments["fold"] != fold, "index"])
        train = df.loc[df["index"].map(lambda idx: int(idx) in train_ids)].copy()
        test = df.loc[df["index"].map(lambda idx: int(idx) in test_ids)].copy()
        train = train.sort_values(["index"], kind="stable").reset_index(drop=True)
        test = test.sort_values(["index"], kind="stable").reset_index(drop=True)
        validate_split(df, train, test, f"{spec.name}/fold_{fold}")

        dataset_output_dir = output_dir / spec.folder / f"fold_{fold}"
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        train_path = dataset_output_dir / f"{spec.name}_train.csv"
        test_path = dataset_output_dir / f"{spec.name}_test.csv"
        assignment_path = dataset_output_dir / f"{spec.name}_fold_assignments.csv"
        train.to_csv(train_path, index=False, encoding="utf-8-sig")
        test.to_csv(test_path, index=False, encoding="utf-8-sig")
        assignments.to_csv(assignment_path, index=False, encoding="utf-8-sig")

        rows.append(
            {
                "fold": fold,
                "dataset": spec.name,
                "input_rows": len(df),
                "train_rows": len(train),
                "test_rows": len(test),
                "input_groups": df["index"].nunique(),
                "train_groups": train["index"].nunique(),
                "test_groups": test["index"].nunique(),
                "train_path": str(train_path),
                "test_path": str(test_path),
                "assignment_path": str(assignment_path),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    output_dir = args.output_dir or (base_dir / "cv_folds")

    specs = [
        DatasetSpec("agiqa_1k_cqupt", "1k_cqupt", base_dir / "1k" / "agiqa_1k_cqupt-indexed.csv", stratify_1k_models=True),
        DatasetSpec("agiqa_3k_cqupt", "3k_cqupt", base_dir / "3k" / "agiqa_3k_cqupt-indexed.csv"),
        DatasetSpec("agiqa_2023_cqupt", "2023_cqupt", base_dir / "2023" / "agiqa_2023_cqupt-indexed.csv"),
    ]

    summaries = []
    for spec in specs:
        summaries.extend(write_dataset_split(spec, output_dir, args.folds, args.seed))
    summary = pd.DataFrame(summaries)
    summary_path = output_dir / "cv_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(f"\nsummary_path={summary_path}")


if __name__ == "__main__":
    main()
