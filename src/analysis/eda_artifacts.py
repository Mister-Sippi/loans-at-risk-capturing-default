from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd


def add_issue_year(
    df: pd.DataFrame,
    *,
    issue_date_column: str = "issue_d",
    output_column: str = "issue_year",
    log: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """
    Return a copy of the clean dataset with an issue-year column derived from
    the issue-date field.

    This helper is intended only for clean-dataset diagnostics. The feature
    base dataset does not retain `issue_d` because post-origination timestamps
    are outside the modeling boundary.
    """
    try:
        if issue_date_column not in df.columns:
            raise KeyError(
                "add_issue_year: missing required column "
                f"'{issue_date_column}'. This helper is intended only for the clean dataset."
            )

        df_with_issue_year = df.copy()

        df_with_issue_year[output_column] = (
            pd.to_datetime(
                df_with_issue_year[issue_date_column],
                errors="coerce",
            )
            .dt.year
        )

        if log:
            non_null_issue_year_count = int(df_with_issue_year[output_column].notna().sum())
            log(
                "add_issue_year completed | "
                f"rows={df_with_issue_year.shape[0]} | "
                f"non_null_issue_year_count={non_null_issue_year_count}"
            )

        return df_with_issue_year

    except Exception as exc:
        if log:
            log(f"add_issue_year failed | error={exc}")
        raise


def build_loan_status_by_issue_year_table(
    df: pd.DataFrame,
    *,
    issue_year_column: str = "issue_year",
    status_column: str = "loan_status",
) -> pd.DataFrame:
    """
    Build a crosstab of loan-status counts by issue year.
    """
    return pd.crosstab(
        df[issue_year_column],
        df[status_column],
        dropna=False,
    )


def build_default_rate_by_issue_year(
    df: pd.DataFrame,
    *,
    issue_year_column: str = "issue_year",
    target_column: str = "target_default",
    dataset_split: str,
) -> pd.DataFrame:
    """
    Build a default-rate table by issue year for a single dataset split.
    """
    default_rate_table = (
        df.groupby(issue_year_column)[target_column]
        .agg(
            default_rate_percent=lambda series: round(float(series.mean() * 100), 2),
            loan_count="size",
        )
        .reset_index()
    )

    default_rate_table["dataset_split"] = dataset_split

    return default_rate_table


def build_issuance_volume_by_issue_year(
    df: pd.DataFrame,
    *,
    issue_year_column: str = "issue_year",
    dataset_split: str,
) -> pd.DataFrame:
    """
    Build an issuance-volume table by issue year for a single dataset split.
    """
    volume_table = (
        df.groupby(issue_year_column)
        .size()
        .reset_index(name="loan_count")
    )

    volume_table["dataset_split"] = dataset_split

    return volume_table


def build_missingness_by_issue_year(
    df: pd.DataFrame,
    *,
    column_names: Sequence[str],
    issue_year_column: str = "issue_year",
    dataset_split: str,
) -> pd.DataFrame:
    """
    Build a long-format missingness table by issue year for one dataset split.
    """
    missingness_tables: list[pd.DataFrame] = []

    for column_name in column_names:
        missingness_table = (
            df.groupby(issue_year_column)[column_name]
            .apply(lambda series: round(float(series.isna().mean() * 100), 2))
            .reset_index(name="missing_percent")
        )

        missingness_table["feature_name"] = column_name
        missingness_table["dataset_split"] = dataset_split
        missingness_table["series_label"] = (
            missingness_table["dataset_split"] + " — " + missingness_table["feature_name"]
        )

        missingness_tables.append(missingness_table)

    return pd.concat(missingness_tables, axis=0, ignore_index=True)


def build_feature_missingness_comparison(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    *,
    column_names: Sequence[str],
    train_label: str = "train",
    test_label: str = "test",
) -> pd.DataFrame:
    """
    Build a train-vs-test missingness summary for selected features.
    """
    summary_rows: list[dict[str, float | str]] = []

    for column_name in column_names:
        summary_rows.append(
            {
                "feature_name": column_name,
                "dataset_split": train_label,
                "missing_percent": round(float(df_train[column_name].isna().mean() * 100), 2),
            }
        )
        summary_rows.append(
            {
                "feature_name": column_name,
                "dataset_split": test_label,
                "missing_percent": round(float(df_test[column_name].isna().mean() * 100), 2),
            }
        )

    return pd.DataFrame(summary_rows)


def build_default_rate_by_train_fitted_quantile(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    *,
    feature_name: str,
    target_column: str = "target_default",
    number_of_bins: int = 10,
    train_label: str = "train",
    test_label: str = "test",
) -> pd.DataFrame:
    """
    Build a train-fitted quantile comparison table for a numeric feature.
    """
    train_non_null = df_train[feature_name].dropna()

    if train_non_null.empty:
        raise ValueError(
            "build_default_rate_by_train_fitted_quantile: "
            f"'{feature_name}' has no non-null train values"
        )

    _, bin_edges = pd.qcut(
        train_non_null,
        q=number_of_bins,
        retbins=True,
        duplicates="drop",
    )

    unique_bin_edges = np.unique(bin_edges)

    if len(unique_bin_edges) < 3:
        raise ValueError(
            "build_default_rate_by_train_fitted_quantile: "
            f"insufficient unique bin edges for '{feature_name}'"
        )

    comparison_tables: list[pd.DataFrame] = []

    for dataset_split, df_split in [
        (train_label, df_train),
        (test_label, df_test),
    ]:
        df_binned = df_split[[feature_name, target_column]].copy()

        df_binned["feature_bin"] = pd.cut(
            df_binned[feature_name],
            bins=unique_bin_edges,
            include_lowest=True,
            duplicates="drop",
        )

        rate_table = (
            df_binned
            .dropna(subset=["feature_bin"])
            .groupby("feature_bin", observed=False)[target_column]
            .agg(
                default_rate_percent=lambda series: round(float(series.mean() * 100), 2),
                loan_count="size",
            )
            .reset_index()
        )

        rate_table["feature_name"] = feature_name
        rate_table["dataset_split"] = dataset_split
        rate_table["feature_bin"] = rate_table["feature_bin"].astype("string")

        comparison_tables.append(rate_table)

    return pd.concat(comparison_tables, axis=0, ignore_index=True)


def build_default_rate_by_category_comparison(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    *,
    feature_name: str,
    target_column: str = "target_default",
    train_label: str = "train",
    test_label: str = "test",
    treat_as_missingness: bool = False,
    log: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """
    Build a default-rate comparison table for train and test splits.

    Parameters
    ----------
    df_train : pd.DataFrame
        Train dataset.
    df_test : pd.DataFrame
        Test dataset.
    feature_name : str
        Feature to evaluate.
    target_column : str, default="target_default"
        Binary target column used to compute default rates.
    train_label : str, default="train"
        Label used for the train split in the output table.
    test_label : str, default="test"
        Label used for the test split in the output table.
    treat_as_missingness : bool, default=False
        If True, derive categories dynamically as "missing" and "not_missing"
        from the specified feature instead of grouping by its raw values.
    log : Callable[[str], None] | None, default=None
        Optional logger.

    Returns
    -------
    pd.DataFrame
        Comparison table containing feature name, dataset split, category,
        loan count, and default rate percent.
    """
    comparison_tables: list[pd.DataFrame] = []

    for dataset_split, df_split in [
        (train_label, df_train),
        (test_label, df_test),
    ]:
        if feature_name not in df_split.columns:
            raise KeyError(
                f"build_default_rate_by_category_comparison: "
                f"missing required column '{feature_name}' in {dataset_split} dataset"
            )

        if target_column not in df_split.columns:
            raise KeyError(
                f"build_default_rate_by_category_comparison: "
                f"missing required target column '{target_column}' in {dataset_split} dataset"
            )

        if treat_as_missingness:
            category_column_name = "category"
            df_grouping = df_split.copy()
            df_grouping[category_column_name] = (
                df_grouping[feature_name]
                .isna()
                .map({True: "missing", False: "not_missing"})
                .astype("string")
            )
        else:
            category_column_name = feature_name
            df_grouping = df_split.copy()
            df_grouping[category_column_name] = (
                df_grouping[feature_name]
                .astype("string")
                .fillna("missing")
            )

        category_table = (
            df_grouping
            .groupby(category_column_name, dropna=False)[target_column]
            .agg(
                default_rate_percent=lambda series: round(float(series.mean() * 100), 2),
                loan_count="size",
            )
            .reset_index()
        )

        category_table["feature_name"] = feature_name
        category_table["dataset_split"] = dataset_split

        if treat_as_missingness:
            category_table = category_table.rename(
                columns={category_column_name: "category"}
            )
            category_table = category_table[
                [
                    "feature_name",
                    "dataset_split",
                    "category",
                    "loan_count",
                    "default_rate_percent",
                ]
            ]
        else:
            category_table = category_table[
                [
                    "feature_name",
                    "dataset_split",
                    feature_name,
                    "loan_count",
                    "default_rate_percent",
                ]
            ]

        comparison_tables.append(category_table)

    comparison_table = pd.concat(comparison_tables, axis=0, ignore_index=True)

    if log:
        if treat_as_missingness:
            log(
                f"Built default-rate comparison for missingness in feature '{feature_name}'"
            )
        else:
            log(
                f"Built default-rate comparison for categorical feature '{feature_name}'"
            )

    return comparison_table


def build_category_volume_comparison(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    *,
    feature_name: str,
    train_label: str = "train",
    test_label: str = "test",
) -> pd.DataFrame:
    """
    Build a category-volume comparison table for train and test splits.
    """
    comparison_tables: list[pd.DataFrame] = []

    for dataset_split, df_split in [
        (train_label, df_train),
        (test_label, df_test),
    ]:
        volume_table = (
            df_split[feature_name]
            .astype("string")
            .fillna("missing")
            .value_counts(dropna=False)
            .rename_axis(feature_name)
            .reset_index(name="loan_count")
        )

        volume_table["dataset_split"] = dataset_split
        comparison_tables.append(volume_table)

    return pd.concat(comparison_tables, axis=0, ignore_index=True)




