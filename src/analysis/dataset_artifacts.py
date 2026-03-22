from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

import config.logging as log_config



# =============================================================================
# Inspection & Audit
# =============================================================================


def initial_inspection(
    df: pd.DataFrame,
    log: Callable[[str], None] | Path | str | None = None,
) -> dict[str, Any]:
    """
    Perform an initial inspection of a DataFrame.

    Logs key dataset information and returns a structured summary suitable for notebook display.

    Summary includes:
    - shape and memory usage
    - numeric and object columns
    - missing values (all columns)
    - constant columns (excluding fully-null columns)
    - mixed type columns
    - numeric columns stored as objects
    - consolidated feature summary table
    """
    try:
        if df is None:
            raise ValueError("df must not be None")

        log_config.emit_log(log, "===== Starting initial data inspection =====")

        row_count, column_count = df.shape
        memory_usage_mb = df.memory_usage(deep=True).sum() / (1024**2)
        log_config.emit_log(
            log,
            f"Shape: {row_count} rows x {column_count} columns | Memory: {memory_usage_mb:.2f} MB",
        )

        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
        object_columns = df.select_dtypes(include=["object", "string"]).columns.tolist()
        log_config.emit_log(
            log,
            f"Numeric columns: {len(numeric_columns)} | Object/string columns: {len(object_columns)}",
        )

        null_counts = df.isna().sum()
        null_percent = (null_counts / len(df)) * 100

        missing_values_df = (
            pd.DataFrame(
                {
                    "null_count": null_counts,
                    "null_percent": null_percent.round(2),
                }
            )
            .sort_values(by="null_percent", ascending=False)
        )

        log_config.emit_log(log, "\nColumns with missing values (% missing):")
        for column_name, missing_row in missing_values_df.iterrows():
            log_config.emit_log(log, f"{column_name:<30} {missing_row['null_percent']:.2f}%")

        fully_null_columns = missing_values_df.index[
            missing_values_df["null_percent"] == 100
        ].tolist()

        constant_columns_all = df.columns[df.nunique(dropna=False) <= 1].tolist()
        constant_columns = [
            column_name
            for column_name in constant_columns_all
            if column_name not in fully_null_columns
        ]

        if constant_columns:
            log_config.emit_log(log, "\nConstant columns (non-null):")
            log_config.emit_log(log, ", ".join(constant_columns))

        mixed_type_columns: list[str] = []
        for column_name in df.columns:
            if df[column_name].dropna().map(type).nunique() > 1:
                mixed_type_columns.append(column_name)

        if mixed_type_columns:
            log_config.emit_log(log, "\nColumns with mixed types:")
            log_config.emit_log(log, ", ".join(mixed_type_columns))

        numeric_object_columns: list[str] = []
        for column_name in object_columns:
            non_null_values = df[column_name].dropna()
            if non_null_values.empty:
                continue

            coerced_values = pd.to_numeric(non_null_values, errors="coerce")
            if coerced_values.notna().all():
                numeric_object_columns.append(column_name)

        if numeric_object_columns:
            log_config.emit_log(log, "\nColumns stored as object but numeric:")
            log_config.emit_log(log, ", ".join(numeric_object_columns))

        feature_summary = pd.DataFrame(index=df.columns)
        feature_summary["dtype"] = df.dtypes
        feature_summary["n_unique"] = df.nunique(dropna=False)
        feature_summary["null_percent"] = missing_values_df["null_percent"]
        feature_summary["is_numeric"] = feature_summary.index.isin(numeric_columns)
        feature_summary["is_object"] = feature_summary.index.isin(object_columns)
        feature_summary["is_mixed_type"] = feature_summary.index.isin(mixed_type_columns)
        feature_summary["is_numeric_object"] = feature_summary.index.isin(numeric_object_columns)
        feature_summary["is_fully_null"] = feature_summary["null_percent"] == 100
        feature_summary["is_constant"] = (
            feature_summary["n_unique"] <= 1
        ) & (~feature_summary["is_fully_null"])

        log_config.emit_log(log, "\nFeature structure summary created.")
        log_config.emit_log(log, f"Total columns: {len(feature_summary)}")
        log_config.emit_log(log, f"Fully null columns: {int(feature_summary['is_fully_null'].sum())}")
        log_config.emit_log(log, f"Constant columns (non-null): {int(feature_summary['is_constant'].sum())}")
        log_config.emit_log(log, f"Mixed-type columns: {int(feature_summary['is_mixed_type'].sum())}")
        log_config.emit_log(log, f"Numeric stored as object: {int(feature_summary['is_numeric_object'].sum())}")
        log_config.emit_log(log, "===== Initial data inspection completed successfully =====")

        return {
            "shape": df.shape,
            "memory_usage_mb": memory_usage_mb,
            "numeric_columns": numeric_columns,
            "object_columns": object_columns,
            "missing_values": missing_values_df,
            "constant_columns": constant_columns,
            "mixed_type_columns": mixed_type_columns,
            "numeric_object_columns": numeric_object_columns,
            "feature_summary": feature_summary,
        }

    except Exception as exc:
        try:
            log_config.emit_log(log, f"Error during initial data inspection: {exc}")
        except Exception:
            pass
        raise


def audit_string_columns(
    df: pd.DataFrame,
    sample_size: int = 5,
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Create an audit table for string-like columns.

    Audited dtypes:
    - object
    - string
    - category
    """
    try:
        if df is None:
            raise ValueError("df must not be None")

        if sample_size <= 0:
            raise ValueError("sample_size must be > 0")

        audited_dtypes = ["object", "string", "category"]
        string_like_columns = df.select_dtypes(include=audited_dtypes).columns.tolist()

        if not string_like_columns:
            log_config.emit_log(log, "[audit_string_columns] no string-like columns found")
            return pd.DataFrame(
                columns=[
                    "column_name",
                    "dtype",
                    "unique_count_including_null",
                    "unique_count_non_null",
                    "null_percent",
                    "sample_values",
                ]
            )

        records: list[dict[str, Any]] = []

        for column_name in string_like_columns:
            series = df[column_name]

            unique_count_including_null = int(series.nunique(dropna=False))
            unique_count_non_null = int(series.nunique(dropna=True))
            null_percent = float((series.isna().mean() * 100).round(2))

            sample_values = (
                series.dropna()
                .astype("string")
                .str.strip()
                .drop_duplicates()
                .head(sample_size)
                .tolist()
            )

            records.append(
                {
                    "column_name": column_name,
                    "dtype": str(series.dtype),
                    "unique_count_including_null": unique_count_including_null,
                    "unique_count_non_null": unique_count_non_null,
                    "null_percent": null_percent,
                    "sample_values": sample_values,
                }
            )

        audit_dataframe = (
            pd.DataFrame(records)
            .sort_values(
                by=["unique_count_non_null", "null_percent"],
                ascending=[False, False],
            )
            .reset_index(drop=True)
        )

        log_config.emit_log(
            log,
            "[audit_string_columns] done | "
            f"audited_cols={int(audit_dataframe.shape[0])} "
            f"dtypes={audited_dtypes}",
        )

        return audit_dataframe

    except Exception as exc:
        try:
            log_config.emit_log(
                log,
                f"[audit_string_columns][error] "
                f"Error={type(exc).__name__}: {exc}",
            )
        except Exception:
            pass
        raise


def audit_numeric_columns(
    df: pd.DataFrame,
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Create an audit table for numeric columns.

    Audited dtypes:
    - integer
    - float
    - boolean (if pandas treats it as numeric)

    Returned metrics:
    - dtype
    - unique_count_including_null
    - unique_count_non_null
    - null_percent
    - mean
    - median
    - std
    - min
    - max
    """
    try:
        if df is None:
            raise ValueError("df must not be None")

        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

        if not numeric_columns:
            log_config.emit_log(log, "[numerical_alignment][audit_numeric_columns] no numeric columns found")
            return pd.DataFrame(
                columns=[
                    "column_name",
                    "dtype",
                    "unique_count_including_null",
                    "unique_count_non_null",
                    "null_percent",
                    "mean",
                    "median",
                    "std",
                    "min",
                    "max",
                ]
            )

        records: list[dict[str, Any]] = []

        for column_name in numeric_columns:
            series = df[column_name]

            records.append(
                {
                    "column_name": column_name,
                    "dtype": str(series.dtype),
                    "unique_count_including_null": int(series.nunique(dropna=False)),
                    "unique_count_non_null": int(series.nunique(dropna=True)),
                    "null_percent": float((series.isna().mean() * 100).round(2)),
                    "mean": float(series.mean()) if series.notna().any() else pd.NA,
                    "median": float(series.median()) if series.notna().any() else pd.NA,
                    "std": float(series.std()) if series.notna().sum() > 1 else pd.NA,
                    "min": float(series.min()) if series.notna().any() else pd.NA,
                    "max": float(series.max()) if series.notna().any() else pd.NA,
                }
            )

        audit_dataframe = (
            pd.DataFrame(records)
            .sort_values(
                by=["null_percent", "column_name"],
                ascending=[False, True],
            )
            .reset_index(drop=True)
        )

        log_config.emit_log(
            log,
            "[numerical_alignment][audit_numeric_columns] done | "
            f"audited_cols={int(audit_dataframe.shape[0])}"
        )

        return audit_dataframe

    except Exception as exc:
        try:
            log_config.emit_log(
                log,
                f"[numerical_alignment][audit_numeric_columns][error] "
                f"Error={type(exc).__name__}: {exc}",
            )
        except Exception:
            pass
        raise


def compare_categorical_column_values(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    column_name: str,
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Compare categorical values between training and test datasets.

    Returns a DataFrame containing:
    - value
    - whether present in training
    - whether present in test

    Intended for schema validation and category alignment prior to encoding.
    """
    try:
        if df_train is None or df_test is None:
            raise ValueError("df_train and df_test must not be None")

        if not column_name or not str(column_name).strip():
            raise ValueError("column_name must be a non-empty string")

        if column_name not in df_train.columns:
            raise ValueError(f"Column not found in training dataset: {column_name}")

        if column_name not in df_test.columns:
            raise ValueError(f"Column not found in test dataset: {column_name}")

        log_config.emit_log(log, f"[categorical_compare] start | column='{column_name}'")

        training_values = set(
            df_train[column_name].dropna().astype("string").str.strip().unique()
        )
        test_values = set(
            df_test[column_name].dropna().astype("string").str.strip().unique()
        )

        all_values = sorted(training_values.union(test_values))

        comparison_dataframe = pd.DataFrame(
            {
                "value": all_values,
                "present_in_training": [value in training_values for value in all_values],
                "present_in_test": [value in test_values for value in all_values],
            }
        )

        log_config.emit_log(
            log,
            f"[categorical_compare] done | column='{column_name}' | "
            f"training_unique={len(training_values)} | test_unique={len(test_values)} | "
            f"combined_unique={len(all_values)}",
        )

        return comparison_dataframe

    except Exception as exc:
        try:
            log_config.emit_log(log, f"[categorical_compare] failed | column='{column_name}' | error={exc}")
        except Exception:
            pass
        raise


def build_combined_schema(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    stage_name: str,
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Build a combined schema comparison between train and test datasets.

    Returns a DataFrame containing:
    - column_name
    - train_dtype
    - test_dtype
    - present_in_train
    - present_in_test
    - dtype_mismatch

    Logs structural deltas.
    """
    try:
        if df_train is None or df_test is None:
            raise ValueError("df_train and df_test must not be None")

        if not stage_name or not str(stage_name).strip():
            raise ValueError("stage_name must be a non-empty string")

        train_schema = (
            df_train.dtypes.rename("train_dtype").reset_index().rename(columns={"index": "column_name"})
        )

        test_schema = (
            df_test.dtypes.rename("test_dtype").reset_index().rename(columns={"index": "column_name"})
        )

        combined_schema = pd.merge(train_schema, test_schema, on="column_name", how="outer")

        combined_schema["present_in_train"] = combined_schema["train_dtype"].notna()
        combined_schema["present_in_test"] = combined_schema["test_dtype"].notna()
        combined_schema["dtype_mismatch"] = (
            combined_schema["present_in_train"]
            & combined_schema["present_in_test"]
            & (combined_schema["train_dtype"] != combined_schema["test_dtype"])
        )

        combined_schema = combined_schema.sort_values("column_name").reset_index(drop=True)

        train_only_columns = combined_schema.loc[
            combined_schema["present_in_train"] & ~combined_schema["present_in_test"],
            "column_name",
        ].tolist()

        test_only_columns = combined_schema.loc[
            ~combined_schema["present_in_train"] & combined_schema["present_in_test"],
            "column_name",
        ].tolist()

        dtype_mismatch_records = combined_schema.loc[
            combined_schema["dtype_mismatch"],
            ["column_name", "train_dtype", "test_dtype"],
        ].to_dict(orient="records")

        log_config.emit_log(log, f"[{stage_name}][schema] train_only_count={len(train_only_columns)}")
        log_config.emit_log(log, f"[{stage_name}][schema] test_only_count={len(test_only_columns)}")
        log_config.emit_log(log, f"[{stage_name}][schema] dtype_mismatch_count={len(dtype_mismatch_records)}")

        if train_only_columns:
            log_config.emit_log(log, f"[{stage_name}][schema] train_only_columns={train_only_columns}")

        if test_only_columns:
            log_config.emit_log(log, f"[{stage_name}][schema] test_only_columns={test_only_columns}")

        if dtype_mismatch_records:
            log_config.emit_log(log, f"[{stage_name}][schema] dtype_mismatches={dtype_mismatch_records}")

        return combined_schema

    except Exception as exc:
        try:
            log_config.emit_log(log, f"[{stage_name}][schema] Failed to build combined schema: {exc}")
        except Exception:
            pass
        raise


def build_structural_issues_report(
    combined_feature_universe_df: pd.DataFrame,
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Build a structural issues report from a combined train/test feature universe table.

    Expected columns in combined_feature_universe_df:
      - column_name
      - present_in_train, present_in_test
      - dtype_mismatch
      - train_is_fully_null, test_is_fully_null
      - train_null_percent, test_null_percent
      - train_is_constant, test_is_constant
    """
    try:
        if combined_feature_universe_df is None:
            raise ValueError("combined_feature_universe_df must not be None")

        required_columns = [
            "column_name",
            "present_in_train",
            "present_in_test",
            "dtype_mismatch",
            "train_is_fully_null",
            "test_is_fully_null",
            "train_null_percent",
            "test_null_percent",
            "train_is_constant",
            "test_is_constant",
        ]
        missing_required_columns = [
            column_name for column_name in required_columns if column_name not in combined_feature_universe_df.columns
        ]
        if missing_required_columns:
            raise ValueError(
                f"Missing required columns for structural issues report: {missing_required_columns}"
            )

        issues_records: list[dict[str, object]] = []

        for _, feature_row in combined_feature_universe_df.iterrows():
            column_name = feature_row["column_name"]

            present_in_train = bool(feature_row["present_in_train"])
            present_in_test = bool(feature_row["present_in_test"])

            if present_in_train and not present_in_test:
                issues_records.append(
                    {"column_name": column_name, "issue": "Present in train only", "applies_to": "train"}
                )

            if present_in_test and not present_in_train:
                issues_records.append(
                    {"column_name": column_name, "issue": "Present in test only", "applies_to": "test"}
                )

            if present_in_train and present_in_test and bool(feature_row["dtype_mismatch"]):
                issues_records.append(
                    {"column_name": column_name, "issue": "Dtype mismatch", "applies_to": "both"}
                )

            if present_in_train:
                if bool(feature_row["train_is_fully_null"]):
                    issues_records.append({"column_name": column_name, "issue": "100% Null", "applies_to": "train"})
                elif float(feature_row["train_null_percent"]) >= 50:
                    issues_records.append(
                        {"column_name": column_name, "issue": "High Missing (>50%)", "applies_to": "train"}
                    )

                if bool(feature_row["train_is_constant"]):
                    issues_records.append({"column_name": column_name, "issue": "Constant", "applies_to": "train"})

            if present_in_test:
                if bool(feature_row["test_is_fully_null"]):
                    issues_records.append({"column_name": column_name, "issue": "100% Null", "applies_to": "test"})
                elif float(feature_row["test_null_percent"]) >= 50:
                    issues_records.append(
                        {"column_name": column_name, "issue": "High Missing (>50%)", "applies_to": "test"}
                    )

                if bool(feature_row["test_is_constant"]):
                    issues_records.append({"column_name": column_name, "issue": "Constant", "applies_to": "test"})

        issues_df = pd.DataFrame(issues_records)
        if issues_df.empty:
            issues_df = pd.DataFrame(columns=["column_name", "issue", "applies_to"])

        severity_order = [
            "Present in train only",
            "Present in test only",
            "Dtype mismatch",
            "100% Null",
            "High Missing (>50%)",
            "Constant",
        ]

        issues_df["issue"] = pd.Categorical(issues_df["issue"], categories=severity_order, ordered=True)
        issues_df = issues_df.sort_values(
            ["issue", "column_name", "applies_to"],
            ascending=[True, True, True],
        ).reset_index(drop=True)

        log_config.emit_log(log, f"[structural_issues] rows={issues_df.shape[0]}")

        return issues_df

    except Exception as exc:
        try:
            log_config.emit_log(log, f"[structural_issues] failed: {exc}")
        except Exception:
            pass
        raise


def build_residual_text_column_audit(
    df_to_audit: pd.DataFrame,
    *,
    dataset_split_name: str,
    log: Callable[[dict[str, Any]], None],
    sample_size: int = 5,
) -> pd.DataFrame:
    """
    Identify columns that remain text-like (object/string/category) after transformation.

    Used after transformation layers (clean, feature_base) to verify whether
    residual text columns are expected categorical features or unhandled
    numeric-like strings.
    """

    try:

        text_like_columns: list[str] = []

        for column_name in df_to_audit.columns:

            column_series: pd.Series = df_to_audit[column_name]

            if (
                pd.api.types.is_object_dtype(column_series)
                or pd.api.types.is_string_dtype(column_series)
                or pd.api.types.is_categorical_dtype(column_series)
            ):
                text_like_columns.append(column_name)

        audit_records: list[dict[str, Any]] = []

        for column_name in text_like_columns:

            column_series: pd.Series = df_to_audit[column_name]

            non_missing_values: pd.Series = column_series.dropna().astype("string")

            sample_values_list: list[str] = (
                non_missing_values.head(sample_size).tolist()
            )

            formatted_sample_values: str = (
                " | ".join([str(value) for value in sample_values_list])
                if sample_values_list
                else "no non-missing values"
            )

            missing_count: int = int(column_series.isna().sum())
            missing_rate_percent: float = float(column_series.isna().mean() * 100)

            unique_count_including_missing: int = int(
                column_series.nunique(dropna=False)
            )

            suspicious_numeric_like_text: bool = False

            for sample_value in sample_values_list:

                value_as_string: str = str(sample_value)

                if (
                    "%" in value_as_string
                    or "$" in value_as_string
                    or "," in value_as_string
                ):
                    suspicious_numeric_like_text = True

                numeric_candidate: str = (
                    value_as_string.replace(".", "", 1)
                    .replace("-", "", 1)
                    .replace("%", "")
                )

                if numeric_candidate.isdigit():
                    suspicious_numeric_like_text = True

            audit_records.append(
                {
                    "dataset_split": dataset_split_name,
                    "column_name": column_name,
                    "dtype": str(column_series.dtype),
                    "missing_count": missing_count,
                    "missing_rate_percent": round(missing_rate_percent, 2),
                    "unique_count_including_missing": unique_count_including_missing,
                    "sample_values": formatted_sample_values,
                    "suspicious_numeric_like_text": suspicious_numeric_like_text,
                }
            )

        df_audit: pd.DataFrame = pd.DataFrame(audit_records)

        if not df_audit.empty:

            df_audit = df_audit.sort_values(
                by=["suspicious_numeric_like_text", "column_name"],
                ascending=[False, True],
            ).reset_index(drop=True)

        log(
            {
                "stage": "residual_text_column_audit_complete",
                "dataset_split": dataset_split_name,
                "text_column_count": len(text_like_columns),
            }
        )

        return df_audit

    except Exception as exc:

        log(
            {
                "stage": "residual_text_column_audit_failed",
                "dataset_split": dataset_split_name,
                "error": str(exc),
            }
        )
        raise


def build_dataset_overview(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    *,
    train_label: str = "train",
    test_label: str = "test",
) -> pd.DataFrame:
    """
    Build a compact overview of dataset dimensions and memory usage.
    """
    return pd.DataFrame(
        [
            {
                "dataset_split": train_label,
                "rows": int(df_train.shape[0]),
                "columns": int(df_train.shape[1]),
                "memory_mb": round(float(df_train.memory_usage(deep=True).sum()) / (1024 ** 2), 2),
            },
            {
                "dataset_split": test_label,
                "rows": int(df_test.shape[0]),
                "columns": int(df_test.shape[1]),
                "memory_mb": round(float(df_test.memory_usage(deep=True).sum()) / (1024 ** 2), 2),
            },
        ]
    )


def build_split_table(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    *,
    train_label: str = "train",
    test_label: str = "test",
    index_name: str | None = None,
) -> pd.DataFrame:
    """
    Stack train and test tables into one comparison table with a split index.
    """
    df_train_display = df_train.copy()
    df_train_display["dataset_split"] = train_label

    df_test_display = df_test.copy()
    df_test_display["dataset_split"] = test_label

    combined_table = pd.concat(
        [df_train_display, df_test_display],
        axis=0,
    ).reset_index()

    if index_name is None:
        index_name = combined_table.columns[0]

    combined_table = combined_table.rename(columns={combined_table.columns[0]: index_name})

    split_order = pd.CategoricalDtype(
        categories=[train_label, test_label],
        ordered=True,
    )
    combined_table["dataset_split"] = combined_table["dataset_split"].astype(split_order)

    combined_table = combined_table.set_index(["dataset_split", index_name])

    return combined_table.sort_index(level=["dataset_split", index_name])


def build_feature_group_audit(
    df: pd.DataFrame,
    *,
    feature_groups: dict[str, list[str]],
) -> dict[str, pd.DataFrame | list[str]]:
    """
    Audit feature-group coverage against the columns present in a dataset.
    """
    grouped_columns: list[str] = []
    feature_group_rows: list[dict[str, str | int | bool]] = []

    for group_name, column_names in feature_groups.items():
        for column_name in column_names:
            grouped_columns.append(column_name)
            feature_group_rows.append(
                {
                    "feature_group": group_name,
                    "feature_name": column_name,
                    "present_in_dataset": column_name in df.columns,
                }
            )

    feature_group_summary = pd.DataFrame(feature_group_rows)

    coverage_summary = (
        feature_group_summary
        .groupby("feature_group", as_index=False)["present_in_dataset"]
        .agg(
            grouped_feature_count="size",
            present_feature_count="sum",
        )
    )

    coverage_summary["missing_feature_count"] = (
        coverage_summary["grouped_feature_count"] - coverage_summary["present_feature_count"]
    )

    duplicate_grouped_columns = sorted(
        pd.Series(grouped_columns).value_counts().loc[lambda series: series > 1].index.tolist()
    )

    ungrouped_columns = sorted(
        set(df.columns) - set(grouped_columns)
    )

    return {
        "coverage_summary": coverage_summary.sort_values("feature_group").reset_index(drop=True),
        "feature_group_summary": feature_group_summary.sort_values(
            ["feature_group", "feature_name"]
        ).reset_index(drop=True),
        "duplicate_grouped_columns": duplicate_grouped_columns,
        "ungrouped_columns": ungrouped_columns,
    }


def build_numeric_feature_summary(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    *,
    feature_name: str,
    train_label: str = "train",
    test_label: str = "test",
) -> pd.DataFrame:
    """
    Build a compact numeric-summary table for one feature across train and test.
    """
    summary_rows: list[dict[str, float | str]] = []

    for dataset_split, df_split in [
        (train_label, df_train),
        (test_label, df_test),
    ]:
        feature_series = df_split[feature_name]

        summary_rows.append(
            {
                "feature_name": feature_name,
                "dataset_split": dataset_split,
                "count": int(feature_series.notna().sum()),
                "missing_percent": round(float(feature_series.isna().mean() * 100), 2),
                "mean": round(float(feature_series.mean()), 2),
                "median": round(float(feature_series.median()), 2),
                "p05": round(float(feature_series.quantile(0.05)), 2),
                "p95": round(float(feature_series.quantile(0.95)), 2),
            }
        )

    return pd.DataFrame(summary_rows)


def build_group_mean_table(
    df: pd.DataFrame,
    *,
    group_column: str,
    value_columns: Sequence[str],
) -> pd.DataFrame:
    """
    Build a grouped mean table for selected numeric columns.
    """
    return (
        df.groupby(group_column)[list(value_columns)]
        .mean()
        .round(2)
    )