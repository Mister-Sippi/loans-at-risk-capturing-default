from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

import config.logging as log_config
import analysis.dataset_artifacts as da


# =============================================================================
# Alignment Analysis
# =============================================================================


def _resolve_export_dir_and_suffix(
    export_dir: Path | str | None,
    export_tag: str | None,
    *,
    log: Callable[[str], None] | Path | str | None = None,
) -> tuple[Path | None, str]:
    try:
        export_dir_path: Path | None = None

        if export_dir is not None and str(export_dir).strip():
            export_dir_path = Path(export_dir)
            export_dir_path.mkdir(parents=True, exist_ok=True)

        export_suffix = f"_{export_tag}" if export_tag else ""

        if export_dir_path is None:
            log_config.emit_log(log, f"[string_alignment][export] disabled | export_tag='{export_tag or ''}'")
        else:
            log_config.emit_log(
                log,
                f"[string_alignment][export] enabled | export_dir='{export_dir_path}' | "
                f"export_suffix='{export_suffix}'",
            )

        return export_dir_path, export_suffix

    except Exception as exc:
        try:
            log_config.emit_log(log, f"[string_alignment][export] failed to resolve export dir | error={exc}")
        except Exception:
            pass
        raise


def _run_string_audits(
    *,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    sample_size: int,
    log: Callable[[str], None] | Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        if df_train is None or df_test is None:
            raise ValueError("df_train and df_test must not be None")

        if sample_size <= 0:
            raise ValueError("sample_size must be > 0")

        log_config.emit_log(
            log,
            "[string_alignment] building audits | "
            f"sample_size={sample_size} | train_shape={df_train.shape} | test_shape={df_test.shape}",
        )

        train_audit_df = da.audit_string_columns(
            df=df_train,
            sample_size=sample_size,
            log=log,
        )

        test_audit_df = da.audit_string_columns(
            df=df_test,
            sample_size=sample_size,
            log=log,
        )

        log_config.emit_log(
            log,
            "[string_alignment] audits built | "
            f"train_string_cols={train_audit_df.shape[0]} | test_string_cols={test_audit_df.shape[0]}",
        )

        return train_audit_df, test_audit_df

    except Exception as exc:
        try:
            log_config.emit_log(log, f"[string_alignment] failed building audits | error={exc}")
        except Exception:
            pass
        raise

    
def _run_numerical_audits(
    *,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    log: Callable[[str], None] | Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        if df_train is None or df_test is None:
            raise ValueError("df_train and df_test must not be None")

        log_config.emit_log(
            log,
            "[numerical_alignment] building audits | "
            f"train_shape={df_train.shape} | test_shape={df_test.shape}",
        )

        train_audit_df = da.audit_numeric_columns(
            df=df_train,
            log=log,
        )

        test_audit_df = da.audit_numeric_columns(
            df=df_test,
            log=log,
        )

        log_config.emit_log(
            log,
            "[numerical_alignment] audits built | "
            f"train_numeric_cols={train_audit_df.shape[0]} | "
            f"test_numeric_cols={test_audit_df.shape[0]}",
        )

        return train_audit_df, test_audit_df

    except Exception as exc:
        try:
            log_config.emit_log(log, f"[numerical_alignment] failed building audits | error={exc}")
        except Exception:
            pass
        raise


def _build_combined_string_alignment_table(
    df_train_audit: pd.DataFrame,
    df_test_audit: pd.DataFrame,
    *,
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    try:
        required_columns = [
            "column_name",
            "dtype",
            "unique_count_including_null",
            "unique_count_non_null",
            "null_percent",
            "sample_values",
        ]

        missing_train = [
            column_name for column_name in required_columns
            if column_name not in df_train_audit.columns
        ]

        missing_test = [
            column_name for column_name in required_columns
            if column_name not in df_test_audit.columns
        ]

        if missing_train or missing_test:
            raise ValueError(
                "audit_string_columns output schema mismatch. "
                f"missing_train={missing_train} missing_test={missing_test}"
            )

        train_renamed_df = df_train_audit.copy().rename(
            columns={
                "dtype": "dtype_train",
                "unique_count_including_null": "unique_including_null_train",
                "unique_count_non_null": "unique_non_null_train",
                "null_percent": "null_percent_train",
                "sample_values": "sample_values_train",
            }
        )

        test_renamed_df = df_test_audit.copy().rename(
            columns={
                "dtype": "dtype_test",
                "unique_count_including_null": "unique_including_null_test",
                "unique_count_non_null": "unique_non_null_test",
                "null_percent": "null_percent_test",
                "sample_values": "sample_values_test",
            }
        )

        combined_df = (
            pd.merge(train_renamed_df, test_renamed_df, on="column_name", how="outer")
            .sort_values("column_name")
            .reset_index(drop=True)
        )

        combined_df["present_in_train"] = combined_df["dtype_train"].notna()
        combined_df["present_in_test"] = combined_df["dtype_test"].notna()

        for column_name in [
            "unique_non_null_train",
            "unique_non_null_test",
            "null_percent_train",
            "null_percent_test",
        ]:
            combined_df[column_name] = pd.to_numeric(combined_df[column_name], errors="coerce")

        combined_df["dtype_mismatch"] = (
            combined_df["present_in_train"]
            & combined_df["present_in_test"]
            & (combined_df["dtype_train"].astype("string") != combined_df["dtype_test"].astype("string"))
        )

        combined_df["unique_non_null_gap_test_minus_train"] = (
            combined_df["unique_non_null_test"].fillna(0) - combined_df["unique_non_null_train"].fillna(0)
        )

        combined_df["null_gap_test_minus_train"] = (
            combined_df["null_percent_test"] - combined_df["null_percent_train"]
        )
        combined_df["max_null_percent"] = combined_df[["null_percent_train", "null_percent_test"]].max(axis=1)

        combined_df["has_difference"] = (
            (combined_df["present_in_train"] ^ combined_df["present_in_test"])
            | combined_df["dtype_mismatch"]
            | (combined_df["unique_non_null_train"].fillna(-1) != combined_df["unique_non_null_test"].fillna(-1))
            | (
                combined_df["null_percent_train"].round(2).fillna(-1)
                != combined_df["null_percent_test"].round(2).fillna(-1)
            )
        )

        log_config.emit_log(
            log,
            "[string_alignment] combined table built | "
            f"union_cols={combined_df.shape[0]} | diffs={int(combined_df['has_difference'].sum())}",
        )

        return combined_df

    except Exception as exc:
        try:
            log_config.emit_log(log, f"[string_alignment] failed building combined alignment table | error={exc}")
        except Exception:
            pass
        raise
    

def _build_combined_numerical_alignment_table(
    df_train_audit: pd.DataFrame,
    df_test_audit: pd.DataFrame,
    *,
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    try:
        required_columns = [
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

        missing_train = [
            column_name
            for column_name in required_columns
            if column_name not in df_train_audit.columns
        ]

        missing_test = [
            column_name
            for column_name in required_columns
            if column_name not in df_test_audit.columns
        ]

        if missing_train or missing_test:
            raise ValueError(
                "audit_numeric_columns output schema mismatch. "
                f"missing_train={missing_train} missing_test={missing_test}"
            )

        train_renamed_df = df_train_audit.copy().rename(
            columns={
                "dtype": "dtype_train",
                "unique_count_including_null": "unique_including_null_train",
                "unique_count_non_null": "unique_non_null_train",
                "null_percent": "null_percent_train",
                "mean": "mean_train",
                "median": "median_train",
                "std": "std_train",
                "min": "min_train",
                "max": "max_train",
            }
        )

        test_renamed_df = df_test_audit.copy().rename(
            columns={
                "dtype": "dtype_test",
                "unique_count_including_null": "unique_including_null_test",
                "unique_count_non_null": "unique_non_null_test",
                "null_percent": "null_percent_test",
                "mean": "mean_test",
                "median": "median_test",
                "std": "std_test",
                "min": "min_test",
                "max": "max_test",
            }
        )

        combined_df = (
            pd.merge(train_renamed_df, test_renamed_df, on="column_name", how="outer")
            .sort_values("column_name")
            .reset_index(drop=True)
        )

        combined_df["present_in_train"] = combined_df["dtype_train"].notna()
        combined_df["present_in_test"] = combined_df["dtype_test"].notna()

        numeric_metric_columns = [
            "unique_non_null_train",
            "unique_non_null_test",
            "null_percent_train",
            "null_percent_test",
            "mean_train",
            "mean_test",
            "median_train",
            "median_test",
            "std_train",
            "std_test",
            "min_train",
            "min_test",
            "max_train",
            "max_test",
        ]

        for column_name in numeric_metric_columns:
            combined_df[column_name] = pd.to_numeric(combined_df[column_name], errors="coerce")

        combined_df["dtype_mismatch"] = (
            combined_df["present_in_train"]
            & combined_df["present_in_test"]
            & (combined_df["dtype_train"].astype("string") != combined_df["dtype_test"].astype("string"))
        )

        combined_df["unique_non_null_gap_test_minus_train"] = (
            combined_df["unique_non_null_test"].fillna(0) - combined_df["unique_non_null_train"].fillna(0)
        )

        combined_df["null_gap_test_minus_train"] = (
            combined_df["null_percent_test"] - combined_df["null_percent_train"]
        )
        combined_df["mean_gap_test_minus_train"] = combined_df["mean_test"] - combined_df["mean_train"]
        combined_df["median_gap_test_minus_train"] = combined_df["median_test"] - combined_df["median_train"]
        combined_df["std_gap_test_minus_train"] = combined_df["std_test"] - combined_df["std_train"]
        combined_df["min_gap_test_minus_train"] = combined_df["min_test"] - combined_df["min_train"]
        combined_df["max_gap_test_minus_train"] = combined_df["max_test"] - combined_df["max_train"]

        combined_df["max_null_percent"] = combined_df[["null_percent_train", "null_percent_test"]].max(axis=1)

        combined_df["has_difference"] = (
            (combined_df["present_in_train"] ^ combined_df["present_in_test"])
            | combined_df["dtype_mismatch"]
            | (
                combined_df["null_percent_train"].round(2).fillna(-1)
                != combined_df["null_percent_test"].round(2).fillna(-1)
            )
            | (
                combined_df["mean_train"].round(6).fillna(-999999)
                != combined_df["mean_test"].round(6).fillna(-999999)
            )
            | (
                combined_df["median_train"].round(6).fillna(-999999)
                != combined_df["median_test"].round(6).fillna(-999999)
            )
            | (
                combined_df["std_train"].round(6).fillna(-999999)
                != combined_df["std_test"].round(6).fillna(-999999)
            )
        )

        log_config.emit_log(
            log,
            "[numerical_alignment] combined table built | "
            f"union_cols={combined_df.shape[0]} | "
            f"diffs={int(combined_df['has_difference'].sum())}",
        )

        return combined_df

    except Exception as exc:
        try:
            log_config.emit_log(log, f"[numerical_alignment] failed building combined alignment table | error={exc}")
        except Exception:
            pass
        raise


def _rank_top_string_deltas(
    deltas_df: pd.DataFrame,
    *,
    top_k: int,
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    try:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        if deltas_df is None or deltas_df.empty:
            return pd.DataFrame(columns=deltas_df.columns if deltas_df is not None else [])

        ranked_df = deltas_df.copy()

        ranked_df["severity_score"] = 0
        ranked_df.loc[(ranked_df["present_in_train"] ^ ranked_df["present_in_test"]), "severity_score"] += 100
        ranked_df.loc[ranked_df["dtype_mismatch"], "severity_score"] += 60
        ranked_df.loc[ranked_df["null_gap_test_minus_train"].abs() >= 5, "severity_score"] += 20
        ranked_df.loc[ranked_df["unique_non_null_gap_test_minus_train"].abs() >= 10, "severity_score"] += 10

        ranked_df["abs_null_gap"] = ranked_df["null_gap_test_minus_train"].abs()
        ranked_df["abs_unique_gap"] = ranked_df["unique_non_null_gap_test_minus_train"].abs()

        top_deltas_df = (
            ranked_df.sort_values(
                ["severity_score", "abs_null_gap", "abs_unique_gap", "column_name"],
                ascending=[False, False, False, True],
            )
            .head(top_k)
            .reset_index(drop=True)
        )

        log_config.emit_log(
            log,
            f"[string_alignment] ranked top deltas | requested={top_k} | returned={top_deltas_df.shape[0]}",
        )
        return top_deltas_df

    except Exception as exc:
        try:
            log_config.emit_log(log, f"[string_alignment] failed ranking top deltas | error={exc}")
        except Exception:
            pass
        raise


def _rank_top_numerical_deltas(
    deltas_df: pd.DataFrame,
    *,
    top_k: int,
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    try:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        if deltas_df is None or deltas_df.empty:
            return pd.DataFrame(columns=deltas_df.columns if deltas_df is not None else [])

        ranked_df = deltas_df.copy()

        ranked_df["severity_score"] = 0
        ranked_df.loc[(ranked_df["present_in_train"] ^ ranked_df["present_in_test"]), "severity_score"] += 100
        ranked_df.loc[ranked_df["dtype_mismatch"], "severity_score"] += 60
        ranked_df.loc[ranked_df["null_gap_test_minus_train"].abs() >= 5, "severity_score"] += 20
        ranked_df.loc[ranked_df["mean_gap_test_minus_train"].abs() > 0, "severity_score"] += 10
        ranked_df.loc[ranked_df["median_gap_test_minus_train"].abs() > 0, "severity_score"] += 10
        ranked_df.loc[ranked_df["std_gap_test_minus_train"].abs() > 0, "severity_score"] += 10

        ranked_df["abs_null_gap"] = ranked_df["null_gap_test_minus_train"].abs()
        ranked_df["abs_mean_gap"] = ranked_df["mean_gap_test_minus_train"].abs()
        ranked_df["abs_median_gap"] = ranked_df["median_gap_test_minus_train"].abs()
        ranked_df["abs_std_gap"] = ranked_df["std_gap_test_minus_train"].abs()

        top_deltas_df = (
            ranked_df.sort_values(
                [
                    "severity_score",
                    "abs_null_gap",
                    "abs_mean_gap",
                    "abs_median_gap",
                    "abs_std_gap",
                    "column_name",
                ],
                ascending=[False, False, False, False, False, True],
            )
            .head(top_k)
            .reset_index(drop=True)
        )

        log_config.emit_log(
            log,
            f"[numerical_alignment] ranked top deltas | requested={top_k} | returned={top_deltas_df.shape[0]}",
        )

        return top_deltas_df

    except Exception as exc:
        try:
            log_config.emit_log(log, f"[numerical_alignment] failed ranking top deltas | error={exc}")
        except Exception:
            pass
        raise


def _build_value_differences(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    top_deltas_df: pd.DataFrame,
    *,
    drilldown_max_columns: int,
    drilldown_top_values_per_side: int,
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    try:
        if drilldown_max_columns <= 0 or top_deltas_df is None or top_deltas_df.empty:
            return pd.DataFrame(columns=["column_name", "value", "present_in"])

        columns_to_drill = top_deltas_df["column_name"].dropna().astype(str).head(drilldown_max_columns).tolist()
        value_records: list[dict[str, Any]] = []

        for column_name in columns_to_drill:
            in_train = column_name in df_train.columns
            in_test = column_name in df_test.columns

            if in_train and in_test:
                diff_df = da.compare_categorical_column_values(
                    df_train=df_train,
                    df_test=df_test,
                    column_name=column_name,
                    log=log,
                )

                required = {"value", "present_in_training", "present_in_test"}
                if not required.issubset(set(diff_df.columns)):
                    log_config.emit_log(
                        log,
                        f"[string_alignment][drilldown] unexpected schema for '{column_name}' | "
                        f"required={sorted(required)} got={sorted(diff_df.columns.tolist())}",
                    )
                    continue

                train_only_values = (
                    diff_df[diff_df["present_in_test"] == False]["value"]
                    .head(drilldown_top_values_per_side)
                    .tolist()
                )
                test_only_values = (
                    diff_df[diff_df["present_in_training"] == False]["value"]
                    .head(drilldown_top_values_per_side)
                    .tolist()
                )

                for value in train_only_values:
                    value_records.append({"column_name": column_name, "value": value, "present_in": "train_only"})

                for value in test_only_values:
                    value_records.append({"column_name": column_name, "value": value, "present_in": "test_only"})

                continue

            if in_train and not in_test:
                log_config.emit_log(log, f"[string_alignment][drilldown] '{column_name}' present in train only; sampling train values")

                train_values = (
                    df_train[column_name]
                    .dropna()
                    .astype("string")
                    .str.strip()
                    .drop_duplicates()
                    .head(drilldown_top_values_per_side)
                    .tolist()
                )

                for value in train_values:
                    value_records.append({"column_name": column_name, "value": value, "present_in": "train_only"})

                continue

            if in_test and not in_train:
                log_config.emit_log(log, f"[string_alignment][drilldown] '{column_name}' present in test only; sampling test values")

                test_values = (
                    df_test[column_name]
                    .dropna()
                    .astype("string")
                    .str.strip()
                    .drop_duplicates()
                    .head(drilldown_top_values_per_side)
                    .tolist()
                )

                for value in test_values:
                    value_records.append({"column_name": column_name, "value": value, "present_in": "test_only"})

                continue

            log_config.emit_log(log, f"[string_alignment][drilldown] skipped missing column in both: {column_name}")

        if not value_records:
            return pd.DataFrame(columns=["column_name", "value", "present_in"])

        value_differences_df = pd.DataFrame(value_records)
        log_config.emit_log(log, f"[string_alignment][drilldown] done | rows={value_differences_df.shape[0]}")

        return value_differences_df

    except Exception as exc:
        try:
            log_config.emit_log(log, f"[string_alignment][drilldown] failed | error={exc}")
        except Exception:
            pass
        raise


def _serialize_sample_values_for_export(
    df: pd.DataFrame,
    *,
    columns_to_serialize: list[str],
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    try:
        export_df = df.copy()

        def serialize_cell(cell_value: Any) -> Any:
            if isinstance(cell_value, list):
                return json.dumps(cell_value, ensure_ascii=False)
            return cell_value

        for column_name in columns_to_serialize:
            if column_name in export_df.columns:
                export_df[column_name] = export_df[column_name].apply(serialize_cell)

        return export_df

    except Exception as exc:
        try:
            log_config.emit_log(log, f"[string_alignment][export] failed serializing sample values | error={exc}")
        except Exception:
            pass
        raise


def _export_string_alignment_reports(
    *,
    summary_df: pd.DataFrame,
    deltas_df: pd.DataFrame,
    top_deltas_df: pd.DataFrame,
    value_differences_df: pd.DataFrame,
    export_dir_path: Path,
    export_base_name: str,
    export_suffix: str,
    export_sample_values_as_json: bool,
    log: Callable[[str], None] | Path | str | None = None,
) -> None:
    try:
        summary_path = export_dir_path / f"{export_base_name}{export_suffix}_summary.csv"
        deltas_path = export_dir_path / f"{export_base_name}{export_suffix}_deltas.csv"
        top_deltas_path = export_dir_path / f"{export_base_name}{export_suffix}_top_deltas.csv"

        deltas_export_df = deltas_df.copy()
        top_deltas_export_df = top_deltas_df.copy()

        if export_sample_values_as_json:
            deltas_export_df = _serialize_sample_values_for_export(
                deltas_export_df,
                columns_to_serialize=["sample_values_train", "sample_values_test"],
                log=log,
            )
            top_deltas_export_df = _serialize_sample_values_for_export(
                top_deltas_export_df,
                columns_to_serialize=["sample_values_train", "sample_values_test"],
                log=log,
            )

        summary_df.to_csv(summary_path, index=False, encoding="utf-8")
        deltas_export_df.to_csv(deltas_path, index=False, encoding="utf-8")
        top_deltas_export_df.to_csv(top_deltas_path, index=False, encoding="utf-8")

        values_path: Path | None = None
        if value_differences_df is not None and not value_differences_df.empty:
            values_path = export_dir_path / f"{export_base_name}{export_suffix}_value_differences.csv"
            value_differences_df.to_csv(values_path, index=False, encoding="utf-8")
            log_config.emit_log(log, f"[string_alignment][export] values={values_path}")
        else:
            log_config.emit_log(log, "[string_alignment][export] values=skipped (empty)")

        log_config.emit_log(
            log,
            "[string_alignment][export] done | "
            f"summary={summary_path} | deltas={deltas_path} | top={top_deltas_path}"
            + (f" | values={values_path}" if values_path is not None else ""),
        )

    except Exception as exc:
        try:
            log_config.emit_log(log, f"[string_alignment][export] failed | error={exc}")
        except Exception:
            pass
        raise


def _export_numerical_alignment_reports(
    *,
    summary_df: pd.DataFrame,
    deltas_df: pd.DataFrame,
    top_deltas_df: pd.DataFrame,
    export_dir_path: Path,
    export_base_name: str,
    export_suffix: str,
    log: Callable[[str], None] | Path | str | None = None,
) -> None:
    try:
        summary_path = export_dir_path / f"{export_base_name}{export_suffix}_summary.csv"
        deltas_path = export_dir_path / f"{export_base_name}{export_suffix}_deltas.csv"
        top_deltas_path = export_dir_path / f"{export_base_name}{export_suffix}_top_deltas.csv"

        summary_df.to_csv(summary_path, index=False, encoding="utf-8")
        deltas_df.to_csv(deltas_path, index=False, encoding="utf-8")
        top_deltas_df.to_csv(top_deltas_path, index=False, encoding="utf-8")

        log_config.emit_log(
            log,
            "[numerical_alignment][export] done | "
            f"summary={summary_path} | deltas={deltas_path} | top={top_deltas_path}",
        )

    except Exception as exc:
        try:
            log_config.emit_log(log, f"[numerical_alignment][export] failed | error={exc}")
        except Exception:
            pass
        raise


def build_string_alignment_report(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    *,
    sample_size: int = 5,
    top_k: int = 10,
    drilldown_max_columns: int = 5,
    drilldown_top_values_per_side: int = 10,
    log: Callable[[str], None] | Path | str | None = None,
    export_dir: Path | str | None = None,
    export_base_name: str = "string_alignment",
    export_tag: str | None = None,
    export_sample_values_as_json: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Compare train vs test for string-like columns using audit_string_columns output.
    """
    try:
        if df_train is None or df_test is None:
            raise ValueError("df_train and df_test must not be None")

        if sample_size <= 0:
            raise ValueError("sample_size must be > 0")

        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        if drilldown_max_columns < 0:
            raise ValueError("drilldown_max_columns must be >= 0")

        if drilldown_top_values_per_side <= 0:
            raise ValueError("drilldown_top_values_per_side must be > 0")

        export_dir_path, export_suffix = _resolve_export_dir_and_suffix(
            export_dir=export_dir,
            export_tag=export_tag,
            log=log,
        )

        log_config.emit_log(
            log,
            "[string_alignment] start | "
            f"train_shape={df_train.shape} | test_shape={df_test.shape} | "
            f"sample_size={sample_size} | top_k={top_k} | drilldown_max_columns={drilldown_max_columns} | "
            f"export_suffix='{export_suffix}'",
        )

        df_train_audit, df_test_audit = _run_string_audits(
            df_train=df_train,
            df_test=df_test,
            sample_size=sample_size,
            log=log,
        )

        combined_df = _build_combined_string_alignment_table(
            df_train_audit=df_train_audit,
            df_test_audit=df_test_audit,
            log=log,
        )

        deltas_df = combined_df[combined_df["has_difference"]].copy()
        top_deltas_df = _rank_top_string_deltas(deltas_df=deltas_df, top_k=top_k, log=log)

        summary_df = pd.DataFrame(
            [
                {
                    "string_like_cols_train": int(df_train_audit.shape[0]),
                    "string_like_cols_test": int(df_test_audit.shape[0]),
                    "string_like_cols_union": int(combined_df.shape[0]),
                    "string_like_cols_with_differences": int(deltas_df.shape[0]),
                    "top_k_returned": int(top_deltas_df.shape[0]),
                }
            ]
        )

        value_differences_df = _build_value_differences(
            df_train=df_train,
            df_test=df_test,
            top_deltas_df=top_deltas_df,
            drilldown_max_columns=drilldown_max_columns,
            drilldown_top_values_per_side=drilldown_top_values_per_side,
            log=log,
        )

        deltas_to_return = deltas_df.drop(
            columns=["severity_score", "abs_null_gap", "abs_unique_gap"],
            errors="ignore",
        ).copy()
        top_deltas_to_return = top_deltas_df.drop(
            columns=["severity_score", "abs_null_gap", "abs_unique_gap"],
            errors="ignore",
        ).copy()

        if export_dir_path is not None:
            _export_string_alignment_reports(
                summary_df=summary_df,
                deltas_df=deltas_to_return,
                top_deltas_df=top_deltas_to_return,
                value_differences_df=value_differences_df,
                export_dir_path=export_dir_path,
                export_base_name=export_base_name,
                export_suffix=export_suffix,
                export_sample_values_as_json=export_sample_values_as_json,
                log=log,
            )

        log_config.emit_log(
            log,
            "[string_alignment] done | "
            f"train_cols={summary_df.loc[0, 'string_like_cols_train']} | "
            f"test_cols={summary_df.loc[0, 'string_like_cols_test']} | "
            f"deltas={summary_df.loc[0, 'string_like_cols_with_differences']} | "
            f"value_differences_rows={value_differences_df.shape[0]}",
        )

        return {
            "summary": summary_df,
            "deltas": deltas_to_return,
            "top_deltas": top_deltas_to_return,
            "value_differences": value_differences_df,
        }

    except Exception as exc:
        try:
            log_config.emit_log(log, f"[string_alignment] failed | error={exc}")
        except Exception:
            pass
        raise
    

def build_numerical_alignment_report(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    *,
    top_k: int = 10,
    log: Callable[[str], None] | Path | str | None = None,
    export_dir: Path | str | None = None,
    export_base_name: str = "numerical_alignment",
    export_tag: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Compare train vs test for numeric columns using audit_numeric_columns output.
    """
    try:
        if df_train is None or df_test is None:
            raise ValueError("df_train and df_test must not be None")

        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        export_dir_path, export_suffix = _resolve_export_dir_and_suffix(
            export_dir=export_dir,
            export_tag=export_tag,
            log=log,
        )

        log_config.emit_log(
            log,
            "[numerical_alignment] start | "
            f"train_shape={df_train.shape} | test_shape={df_test.shape} | "
            f"top_k={top_k} | export_suffix='{export_suffix}'",
        )

        df_train_audit, df_test_audit = _run_numerical_audits(
            df_train=df_train,
            df_test=df_test,
            log=log,
        )

        combined_df = _build_combined_numerical_alignment_table(
            df_train_audit=df_train_audit,
            df_test_audit=df_test_audit,
            log=log,
        )

        deltas_df = combined_df[combined_df["has_difference"]].copy()
        top_deltas_df = _rank_top_numerical_deltas(
            deltas_df=deltas_df,
            top_k=top_k,
            log=log,
        )

        summary_df = pd.DataFrame(
            [
                {
                    "numeric_cols_train": int(df_train_audit.shape[0]),
                    "numeric_cols_test": int(df_test_audit.shape[0]),
                    "numeric_cols_union": int(combined_df.shape[0]),
                    "numeric_cols_with_differences": int(deltas_df.shape[0]),
                    "top_k_returned": int(top_deltas_df.shape[0]),
                }
            ]
        )

        deltas_to_return = deltas_df.drop(
            columns=[
                "severity_score",
                "abs_null_gap",
                "abs_mean_gap",
                "abs_median_gap",
                "abs_std_gap",
            ],
            errors="ignore",
        ).copy()

        top_deltas_to_return = top_deltas_df.drop(
            columns=[
                "severity_score",
                "abs_null_gap",
                "abs_mean_gap",
                "abs_median_gap",
                "abs_std_gap",
            ],
            errors="ignore",
        ).copy()

        if export_dir_path is not None:
            _export_numerical_alignment_reports(
                summary_df=summary_df,
                deltas_df=deltas_to_return,
                top_deltas_df=top_deltas_to_return,
                export_dir_path=export_dir_path,
                export_base_name=export_base_name,
                export_suffix=export_suffix,
                log=log,
            )

        log_config.emit_log(
            log,
            "[numerical_alignment] done | "
            f"train_cols={summary_df.loc[0, 'numeric_cols_train']} | "
            f"test_cols={summary_df.loc[0, 'numeric_cols_test']} | "
            f"deltas={summary_df.loc[0, 'numeric_cols_with_differences']}",
        )

        return {
            "summary": summary_df,
            "deltas": deltas_to_return,
            "top_deltas": top_deltas_to_return,
        }

    except Exception as exc:
        try:
            log_config.emit_log(log, f"[numerical_alignment] failed | error={exc}")
        except Exception:
            pass
        raise