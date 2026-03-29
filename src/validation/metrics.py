from pathlib import Path
from typing import Callable

import pandas as pd
from sklearn.metrics import auc, roc_curve

import config.logging as log_config


def map_subgrade_to_rank(
    subgrade_series: pd.Series,
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.Series:
    """
    Map LendingClub subgrades from a1-g5 to an ordered numeric risk rank.

    Lower-quality subgrades receive higher rank values:
    - a1 -> 1
    - ...
    - g5 -> 35
    """

    try:
        log_config.emit_log(log, "Mapping subgrades to numeric rank")

        if subgrade_series is None:
            raise ValueError(
                {
                    "stage": "map_subgrade_to_rank",
                    "error": "input_is_none",
                }
            )

        if not isinstance(subgrade_series, pd.Series):
            raise ValueError(
                {
                    "stage": "map_subgrade_to_rank",
                    "error": "input_is_not_series",
                    "input_type": type(subgrade_series).__name__,
                }
            )

        if subgrade_series.empty:
            raise ValueError(
                {
                    "stage": "map_subgrade_to_rank",
                    "error": "input_is_empty",
                }
            )

        if subgrade_series.isna().any():
            raise ValueError(
                {
                    "stage": "map_subgrade_to_rank",
                    "error": "missing_subgrade_values",
                    "missing_rows": int(subgrade_series.isna().sum()),
                }
            )

        ordered_subgrades = [
            f"{grade}{subgrade_number}"
            for grade in ["a", "b", "c", "d", "e", "f", "g"]
            for subgrade_number in [1, 2, 3, 4, 5]
        ]

        subgrade_to_rank_mapping = {
            subgrade_value: rank_value
            for rank_value, subgrade_value in enumerate(ordered_subgrades, start=1)
        }

        observed_subgrades = set(subgrade_series.astype(str).unique().tolist())
        unknown_subgrades = sorted(observed_subgrades - set(subgrade_to_rank_mapping.keys()))

        if unknown_subgrades:
            raise ValueError(
                {
                    "stage": "map_subgrade_to_rank",
                    "error": "unknown_subgrade_values",
                    "unknown_subgrades": unknown_subgrades,
                }
            )

        subgrade_rank_series = (
            subgrade_series.astype(str)
            .map(subgrade_to_rank_mapping)
            .astype(int)
        )

        log_config.emit_log(
            log,
            {
                "stage": "map_subgrade_to_rank_complete",
                "rows": int(subgrade_rank_series.shape[0]),
                "min_rank": int(subgrade_rank_series.min()),
                "max_rank": int(subgrade_rank_series.max()),
            },
        )

        return subgrade_rank_series

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "map_subgrade_to_rank_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise


def compute_model_roc_auc(
    y_true: pd.Series,
    y_score: pd.Series,
    system_name: str,
    dataset_name: str,
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Compute ROC curve points and AUC for model-based probability scores.

    Returns a long-format dataframe containing ROC coordinates, thresholds,
    and the AUC value repeated for traceability.
    """

    try:
        log_config.emit_log(
            log,
            f"Computing model ROC AUC for system={system_name}, dataset={dataset_name}",
        )

        if y_true is None:
            raise ValueError(
                {
                    "stage": "compute_model_roc_auc",
                    "error": "y_true_is_none",
                }
            )

        if y_score is None:
            raise ValueError(
                {
                    "stage": "compute_model_roc_auc",
                    "error": "y_score_is_none",
                }
            )

        if not isinstance(y_true, pd.Series):
            raise ValueError(
                {
                    "stage": "compute_model_roc_auc",
                    "error": "y_true_is_not_series",
                    "input_type": type(y_true).__name__,
                }
            )

        if not isinstance(y_score, pd.Series):
            raise ValueError(
                {
                    "stage": "compute_model_roc_auc",
                    "error": "y_score_is_not_series",
                    "input_type": type(y_score).__name__,
                }
            )

        if y_true.empty:
            raise ValueError(
                {
                    "stage": "compute_model_roc_auc",
                    "error": "y_true_is_empty",
                }
            )

        if y_score.empty:
            raise ValueError(
                {
                    "stage": "compute_model_roc_auc",
                    "error": "y_score_is_empty",
                }
            )

        if len(y_true) != len(y_score):
            raise ValueError(
                {
                    "stage": "compute_model_roc_auc",
                    "error": "length_mismatch",
                    "y_true_rows": len(y_true),
                    "y_score_rows": len(y_score),
                }
            )

        if not y_true.index.equals(y_score.index):
            raise ValueError(
                {
                    "stage": "compute_model_roc_auc",
                    "error": "index_mismatch",
                }
            )

        if y_true.isna().any():
            raise ValueError(
                {
                    "stage": "compute_model_roc_auc",
                    "error": "missing_y_true_values",
                    "missing_rows": int(y_true.isna().sum()),
                }
            )

        if y_score.isna().any():
            raise ValueError(
                {
                    "stage": "compute_model_roc_auc",
                    "error": "missing_y_score_values",
                    "missing_rows": int(y_score.isna().sum()),
                }
            )

        observed_target_values = set(y_true.unique().tolist())
        if not observed_target_values.issubset({0, 1}):
            raise ValueError(
                {
                    "stage": "compute_model_roc_auc",
                    "error": "invalid_target_values",
                    "observed_values": sorted(observed_target_values),
                }
            )

        false_positive_rate, true_positive_rate, threshold_values = roc_curve(
            y_true=y_true,
            y_score=y_score,
        )

        auc_value = auc(false_positive_rate, true_positive_rate)

        roc_auc_dataframe = pd.DataFrame(
            {
                "system_name": system_name,
                "dataset_name": dataset_name,
                "false_positive_rate": false_positive_rate,
                "true_positive_rate": true_positive_rate,
                "threshold": threshold_values,
                "auc": auc_value,
            }
        )

        log_config.emit_log(
            log,
            {
                "stage": "compute_model_roc_auc_complete",
                "system_name": system_name,
                "dataset_name": dataset_name,
                "rows": roc_auc_dataframe.shape[0],
                "auc": float(auc_value),
            },
        )

        return roc_auc_dataframe

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "compute_model_roc_auc_failed",
                "system_name": system_name if "system_name" in locals() else "unknown",
                "dataset_name": dataset_name if "dataset_name" in locals() else "unknown",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise


def compute_baseline_roc_auc_from_subgrade(
    y_true: pd.Series,
    subgrade_series: pd.Series,
    system_name: str,
    dataset_name: str,
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Compute ROC curve points and AUC for the subgrade-based baseline system.

    Subgrades are first converted to a numeric risk rank, which is then used
    as the baseline score.
    """

    try:
        log_config.emit_log(
            log,
            f"Computing baseline ROC AUC from subgrade for system={system_name}, dataset={dataset_name}",
        )

        if y_true is None:
            raise ValueError(
                {
                    "stage": "compute_baseline_roc_auc_from_subgrade",
                    "error": "y_true_is_none",
                }
            )

        if subgrade_series is None:
            raise ValueError(
                {
                    "stage": "compute_baseline_roc_auc_from_subgrade",
                    "error": "subgrade_series_is_none",
                }
            )

        if not isinstance(y_true, pd.Series):
            raise ValueError(
                {
                    "stage": "compute_baseline_roc_auc_from_subgrade",
                    "error": "y_true_is_not_series",
                    "input_type": type(y_true).__name__,
                }
            )

        if not isinstance(subgrade_series, pd.Series):
            raise ValueError(
                {
                    "stage": "compute_baseline_roc_auc_from_subgrade",
                    "error": "subgrade_series_is_not_series",
                    "input_type": type(subgrade_series).__name__,
                }
            )

        if y_true.empty:
            raise ValueError(
                {
                    "stage": "compute_baseline_roc_auc_from_subgrade",
                    "error": "y_true_is_empty",
                }
            )

        if subgrade_series.empty:
            raise ValueError(
                {
                    "stage": "compute_baseline_roc_auc_from_subgrade",
                    "error": "subgrade_series_is_empty",
                }
            )

        if len(y_true) != len(subgrade_series):
            raise ValueError(
                {
                    "stage": "compute_baseline_roc_auc_from_subgrade",
                    "error": "length_mismatch",
                    "y_true_rows": len(y_true),
                    "subgrade_rows": len(subgrade_series),
                }
            )

        if not y_true.index.equals(subgrade_series.index):
            raise ValueError(
                {
                    "stage": "compute_baseline_roc_auc_from_subgrade",
                    "error": "index_mismatch",
                }
            )

        if y_true.isna().any():
            raise ValueError(
                {
                    "stage": "compute_baseline_roc_auc_from_subgrade",
                    "error": "missing_y_true_values",
                    "missing_rows": int(y_true.isna().sum()),
                }
            )

        observed_target_values = set(y_true.unique().tolist())
        if not observed_target_values.issubset({0, 1}):
            raise ValueError(
                {
                    "stage": "compute_baseline_roc_auc_from_subgrade",
                    "error": "invalid_target_values",
                    "observed_values": sorted(observed_target_values),
                }
            )

        subgrade_rank_series = map_subgrade_to_rank(
            subgrade_series=subgrade_series,
            log=log,
        )

        false_positive_rate, true_positive_rate, threshold_values = roc_curve(
            y_true=y_true,
            y_score=subgrade_rank_series,
        )

        auc_value = auc(false_positive_rate, true_positive_rate)

        roc_auc_dataframe = pd.DataFrame(
            {
                "system_name": system_name,
                "dataset_name": dataset_name,
                "false_positive_rate": false_positive_rate,
                "true_positive_rate": true_positive_rate,
                "threshold": threshold_values,
                "auc": auc_value,
            }
        )

        log_config.emit_log(
            log,
            {
                "stage": "compute_baseline_roc_auc_from_subgrade_complete",
                "system_name": system_name,
                "dataset_name": dataset_name,
                "rows": roc_auc_dataframe.shape[0],
                "auc": float(auc_value),
            },
        )

        return roc_auc_dataframe

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "compute_baseline_roc_auc_from_subgrade_failed",
                "system_name": system_name if "system_name" in locals() else "unknown",
                "dataset_name": dataset_name if "dataset_name" in locals() else "unknown",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise


def compute_calibration_table(
    y_true: pd.Series,
    y_score: pd.Series,
    system_name: str,
    dataset_name: str,
    n_bins: int = 10,
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Compute a calibration table by grouping predicted probabilities into
    quantile-based bins and comparing average predicted risk to observed
    default rate.

    Parameters
    ----------
    y_true : pd.Series
        Binary observed outcome series.
    y_score : pd.Series
        Predicted probability series.
    system_name : str
        Name of the evaluated system.
    dataset_name : str
        Name of the dataset split.
    n_bins : int, default=10
        Number of quantile bins used for calibration.
    log : Callable[[str], None] | Path | str | None, default=None
        Logger callable or log destination accepted by ``log_config.emit_log``.

    Returns
    -------
    pd.DataFrame
        Calibration table with one row per bin.

    Raises
    ------
    ValueError
        If required inputs are invalid, indices do not match, or calibration
        bins cannot be constructed.
    """

    try:
        log_config.emit_log(
            log,
            f"Computing calibration table for system={system_name}, dataset={dataset_name}",
        )

        if y_true is None:
            raise ValueError(
                {
                    "stage": "compute_calibration_table",
                    "error": "y_true_is_none",
                }
            )

        if y_score is None:
            raise ValueError(
                {
                    "stage": "compute_calibration_table",
                    "error": "y_score_is_none",
                }
            )

        if not isinstance(y_true, pd.Series):
            raise ValueError(
                {
                    "stage": "compute_calibration_table",
                    "error": "y_true_is_not_series",
                    "input_type": type(y_true).__name__,
                }
            )

        if not isinstance(y_score, pd.Series):
            raise ValueError(
                {
                    "stage": "compute_calibration_table",
                    "error": "y_score_is_not_series",
                    "input_type": type(y_score).__name__,
                }
            )

        if y_true.empty:
            raise ValueError(
                {
                    "stage": "compute_calibration_table",
                    "error": "y_true_is_empty",
                }
            )

        if y_score.empty:
            raise ValueError(
                {
                    "stage": "compute_calibration_table",
                    "error": "y_score_is_empty",
                }
            )

        if len(y_true) != len(y_score):
            raise ValueError(
                {
                    "stage": "compute_calibration_table",
                    "error": "length_mismatch",
                    "y_true_rows": len(y_true),
                    "y_score_rows": len(y_score),
                }
            )

        if not y_true.index.equals(y_score.index):
            raise ValueError(
                {
                    "stage": "compute_calibration_table",
                    "error": "index_mismatch",
                }
            )

        if y_true.isna().any():
            raise ValueError(
                {
                    "stage": "compute_calibration_table",
                    "error": "missing_y_true_values",
                    "missing_rows": int(y_true.isna().sum()),
                }
            )

        if y_score.isna().any():
            raise ValueError(
                {
                    "stage": "compute_calibration_table",
                    "error": "missing_y_score_values",
                    "missing_rows": int(y_score.isna().sum()),
                }
            )

        observed_target_values = set(y_true.unique().tolist())
        if not observed_target_values.issubset({0, 1}):
            raise ValueError(
                {
                    "stage": "compute_calibration_table",
                    "error": "invalid_target_values",
                    "observed_values": sorted(observed_target_values),
                }
            )

        if not isinstance(n_bins, int):
            raise ValueError(
                {
                    "stage": "compute_calibration_table",
                    "error": "n_bins_is_not_int",
                    "input_type": type(n_bins).__name__,
                }
            )

        if n_bins < 2:
            raise ValueError(
                {
                    "stage": "compute_calibration_table",
                    "error": "n_bins_too_small",
                    "n_bins": n_bins,
                }
            )

        if not system_name or not str(system_name).strip():
            raise ValueError(
                {
                    "stage": "compute_calibration_table",
                    "error": "system_name_is_empty",
                }
            )

        if not dataset_name or not str(dataset_name).strip():
            raise ValueError(
                {
                    "stage": "compute_calibration_table",
                    "error": "dataset_name_is_empty",
                }
            )

        df_calibration_input = pd.DataFrame(
            {
                "y_true": y_true.astype(int),
                "y_score": y_score.astype(float),
            }
        ).copy()

        df_calibration_input["calibration_bin"] = pd.qcut(
            df_calibration_input["y_score"],
            q=n_bins,
            duplicates="drop",
        )

        if df_calibration_input["calibration_bin"].isna().any():
            raise ValueError(
                {
                    "stage": "compute_calibration_table",
                    "error": "missing_calibration_bins",
                    "missing_rows": int(df_calibration_input["calibration_bin"].isna().sum()),
                }
            )

        df_calibration_table = (
            df_calibration_input.groupby("calibration_bin", observed=True)
            .agg(
                row_count=("y_true", "size"),
                predicted_probability_mean=("y_score", "mean"),
                observed_default_rate=("y_true", "mean"),
                predicted_probability_min=("y_score", "min"),
                predicted_probability_max=("y_score", "max"),
            )
            .reset_index()
        )

        df_calibration_table["system_name"] = system_name
        df_calibration_table["dataset_name"] = dataset_name
        df_calibration_table["bin_order"] = range(1, len(df_calibration_table) + 1)

        df_calibration_table = df_calibration_table[
            [
                "system_name",
                "dataset_name",
                "bin_order",
                "calibration_bin",
                "row_count",
                "predicted_probability_mean",
                "observed_default_rate",
                "predicted_probability_min",
                "predicted_probability_max",
            ]
        ].reset_index(drop=True)

        log_config.emit_log(
            log,
            {
                "stage": "compute_calibration_table_complete",
                "system_name": system_name,
                "dataset_name": dataset_name,
                "rows": df_calibration_table.shape[0],
                "bins": int(df_calibration_table["bin_order"].nunique()),
            },
        )

        return df_calibration_table

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "compute_calibration_table_failed",
                "system_name": system_name if "system_name" in locals() else "unknown",
                "dataset_name": dataset_name if "dataset_name" in locals() else "unknown",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise