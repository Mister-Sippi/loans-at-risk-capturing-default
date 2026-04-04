# src/validation/policy.py

from pathlib import Path
from typing import Callable

import pandas as pd

import config.logging as log_config
import validation.metrics as vm


def apply_model_threshold_policy(
    score_series: pd.Series,
    threshold: float,
    score_column_name: str = "predicted_default_probability",
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Apply a probability-threshold lending policy.

    Loans with predicted default probability less than or equal to the threshold
    are accepted. Loans above the threshold are rejected.
    """

    try:
        log_config.emit_log(
            log,
            f"Applying model threshold policy at threshold={threshold}",
        )

        if score_series is None:
            raise ValueError(
                {
                    "stage": "apply_model_threshold_policy",
                    "error": "score_series_is_none",
                }
            )

        if not isinstance(score_series, pd.Series):
            raise ValueError(
                {
                    "stage": "apply_model_threshold_policy",
                    "error": "score_series_is_not_series",
                    "input_type": type(score_series).__name__,
                }
            )

        if score_series.empty:
            raise ValueError(
                {
                    "stage": "apply_model_threshold_policy",
                    "error": "score_series_is_empty",
                }
            )

        if score_series.isna().any():
            raise ValueError(
                {
                    "stage": "apply_model_threshold_policy",
                    "error": "missing_score_values",
                    "missing_rows": int(score_series.isna().sum()),
                }
            )

        if not isinstance(threshold, (int, float)):
            raise ValueError(
                {
                    "stage": "apply_model_threshold_policy",
                    "error": "threshold_is_not_numeric",
                    "input_type": type(threshold).__name__,
                }
            )

        threshold_float = float(threshold)

        if not 0.0 <= threshold_float <= 1.0:
            raise ValueError(
                {
                    "stage": "apply_model_threshold_policy",
                    "error": "threshold_out_of_bounds",
                    "threshold": threshold_float,
                }
            )

        decision_dataframe = pd.DataFrame(
            {
                score_column_name: score_series.astype(float).copy(),
            },
            index=score_series.index.copy(),
        )

        decision_dataframe["policy_threshold"] = threshold_float
        decision_dataframe["accepted_flag"] = (
            decision_dataframe[score_column_name] <= threshold_float
        ).astype(int)
        decision_dataframe["rejected_flag"] = 1 - decision_dataframe["accepted_flag"]
        decision_dataframe["decision_label"] = decision_dataframe["accepted_flag"].map(
            {1: "accept", 0: "reject"}
        )

        log_config.emit_log(
            log,
            {
                "stage": "apply_model_threshold_policy_complete",
                "rows": decision_dataframe.shape[0],
                "threshold": threshold_float,
                "accepted_rows": int(decision_dataframe["accepted_flag"].sum()),
                "rejected_rows": int(decision_dataframe["rejected_flag"].sum()),
            },
        )

        return decision_dataframe

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "apply_model_threshold_policy_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise


def apply_subgrade_cutoff_policy(
    subgrade_series: pd.Series,
    cutoff_subgrade: str,
    subgrade_column_name: str = "sub_grade",
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Apply a subgrade-cutoff lending policy.

    Loans with subgrade rank less than or equal to the cutoff rank are accepted.
    Loans with worse subgrades are rejected.
    """

    try:
        log_config.emit_log(
            log,
            f"Applying subgrade cutoff policy at cutoff_subgrade={cutoff_subgrade}",
        )

        if subgrade_series is None:
            raise ValueError(
                {
                    "stage": "apply_subgrade_cutoff_policy",
                    "error": "subgrade_series_is_none",
                }
            )

        if not isinstance(subgrade_series, pd.Series):
            raise ValueError(
                {
                    "stage": "apply_subgrade_cutoff_policy",
                    "error": "subgrade_series_is_not_series",
                    "input_type": type(subgrade_series).__name__,
                }
            )

        if subgrade_series.empty:
            raise ValueError(
                {
                    "stage": "apply_subgrade_cutoff_policy",
                    "error": "subgrade_series_is_empty",
                }
            )

        if not cutoff_subgrade or not str(cutoff_subgrade).strip():
            raise ValueError(
                {
                    "stage": "apply_subgrade_cutoff_policy",
                    "error": "cutoff_subgrade_is_empty",
                }
            )

        cutoff_subgrade = str(cutoff_subgrade).strip().lower()

        cutoff_rank = int(
            vm.map_subgrade_to_rank(
                subgrade_series=pd.Series([cutoff_subgrade]),
                log=log,
            ).iloc[0]
        )

        subgrade_rank_series = vm.map_subgrade_to_rank(
            subgrade_series=subgrade_series,
            log=log,
        )

        decision_dataframe = pd.DataFrame(
            {
                subgrade_column_name: subgrade_series.astype(str).copy(),
                "subgrade_rank": subgrade_rank_series.copy(),
            },
            index=subgrade_series.index.copy(),
        )

        decision_dataframe["policy_cutoff_subgrade"] = cutoff_subgrade
        decision_dataframe["policy_cutoff_rank"] = cutoff_rank
        decision_dataframe["accepted_flag"] = (
            decision_dataframe["subgrade_rank"] <= cutoff_rank
        ).astype(int)
        decision_dataframe["rejected_flag"] = 1 - decision_dataframe["accepted_flag"]
        decision_dataframe["decision_label"] = decision_dataframe["accepted_flag"].map(
            {1: "accept", 0: "reject"}
        )

        log_config.emit_log(
            log,
            {
                "stage": "apply_subgrade_cutoff_policy_complete",
                "rows": decision_dataframe.shape[0],
                "cutoff_subgrade": cutoff_subgrade,
                "cutoff_rank": cutoff_rank,
                "accepted_rows": int(decision_dataframe["accepted_flag"].sum()),
                "rejected_rows": int(decision_dataframe["rejected_flag"].sum()),
            },
        )

        return decision_dataframe

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "apply_subgrade_cutoff_policy_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise


def evaluate_policy_outcomes(
    y_true: pd.Series,
    loan_amounts: pd.Series,
    decision_dataframe: pd.DataFrame,
    accepted_flag_column: str = "accepted_flag",
    rejected_flag_column: str = "rejected_flag",
    policy_name: str | None = None,
    policy_value: str | float | int | None = None,
    dataset_name: str | None = None,
    log: Callable[[str], None] | Path | str | None = None,
) -> pd.DataFrame:
    """
    Evaluate lending-policy outcomes from accept/reject decisions.

    Accepted loans with default outcome are false positives from a lending
    perspective (bad loans accepted). Rejected loans with non-default outcome
    are false negatives (good loans rejected).
    """

    try:
        log_config.emit_log(log, "Evaluating policy outcomes")

        if y_true is None:
            raise ValueError(
                {
                    "stage": "evaluate_policy_outcomes",
                    "error": "y_true_is_none",
                }
            )

        if loan_amounts is None:
            raise ValueError(
                {
                    "stage": "evaluate_policy_outcomes",
                    "error": "loan_amounts_is_none",
                }
            )

        if decision_dataframe is None:
            raise ValueError(
                {
                    "stage": "evaluate_policy_outcomes",
                    "error": "decision_dataframe_is_none",
                }
            )

        if not isinstance(y_true, pd.Series):
            raise ValueError(
                {
                    "stage": "evaluate_policy_outcomes",
                    "error": "y_true_is_not_series",
                    "input_type": type(y_true).__name__,
                }
            )

        if not isinstance(loan_amounts, pd.Series):
            raise ValueError(
                {
                    "stage": "evaluate_policy_outcomes",
                    "error": "loan_amounts_is_not_series",
                    "input_type": type(loan_amounts).__name__,
                }
            )

        if not isinstance(decision_dataframe, pd.DataFrame):
            raise ValueError(
                {
                    "stage": "evaluate_policy_outcomes",
                    "error": "decision_dataframe_is_not_dataframe",
                    "input_type": type(decision_dataframe).__name__,
                }
            )

        if y_true.empty:
            raise ValueError(
                {
                    "stage": "evaluate_policy_outcomes",
                    "error": "y_true_is_empty",
                }
            )

        if loan_amounts.empty:
            raise ValueError(
                {
                    "stage": "evaluate_policy_outcomes",
                    "error": "loan_amounts_is_empty",
                }
            )

        if decision_dataframe.empty:
            raise ValueError(
                {
                    "stage": "evaluate_policy_outcomes",
                    "error": "decision_dataframe_is_empty",
                }
            )

        if len(y_true) != len(loan_amounts):
            raise ValueError(
                {
                    "stage": "evaluate_policy_outcomes",
                    "error": "length_mismatch_y_true_loan_amounts",
                    "y_true_rows": len(y_true),
                    "loan_amount_rows": len(loan_amounts),
                }
            )

        if len(y_true) != len(decision_dataframe):
            raise ValueError(
                {
                    "stage": "evaluate_policy_outcomes",
                    "error": "length_mismatch_y_true_decision_dataframe",
                    "y_true_rows": len(y_true),
                    "decision_rows": len(decision_dataframe),
                }
            )

        if not y_true.index.equals(loan_amounts.index):
            raise ValueError(
                {
                    "stage": "evaluate_policy_outcomes",
                    "error": "index_mismatch_y_true_loan_amounts",
                }
            )

        if not y_true.index.equals(decision_dataframe.index):
            raise ValueError(
                {
                    "stage": "evaluate_policy_outcomes",
                    "error": "index_mismatch_y_true_decision_dataframe",
                }
            )

        required_decision_columns = {
            accepted_flag_column,
            rejected_flag_column,
        }
        missing_decision_columns = required_decision_columns - set(decision_dataframe.columns)

        if missing_decision_columns:
            raise KeyError(
                {
                    "stage": "evaluate_policy_outcomes",
                    "error": "missing_required_decision_columns",
                    "missing_columns": sorted(missing_decision_columns),
                }
            )

        if y_true.isna().any():
            raise ValueError(
                {
                    "stage": "evaluate_policy_outcomes",
                    "error": "missing_y_true_values",
                    "missing_rows": int(y_true.isna().sum()),
                }
            )

        if loan_amounts.isna().any():
            raise ValueError(
                {
                    "stage": "evaluate_policy_outcomes",
                    "error": "missing_loan_amount_values",
                    "missing_rows": int(loan_amounts.isna().sum()),
                }
            )

        observed_target_values = set(y_true.unique().tolist())
        if not observed_target_values.issubset({0, 1}):
            raise ValueError(
                {
                    "stage": "evaluate_policy_outcomes",
                    "error": "invalid_target_values",
                    "observed_values": sorted(observed_target_values),
                }
            )

        decision_values = set(
            pd.concat(
                [
                    decision_dataframe[accepted_flag_column],
                    decision_dataframe[rejected_flag_column],
                ]
            )
            .dropna()
            .unique()
            .tolist()
        )
        if not decision_values.issubset({0, 1}):
            raise ValueError(
                {
                    "stage": "evaluate_policy_outcomes",
                    "error": "invalid_decision_flag_values",
                    "observed_values": sorted(decision_values),
                }
            )

        accepted_flag_sum = (
            decision_dataframe[accepted_flag_column]
            + decision_dataframe[rejected_flag_column]
        )

        if not (accepted_flag_sum == 1).all():
            raise ValueError(
                {
                    "stage": "evaluate_policy_outcomes",
                    "error": "decision_flags_not_complementary",
                }
            )

        df_policy_evaluation = pd.DataFrame(
            {
                "y_true": y_true.astype(int).copy(),
                "loan_amnt": loan_amounts.astype(float).copy(),
                "accepted_flag": decision_dataframe[accepted_flag_column].astype(int).copy(),
                "rejected_flag": decision_dataframe[rejected_flag_column].astype(int).copy(),
            },
            index=y_true.index.copy(),
        )

        df_policy_evaluation["accepted_non_default"] = (
            (df_policy_evaluation["accepted_flag"] == 1)
            & (df_policy_evaluation["y_true"] == 0)
        ).astype(int)

        df_policy_evaluation["accepted_default"] = (
            (df_policy_evaluation["accepted_flag"] == 1)
            & (df_policy_evaluation["y_true"] == 1)
        ).astype(int)

        df_policy_evaluation["rejected_non_default"] = (
            (df_policy_evaluation["rejected_flag"] == 1)
            & (df_policy_evaluation["y_true"] == 0)
        ).astype(int)

        df_policy_evaluation["rejected_default"] = (
            (df_policy_evaluation["rejected_flag"] == 1)
            & (df_policy_evaluation["y_true"] == 1)
        ).astype(int)

        total_rows = int(df_policy_evaluation.shape[0])
        accepted_rows = int(df_policy_evaluation["accepted_flag"].sum())
        rejected_rows = int(df_policy_evaluation["rejected_flag"].sum())

        accepted_default_count = int(df_policy_evaluation["accepted_default"].sum())
        accepted_non_default_count = int(df_policy_evaluation["accepted_non_default"].sum())
        rejected_non_default_count = int(df_policy_evaluation["rejected_non_default"].sum())
        rejected_default_count = int(df_policy_evaluation["rejected_default"].sum())

        total_loan_amount = float(df_policy_evaluation["loan_amnt"].sum())
        accepted_loan_amount = float(
            df_policy_evaluation.loc[
                df_policy_evaluation["accepted_flag"] == 1,
                "loan_amnt",
            ].sum()
        )
        rejected_loan_amount = float(
            df_policy_evaluation.loc[
                df_policy_evaluation["rejected_flag"] == 1,
                "loan_amnt",
            ].sum()
        )

        accepted_default_loan_amount = float(
            df_policy_evaluation.loc[
                df_policy_evaluation["accepted_default"] == 1,
                "loan_amnt",
            ].sum()
        )
        accepted_non_default_loan_amount = float(
            df_policy_evaluation.loc[
                df_policy_evaluation["accepted_non_default"] == 1,
                "loan_amnt",
            ].sum()
        )
        rejected_non_default_loan_amount = float(
            df_policy_evaluation.loc[
                df_policy_evaluation["rejected_non_default"] == 1,
                "loan_amnt",
            ].sum()
        )
        rejected_default_loan_amount = float(
            df_policy_evaluation.loc[
                df_policy_evaluation["rejected_default"] == 1,
                "loan_amnt",
            ].sum()
        )

        policy_outcome_summary = pd.DataFrame(
            [
                {
                    "policy_name": policy_name,
                    "policy_value": policy_value,
                    "dataset_name": dataset_name,
                    "row_count": total_rows,
                    "accepted_count": accepted_rows,
                    "rejected_count": rejected_rows,
                    "acceptance_rate": accepted_rows / total_rows,
                    "rejection_rate": rejected_rows / total_rows,
                    "accepted_default_count": accepted_default_count,
                    "accepted_non_default_count": accepted_non_default_count,
                    "rejected_non_default_count": rejected_non_default_count,
                    "rejected_default_count": rejected_default_count,
                    "default_rate_among_accepted": (
                        accepted_default_count / accepted_rows if accepted_rows > 0 else 0.0
                    ),
                    "non_default_rate_among_accepted": (
                        accepted_non_default_count / accepted_rows if accepted_rows > 0 else 0.0
                    ),
                    "total_loan_amnt": total_loan_amount,
                    "accepted_loan_amnt": accepted_loan_amount,
                    "rejected_loan_amnt": rejected_loan_amount,
                    "accepted_default_loan_amnt": accepted_default_loan_amount,
                    "accepted_non_default_loan_amnt": accepted_non_default_loan_amount,
                    "rejected_non_default_loan_amnt": rejected_non_default_loan_amount,
                    "rejected_default_loan_amnt": rejected_default_loan_amount,
                }
            ]
        )

        log_config.emit_log(
            log,
            {
                "stage": "evaluate_policy_outcomes_complete",
                "policy_name": policy_name,
                "policy_value": policy_value,
                "dataset_name": dataset_name,
                "accepted_count": accepted_rows,
                "rejected_count": rejected_rows,
                "acceptance_rate": (
                    accepted_rows / total_rows if total_rows > 0 else 0.0
                ),
            },
        )

        return policy_outcome_summary

    except Exception as exc:
        log_config.emit_log(
            log,
            {
                "stage": "evaluate_policy_outcomes_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise