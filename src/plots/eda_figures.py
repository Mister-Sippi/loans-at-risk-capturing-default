from __future__ import annotations

from math import ceil
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd


def plot_line_comparison(
    df: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    split_column: str,
    title: str,
    xlabel: str,
    ylabel: str,
    split_order: Sequence[str] = ("train", "test"),
) -> None:
    """
    Plot a line chart that compares train and test series in one figure.
    """
    plt.figure(figsize=(9, 4.5))

    for split_name in split_order:
        df_split = df[df[split_column] == split_name].copy()

        if df_split.empty:
            continue

        plt.plot(
            df_split[x_column],
            df_split[y_column],
            marker="o",
            label=split_name,
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_multi_series_line_chart(
    df: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    series_column: str,
    series_order: Sequence[str],
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """
    Plot a line chart with an explicit series order for multi-series diagnostics.
    """
    plt.figure(figsize=(10, 5))

    for series_name in series_order:
        df_series = df[df[series_column] == series_name].copy()

        if df_series.empty:
            continue

        plt.plot(
            df_series[x_column],
            df_series[y_column],
            marker="o",
            label=series_name,
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_numeric_distribution_grid(
    df: pd.DataFrame,
    *,
    column_names: Sequence[str],
    title: str,
    bins: int = 40,
    ncols: int = 2,
) -> None:
    """
    Plot histograms for a group of numeric variables.
    """
    if not column_names:
        raise ValueError("plot_numeric_distribution_grid: column_names must not be empty")

    number_of_columns = len(column_names)
    number_of_rows = ceil(number_of_columns / ncols)

    fig, axes = plt.subplots(
        number_of_rows,
        ncols,
        figsize=(6 * ncols, 3.8 * number_of_rows),
    )

    if hasattr(axes, "flatten"):
        axes_list = axes.flatten()
    else:
        axes_list = [axes]

    for axis, column_name in zip(axes_list, column_names):
        axis.hist(
            df[column_name].dropna(),
            bins=bins,
            color="steelblue",
            alpha=0.9,
            edgecolor="black",
            linewidth=1.0,
        )
        axis.set_title(column_name)
        axis.set_xlabel(column_name)
        axis.set_ylabel("Count")

    for axis in axes_list[number_of_columns:]:
        axis.axis("off")

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    plt.show()


def plot_default_rate_comparison(
    df: pd.DataFrame,
    *,
    category_column: str,
    title: str,
    xlabel: str,
    ylabel: str = "Default Rate (%)",
    split_column: str = "dataset_split",
    value_column: str = "default_rate_percent",
    rotate_labels: bool = False,
    split_order: Sequence[str] = ("train", "test"),
) -> None:
    """
    Plot a grouped bar chart for train-vs-test default-rate comparisons.
    """
    plot_df = df.pivot(
        index=category_column,
        columns=split_column,
        values=value_column,
    )

    present_split_order = [
        split_name for split_name in split_order
        if split_name in plot_df.columns
    ]
    plot_df = plot_df.reindex(columns=present_split_order).sort_index()

    plt.figure(figsize=(10, 4.8))
    plot_df.plot(
        kind="bar",
        ax=plt.gca(),
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if rotate_labels:
        plt.xticks(rotation=45, ha="right")
    else:
        plt.xticks(rotation=0)

    plt.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4)
    plt.tight_layout()
    plt.show()
