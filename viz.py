"""
viz.py
Visualization utilities for contingency tables and batch summaries.
Uses matplotlib only (no seaborn) — suitable for publishing.
"""

from typing import Optional
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_heatmap(table: pd.DataFrame, title: str = None, annotate: bool = True, figsize=(8, 6), cmap=None):
    """
    Simple heatmap for a contingency table using matplotlib.
    Do not set colors by default (project rule); user can modify cmap if desired.
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(table.values, aspect='auto')
    # ticks and labels
    ax.set_xticks(np.arange(table.shape[1]))
    ax.set_xticklabels(table.columns, rotation=45, ha='right')
    ax.set_yticks(np.arange(table.shape[0]))
    ax.set_yticklabels(table.index)
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")
    if title:
        ax.set_title(title)
    # annotate
    if annotate:
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                text = ax.text(j, i, int(table.iat[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, orientation='vertical')
    plt.tight_layout()
    return fig, ax


def plot_stacked_bar(table: pd.DataFrame, title: str = None, figsize=(8,6)):
    """
    Plot stacked bar chart by rows for contingency table.
    """
    ax = table.div(table.sum(axis=1), axis=0).plot(kind='bar', stacked=True, figsize=figsize)
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Rows")
    if title:
        ax.set_title(title)
    plt.legend(title="Columns", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return ax.figure, ax
