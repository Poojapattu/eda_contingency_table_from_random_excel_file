"""
analysis.py
Functions to construct contingency tables and perform statistical tests.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from scipy.stats import chi2_contingency, fisher_exact, f_oneway
import math


def build_contingency_table(df: pd.DataFrame, row: str, col: str) -> pd.DataFrame:
    """
    Build contingency table (cross-tab) for two categorical variables.
    Returns a pandas DataFrame (rows x columns).
    """
    if row not in df.columns or col not in df.columns:
        raise KeyError("Row/Col not present in dataframe.")
    table = pd.crosstab(df[row], df[col], margins=False)
    return table


def chi_square_test(table: pd.DataFrame, correction: bool = True) -> Dict:
    """
    Perform Chi-square test on contingency table.
    If expected counts are small, user may want to prefer Fisher's exact (only for 2x2).
    Returns test statistic, p-value, dof, expected frequencies.
    """
    if table.size == 0:
        raise ValueError("Empty contingency table.")
    stat, p, dof, expected = chi2_contingency(table, correction=correction)
    return {"chi2": float(stat), "p_value": float(p), "dof": int(dof), "expected": pd.DataFrame(expected, index=table.index, columns=table.columns)}


def fisher_test_if_applicable(table: pd.DataFrame) -> Optional[Dict]:
    """
    If table is 2x2 and counts are small, run Fisher exact.
    Returns oddsratio and p-value or None if not applicable.
    """
    if table.shape == (2, 2):
        # SciPy's fisher_exact returns (oddsratio, p-value)
        oddsratio, p = fisher_exact(table.values)
        return {"oddsratio": float(oddsratio), "p_value": float(p)}
    return None


def cramers_v(table: pd.DataFrame) -> float:
    """
    Calculate Cramér's V effect size for contingency table association.
    V = sqrt(chi2 / (n * (k-1))) where k = min(n_rows, n_cols)
    """
    chi2, p, dof, expected = chi2_contingency(table)
    n = table.to_numpy().sum()
    if n == 0:
        return float("nan")
    k = min(table.shape[0], table.shape[1])
    if k == 1:
        return 0.0
    v = math.sqrt(chi2 / (n * (k - 1)))
    return float(v)


def pairwise_chi2_posthoc(table: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Perform pairwise chi-square tests between all pairs of rows (or columns) with Bonferroni correction.
    Returns a pandas DataFrame with p-values and corrected p-values (rows vs rows).
    NOTE: This is a simple post-hoc that collapses columns; for rigorous methods consider
    specialized packages.
    """
    # We'll do pairwise on rows (categories)
    rows = table.index.tolist()
    results = []
    m = 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            sub = table.loc[[rows[i], rows[j]], :]
            # if any expected < 5, chi2 may be invalid; but we still compute
            stat, p, dof, expected = chi2_contingency(sub)
            results.append({"row1": rows[i], "row2": rows[j], "chi2": float(stat), "p": float(p)})
            m += 1
    # Bonferroni correction
    for r in results:
        r["p_adj"] = min(1.0, r["p"] * m) if m > 0 else r["p"]
        r["significant"] = r["p_adj"] < alpha
    return pd.DataFrame(results)


def anova_numeric_by_category(df: pd.DataFrame, numeric_col: str, category_col: str) -> Dict:
    """
    Perform one-way ANOVA of a numeric variable across categories.
    Returns F-statistic and p-value.
    """
    groups = [group[numeric_col].dropna().values for _, group in df.groupby(category_col)]
    # Need at least 2 groups with >0 observations
    if len([g for g in groups if len(g) > 0]) < 2:
        return {"F": float("nan"), "p": float("nan")}
    stat, p = f_oneway(*[g for g in groups if len(g) > 0])
    return {"F": float(stat), "p": float(p)}
