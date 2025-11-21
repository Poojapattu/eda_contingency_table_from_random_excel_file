"""
utils.py
Small helpers: pretty printing tables as LaTeX or markdown for a paper.
"""

import pandas as pd

def table_to_latex(table: pd.DataFrame, caption: str = "", label: str = "") -> str:
    """
    Convert contingency table to a LaTeX tabular environment (for paper).
    """
    return table.to_latex(index=True, caption=caption, label=label, bold_rows=True)
