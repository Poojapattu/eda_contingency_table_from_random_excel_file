"""
reporting.py
Export summaries for each batch: contingency table, test results, effect sizes.
"""

import pandas as pd
from typing import Dict, List
import os
from analysis import chi_square_test, cramers_v, fisher_test_if_applicable, pairwise_chi2_posthoc

def summarize_batch(df_batch: pd.DataFrame, row: str, col: str) -> Dict:
    """
    Create summary dict including contingency table, test results and effect sizes.
    """
    table = pd.crosstab(df_batch[row], df_batch[col], margins=False)
    test = chi_square_test(table)
    fisher = fisher_test_if_applicable(table)
    v = cramers_v(table)
    posthoc = pairwise_chi2_posthoc(table)
    return {"table": table, "chi2": test, "fisher": fisher, "cramers_v": v, "posthoc": posthoc}

def export_summary_to_csv(summary: Dict, out_dir: str, batch_name: str):
    """
    Exports contingency table and results to CSV files in out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    table: pd.DataFrame = summary["table"]
    table.to_csv(os.path.join(out_dir, f"{batch_name}_contingency.csv"))
    # write chi2 basic
    chi2 = summary["chi2"]
    with open(os.path.join(out_dir, f"{batch_name}_chi2.txt"), "w") as f:
        f.write(f"Chi2: {chi2['chi2']}\nP-value: {chi2['p_value']}\nDOF: {chi2['dof']}\n")
    if summary.get("fisher") is not None:
        with open(os.path.join(out_dir, f"{batch_name}_fisher.txt"), "w") as f:
            f.write(str(summary["fisher"]))
    # export cramers v
    with open(os.path.join(out_dir, f"{batch_name}_cramers_v.txt"), "w") as f:
        f.write(str(summary["cramers_v"]))
    # posthoc
    summary["posthoc"].to_csv(os.path.join(out_dir, f"{batch_name}_posthoc.csv"), index=False)
