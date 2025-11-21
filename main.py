"""
main.py
End-to-end runner demonstrating the pipeline on a synthetic dataset.
"""

from data_processing import generate_synthetic_dataset, clean_categorical_columns, define_batches_by_column
from analysis import build_contingency_table, chi_square_test, cramers_v, anova_numeric_by_category
from viz import plot_heatmap, plot_stacked_bar
from reporting import summarize_batch, export_summary_to_csv
import matplotlib.pyplot as plt
import os

def run_example(out_dir="output_demo"):
    # 1. Generate synthetic data
    df = generate_synthetic_dataset(n_rows=2000, random_state=42)
    categorical_cols = ["Region", "PropertyType", "Satisfaction", "District", "BatchID"]
    df = clean_categorical_columns(df, categorical_cols)

    # 2. Define batches by BatchID
    batches = define_batches_by_column(df, "BatchID")
    os.makedirs(out_dir, exist_ok=True)

    # 3. For each batch compute contingency table between Region and Satisfaction
    for i, b in enumerate(batches):
        batch_name = b["BatchID"].iloc[0]
        print(f"Processing {batch_name} | size: {len(b)}")
        table = build_contingency_table(b, row="Region", col="Satisfaction")
        chi2_res = chi_square_test(table)
        v = cramers_v(table)
        print(f"Chi2: {chi2_res['chi2']:.3f}, p={chi2_res['p_value']:.4f}, Cramer's V={v:.3f}")

        # Export summary
        summary = summarize_batch(b, row="Region", col="Satisfaction")
        export_summary_to_csv(summary, out_dir=out_dir, batch_name=batch_name)

        # Plot heatmap and save
        fig, ax = plot_heatmap(summary['table'], title=f"{batch_name}: Region vs Satisfaction")
        fig.savefig(os.path.join(out_dir, f"{batch_name}_heatmap.png"))
        plt.close(fig)

        # Plot stacked bar
        fig2, ax2 = plot_stacked_bar(summary['table'], title=f"{batch_name}: Region vs Satisfaction (Proportions)")
        fig2.savefig(os.path.join(out_dir, f"{batch_name}_stackedbar.png"))
        plt.close(fig2)

    # 4. ANOVA example: test if price differs by PropertyType on the full dataset
    anova_res = anova_numeric_by_category(df, numeric_col="Price", category_col="PropertyType")
    print(f"ANOVA Price ~ PropertyType: F={anova_res['F']:.3f}, p={anova_res['p']:.4f}")

if __name__ == "__main__":
    run_example()
