import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Batch Contingency Analysis", layout="wide")

st.title("📊 Handling Batches in Contingency Tables")
st.write("Upload your dataset and perform Chi-Square / Fisher’s Test with effect size.")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📂 Dataset Preview")
    st.dataframe(df.head())

    # Select categorical variables
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if len(categorical_cols) >= 2:
        row_var = st.selectbox("Select Row Variable", categorical_cols, index=0)
        col_var = st.selectbox("Select Column Variable", categorical_cols, index=1)

        # Create contingency table
        contingency = pd.crosstab(df[row_var], df[col_var])
        st.subheader("📌 Contingency Table")
        st.dataframe(contingency)

        # Chi-square test
        chi2, p, dof, expected = chi2_contingency(contingency)
        st.markdown(f"**Chi-Square Test Result**")
        st.write(f"Chi2 = {chi2:.3f}, p-value = {p:.4f}, dof = {dof}")

        # Cramer's V
        n = contingency.sum().sum()
        phi2 = chi2 / n
        r, k = contingency.shape
        cramers_v = np.sqrt(phi2 / min(k - 1, r - 1))
        st.write(f"**Cramer’s V (Effect Size):** {cramers_v:.3f}")

        # Fisher’s Exact Test (only if 2x2)
        if contingency.shape == (2, 2):
            odds_ratio, fisher_p = fisher_exact(contingency)
            st.write(f"**Fisher’s Exact Test:** OR={odds_ratio:.3f}, p-value={fisher_p:.4f}")

        # Visualization
        st.subheader("📊 Heatmap of Contingency Table")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(contingency, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

    else:
        st.warning("⚠️ Dataset must have at least 2 categorical columns.")
else:
    st.info("Please upload a CSV file to begin.")
