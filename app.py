# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 10:39:53 2026

@author: ADMIN
"""

# =========================================================
# IMPORT LIBRARIES
# =========================================================
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS (professional theme)
# =========================================================
st.markdown("""
<style>
    /* Overall background */
    .main {
        background-color: #f5f7fa;
    }

    /* Header banner */
    .app-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem 2.5rem;
        border-radius: 14px;
        margin-bottom: 1.8rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.15);
    }
    .app-header h1 {
        color: #ffffff;
        font-size: 2.1rem;
        margin-bottom: 0.3rem;
    }
    .app-header p {
        color: #dbe4f0;
        font-size: 1rem;
        margin: 0;
    }

    /* Section card look */
    .block-container {
        padding-top: 2rem;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e3e7ee;
        border-radius: 12px;
        padding: 1rem 1rem 0.5rem 1rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        transition: 0.2s ease-in-out;
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 10px rgba(30,60,114,0.35);
    }

    /* Download button */
    div.stDownloadButton > button {
        background: linear-gradient(135deg, #0f9b6e 0%, #16c784 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
    }

    /* File uploader */
    section[data-testid="stFileUploaderDropzone"] {
        border: 2px dashed #2a5298;
        border-radius: 12px;
        background-color: #ffffff;
    }

    /* Subheaders */
    h2, h3 {
        color: #1e3c72;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1e3c72;
    }
    section[data-testid="stSidebar"] * {
        color: #f0f2f6 !important;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 💰 Loan Default App")
    st.markdown("---")
    st.markdown(
        """
        **How to use:**
        1. Upload a customer CSV file
        2. Preview the uploaded data
        3. Click **Predict**
        4. Download the results

        ---
        **Note:** If your CSV has a `default`
        column, it will be automatically
        excluded before prediction.
        """
    )
    st.markdown("---")
    st.caption("Built with Streamlit • Decision Tree Pipeline")

# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div class="app-header">
        <h1>💰 Loan Default Prediction System</h1>
        <p>Upload a customer dataset to instantly predict loan default risk.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# LOAD COMPLETE PIPELINE
# =========================================================
try:
    pipeline = joblib.load(
        r"C:\Users\ADMIN\Desktop\Data science practice\Machine_learning\Decision Tree\loan_default_pipeline.pkl"
    )
except Exception as e:
    st.error(f"⚠️ Could not load model pipeline: {e}")
    st.stop()

# =========================================================
# FILE UPLOADER
# =========================================================
st.subheader("📂 Upload Dataset")
uploaded_file = st.file_uploader(
    "Drag and drop a CSV file here, or click to browse",
    type=["csv"]
)

# =========================================================
# IF FILE EXISTS
# =========================================================
if uploaded_file is not None:
    # =====================================================
    # READ CSV
    # =====================================================
    df = pd.read_csv(uploaded_file)

    # =====================================================
    # DISPLAY DATA
    # =====================================================
    st.subheader("🔍 Uploaded Dataset Preview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", f"{df.shape[0]:,}")
    col2.metric("Total Columns", f"{df.shape[1]:,}")
    col3.metric("Missing Values", f"{int(df.isna().sum().sum()):,}")

    st.dataframe(df.head(), use_container_width=True)

    # =====================================================
    # REMOVE TARGET COLUMN IF PRESENT
    # =====================================================
    if "default" in df.columns:
        test_data = df.drop("default", axis=1)
    else:
        test_data = df.copy()

    st.markdown("---")

    # =====================================================
    # PREDICTION BUTTON
    # =====================================================
    if st.button("🚀 Predict"):
        try:
            with st.spinner("Running predictions..."):
                # =============================================
                # MAKE PREDICTIONS
                # =============================================
                prediction = pipeline.predict(test_data)

                # =============================================
                # ADD PREDICTIONS
                # =============================================
                df["Prediction"] = prediction

            st.success("✅ Prediction completed successfully!")

            # =============================================
            # DISPLAY RESULTS
            # =============================================
            st.subheader("📊 Prediction Results")
            st.dataframe(df, use_container_width=True)

            # =============================================
            # SUMMARY
            # =============================================
            st.subheader("📈 Prediction Summary")

            counts = df["Prediction"].value_counts().reset_index()
            counts.columns = ["Prediction", "Count"]

            summary_col1, summary_col2 = st.columns([1, 1.4])

            with summary_col1:
                st.dataframe(counts, use_container_width=True, hide_index=True)

            with summary_col2:
                fig = px.pie(
                    counts,
                    names="Prediction",
                    values="Count",
                    hole=0.45,
                    color_discrete_sequence=["#2a5298", "#e74c3c", "#16c784", "#f39c12"]
                )
                fig.update_layout(
                    margin=dict(t=10, b=10, l=10, r=10),
                    legend_title_text="Prediction"
                )
                st.plotly_chart(fig, use_container_width=True)

            # =============================================
            # DOWNLOAD BUTTON
            # =============================================
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Results",
                data=csv,
                file_name="loan_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"❌ Prediction Error: {e}")
else:
    st.info("👆 Please upload a CSV file to get started.")