"""
Streamlit web application for ML Cutoff Optimizer.

This interactive app allows users to:
- Upload their own data or use example datasets
- Configure optimization parameters
- Visualize probability distributions
- Get optimal cutoff suggestions
- Download results
"""

import sys
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import json

# Add src to path
sys.path.insert(0, "../src")

from ml_cutoff_optimizer import ThresholdVisualizer, CutoffOptimizer, MetricsCalculator


# Page configuration
st.set_page_config(
    page_title="ML Cutoff Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .zone-negative {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .zone-manual {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .zone-positive {
        background-color: #d1ecf1;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
    }
</style>
""",
    unsafe_allow_html=True,
)


def generate_example_data():
    """Generate synthetic example dataset."""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    np.random.seed(42)
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_classes=2,
        weights=[0.6, 0.4],
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]

    df = pd.DataFrame({"y_true": y_test, "y_proba": y_proba})

    return df


def main():
    """Main application function."""

    # Header
    st.markdown(
        '<p class="main-header">📊 ML Cutoff Optimizer</p>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">Find optimal probability thresholds for binary classification with intelligent three-zone analysis</p>',
        unsafe_allow_html=True,
    )

    # Sidebar
    st.sidebar.header("⚙️ Configuration")

    # Data source selection
    data_source = st.sidebar.radio("Data Source", ["Upload CSV", "Use Example Data"])

    df = None

    if data_source == "Upload CSV":
        st.sidebar.markdown("### 📁 Upload Your Data")
        st.sidebar.markdown(
            """
        **Required format:**
        - CSV file with at least 2 columns
        - One column with true labels (0/1)
        - One column with predicted probabilities (0-1)
        """
        )

        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.sidebar.success(f"✅ File loaded: {len(df)} rows")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading file: {e}")

    else:  # Use example data
        df = generate_example_data()
        st.sidebar.info("📊 Using synthetic example data (300 samples)")

    # Only proceed if we have data
    if df is not None and len(df) > 0:

        # Column selection
        st.sidebar.markdown("### 🎯 Column Selection")

        columns = df.columns.tolist()

        col_y_true = st.sidebar.selectbox(
            "True Labels Column (0/1)", columns, index=0 if "y_true" in columns else 0
        )

        col_y_proba = st.sidebar.selectbox(
            "Predicted Probabilities Column (0-1)",
            columns,
            index=1 if "y_proba" in columns else min(1, len(columns) - 1),
        )

        # Extract data
        try:
            y_true = df[col_y_true].values
            y_proba = df[col_y_proba].values

            # Validate
            if not np.all(np.isin(y_true, [0, 1])):
                st.sidebar.error("❌ True labels must contain only 0s and 1s")
                return

            if np.any(y_proba < 0) or np.any(y_proba > 1):
                st.sidebar.error("❌ Probabilities must be between 0 and 1")
                return

            st.sidebar.success("✅ Data validated successfully")

        except Exception as e:
            st.sidebar.error(f"❌ Error processing data: {e}")
            return

        # Optimization parameters
        st.sidebar.markdown("### 🔧 Optimization Parameters")

        step = st.sidebar.slider(
            "Histogram Bin Size",
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            step=0.01,
            help="Smaller values = more detailed histogram",
        )

        min_metric_value = st.sidebar.slider(
            "Minimum Metric Value",
            min_value=0.50,
            max_value=0.99,
            value=0.80,
            step=0.01,
            help="Minimum acceptable performance for automated zones",
        )

        max_manual_zone = st.sidebar.slider(
            "Max Manual Zone Width",
            min_value=0.10,
            max_value=0.60,
            value=0.40,
            step=0.05,
            help="Maximum percentage of data requiring manual review",
        )

        # Main content
        st.markdown("---")

        # Data overview
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("📊 Total Samples", len(y_true))

        with col2:
            pos_count = np.sum(y_true == 1)
            st.metric(
                "✅ Positive Class", f"{pos_count} ({pos_count/len(y_true)*100:.1f}%)"
            )

        with col3:
            neg_count = np.sum(y_true == 0)
            st.metric(
                "❌ Negative Class", f"{neg_count} ({neg_count/len(y_true)*100:.1f}%)"
            )

        with col4:
            st.metric(
                "📈 Probability Range", f"{y_proba.min():.3f} - {y_proba.max():.3f}"
            )

        st.markdown("---")

        # Run optimization
        if st.button("🚀 Run Optimization", type="primary", use_container_width=True):

            with st.spinner("Optimizing cutoffs..."):

                # Create optimizer
                optimizer = CutoffOptimizer(y_true, y_proba)

                # Get cutoffs
                cutoffs = optimizer.suggest_three_zones(
                    min_metric_value=min_metric_value,
                    max_manual_zone_width=max_manual_zone,
                )

                # Store in session state
                st.session_state.cutoffs = cutoffs
                st.session_state.y_true = y_true
                st.session_state.y_proba = y_proba
                st.session_state.step = step

        # Display results if available
        if "cutoffs" in st.session_state:

            cutoffs = st.session_state.cutoffs

            st.markdown("## 🎯 Suggested Cutoffs")

            # Cutoff zones display
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(
                    f"""
                <div class="zone-negative">
                    <h3>🟢 Negative Zone</h3>
                    <p style="font-size: 1.5rem; font-weight: bold;">0% - {cutoffs['negative_cutoff']*100:.1f}%</p>
                    <p><strong>Auto-Reject</strong></p>
                    <p>{cutoffs['population']['negative_zone_pct']:.1f}% of population</p>
                    <p>({cutoffs['population']['negative_zone_count']} samples)</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    f"""
                <div class="zone-manual">
                    <h3>🟡 Manual Zone</h3>
                    <p style="font-size: 1.5rem; font-weight: bold;">{cutoffs['negative_cutoff']*100:.1f}% - {cutoffs['positive_cutoff']*100:.1f}%</p>
                    <p><strong>Human Review</strong></p>
                    <p>{cutoffs['population']['manual_zone_pct']:.1f}% of population</p>
                    <p>({cutoffs['population']['manual_zone_count']} samples)</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col3:
                st.markdown(
                    f"""
                <div class="zone-positive">
                    <h3>🔵 Positive Zone</h3>
                    <p style="font-size: 1.5rem; font-weight: bold;">{cutoffs['positive_cutoff']*100:.1f}% - 100%</p>
                    <p><strong>Auto-Accept</strong></p>
                    <p>{cutoffs['population']['positive_zone_pct']:.1f}% of population</p>
                    <p>({cutoffs['population']['positive_zone_count']} samples)</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            # Visualization
            st.markdown("## 📊 Probability Distribution")

            viz = ThresholdVisualizer(
                st.session_state.y_true,
                st.session_state.y_proba,
                step=st.session_state.step,
            )

            fig, ax = viz.plot_distributions(figsize=(14, 6))
            viz.add_cutoff_lines(
                {
                    "negative_cutoff": cutoffs["negative_cutoff"],
                    "positive_cutoff": cutoffs["positive_cutoff"],
                }
            )

            st.pyplot(fig)
            plt.close(fig)

            st.markdown("---")

            # Performance metrics
            st.markdown("## 📈 Performance Metrics")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🟢 Negative Zone Performance")
                neg_metrics = cutoffs["metrics"]["negative_zone"]

                metric_df = pd.DataFrame(
                    {
                        "Metric": [
                            "Specificity",
                            "False Positive Rate",
                            "Accuracy",
                            "Precision",
                        ],
                        "Value": [
                            f"{neg_metrics['specificity']:.2%}",
                            f"{neg_metrics['fpr']:.2%}",
                            f"{neg_metrics['accuracy']:.2%}",
                            f"{neg_metrics['precision']:.2%}",
                        ],
                    }
                )
                st.dataframe(metric_df, use_container_width=True, hide_index=True)

                st.markdown(
                    f"""
                **Confusion Matrix:**
                - True Negatives: {neg_metrics['tn']}
                - False Positives: {neg_metrics['fp']}
                - False Negatives: {neg_metrics['fn']}
                - True Positives: {neg_metrics['tp']}
                """
                )

            with col2:
                st.markdown("### 🔵 Positive Zone Performance")
                pos_metrics = cutoffs["metrics"]["positive_zone"]

                metric_df = pd.DataFrame(
                    {
                        "Metric": [
                            "Recall",
                            "Precision",
                            "False Negative Rate",
                            "Accuracy",
                        ],
                        "Value": [
                            f"{pos_metrics['recall']:.2%}",
                            f"{pos_metrics['precision']:.2%}",
                            f"{pos_metrics['fnr']:.2%}",
                            f"{pos_metrics['accuracy']:.2%}",
                        ],
                    }
                )
                st.dataframe(metric_df, use_container_width=True, hide_index=True)

                st.markdown(
                    f"""
                **Confusion Matrix:**
                - True Negatives: {pos_metrics['tn']}
                - False Positives: {pos_metrics['fp']}
                - False Negatives: {pos_metrics['fn']}
                - True Positives: {pos_metrics['tp']}
                """
                )

            st.markdown("---")

            # Justification report
            st.markdown("## 📋 Detailed Justification Report")
            st.text(cutoffs["justification"])

            st.markdown("---")

            # Download options
            st.markdown("## 💾 Download Results")

            col1, col2 = st.columns(2)

            with col1:
                # JSON download
                results_json = json.dumps(
                    {
                        "negative_cutoff": cutoffs["negative_cutoff"],
                        "positive_cutoff": cutoffs["positive_cutoff"],
                        "population": cutoffs["population"],
                        "metrics": {
                            "negative_zone": {
                                k: (
                                    float(v)
                                    if isinstance(v, (np.integer, np.floating))
                                    else v
                                )
                                for k, v in cutoffs["metrics"]["negative_zone"].items()
                            },
                            "positive_zone": {
                                k: (
                                    float(v)
                                    if isinstance(v, (np.integer, np.floating))
                                    else v
                                )
                                for k, v in cutoffs["metrics"]["positive_zone"].items()
                            },
                        },
                    },
                    indent=2,
                )

                st.download_button(
                    label="📥 Download Results (JSON)",
                    data=results_json,
                    file_name="cutoff_results.json",
                    mime="application/json",
                    use_container_width=True,
                )

            with col2:
                # CSV download
                results_csv = pd.DataFrame(
                    {
                        "Metric": [
                            "Negative Cutoff",
                            "Positive Cutoff",
                            "Negative Zone %",
                            "Manual Zone %",
                            "Positive Zone %",
                            "Neg Zone Specificity",
                            "Pos Zone Recall",
                        ],
                        "Value": [
                            f"{cutoffs['negative_cutoff']:.4f}",
                            f"{cutoffs['positive_cutoff']:.4f}",
                            f"{cutoffs['population']['negative_zone_pct']:.2f}",
                            f"{cutoffs['population']['manual_zone_pct']:.2f}",
                            f"{cutoffs['population']['positive_zone_pct']:.2f}",
                            f"{cutoffs['metrics']['negative_zone']['specificity']:.4f}",
                            f"{cutoffs['metrics']['positive_zone']['recall']:.4f}",
                        ],
                    }
                )

                csv_buffer = BytesIO()
                results_csv.to_csv(csv_buffer, index=False)

                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="cutoff_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    else:
        st.info(
            "👈 Please upload a CSV file or select 'Use Example Data' in the sidebar to get started."
        )

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; color: #888; padding: 2rem;">
        <p>Built with ❤️ using <strong>Streamlit</strong> and <strong>ML Cutoff Optimizer</strong></p>
        <p>Created by <strong>Luan Drulla</strong> | 
        <a href="https://github.com/Xdrulla/ml-cutoff-optimizer" target="_blank">GitHub</a> | 
        <a href="https://www.linkedin.com/in/luan-drulla-822a24189/" target="_blank">LinkedIn</a></p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
