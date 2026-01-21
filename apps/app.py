"""
TabPFN Prediction App for Databricks

This Streamlit app allows users to:
- Select datasets from Unity Catalog Delta tables
- Choose target variables for prediction
- Run TabPFN classification or regression
- View predictions and model performance metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
from databricks import sql
from databricks.sdk.core import Config
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
import tabpfn_client
from tabpfn_client import TabPFNClassifier, TabPFNRegressor
import os

# Page configuration
st.set_page_config(
    page_title="TabPFN Predictions",
    page_icon="🔮",
    layout="wide",
)

# App title and description
st.title("🔮 TabPFN Prediction App")
st.markdown("""
Make predictions on your data using **TabPFN**, a foundation model for tabular data.
TabPFN provides state-of-the-art performance without hyperparameter tuning.
""")

# Databricks configuration
cfg = Config()

# Dataset configurations
CATALOG = "tabpfn_databricks"
SCHEMA = "default"

AVAILABLE_DATASETS = {
    "Breast Cancer (Classification)": {
        "table": f"{CATALOG}.{SCHEMA}.breast_cancer",
        "task": "classification",
        "description": "Binary classification - Predict malignant vs benign tumors",
        "default_target": "target",
    },
    "Iris (Classification)": {
        "table": f"{CATALOG}.{SCHEMA}.iris",
        "task": "classification",
        "description": "Multi-class classification - Predict iris species",
        "default_target": "target",
    },
    "California Housing (Regression)": {
        "table": f"{CATALOG}.{SCHEMA}.california_housing",
        "task": "regression",
        "description": "Regression - Predict median house values",
        "default_target": "target",
    },
}


@st.cache_resource(ttl=300, show_spinner="Connecting to Databricks...")
def get_connection(http_path: str):
    """Create a cached connection to Databricks SQL warehouse."""
    return sql.connect(
        server_hostname=cfg.host,
        http_path=http_path,
        credentials_provider=lambda: cfg.authenticate,
    )


@st.cache_data(ttl=600, show_spinner="Loading data...")
def load_table(_conn, table_name: str) -> pd.DataFrame:
    """Load a table from Unity Catalog into a pandas DataFrame."""
    with _conn.cursor() as cursor:
        cursor.execute(f"SELECT * FROM {table_name}")
        return cursor.fetchall_arrow().to_pandas()


def authenticate_tabpfn():
    """Authenticate with TabPFN using the stored token."""
    token = os.environ.get("TABPFN_TOKEN")
    if token:
        tabpfn_client.set_access_token(token)
        return True
    return False


def run_classification(X_train, X_test, y_train, y_test, class_names=None):
    """Run TabPFN classification and return results."""
    clf = TabPFNClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate ROC AUC (handle binary and multi-class)
    n_classes = len(np.unique(y_test))
    if n_classes == 2:
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    else:
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr", average="weighted")

    return {
        "predictions": y_pred,
        "probabilities": y_pred_proba,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "y_test": y_test,
        "model": clf,
    }


def run_regression(X_train, X_test, y_train, y_test):
    """Run TabPFN regression and return results."""
    reg = TabPFNRegressor()
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "predictions": y_pred,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "y_test": y_test,
        "model": reg,
    }


# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# SQL Warehouse HTTP Path
http_path = st.sidebar.text_input(
    "SQL Warehouse HTTP Path",
    value=os.environ.get("DATABRICKS_HTTP_PATH", ""),
    placeholder="/sql/1.0/warehouses/xxxxxx",
    help="Enter the HTTP path for your Databricks SQL warehouse",
)

# Dataset selection
st.sidebar.header("📊 Dataset Selection")
selected_dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    options=list(AVAILABLE_DATASETS.keys()),
    help="Choose a dataset for prediction",
)

selected_dataset = AVAILABLE_DATASETS[selected_dataset_name]
st.sidebar.info(selected_dataset["description"])

# Main content
if not http_path:
    st.warning("⚠️ Please enter your SQL Warehouse HTTP Path in the sidebar to continue.")
    st.stop()

# Authenticate TabPFN
if not authenticate_tabpfn():
    st.error("❌ TabPFN token not found. Please set the TABPFN_TOKEN environment variable.")
    st.info("""
    **To set up TabPFN authentication:**
    1. Get your TabPFN token from [Prior Labs](https://docs.priorlabs.ai/)
    2. Set it as an environment variable in your app configuration
    """)
    st.stop()

try:
    conn = get_connection(http_path)

    # Load selected dataset
    st.header(f"📈 {selected_dataset_name}")

    with st.spinner(f"Loading {selected_dataset['table']}..."):
        df = load_table(conn, selected_dataset["table"])

    # Display dataset info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Features", df.shape[1] - 1)  # Exclude target
    with col3:
        st.metric("Task Type", selected_dataset["task"].capitalize())

    # Target variable selection
    st.subheader("🎯 Target Variable Selection")
    all_columns = df.columns.tolist()
    default_target = selected_dataset["default_target"]
    default_idx = all_columns.index(default_target) if default_target in all_columns else 0

    target_column = st.selectbox(
        "Select Target Variable",
        options=all_columns,
        index=default_idx,
        help="Choose the column to predict",
    )

    # Feature columns (all except target)
    feature_columns = [col for col in all_columns if col != target_column]

    # Show data preview
    with st.expander("📋 Data Preview", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)

    # Model configuration
    st.subheader("🔧 Model Configuration")
    col1, col2 = st.columns(2)

    with col1:
        test_size = st.slider(
            "Test Set Size (%)",
            min_value=10,
            max_value=50,
            value=20,
            help="Percentage of data to use for testing",
        )

    with col2:
        random_state = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=9999,
            value=42,
            help="Random seed for reproducibility",
        )

    # Sample size limit for regression (TabPFN works best with smaller datasets)
    max_samples = None
    if selected_dataset["task"] == "regression" and len(df) > 3000:
        max_samples = st.slider(
            "Max Training Samples",
            min_value=500,
            max_value=min(5000, len(df)),
            value=2000,
            help="TabPFN works best with smaller datasets. Limit samples for faster inference.",
        )

    # Run prediction button
    if st.button("🚀 Run TabPFN Prediction", type="primary", use_container_width=True):
        with st.spinner("Training TabPFN model..."):
            # Prepare data
            X = df[feature_columns].values
            y = df[target_column].values

            # Sample if needed
            if max_samples and len(X) > max_samples:
                np.random.seed(random_state)
                sample_idx = np.random.choice(len(X), size=max_samples, replace=False)
                X = X[sample_idx]
                y = y[sample_idx]
                st.info(f"Using {max_samples} samples for training (sampled from {len(df)} total)")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size / 100, random_state=random_state
            )

            # Run appropriate model
            if selected_dataset["task"] == "classification":
                results = run_classification(X_train, X_test, y_train, y_test)

                # Display results
                st.header("📊 Classification Results")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Accuracy", f"{results['accuracy']:.4f}")
                with col2:
                    st.metric("ROC AUC", f"{results['roc_auc']:.4f}")

                # Prediction details
                st.subheader("🔍 Prediction Details")
                results_df = pd.DataFrame({
                    "Actual": results["y_test"],
                    "Predicted": results["predictions"],
                    "Correct": results["y_test"] == results["predictions"],
                })

                # Add probability columns
                n_classes = results["probabilities"].shape[1]
                for i in range(n_classes):
                    results_df[f"Prob_Class_{i}"] = results["probabilities"][:, i]

                st.dataframe(results_df, use_container_width=True)

                # Class distribution
                st.subheader("📈 Class Distribution")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Actual Distribution:**")
                    actual_counts = pd.Series(results["y_test"]).value_counts().sort_index()
                    st.bar_chart(actual_counts)
                with col2:
                    st.write("**Predicted Distribution:**")
                    pred_counts = pd.Series(results["predictions"]).value_counts().sort_index()
                    st.bar_chart(pred_counts)

            else:  # regression
                results = run_regression(X_train, X_test, y_train, y_test)

                # Display results
                st.header("📊 Regression Results")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("RMSE", f"{results['rmse']:.4f}")
                with col2:
                    st.metric("MAE", f"{results['mae']:.4f}")
                with col3:
                    st.metric("R²", f"{results['r2']:.4f}")

                # Prediction details
                st.subheader("🔍 Prediction Details")
                results_df = pd.DataFrame({
                    "Actual": results["y_test"],
                    "Predicted": results["predictions"],
                    "Error": results["y_test"] - results["predictions"],
                    "Abs_Error": np.abs(results["y_test"] - results["predictions"]),
                })
                st.dataframe(results_df, use_container_width=True)

                # Scatter plot
                st.subheader("📈 Predicted vs Actual")
                chart_data = pd.DataFrame({
                    "Actual": results["y_test"],
                    "Predicted": results["predictions"],
                })
                st.scatter_chart(chart_data, x="Actual", y="Predicted")

        st.success("✅ Prediction complete!")

except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.exception(e)

# Footer
st.divider()
st.markdown("""
**About TabPFN:**
TabPFN is a foundation model for tabular data developed by [Prior Labs](https://priorlabs.ai/).
It provides state-of-the-art performance without hyperparameter tuning.

**Resources:**
- [TabPFN Documentation](https://docs.priorlabs.ai/)
- [TabPFN Client GitHub](https://github.com/PriorLabs/tabpfn-client)
""")
