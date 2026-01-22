"""
Retail/CPG Planning Analytics App with TabPFN

This Streamlit app provides an interactive interface for supply chain planning
analytics using TabPFN, a foundation model for tabular data.

Use Cases:
- Supplier Delay Risk Prediction (Classification)
- Material Shortage Prediction (Multi-class Classification)
- Price Elasticity Prediction (Regression)
- Promotion Lift Prediction (Regression)
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
    classification_report,
)
import tabpfn_client
from tabpfn_client import TabPFNClassifier, TabPFNRegressor
import os

# Page configuration
st.set_page_config(
    page_title="Retail/CPG Planning Analytics",
    page_icon="📊",
    layout="wide",
)

# App title and description
st.title("📊 Retail/CPG Planning Analytics")
st.markdown("""
**End-to-end supply chain planning powered by TabPFN**, a foundation model for tabular data.

This app demonstrates predictive analytics across the retail/CPG planning value chain:
- **Demand Planning**: Price elasticity, promotion lift prediction
- **Supply Planning**: Supplier delay risk, material shortage prediction
""")

# Databricks configuration
cfg = Config()

# Dataset configurations
CATALOG = "tabpfn_databricks"
SCHEMA = "default"

AVAILABLE_DATASETS = {
    "Supplier Delay Risk (Classification)": {
        "table": f"{CATALOG}.{SCHEMA}.supplier_delay_risk",
        "task": "classification",
        "description": "Predict which supplier deliveries will be delayed",
        "default_target": "is_delayed",
        "target_names": ["On-Time", "Delayed"],
        "exclude_cols": [],
        "business_context": """
        **Business Value**: Enable proactive supply risk mitigation by identifying 
        high-risk deliveries before they impact production.
        """
    },
    "Material Shortage (Multi-class)": {
        "table": f"{CATALOG}.{SCHEMA}.material_shortage",
        "task": "classification",
        "description": "Predict material shortage risk levels (No Risk, At Risk, Critical)",
        "default_target": "shortage_risk",
        "target_names": ["No Risk", "At Risk", "Critical"],
        "exclude_cols": [],
        "business_context": """
        **Business Value**: Prioritize procurement actions based on shortage risk levels
        to prevent stockouts and production disruptions.
        """
    },
    "Price Elasticity (Regression)": {
        "table": f"{CATALOG}.{SCHEMA}.price_elasticity",
        "task": "regression",
        "description": "Predict price elasticity of demand for pricing optimization",
        "default_target": "price_elasticity",
        "exclude_cols": [],
        "business_context": """
        **Business Value**: Optimize pricing strategies by understanding how price 
        changes affect demand for different products and markets.
        """
    },
    "Promotion Lift (Regression)": {
        "table": f"{CATALOG}.{SCHEMA}.promotion_lift",
        "task": "regression",
        "description": "Predict promotional sales lift for trade promotion planning",
        "default_target": "promotion_lift_pct",
        "exclude_cols": [],
        "business_context": """
        **Business Value**: Plan promotions with accurate ROI forecasts to optimize
        trade spend and inventory planning.
        """
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


def prepare_features(df: pd.DataFrame, target_col: str, exclude_cols: list = None):
    """Prepare features by encoding categoricals and separating target."""
    exclude = set(exclude_cols or [])
    exclude.add(target_col)
    
    # Get all columns except target and excluded
    feature_cols = [c for c in df.columns if c not in exclude]
    
    # Identify categorical columns
    cat_cols = df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
    
    if cat_cols:
        df_encoded = pd.get_dummies(df[feature_cols], columns=cat_cols, drop_first=True)
    else:
        df_encoded = df[feature_cols].copy()
    
    X = df_encoded.values
    y = df[target_col].values
    
    return X, y, df_encoded.columns.tolist()


def run_classification(X_train, X_test, y_train, y_test, target_names=None):
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
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr", average="weighted")
        except ValueError:
            roc_auc = None

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


# Get configuration from environment variables (set in app.yaml)
http_path = os.environ.get("DATABRICKS_HTTP_PATH", "")

# Sidebar - Use Case Selection
st.sidebar.header("📊 Use Case Selection")
selected_dataset_name = st.sidebar.selectbox(
    "Select Planning Use Case",
    options=list(AVAILABLE_DATASETS.keys()),
    help="Choose a planning use case for prediction",
)

selected_dataset = AVAILABLE_DATASETS[selected_dataset_name]

# Show description and business context
st.sidebar.info(selected_dataset["description"])
st.sidebar.markdown(selected_dataset["business_context"])

# Show connection status in sidebar
st.sidebar.divider()
st.sidebar.header("🔗 Connection Status")
if http_path and not http_path.startswith("YOUR_"):
    st.sidebar.success("SQL Warehouse: Configured")
else:
    st.sidebar.error("SQL Warehouse: Not configured")

tabpfn_token = os.environ.get("TABPFN_TOKEN", "")
if tabpfn_token and not tabpfn_token.startswith("YOUR_"):
    st.sidebar.success("TabPFN: Configured")
else:
    st.sidebar.error("TabPFN: Not configured")

# Main content - Check configuration
if not http_path or http_path.startswith("YOUR_"):
    st.error("⚠️ SQL Warehouse not configured")
    st.info("""
    **To configure the SQL Warehouse:**
    
    Edit the `app.yaml` file and set the `DATABRICKS_HTTP_PATH` environment variable:
    
    ```yaml
    env:
      - name: DATABRICKS_HTTP_PATH
        value: "/sql/1.0/warehouses/your_warehouse_id"
    ```
    
    **To find your SQL Warehouse HTTP Path:**
    1. Go to your Databricks workspace
    2. Navigate to **SQL Warehouses**
    3. Select your warehouse and click **Connection details**
    4. Copy the **HTTP Path**
    """)
    st.stop()

# Authenticate TabPFN
if not authenticate_tabpfn():
    st.error("⚠️ TabPFN token not configured")
    st.info("""
    **To configure TabPFN authentication:**
    
    Edit the `app.yaml` file and set the `TABPFN_TOKEN` environment variable:
    
    ```yaml
    env:
      - name: TABPFN_TOKEN
        value: "your_tabpfn_token"
    ```
    
    **To get your TabPFN token:**
    1. Sign up at [Prior Labs](https://docs.priorlabs.ai/)
    2. Run `tabpfn_client.get_access_token()` to retrieve your token
    """)
    st.stop()

try:
    conn = get_connection(http_path)

    # Load selected dataset
    st.header(f"📈 {selected_dataset_name}")

    with st.spinner(f"Loading {selected_dataset['table']}..."):
        df = load_table(conn, selected_dataset["table"])

    # Display dataset info in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("Features", df.shape[1] - 1)
    with col3:
        st.metric("Task Type", selected_dataset["task"].capitalize())
    with col4:
        if selected_dataset["task"] == "classification":
            n_classes = df[selected_dataset["default_target"]].nunique()
            st.metric("Classes", n_classes)
        else:
            target_range = df[selected_dataset["default_target"]].max() - df[selected_dataset["default_target"]].min()
            st.metric("Target Range", f"{target_range:.2f}")

    # Target variable
    target_column = selected_dataset["default_target"]
    feature_columns = [col for col in df.columns if col != target_column]

    # Show data preview
    with st.expander("📋 Data Preview", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)
        
        # Show target distribution
        st.subheader("Target Distribution")
        if selected_dataset["task"] == "classification":
            target_counts = df[target_column].value_counts()
            if "target_names" in selected_dataset:
                target_counts.index = [selected_dataset["target_names"][i] for i in target_counts.index]
            st.bar_chart(target_counts)
        else:
            st.write(df[target_column].describe())

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

    # Sample size limit for large datasets
    max_samples = None
    if len(df) > 3000:
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
            # Prepare features
            X, y, feature_names = prepare_features(
                df, target_column, selected_dataset.get("exclude_cols", [])
            )

            # Sample if needed
            if max_samples and len(X) > max_samples:
                np.random.seed(random_state)
                sample_idx = np.random.choice(len(X), size=max_samples, replace=False)
                X = X[sample_idx]
                y = y[sample_idx]
                st.info(f"Using {max_samples:,} samples for training (sampled from {len(df):,} total)")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size / 100, random_state=random_state,
                stratify=y if selected_dataset["task"] == "classification" else None
            )

            # Run appropriate model
            if selected_dataset["task"] == "classification":
                results = run_classification(
                    X_train, X_test, y_train, y_test,
                    target_names=selected_dataset.get("target_names")
                )

                # Display results
                st.header("📊 Classification Results")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Accuracy", f"{results['accuracy']:.4f}")
                with col2:
                    if results['roc_auc']:
                        st.metric("ROC AUC", f"{results['roc_auc']:.4f}")
                with col3:
                    st.metric("Test Samples", len(y_test))

                # Class distribution comparison
                st.subheader("📈 Prediction Distribution")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Actual Distribution:**")
                    actual_counts = pd.Series(results["y_test"]).value_counts().sort_index()
                    if "target_names" in selected_dataset:
                        actual_counts.index = [selected_dataset["target_names"][i] for i in actual_counts.index]
                    st.bar_chart(actual_counts)
                    
                with col2:
                    st.write("**Predicted Distribution:**")
                    pred_counts = pd.Series(results["predictions"]).value_counts().sort_index()
                    if "target_names" in selected_dataset:
                        pred_counts.index = [selected_dataset["target_names"][i] for i in pred_counts.index]
                    st.bar_chart(pred_counts)

                # Prediction details
                st.subheader("🔍 Prediction Details")
                results_df = pd.DataFrame({
                    "Actual": results["y_test"],
                    "Predicted": results["predictions"],
                    "Correct": results["y_test"] == results["predictions"],
                })
                
                # Add probability columns
                n_classes = results["probabilities"].shape[1]
                target_names = selected_dataset.get("target_names", [f"Class_{i}" for i in range(n_classes)])
                for i, name in enumerate(target_names[:n_classes]):
                    results_df[f"Prob_{name}"] = results["probabilities"][:, i].round(4)

                st.dataframe(results_df, use_container_width=True)
                
                # High-risk items (for supply chain use cases)
                if selected_dataset["default_target"] in ["is_delayed", "shortage_risk"]:
                    st.subheader("⚠️ High-Risk Items")
                    if selected_dataset["default_target"] == "is_delayed":
                        high_risk_col = "Prob_Delayed"
                        threshold = 0.5
                    else:
                        high_risk_col = "Prob_Critical"
                        threshold = 0.3
                    
                    if high_risk_col in results_df.columns:
                        high_risk = results_df[results_df[high_risk_col] >= threshold].sort_values(
                            high_risk_col, ascending=False
                        ).head(10)
                        st.write(f"Items with {high_risk_col} >= {threshold}:")
                        st.dataframe(high_risk, use_container_width=True)

            else:  # regression
                results = run_regression(X_train, X_test, y_train, y_test)

                # Display results
                st.header("📊 Regression Results")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if "elasticity" in selected_dataset_name.lower():
                        st.metric("RMSE", f"{results['rmse']:.4f}")
                    else:
                        st.metric("RMSE", f"{results['rmse']:.2f}%")
                with col2:
                    if "elasticity" in selected_dataset_name.lower():
                        st.metric("MAE", f"{results['mae']:.4f}")
                    else:
                        st.metric("MAE", f"{results['mae']:.2f}%")
                with col3:
                    st.metric("R²", f"{results['r2']:.4f}")
                with col4:
                    st.metric("Test Samples", len(y_test))

                # Prediction details
                st.subheader("🔍 Prediction Details")
                results_df = pd.DataFrame({
                    "Actual": results["y_test"],
                    "Predicted": results["predictions"],
                    "Error": results["y_test"] - results["predictions"],
                    "Abs_Error": np.abs(results["y_test"] - results["predictions"]),
                })
                
                # Round appropriately
                for col in results_df.columns:
                    if "elasticity" in selected_dataset_name.lower():
                        results_df[col] = results_df[col].round(4)
                    else:
                        results_df[col] = results_df[col].round(2)
                
                st.dataframe(results_df, use_container_width=True)

                # Scatter plot
                st.subheader("📈 Predicted vs Actual")
                chart_data = pd.DataFrame({
                    "Actual": results["y_test"],
                    "Predicted": results["predictions"],
                })
                st.scatter_chart(chart_data, x="Actual", y="Predicted")
                
                # Summary statistics
                st.subheader("📊 Prediction Summary")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Actual Values:**")
                    st.write(pd.Series(results["y_test"]).describe())
                with col2:
                    st.write("**Predicted Values:**")
                    st.write(pd.Series(results["predictions"]).describe())

        st.success("✅ Prediction complete!")

except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.exception(e)

# Footer
st.divider()
st.markdown("""
### About This App

This app demonstrates **TabPFN** for retail/CPG supply chain planning analytics.

**Planning Value Chain Coverage:**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Demand Planning │───▶│ Supply Planning │───▶│ Production      │
│                 │    │                 │    │ Planning        │
│ • Price Elast.  │    │ • Supplier Risk │    │ • Scrap Detect. │
│ • Promo Lift    │    │ • Material      │    │ • Yield Pred.   │
│ • Forecasting   │    │   Shortage      │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

**Resources:**
- [TabPFN Documentation](https://docs.priorlabs.ai/)
- [TabPFN Client GitHub](https://github.com/PriorLabs/tabpfn-client)
""")
