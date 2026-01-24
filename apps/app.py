"""
Predictive Planning Hub - Powered by TabPFN & Databricks

This Streamlit app provides an interactive interface for supply chain planning
analytics using TabPFN, a foundation model for tabular data.
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
import matplotlib.pyplot as plt
import os

# Prior Labs logo as base64
PRIOR_LABS_LOGO = "data:image/png;base64,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"

# Page configuration
st.set_page_config(
    page_title="Predictive Planning Hub | TabPFN + Databricks",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"

# ============================================================================
# LIGHT PURPLE THEME - PRIOR LABS INSPIRED
# ============================================================================
st.markdown("""
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global light purple background */
    .stApp {
        background: linear-gradient(180deg, #f8f7ff 0%, #f0eeff 50%, #ebe8ff 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    
    /* Headers - dark text for readability */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        color: #1a1a2e !important;
    }
    
    /* Body text */
    p, span, div {
        color: #2d2d44;
    }
    
    /* Hero section */
    .hero-container {
        background: linear-gradient(135deg, #1a1a2e 0%, #2d2d5a 50%, #1a1a2e 100%);
        border-radius: 24px;
        padding: 3rem 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.15);
        position: relative;
        overflow: hidden;
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 20% 30%, rgba(102, 126, 234, 0.2) 0%, transparent 50%),
                    radial-gradient(circle at 80% 70%, rgba(167, 139, 250, 0.2) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        color: #ffffff !important;
        text-align: center;
        margin-bottom: 1rem;
        position: relative;
        z-index: 1;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        color: #c4c4e0 !important;
        text-align: center;
        max-width: 700px;
        margin: 0 auto;
        line-height: 1.7;
        position: relative;
        z-index: 1;
    }
    
    /* Logo container in hero */
    .logo-container {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 2rem;
        margin-bottom: 2rem;
        position: relative;
        z-index: 1;
    }
    
    .logo-divider {
        color: #8888aa;
        font-size: 1.5rem;
        font-weight: 300;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.25);
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
    }
    
    .metric-card-pink {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        box-shadow: 0 8px 32px rgba(240, 147, 251, 0.25);
    }
    
    .metric-card-cyan {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        box-shadow: 0 8px 32px rgba(79, 172, 254, 0.25);
    }
    
    .metric-number {
        font-size: 3rem;
        font-weight: 800;
        color: #ffffff !important;
        margin: 0;
        line-height: 1;
    }
    
    .metric-label {
        font-size: 0.95rem;
        color: rgba(255, 255, 255, 0.95) !important;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* Process cards - light with colored accents */
    .process-card {
        background: #ffffff;
        border: 1px solid rgba(102, 126, 234, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        height: 220px;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.08);
        transition: all 0.3s ease;
    }
    
    .process-card:hover {
        box-shadow: 0 8px 40px rgba(102, 126, 234, 0.15);
        transform: translateY(-2px);
    }
    
    .process-card-demand { border-top: 4px solid #667eea; }
    .process-card-supply { border-top: 4px solid #f093fb; }
    .process-card-production { border-top: 4px solid #4facfe; }
    .process-card-distribution { border-top: 4px solid #00d4aa; }
    
    .process-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
    }
    
    .process-title-demand { color: #667eea !important; }
    .process-title-supply { color: #e056a0 !important; }
    .process-title-production { color: #4facfe !important; }
    .process-title-distribution { color: #00b894 !important; }
    
    .process-item {
        font-size: 0.9rem;
        color: #4a4a6a !important;
        margin: 0.5rem 0;
        padding-left: 0.5rem;
    }
    
    /* Feature cards */
    .feature-card {
        background: #ffffff;
        border: 1px solid rgba(102, 126, 234, 0.12);
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 12px rgba(102, 126, 234, 0.06);
    }
    
    .feature-title {
        font-size: 1rem;
        font-weight: 600;
        color: #1a1a2e !important;
        margin-bottom: 0.5rem;
    }
    
    .feature-text {
        font-size: 0.9rem;
        color: #5a5a7a !important;
        line-height: 1.5;
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(167, 139, 250, 0.08) 100%);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-left: 4px solid #667eea;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.5rem 0;
    }
    
    .info-box-text {
        color: #2d2d44 !important;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1a1a2e !important;
        margin: 2rem 0 1.25rem 0;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.4rem 0.9rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .badge-demand { background: rgba(102, 126, 234, 0.15); color: #4a5fd9 !important; }
    .badge-supply { background: rgba(240, 147, 251, 0.15); color: #c044a0 !important; }
    .badge-production { background: rgba(79, 172, 254, 0.15); color: #3498db !important; }
    .badge-distribution { background: rgba(0, 212, 170, 0.15); color: #00a885 !important; }
    
    /* Powered by section */
    .powered-by {
        background: #ffffff;
        border: 1px solid rgba(102, 126, 234, 0.15);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.08);
    }
    
    .powered-by-title {
        font-size: 0.85rem;
        color: #8888aa !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
    }
    
    .powered-by-name {
        font-size: 1.8rem;
        font-weight: 800;
        color: #667eea !important;
    }
    
    .powered-by-desc {
        color: #6a6a8a !important;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8f7ff 100%);
        border-right: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    section[data-testid="stSidebar"] .stRadio > label {
        color: #2d2d44 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background: #ffffff !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
        border-radius: 10px !important;
        color: #2d2d44 !important;
    }
    
    /* Dataframe */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Divider */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.2), transparent);
        margin: 2rem 0;
    }
    
    /* Status indicators */
    .status-success { color: #00b894 !important; font-weight: 500; }
    .status-warning { color: #fdcb6e !important; font-weight: 500; }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #6a6a8a !important;
        font-size: 0.85rem;
    }
    
    .footer a {
        color: #667eea !important;
        text-decoration: none;
    }
    
    .footer a:hover {
        text-decoration: underline;
    }
    
    /* Fix metric text colors */
    [data-testid="stMetricValue"] {
        color: #1a1a2e !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #4a4a6a !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #ffffff !important;
        border-radius: 10px !important;
        color: #2d2d44 !important;
    }
</style>
""", unsafe_allow_html=True)

# Databricks configuration
cfg = Config()

# Dataset configurations
CATALOG = "tabpfn_databricks"
SCHEMA = "default"

AVAILABLE_DATASETS = {
    "Demand Forecasting (Time Series)": {
        "table": f"{CATALOG}.{SCHEMA}.demand_forecast",
        "task": "forecasting",
        "description": "Forecast product demand by category and region using lag features",
        "default_target": "demand_units",
        "series_id_col": "series_id",
        "date_col": "date",
        "exclude_cols": ["series_id", "date", "category", "region"],
        "planning_process": "Demand Planning",
        "business_context": """**Business Value**: Drive inventory planning, production scheduling, and distribution requirements with accurate demand forecasts."""
    },
    "Price Elasticity (Regression)": {
        "table": f"{CATALOG}.{SCHEMA}.price_elasticity",
        "task": "regression",
        "description": "Predict price elasticity of demand for pricing optimization",
        "default_target": "price_elasticity",
        "exclude_cols": [],
        "planning_process": "Demand Planning",
        "business_context": """**Business Value**: Optimize pricing strategies by understanding how price changes affect demand for different products and markets."""
    },
    "Promotion Lift (Regression)": {
        "table": f"{CATALOG}.{SCHEMA}.promotion_lift",
        "task": "regression",
        "description": "Predict promotional sales lift for trade promotion planning",
        "default_target": "promotion_lift_pct",
        "exclude_cols": [],
        "planning_process": "Demand Planning",
        "business_context": """**Business Value**: Plan promotions with accurate ROI forecasts to optimize trade spend and inventory planning."""
    },
    "Supplier Delay Risk (Classification)": {
        "table": f"{CATALOG}.{SCHEMA}.supplier_delay_risk",
        "task": "classification",
        "description": "Predict which supplier deliveries will be delayed",
        "default_target": "is_delayed",
        "target_names": ["On-Time", "Delayed"],
        "exclude_cols": [],
        "planning_process": "Supply Planning",
        "business_context": """**Business Value**: Enable proactive supply risk mitigation by identifying high-risk deliveries before they impact production."""
    },
    "Supplier Lead Time (Regression)": {
        "table": f"{CATALOG}.{SCHEMA}.supplier_lead_time",
        "task": "regression",
        "description": "Predict actual supplier delivery lead times for planning accuracy",
        "default_target": "actual_lead_time_days",
        "exclude_cols": [],
        "planning_process": "Supply Planning",
        "business_context": """**Business Value**: Improve planning accuracy by predicting actual lead times vs. contracted times, reducing stockouts and expediting costs."""
    },
    "Material Shortage (Multi-class)": {
        "table": f"{CATALOG}.{SCHEMA}.material_shortage",
        "task": "classification",
        "description": "Predict material shortage risk levels (No Risk, At Risk, Critical)",
        "default_target": "shortage_risk",
        "target_names": ["No Risk", "At Risk", "Critical"],
        "exclude_cols": [],
        "planning_process": "Supply Planning",
        "business_context": """**Business Value**: Prioritize procurement actions based on shortage risk levels to prevent stockouts and production disruptions."""
    },
    "Labor Shortage (Multi-class)": {
        "table": f"{CATALOG}.{SCHEMA}.labor_shortage",
        "task": "classification",
        "description": "Predict labor shortage risk at facilities (Adequate, At Risk, Critical)",
        "default_target": "labor_shortage_risk",
        "target_names": ["Adequate", "At Risk", "Critical"],
        "exclude_cols": [],
        "planning_process": "Production Planning",
        "business_context": """**Business Value**: Anticipate workforce availability issues to enable proactive overtime scheduling, temp staffing, and cross-training."""
    },
    "Yield Prediction (Regression)": {
        "table": f"{CATALOG}.{SCHEMA}.yield_prediction",
        "task": "regression",
        "description": "Predict production yield percentage for capacity planning",
        "default_target": "yield_percentage",
        "exclude_cols": [],
        "planning_process": "Production Planning",
        "business_context": """**Business Value**: Optimize capacity planning and raw material requirements by accurately predicting production output yield."""
    },
    "Transportation Lead Time (Regression)": {
        "table": f"{CATALOG}.{SCHEMA}.transportation_lead_time",
        "task": "regression",
        "description": "Predict shipment transit times for delivery planning",
        "default_target": "actual_transit_days",
        "exclude_cols": [],
        "planning_process": "Distribution Planning",
        "business_context": """**Business Value**: Improve delivery promises and warehouse planning by accurately predicting actual transit times."""
    },
    "OTIF Risk (Multi-class)": {
        "table": f"{CATALOG}.{SCHEMA}.otif_risk",
        "task": "classification",
        "description": "Predict On-Time-In-Full delivery risk (Low, Medium, High Risk)",
        "default_target": "otif_risk",
        "target_names": ["Low Risk", "Medium Risk", "High Risk"],
        "exclude_cols": [],
        "planning_process": "Distribution Planning",
        "business_context": """**Business Value**: Proactively identify orders at risk of OTIF failure to enable intervention and improve customer satisfaction."""
    },
}


@st.cache_resource(ttl=300, show_spinner="Connecting to Databricks...")
def get_connection(http_path: str):
    return sql.connect(
        server_hostname=cfg.host,
        http_path=http_path,
        credentials_provider=lambda: cfg.authenticate,
    )


@st.cache_data(ttl=600, show_spinner="Loading data...")
def load_table(_conn, table_name: str) -> pd.DataFrame:
    with _conn.cursor() as cursor:
        cursor.execute(f"SELECT * FROM {table_name}")
        return cursor.fetchall_arrow().to_pandas()


def authenticate_tabpfn():
    token = os.environ.get("TABPFN_TOKEN")
    if token:
        tabpfn_client.set_access_token(token)
        return True
    return False


def prepare_features(df: pd.DataFrame, target_col: str, exclude_cols: list = None):
    exclude = set(exclude_cols or [])
    exclude.add(target_col)
    feature_cols = [c for c in df.columns if c not in exclude]
    cat_cols = df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        df_encoded = pd.get_dummies(df[feature_cols], columns=cat_cols, drop_first=True)
    else:
        df_encoded = df[feature_cols].copy()
    return df_encoded.values, df[target_col].values, df_encoded.columns.tolist()


def run_classification(X_train, X_test, y_train, y_test, target_names=None):
    clf = TabPFNClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    n_classes = len(np.unique(y_test))
    if n_classes == 2:
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    else:
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr", average="weighted")
        except ValueError:
            roc_auc = None
    return {"predictions": y_pred, "probabilities": y_pred_proba, "accuracy": accuracy, "roc_auc": roc_auc, "y_test": y_test, "model": clf}


def run_regression(X_train, X_test, y_train, y_test):
    reg = TabPFNRegressor()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {"predictions": y_pred, "rmse": rmse, "mae": mae, "r2": r2, "y_test": y_test, "model": reg}


def create_lag_features(series: np.ndarray, n_lags: int = 12):
    X, y = [], []
    for i in range(n_lags, len(series)):
        X.append(series[i-n_lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)


def add_calendar_features(X: np.ndarray, dates, n_lags: int):
    dates_subset = pd.to_datetime(dates[n_lags:])
    months = np.array([d.month for d in dates_subset])
    years = np.array([d.year for d in dates_subset])
    month_sin = np.sin(2 * np.pi * months / 12)
    month_cos = np.cos(2 * np.pi * months / 12)
    return np.column_stack([X, month_sin, month_cos, years - years.min()])


def run_forecasting(X_train, X_test, y_train, y_test):
    reg = TabPFNRegressor()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
    try:
        y_lower = reg.predict(X_test, output_type="quantiles", quantiles=[0.1]).flatten()
        y_upper = reg.predict(X_test, output_type="quantiles", quantiles=[0.9]).flatten()
        coverage = np.mean((y_test >= y_lower) & (y_test <= y_upper))
    except Exception:
        y_lower, y_upper, coverage = None, None, None
    return {"predictions": y_pred, "y_lower": y_lower, "y_upper": y_upper, "mae": mae, "rmse": rmse, "mape": mape, "coverage": coverage, "y_test": y_test, "model": reg}


# Environment variables
http_path = os.environ.get("DATABRICKS_HTTP_PATH", "")
tabpfn_token = os.environ.get("TABPFN_TOKEN", "")

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    # Logos
    # Databricks logo as inline SVG (red/orange geometric icon)
    databricks_icon = '''<svg width="50" height="50" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <path d="M50 10L90 30V50L50 70L10 50V30L50 10Z" fill="#FF3621"/>
        <path d="M50 30L90 50V70L50 90L10 70V50L50 30Z" fill="#FF3621" opacity="0.7"/>
        <path d="M50 10L90 30L50 50L10 30L50 10Z" fill="#FF6B4A"/>
    </svg>'''
    
    st.markdown(f"""
    <div style="padding: 1rem 0; text-align: center;">
        <div style="margin-bottom: 0.5rem;">
            {databricks_icon}
            <div style="font-family: 'Inter', sans-serif; font-size: 1.1rem; font-weight: 600; color: #1a1a2e; letter-spacing: -0.5px;">databricks</div>
        </div>
        <div style="color: #8888aa; font-size: 1.5rem; font-weight: 300; margin: 0.5rem 0;">×</div>
        <img src="{PRIOR_LABS_LOGO}" height="40" alt="Prior Labs" style="margin-top: 0.5rem;">
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Navigation
    page = st.radio("Navigation", ["🏠 Home", "⚡ Predictions"], label_visibility="collapsed")
    st.session_state.current_page = "home" if page == "🏠 Home" else "predictions"
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Status
    st.markdown("**Connection Status**")
    if http_path and not http_path.startswith("YOUR_"):
        st.markdown('<span class="status-success">✓ SQL Warehouse Connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-warning">○ SQL Warehouse</span>', unsafe_allow_html=True)
    
    if tabpfn_token and not tabpfn_token.startswith("YOUR_"):
        st.markdown('<span class="status-success">✓ TabPFN API Connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-warning">○ TabPFN API</span>', unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="font-size: 0.75rem; color: #6a6a8a; text-align: center;">
        <a href="https://priorlabs.ai/" target="_blank" style="color: #667eea;">Prior Labs</a> • 
        <a href="https://databricks.com/" target="_blank" style="color: #ff3621;">Databricks</a>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# HOME PAGE
# ============================================================================
if st.session_state.current_page == "home":
    
    # Hero Section
    st.markdown(f"""
    <div class="hero-container">
        <div class="logo-container">
            <img src="https://www.databricks.com/wp-content/uploads/2022/06/db-nav-logo.svg" width="150" alt="Databricks">
            <span class="logo-divider">×</span>
            <img src="{PRIOR_LABS_LOGO}" height="45" alt="Prior Labs">
        </div>
        <h1 class="hero-title">Predictive Planning Hub</h1>
        <p class="hero-subtitle">
            Your centralized platform for predictive analytics across the entire planning value chain. 
            Powered by TabPFN — the foundation model for tabular data.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-number">10</div><div class="metric-label">Use Cases</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card metric-card-pink"><div class="metric-number">4</div><div class="metric-label">Planning Processes</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card metric-card-cyan"><div class="metric-number">1</div><div class="metric-label">Foundation Model</div></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Why section
    st.markdown('<div class="section-header">Why a Centralized Analytics Hub?</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="feature-card"><div class="feature-title">🔄 Model Proliferation</div><div class="feature-text">Traditional approaches require dozens or hundreds of models to maintain across the planning value chain.</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-card"><div class="feature-title">🔧 High Maintenance</div><div class="feature-text">Continuous retraining, monitoring, and updates consume valuable data science resources.</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="feature-card"><div class="feature-title">📊 Inconsistent Approaches</div><div class="feature-text">Different teams using different tools and methods leads to fragmented insights.</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-card"><div class="feature-title">🚧 Siloed Insights</div><div class="feature-text">Disconnected predictions that don\'t flow across the value chain limit business impact.</div></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box"><div class="info-box-text"><strong>This platform solves these challenges</strong> by providing a single entry point for all predictive model use cases, powered by <strong>TabPFN</strong> — a foundation model that works out-of-the-box on any tabular prediction task without training or tuning.</div></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Value Chain
    st.markdown('<div class="section-header">End-to-End Planning Value Chain</div>', unsafe_allow_html=True)
    
    vc1, vc2, vc3, vc4 = st.columns(4)
    with vc1:
        st.markdown('<div class="process-card process-card-demand"><div class="process-title process-title-demand">📈 Demand Planning</div><hr style="border-color: rgba(102, 126, 234, 0.2);"><p class="process-item">• Demand Forecasting</p><p class="process-item">• Price Elasticity</p><p class="process-item">• Promotion Lift</p></div>', unsafe_allow_html=True)
    with vc2:
        st.markdown('<div class="process-card process-card-supply"><div class="process-title process-title-supply">🚚 Supply Planning</div><hr style="border-color: rgba(240, 147, 251, 0.2);"><p class="process-item">• Supplier Delay Risk</p><p class="process-item">• Supplier Lead Time</p><p class="process-item">• Material Shortage</p></div>', unsafe_allow_html=True)
    with vc3:
        st.markdown('<div class="process-card process-card-production"><div class="process-title process-title-production">🏭 Production Planning</div><hr style="border-color: rgba(79, 172, 254, 0.2);"><p class="process-item">• Labor Shortage</p><p class="process-item">• Yield Prediction</p></div>', unsafe_allow_html=True)
    with vc4:
        st.markdown('<div class="process-card process-card-distribution"><div class="process-title process-title-distribution">📦 Distribution</div><hr style="border-color: rgba(0, 212, 170, 0.2);"><p class="process-item">• Transportation Lead Time</p><p class="process-item">• OTIF Risk</p></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Use Cases Table
    st.markdown('<div class="section-header">Available Use Cases</div>', unsafe_allow_html=True)
    use_cases_data = [{"Use Case": name.split(" (")[0], "Task Type": config["task"].capitalize() if "Multi-class" not in name else "Multi-class", "Planning Process": config.get("planning_process", ""), "Description": config["description"]} for name, config in AVAILABLE_DATASETS.items()]
    st.dataframe(pd.DataFrame(use_cases_data), use_container_width=True, hide_index=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Powered by TabPFN
    st.markdown('<div class="section-header">Powered by TabPFN</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **TabPFN** (Tabular Prior-Data Fitted Network) is a foundation model for tabular data 
        developed by [Prior Labs](https://priorlabs.ai/). It has been pretrained on millions of 
        synthetic datasets, enabling it to make accurate predictions on new data **without any training**.
        
        **Key Benefits:**
        - ✅ **Zero training time** — Predictions in seconds
        - ✅ **No hyperparameter tuning** — Works out of the box  
        - ✅ **Uncertainty quantification** — Built-in prediction intervals
        - ✅ **Strong performance** — Competitive with tuned XGBoost, Random Forest
        - ✅ **Published in Nature** — Rigorous scientific validation
        """)
    with col2:
        st.markdown(f'<div class="powered-by"><div class="powered-by-title">Powered by</div><img src="{PRIOR_LABS_LOGO}" height="50" alt="Prior Labs" style="margin: 0.75rem 0;"><div class="powered-by-desc">Foundation Model for Tabular Data</div></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    cta1, cta2, cta3 = st.columns([1, 2, 1])
    with cta2:
        if st.button("⚡ Start Making Predictions", type="primary", use_container_width=True):
            st.session_state.current_page = "predictions"
            st.rerun()
    
    st.markdown('<div class="footer">Built with <a href="https://databricks.com">Databricks</a> and <a href="https://priorlabs.ai">TabPFN by Prior Labs</a></div>', unsafe_allow_html=True)

# ============================================================================
# PREDICTIONS PAGE
# ============================================================================
else:
    st.markdown('<div class="section-header" style="margin-top: 0;">⚡ Planning Predictions</div>', unsafe_allow_html=True)
    st.markdown("Select a use case from the dropdown below to run predictions with TabPFN.")
    
    selected_dataset_name = st.selectbox("Select Planning Use Case", options=list(AVAILABLE_DATASETS.keys()))
    selected_dataset = AVAILABLE_DATASETS[selected_dataset_name]
    
    # Badge
    process = selected_dataset.get("planning_process", "")
    badge_class = {"Demand Planning": "badge-demand", "Supply Planning": "badge-supply", "Production Planning": "badge-production", "Distribution Planning": "badge-distribution"}.get(process, "badge-demand")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(f'<span class="badge {badge_class}">{process}</span>', unsafe_allow_html=True)
    with col2:
        st.markdown(f"*{selected_dataset['description']}*")
    
    st.markdown(selected_dataset["business_context"])
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    if not http_path or http_path.startswith("YOUR_"):
        st.error("⚠️ SQL Warehouse not configured. Edit `app.yaml` to set `DATABRICKS_HTTP_PATH`.")
        st.stop()

    if not authenticate_tabpfn():
        st.error("⚠️ TabPFN token not configured. Edit `app.yaml` to set `TABPFN_TOKEN`.")
        st.stop()

    try:
        conn = get_connection(http_path)
        st.markdown(f'<div class="section-header">📊 {selected_dataset_name}</div>', unsafe_allow_html=True)

        with st.spinner(f"Loading {selected_dataset['table']}..."):
            df = load_table(conn, selected_dataset["table"])

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", f"{df.shape[0]:,}")
        with col2:
            if selected_dataset["task"] == "forecasting":
                st.metric("Time Series", df[selected_dataset["series_id_col"]].nunique())
            else:
                st.metric("Features", df.shape[1] - 1)
        with col3:
            st.metric("Task Type", selected_dataset["task"].capitalize())
        with col4:
            if selected_dataset["task"] == "classification":
                st.metric("Classes", df[selected_dataset["default_target"]].nunique())
            elif selected_dataset["task"] == "forecasting":
                df[selected_dataset["date_col"]] = pd.to_datetime(df[selected_dataset["date_col"]])
                st.metric("Time Range", f"{df[selected_dataset['date_col']].min().strftime('%Y-%m')} to {df[selected_dataset['date_col']].max().strftime('%Y-%m')}")
            else:
                st.metric("Target Range", f"{df[selected_dataset['default_target']].max() - df[selected_dataset['default_target']].min():.2f}")

        target_column = selected_dataset["default_target"]

        with st.expander("📋 Data Preview", expanded=False):
            st.dataframe(df.head(20), use_container_width=True)

        st.markdown('<div class="section-header">🔧 Model Configuration</div>', unsafe_allow_html=True)
        
        if selected_dataset["task"] == "forecasting":
            col1, col2 = st.columns(2)
            with col1:
                selected_series = st.selectbox("Select Time Series", options=df[selected_dataset["series_id_col"]].unique().tolist())
            with col2:
                n_lags = st.slider("Number of Lag Features", 3, 24, 12)
            col3, col4 = st.columns(2)
            with col3:
                forecast_horizon = st.slider("Forecast Horizon", 1, 12, 6)
            with col4:
                random_state = st.number_input("Random Seed", 0, 9999, 42)
            test_size, max_samples = None, None
        else:
            col1, col2 = st.columns(2)
            with col1:
                test_size = st.slider("Test Set Size (%)", 10, 50, 20)
            with col2:
                random_state = st.number_input("Random Seed", 0, 9999, 42)
            max_samples = st.slider("Max Training Samples", 500, min(5000, len(df)), 2000) if len(df) > 3000 else None

        button_label = "⚡ Run TabPFN Forecast" if selected_dataset["task"] == "forecasting" else "⚡ Run TabPFN Prediction"
        if st.button(button_label, type="primary", use_container_width=True):
            with st.spinner("Running TabPFN model..."):
                
                if selected_dataset["task"] == "forecasting":
                    df_series = df[df[selected_dataset["series_id_col"]] == selected_series].sort_values(selected_dataset["date_col"]).reset_index(drop=True)
                    values, dates = df_series[target_column].values, df_series[selected_dataset["date_col"]].values
                    if len(values) < n_lags + forecast_horizon + 5:
                        st.error("Not enough data points.")
                        st.stop()
                    X, y = create_lag_features(values, n_lags)
                    X_enhanced = add_calendar_features(X, dates, n_lags)
                    X_train, X_test = X_enhanced[:-forecast_horizon], X_enhanced[-forecast_horizon:]
                    y_train, y_test = y[:-forecast_horizon], y[-forecast_horizon:]
                    test_dates = pd.to_datetime(dates[n_lags:])[-forecast_horizon:]
                    train_dates = pd.to_datetime(dates[n_lags:])[:-forecast_horizon]
                    results = run_forecasting(X_train, X_test, y_train, y_test)
                    
                    st.markdown('<div class="section-header">📊 Forecast Results</div>', unsafe_allow_html=True)
                    col1, col2, col3, col4 = st.columns(4)
                    with col1: st.metric("MAE", f"{results['mae']:,.0f}")
                    with col2: st.metric("RMSE", f"{results['rmse']:,.0f}")
                    with col3: st.metric("MAPE", f"{results['mape']:.1f}%")
                    with col4: st.metric("80% Coverage" if results['coverage'] else "Horizon", f"{results['coverage']:.0%}" if results['coverage'] else f"{forecast_horizon}")
                    
                    fig, ax = plt.subplots(figsize=(12, 5))
                    fig.patch.set_facecolor('#f8f7ff')
                    ax.set_facecolor('#f8f7ff')
                    ax.plot(train_dates, y_train, color='#667eea', linewidth=1.5, label='Training', alpha=0.7)
                    ax.plot(test_dates, y_test, color='#00b894', linewidth=2, marker='o', markersize=6, label='Actual')
                    ax.plot(test_dates, results['predictions'], color='#e056a0', linewidth=2, marker='s', markersize=6, linestyle='--', label='Forecast')
                    if results['y_lower'] is not None:
                        ax.fill_between(test_dates, results['y_lower'], results['y_upper'], alpha=0.15, color='#e056a0', label='80% Interval')
                    ax.set_xlabel('Date', color='#2d2d44')
                    ax.set_ylabel('Demand', color='#2d2d44')
                    ax.set_title(f'Forecast - {selected_series}', color='#1a1a2e', fontweight='bold')
                    ax.legend(facecolor='#ffffff', edgecolor='#e0e0e0', labelcolor='#2d2d44')
                    ax.tick_params(colors='#4a4a6a')
                    ax.grid(True, alpha=0.3, color='#667eea')
                    for spine in ax.spines.values():
                        spine.set_color('#e0e0e0')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    results_df = pd.DataFrame({"Date": test_dates.strftime('%Y-%m'), "Actual": y_test.round(0).astype(int), "Forecast": results['predictions'].round(0).astype(int), "Error": (y_test - results['predictions']).round(0).astype(int)})
                    st.dataframe(results_df, use_container_width=True)
                    st.success("✅ Forecast complete!")
                
                else:
                    X, y, _ = prepare_features(df, target_column, selected_dataset.get("exclude_cols", []))
                    if max_samples and len(X) > max_samples:
                        np.random.seed(random_state)
                        idx = np.random.choice(len(X), max_samples, replace=False)
                        X, y = X[idx], y[idx]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=random_state, stratify=y if selected_dataset["task"] == "classification" else None)

                    if selected_dataset["task"] == "classification":
                        results = run_classification(X_train, X_test, y_train, y_test, selected_dataset.get("target_names"))
                        st.markdown('<div class="section-header">📊 Classification Results</div>', unsafe_allow_html=True)
                        col1, col2, col3 = st.columns(3)
                        with col1: st.metric("Accuracy", f"{results['accuracy']:.4f}")
                        with col2:
                            if results['roc_auc']: st.metric("ROC AUC", f"{results['roc_auc']:.4f}")
                        with col3: st.metric("Test Samples", len(y_test))
                        
                        results_df = pd.DataFrame({"Actual": results["y_test"], "Predicted": results["predictions"], "Correct": results["y_test"] == results["predictions"]})
                        target_names = selected_dataset.get("target_names", [f"Class_{i}" for i in range(results["probabilities"].shape[1])])
                        for i, name in enumerate(target_names[:results["probabilities"].shape[1]]):
                            results_df[f"Prob_{name}"] = results["probabilities"][:, i].round(4)
                        st.dataframe(results_df, use_container_width=True)
                    else:
                        results = run_regression(X_train, X_test, y_train, y_test)
                        st.markdown('<div class="section-header">📊 Regression Results</div>', unsafe_allow_html=True)
                        col1, col2, col3, col4 = st.columns(4)
                        fmt = ".4f" if "elasticity" in selected_dataset_name.lower() else ".2f"
                        with col1: st.metric("RMSE", f"{results['rmse']:{fmt}}")
                        with col2: st.metric("MAE", f"{results['mae']:{fmt}}")
                        with col3: st.metric("R²", f"{results['r2']:.4f}")
                        with col4: st.metric("Test Samples", len(y_test))
                        
                        st.scatter_chart(pd.DataFrame({"Actual": results["y_test"], "Predicted": results["predictions"]}), x="Actual", y="Predicted")
                    
                    st.success("✅ Prediction complete!")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)
