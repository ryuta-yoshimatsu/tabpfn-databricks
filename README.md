# TabPFN on Databricks

[![Databricks](https://img.shields.io/badge/Databricks-Solution_Accelerator-FF3621?style=for-the-badge&logo=databricks)](https://databricks.com)
[![Unity Catalog](https://img.shields.io/badge/Unity_Catalog-Enabled-00A1C9?style=for-the-badge)](https://docs.databricks.com/en/data-governance/unity-catalog/index.html)
[![Serverless](https://img.shields.io/badge/Serverless-Compute-00C851?style=for-the-badge)](https://docs.databricks.com/en/compute/serverless.html)

A comprehensive solution accelerator demonstrating how to use **TabPFN** (Tabular Prior-Data Fitted Network) on Databricks for classification, regression, outlier detection, and time series forecasting tasks.

## Overview

[TabPFN](https://priorlabs.ai/) is a foundation model for tabular data developed by Prior Labs. It provides state-of-the-art performance without hyperparameter tuning, making it ideal for rapid prototyping and production ML workflows.

This project provides:
- **Interactive Notebooks** demonstrating TabPFN capabilities for various ML tasks
- **Streamlit App** for interactive predictions on Unity Catalog data
- **Databricks Asset Bundle** for easy deployment and CI/CD integration

## Features

| Feature | Description |
|---------|-------------|
| **Classification** | Binary and multi-class classification with probability estimates |
| **Regression** | Continuous value prediction with uncertainty quantification |
| **Outlier Detection** | Anomaly scoring using semi-supervised learning |
| **Time Series Forecasting** | Lag-based forecasting with TabPFN Regressor |
| **Unity Catalog Integration** | Read/write data from Delta tables |
| **Databricks App** | Interactive Streamlit UI for predictions |

## Project Structure

```
tabpfn-databricks/
├── apps/                           # Databricks App (Streamlit)
│   ├── app.py                      # Main Streamlit application
│   ├── app.yaml                    # App deployment configuration
│   └── requirements.txt            # App dependencies
├── notebooks/                      # Jupyter notebooks
│   ├── 00_data_preparation.ipynb   # Dataset setup and preparation
│   ├── 01_classification.ipynb     # Binary & multi-class classification
│   ├── 02_regression.ipynb         # Regression with uncertainty
│   ├── 03_outlier_detection.ipynb  # Anomaly detection
│   └── 04_time_series_forecasting.ipynb  # Time series forecasting
├── .github/workflows/              # CI/CD pipelines
│   └── databricks-ci.yml           # Databricks Asset Bundle CI
├── scripts/                        # Utility scripts
│   └── cleanup.sh                  # Resource cleanup
├── dashboards/                     # Databricks dashboards (placeholder)
├── databricks.yml                  # Databricks Asset Bundle configuration
├── requirements.txt                # Project dependencies
├── CONTRIBUTING.md                 # Contribution guidelines
├── LICENSE.md                      # Databricks license
└── SECURITY.md                     # Security policy
```

## Prerequisites

1. **Databricks Workspace** with Unity Catalog enabled
2. **TabPFN API Token** from [Prior Labs](https://docs.priorlabs.ai/)
3. **Databricks CLI** (optional, for local development)

## Getting Started

### Option 1: Clone via Databricks UI (Recommended)

1. **Clone the repository** into your Databricks Workspace:
   - Navigate to **Repos** in the Databricks UI
   - Click **Add Repo** and enter the repository URL

2. **Open the Asset Bundle Editor**:
   - Click on the workspace folder containing the cloned repo
   - Open the Asset Bundle Editor from the UI

3. **Deploy the bundle**:
   - Click **Deploy** to deploy all resources

4. **Run the notebooks**:
   - Navigate to the Deployments tab (🚀 icon)
   - Click **Run** to execute the workflow

### Option 2: Deploy via Databricks CLI

1. **Install the Databricks CLI**:
   ```bash
   pip install databricks-cli
   ```

2. **Configure authentication**:
   ```bash
   databricks configure --token
   ```

3. **Clone and deploy**:
   ```bash
   git clone <repository-url>
   cd tabpfn-databricks
   databricks bundle deploy
   ```

### Setting Up TabPFN Authentication

Store your TabPFN token as a Databricks Secret:

```bash
# Create a secret scope
databricks secrets create-scope tabpfn-client

# Store your token
databricks secrets put-secret tabpfn-client token
```

To retrieve your token on another machine:
```python
import tabpfn_client
token = tabpfn_client.get_access_token()
print(token)
```

## Notebooks Overview

### 00_data_preparation.ipynb
Prepares all datasets used in the demo notebooks and stores them as Delta tables in Unity Catalog.

**Datasets:**
| Table | Description | Task |
|-------|-------------|------|
| `breast_cancer` | Binary classification (569 samples, 30 features) | Classification |
| `iris` | Multi-class classification (150 samples, 4 features) | Classification |
| `california_housing` | Regression (20,640 samples, 8 features) | Regression |
| `tourism_monthly` | Synthetic time series (50 series, 120 months each) | Forecasting |

### 01_classification.ipynb
Demonstrates binary and multi-class classification:
- TabPFN classifier setup and training
- Probability estimates and confidence scores
- Model comparison with Random Forest, Gradient Boosting, and Logistic Regression
- Cross-validation evaluation

### 02_regression.ipynb
Shows regression capabilities with uncertainty quantification:
- TabPFN regressor for continuous predictions
- 90% prediction intervals using quantile regression
- Model comparison with traditional regressors
- Visualization of predictions vs actuals

### 03_outlier_detection.ipynb
Demonstrates anomaly detection using TabPFN:
- Semi-supervised anomaly detection approach
- Comparison with Isolation Forest and Local Outlier Factor
- ROC AUC evaluation for anomaly scoring

### 04_time_series_forecasting.ipynb
Shows time series forecasting with lag features:
- Feature engineering with lag variables
- Multi-step ahead forecasting
- Batch forecasting for multiple series
- Evaluation with MAE, RMSE, and MAPE metrics

## Databricks App

The project includes a Streamlit-based Databricks App for interactive predictions.

### Features
- Select datasets from Unity Catalog Delta tables
- Choose target variables for prediction
- Run TabPFN classification or regression
- View predictions and model performance metrics

### Configuration

The app is configured via `apps/app.yaml`:

| Environment Variable | Description |
|---------------------|-------------|
| `DATABRICKS_HTTP_PATH` | SQL Warehouse HTTP path (e.g., `/sql/1.0/warehouses/abc123`) |
| `TABPFN_TOKEN` | Your TabPFN API token |

### Running Locally

```bash
cd apps
pip install -r requirements.txt
streamlit run app.py
```

## Compute Requirements

All notebooks are optimized for **Serverless Compute** with **Base Environment V4**:

1. Click on the compute selector in the notebook toolbar
2. Select **Serverless**
3. Under Environment, choose **Base Environment V4**

Serverless compute provides fast startup times and automatic scaling, ideal for interactive notebook workflows.

## CI/CD Pipeline

The project includes a GitHub Actions workflow (`.github/workflows/databricks-ci.yml`) that:

1. Validates the Databricks Asset Bundle
2. Deploys resources to the workspace
3. Runs the demo workflow
4. Cleans up PR deployments

### Required Secrets

| Secret | Description |
|--------|-------------|
| `DEPLOY_NOTEBOOK_TOKEN` | Databricks personal access token |

## Dependencies

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `tabpfn-client` | >=0.1.0 | TabPFN API client |
| `databricks-sdk` | >=0.20.0 | Databricks SDK |
| `databricks-sql-connector` | >=3.0.0 | SQL connectivity |
| `scikit-learn` | >=1.3.0 | ML utilities |
| `pandas` | >=2.0.0 | Data manipulation |
| `numpy` | >=1.24.0 | Numerical computing |
| `streamlit` | >=1.30.0 | Web app framework |

## Resources

- [TabPFN Documentation](https://docs.priorlabs.ai/)
- [TabPFN Client GitHub](https://github.com/PriorLabs/tabpfn-client)
- [Prior Labs Website](https://priorlabs.ai/)
- [Databricks Asset Bundles](https://docs.databricks.com/en/dev-tools/bundles/index.html)
- [Unity Catalog](https://docs.databricks.com/en/data-governance/unity-catalog/index.html)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines and CLA information.

## Third-Party Package Licenses

| Package | License | Copyright |
|---------|---------|-----------|
| tabpfn-client | Apache 2.0 | Prior Labs GmbH |
| scikit-learn | BSD-3-Clause | scikit-learn developers |
| pandas | BSD-3-Clause | pandas development team |
| numpy | BSD-3-Clause | NumPy developers |
| streamlit | Apache 2.0 | Streamlit Inc. |
| databricks-sdk | Apache 2.0 | Databricks, Inc. |

## License

This project is provided subject to the [Databricks License](LICENSE.md).

---

© 2025 Databricks, Inc. All rights reserved.
