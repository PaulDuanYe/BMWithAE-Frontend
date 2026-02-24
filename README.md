# BMWithAE - Bias Mitigation with Accuracy Enhancement

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Flask Version](https://img.shields.io/badge/flask-3.0.0-green)
![License](https://img.shields.io/badge/license-MIT-orange)

An interactive visual analytics system for bias mitigation and accuracy enhancement in machine learning models.

[Features](#features) • [Quick Start](#quick-start) • [Documentation](#documentation) • [API](#api-reference) • [Deployment](#production-deployment)

</div>

---

## Overview

BMWithAE is a comprehensive visual analytics system designed to help data scientists and researchers identify, analyze, and mitigate bias in machine learning models while maintaining or improving model accuracy. The system provides an intuitive web interface for exploring data distributions, analyzing fairness metrics, and executing iterative debiasing processes.

## Features

### 🎯 Core Capabilities

#### **1. Interactive Data Explorer Dashboard**

The Data Explorer provides comprehensive data analysis and visualization capabilities:

- **Feature Distribution Visualization**: Interactive histograms and bar charts showing the distribution of each feature in your dataset
- **Statistical Summary**: Automatic calculation of min, max, mean, std for continuous features, and value counts for categorical features
- **Missing Data Analysis**: Visual indicators for missing values with percentage calculations
- **Target Distribution**: Clear visualization of the target variable distribution to identify class imbalance
- **Protected Attribute Selection**: Multi-select interface for choosing sensitive attributes (e.g., gender, race, age)
- **Real-time Bias Metrics**: Instant calculation and display of fairness metrics for selected protected attributes

#### **2. Advanced Subgroup Analysis**

Inspired by FairVis, our subgroup analysis tool enables granular fairness inspection:

- **Custom Subgroup Builder**: Interactive UI to define subgroups by selecting specific feature values (e.g., "25-year-old males with college education")
- **Multiple Condition Support**: Combine multiple features to create complex subgroup definitions
- **Performance Metrics Comparison**: For each subgroup, view:
  - Accuracy, Precision, Recall, F1 Score, False Positive Rate
  - Comparison against overall dataset averages
  - Visual bar charts with hover-to-reveal labels
  - Percentage differences highlighting disparities
- **Subgroup Size Statistics**: Track the number of samples in each subgroup and their percentage of the total dataset
- **Multi-Subgroup Management**: Create, save, and compare multiple subgroups simultaneously

#### **3. Multidimensional Bias Metrics**

Comprehensive fairness evaluation across multiple dimensions:

- **Statistical Parity (SP)**: Measures whether different groups receive positive outcomes at equal rates
- **Equal Opportunity (EOpp)**: Ensures equal true positive rates across groups
- **Equalized Odds (EO)**: Equalizes both true positive and false positive rates
- **Disparate Impact (DI)**: Quantifies the ratio of positive outcome rates between groups
- **Bias Concentration (Epsilon)**: Novel metric measuring bias distribution across features
- **Overall Fairness Score**: Aggregated metric for quick assessment (0-1 scale, higher is fairer)

#### **4. Iterative Debiasing Process**

Systematic bias reduction through coordinated BM and AE modules:

**Bias Mitigation (BM) Module:**
- Identifies the protected attribute and value with maximum bias concentration
- Applies targeted transformations to reduce bias for specific subgroups
- Preserves data distribution characteristics while improving fairness
- Supports multiple transformation strategies (SMOTE, data augmentation)

**Accuracy Enhancement (AE) Module:**
- Recovers accuracy lost during bias mitigation
- Uses ensemble methods and instance weighting
- Balances fairness improvements with model performance
- Configurable training rates and classifier selection

**Iteration Control:**
- **Step-by-Step Mode**: Execute one iteration at a time for detailed observation and analysis
- **Run All Mode**: Automatic execution of all iterations until convergence or threshold reached
- **Real-time Monitoring**: Live updates of metrics during the debiasing process
- **Early Stopping**: Automatic termination when fairness goals are met or accuracy drops below threshold

#### **5. Rich Visualizations & Analytics**

Professional-grade visualizations for comprehensive model understanding:

- **Dual-Chart Real-time Dashboard**:
  - Left Chart: Max Epsilon evolution tracking bias concentration over iterations
  - Right Chart: Accuracy evolution showing model performance maintenance
  - Synchronized tooltips and zoom controls
  
- **Iteration History Table**: Detailed records of each iteration including:
  - Selected attribute and value for transformation
  - Current max epsilon and accuracy values
  - Number of samples changed
  - Timestamp and iteration number

- **Comparative Analysis Views**:
  - Before/After comparison of bias metrics
  - Subgroup performance heat maps
  - Feature importance for fairness

- **Export Capabilities**: Download charts, metrics, and complete experiment logs

### Technical Stack

- **Backend**: Python, Flask, Pandas, Scikit-learn
- **Frontend**: Native HTML/CSS/JavaScript (no build required)
- **Machine Learning**: Multiple classifiers (XGBoost, LightGBM, CatBoost)
- **Data Formats**: Excel (.xlsx), CSV
- **Architecture**: RESTful API design

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository

```bash
git clone https://github.com/PaulDuanYe/BMWithAE.git
cd BMWithAE
```

2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Start the backend server

```bash
cd backend
python app.py
```

The backend will start at `http://localhost:5000`

5. Open the frontend

Open `frontend/index.html` in your browser, or serve it via a local server:

```bash
cd frontend
python -m http.server 8000
# Visit http://localhost:8000
```

## Documentation

### Project Structure

```
BMWithAE/
├── backend/              # Backend Python code
│   ├── app.py           # Flask main application
│   ├── core_config.py   # Core configuration
│   ├── module_BM.py     # Bias mitigation module
│   ├── module_AE.py     # Accuracy enhancement module
│   ├── eval.py          # Evaluation module
│   └── ...
├── frontend/            # Frontend code
│   ├── index.html       # Main interface
│   ├── styles.css       # Stylesheet
│   └── api.js           # API client
├── data/                # Data files
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

### Usage Guide

#### **Getting Started**

**Step 1: Load Your Dataset**

- **Option A - Demo Dataset**: Click "Load Demo" to load the credit card default dataset (30,000 samples)
- **Option B - Custom Dataset**: Click "Upload Data" to upload your CSV or Excel file
  - Select the target column (dependent variable)
  - Choose protected attributes (e.g., gender, race, age)

**Step 2: Explore Your Data (Data Explorer)**

The Data Explorer dashboard provides comprehensive data analysis:

1. **View Feature Distributions**: Scroll through the feature cards to see distributions
   - Categorical features: Bar charts with value counts
   - Continuous features: Histograms with statistics (min, max, mean, std)
   - Click any feature card to see detailed tooltips with full distribution

2. **Select Protected Attributes**: 
   - Click the "Protected Attributes" button
   - Select one or more sensitive attributes for bias analysis
   - View real-time bias metrics after selection

3. **Analyze Subgroups** ⭐ NEW:
   - Click "Generate Subgroups" button next to "Feature Distributions"
   - In the Subgroup Builder modal:
     - Define conditions by clicking feature values (e.g., "Female", "Age 25-30", "College Educated")
     - Click "Add Subgroup" to save the defined subgroup
     - View performance metrics (Accuracy, Precision, Recall, F1, FPR) for each subgroup
     - Compare multiple subgroups side-by-side
     - Click any subgroup card to see detailed comparison charts with overall averages

**Step 3: Configure Debiasing Parameters**

Click "Configuration" to customize the debiasing process:

- **Classifier**: Choose from XGBoost, LightGBM, CatBoost, Random Forest, etc.
- **Max Iterations**: Set the maximum number of debiasing iterations (default: 20)
- **Epsilon Threshold**: Target bias concentration level (default: 0.9)
- **Accuracy Threshold**: Minimum acceptable accuracy (default: 0.7)
- **Training Rate**: Train/test split ratio (default: 0.7)

**Step 4: Execute Debiasing Process**

Two execution modes available:

- **"Run All Steps"**: Automatically execute all iterations
  - Process runs in background
  - Real-time chart updates
  - Auto-stops when fairness goals met or max iterations reached
  
- **"Step by Step"**: Execute one iteration at a time
  - Click "Next Step" for each iteration
  - Review metrics after each step
  - Ideal for understanding the debiasing process

**Step 5: Analyze Results**

- **Dual-Chart Dashboard**:
  - Left Chart: Max Epsilon evolution (tracks bias concentration reduction)
  - Right Chart: Accuracy evolution (monitors model performance)
  - Hover over points to see detailed values

- **Iteration History**:
  - Table showing all iterations
  - Selected attribute/value for each transformation
  - Metrics and changes per iteration

- **Download Results**:
  - Export experiment logs (JSON format)
  - Save charts as images
  - Export final debiased dataset

## API Reference

### Data Management

#### `POST /api/data/upload`
Upload a custom dataset (CSV or Excel)
- **Body**: FormData with file, target_column, protected_columns[]
- **Returns**: Dataset ID and basic information

#### `POST /api/data/demo`
Load a demo dataset (e.g., credit card default dataset)
- **Body**: `{ "dataset_name": "credit" }`
- **Returns**: Dataset ID and preview

#### `GET /api/data/<dataset_id>/info`
Get comprehensive dataset information
- **Returns**: Feature types, statistics, value distributions, target info

#### `POST /api/data/<dataset_id>/bias-metrics`
Calculate bias metrics for protected attributes
- **Body**: `{ "protected_attributes": ["SEX", "AGE"] }`
- **Returns**: Statistical Parity, Equal Opportunity, Disparate Impact, etc.

#### `POST /api/data/<dataset_id>/subgroup-metrics` ⭐ NEW
Calculate performance metrics for a specific subgroup
- **Body**: `{ "conditions": { "SEX": 2, "AGE": 25, "EDUCATION": 3 } }`
- **Returns**: 
  - Subgroup size and percentage
  - Subgroup metrics (accuracy, precision, recall, F1, FPR)
  - Overall dataset metrics for comparison

### Debiasing Process

#### `POST /api/debias/init`
Initialize a debiasing job
- **Body**: `{ "dataset_id": "...", "protected_attributes": ["SEX"] }`
- **Returns**: Job ID and initial metrics

#### `POST /api/debias/<job_id>/step`
Execute one iteration of BM + AE
- **Returns**: Updated metrics, selected attribute/value, changes made

#### `POST /api/debias/<job_id>/run-full`
Execute all remaining iterations automatically
- **Returns**: Job ID and running state

#### `GET /api/debias/<job_id>/status`
Get current job status and history
- **Returns**: Current iteration, metrics history, progress, termination info

### Configuration

#### `GET /api/config`
Get current system configuration
- **Returns**: All configuration parameters from core_config.py

#### `POST /api/config`
Update configuration parameters
- **Body**: `{ "PARAMS_MAIN_MAX_ITERATION": 30, "PARAMS_MAIN_CLASSIFIER": "XGB" }`
- **Returns**: Updated configuration

For detailed API documentation and examples, see [PROJECT_SETUP.md](PROJECT_SETUP.md)

## Configuration

Main configuration parameters in `backend/core_config.py`:

```python
PARAMS_MAIN_MAX_ITERATION = 20        # Maximum iterations
PARAMS_MAIN_THRESHOLD_EPSILON = 0.9   # Epsilon threshold
PARAMS_MAIN_CLASSIFIER = 'XGB'        # Classifier selection
PARAMS_MAIN_TRAINING_RATE = 0.7       # Training set ratio
```

## Production Deployment

### For Long-term Server Deployment

If you plan to deploy BMWithAE on a production server for long-term operation, **do not use the development server** (`python app.py`). Instead, use a production-grade WSGI server.

**Quick Start (Production):**

```bash
# Linux/Mac
cd backend
./start_production.sh

# Windows
cd backend
start_production.bat
```

**Features of Production Deployment:**
- ✅ Multi-worker process support for better performance
- ✅ Automatic restart on failure
- ✅ Process management (systemd/supervisor)
- ✅ Nginx reverse proxy support
- ✅ SSL/TLS configuration
- ✅ Log rotation and monitoring
- ✅ Environment-based configuration

For detailed production deployment instructions, see **[DEPLOYMENT.md](DEPLOYMENT.md)**.

### Deployment Options

1. **Systemd** (Linux, recommended)
2. **Supervisor** (Cross-platform)
3. **PM2** (Node.js-based)
4. **Docker** (containerized)

### Environment Variables

Production configuration is controlled via environment variables:

```bash
export FLASK_ENV=production    # Enable production mode
export FLASK_HOST=0.0.0.0      # Listen on all interfaces
export FLASK_PORT=5001         # Port to bind
```

## Troubleshooting

### Port Conflict

Modify the port via environment variable or in `backend/backend_config.py`:

```python
PORT = 5001  # Change to another port
```

Or use environment variable:
```bash
export FLASK_PORT=8080
```

### Dependency Installation Issues

```bash
# Upgrade pip
pip install --upgrade pip

# Install problematic packages separately
pip install xgboost --no-cache-dir
```

### CORS Errors

Ensure the backend is running and Flask-CORS is properly configured.

### Production Server Not Starting

Check logs in `backend/logs/` or use:

```bash
# Systemd
sudo journalctl -u bmwithae -n 50

# Supervisor
sudo supervisorctl tail bmwithae
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or suggestions:

- Submit an Issue: [GitHub Issues](https://github.com/KuumaSan/BMWithAE/issues)
- Email: your.email@example.com

## Acknowledgments

- Inspired by FairSight and FairVis visual analytics systems
- Based on fairness in machine learning research
- Thanks to the open-source community

---

<div align="center">

Made with care for Fair ML

</div>
