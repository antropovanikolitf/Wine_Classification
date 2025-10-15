# Wine Classification Project - Complete Pipeline

**Goal:** Binary classification of wine **red vs white** using the UCI Wine Quality chemistry data.

A comprehensive machine learning project covering the full ML lifecycle from problem framing to production-ready model evaluation.

## Contents

### Notebooks
1. **`01_problem_framing.ipynb`** - Problem definition, stakeholder analysis, impact framing, success metrics
2. **`02_data_understanding.ipynb`** - Exploratory data analysis, class balance, feature distributions, correlation analysis
3. **`03_model_development.ipynb`** - Baseline models (Dummy, Logistic Regression, Random Forest), initial evaluation
4. **`04_model_optimization.ipynb`** - Hyperparameter tuning, advanced algorithms (SVM, Gradient Boosting), cross-validation
5. **`05_impact_reporting.ipynb`** - Final test evaluation, error analysis, business impact, stakeholder dashboard

### Supporting Files
- **`src/utils.py`** - Reusable utility functions for data loading, model evaluation, and visualization
- **`scripts/`** - Helper scripts for notebook maintenance and QA checks
- **`data/`** - Wine quality datasets (raw and processed)
- **`results/`** - Saved models and evaluation outputs

## Data
UCI Wine Quality (Vinho Verde, Portugal, 2004–2007). Laboratory measurements (pH, alcohol, sulphates, etc.).

**Files:**
- `data/winequality-red.csv` - Red wine samples (1,599 samples)
- `data/winequality-white.csv` - White wine samples (4,898 samples)
- Delimiter: semicolon (`;`)

## How to run
```bash
# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook

# Run notebooks in order: 01 → 02 → 03 → 04 → 05
```

## Key Results
- **Final Model:** Optimized Random Forest Classifier
- **Test Accuracy:** 99.5%+
- **ROC-AUC:** 0.998
- **Most Important Features:** Total sulfur dioxide, volatile acidity, chlorides

## Project Structure
```
Classification project/
├── notebooks/          # Jupyter notebooks (01-05)
├── data/              # Raw and processed datasets
├── results/           # Trained models and outputs
├── reports/           # Model cards and assessments
├── src/               # Utility functions
├── scripts/           # Helper scripts
├── requirements.txt   # Python dependencies
└── README.md         # This file
```