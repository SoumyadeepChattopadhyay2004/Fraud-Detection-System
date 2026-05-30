# 🔐 Real-Time Fraud Detection System with Explainable AI

> **IEEE-CIS Fraud Detection**

## Overview
End-to-end fraud detection system using LightGBM, XGBoost, Isolation Forest + SHAP explainability.

## Project Structure
```
FraudDetection/
├── analysis.ipynb          # Main notebook (all 8 tasks)
├── data/
│   ├── train_transaction.csv
│   └── train_identity.csv
├── dashboard/
│   ├── app.py              # Streamlit dashboard
│   ├── model.pkl           # Trained LightGBM model
│   ├── scaler.pkl          # RobustScaler
│   ├── shap_explainer.pkl  # SHAP TreeExplainer
│   └── transactions.csv    # Test set with risk scores
├── charts/                 # All visualisation outputs
├── model_comparison.png    # Model comparison chart
├── shap_summary.png        # SHAP global summary
├── requirements.txt
└── README.md
```

## Setup
```bash
pip install -r requirements.txt
```

## Dataset
Download from [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data):
- `train_transaction.csv`
- `train_identity.csv`

Place both files in the `data/` folder.

## Run Notebook
Open `analysis.ipynb` in Google Colab (GPU runtime recommended) or Jupyter.

## Run Dashboard
```bash
cd dashboard
streamlit run app.py
```

## Live Dashboard
🌐 **Streamlit Community Cloud URL:** _[Deploy and paste URL here]_

To deploy:
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set `dashboard/app.py` as main file

## Key Results
| Model | ROC-AUC | PR-AUC | F1-Score |
|-------|---------|--------|----------|
| LightGBM (Tuned) | ~0.97 | ~0.82 | ~0.88 |
| XGBoost | ~0.96 | ~0.79 | ~0.86 |
| Isolation Forest | ~0.84 | ~0.31 | ~0.41 |

## Tools
Python · LightGBM · XGBoost · SHAP · Optuna · Streamlit · Plotly · SMOTE
