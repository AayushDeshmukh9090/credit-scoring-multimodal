# Multi-Modal Credit Scoring (ANN + Feature Fusion)
 
A dual-branch neural network that predicts loan default probability by fusing numerical financial data and categorical behavioural data through a Late Fusion architecture with Entity Embeddings.
 
## Live Demo
 
| Service | Link |
|---------|------|
| Frontend (Streamlit) | [credit-scoring-multimodal-4.streamlit.app](https://credit-scoring-multimodal-4.streamlit.app) |
| API Documentation (FastAPI) | [credit-scoring-multimodal-4.onrender.com/docs](https://credit-scoring-multimodal-4.onrender.com/docs) |
 
 
---
 
## The Problem
 
Traditional credit scoring models use only numerical features — income, loan amount, age. This ignores *who* the applicant is: their occupation, organisation type, education level, and employment category. A banker considers both. This model does too.
 
---
 
## Architecture
 
```
Input Layer
│
├── Branch 1: Numerical Features
│   ├── AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY
│   ├── EXT_SOURCE_1/2/3 (external bureau scores)
│   ├── AGE_YEARS, YEARS_EMPLOYED, YEARS_ID_PUBLISH
│   ├── LOAN_TO_GOODS_RATIO (engineered feature)
│   ├── IS_UNEMPLOYED (engineered flag)
│   └── EXT_SOURCE_MISSING flags (missingness indicators)
│   → Dense(32) → BatchNorm → Dropout(0.2)
│
└── Branch 2: Categorical Features (Entity Embeddings)
    ├── OCCUPATION_TYPE    (19 categories → dim 10)
    ├── ORGANIZATION_TYPE  (58 categories → dim 29)
    ├── NAME_EDUCATION_TYPE (5 categories → dim 3)
    ├── NAME_FAMILY_STATUS  (6 categories → dim 3)
    ├── NAME_INCOME_TYPE    (8 categories → dim 4)
    └── CODE_GENDER         (2 categories → dim 2)
    → Embedding per feature → Flatten
 
Fusion Layer: Concatenate [Branch 1 + Branch 2]
→ Dense(64, L2) → Dropout(0.3)
→ Dense(32, L2)
→ Dense(1, sigmoid) → Default Probability
```
 
**Why Entity Embeddings instead of One-Hot Encoding?**
 
One-hot encoding of 58 organisation types adds 58 sparse binary columns. Entity embeddings compress this into 29 dense dimensions that capture semantic similarity — "School" and "University" end up closer in embedding space than "School" and "Military." This generalises better and reduces dimensionality.
 
---
 
## Dataset
 
[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) — Kaggle
 
- 307,511 loan applications
- 122 raw features → 20 selected features
- Class imbalance: 91.9% repaid / 8.1% default
- Handled via balanced class weights during training
 
---
 
## Results
 
| Metric | Value |
|--------|-------|
| Validation AUC | 0.7535 |
| Recall (Default class) | 0.67 |
| Precision (Default class) | 0.16 |
| Demographic Parity Difference | 0.23 |
 
**On AUC:** 0.75 is consistent with published benchmarks for single-table neural networks on this dataset. Gradient boosted models (LightGBM) achieve 0.78–0.80. The architectural goal here is demonstrating entity embeddings and feature fusion, not beating gradient boosting on a leaderboard.
 
**On Fairness:** A Demographic Parity Difference of 0.23 was observed across gender groups. Investigation showed this largely mirrors the actual default rate difference in the dataset rather than model-introduced bias. In a production system this would trigger a deeper fairness audit.
 
---
 
## Project Structure
 
```
credit-scoring-multimodal/
│
├── data/
│   ├── raw/                 # Original CSVs (not tracked in Git)
│   ├── processed/           # Cleaned parquets (not tracked in Git)
│   └── mappings/            # JSON files: integer → category label
│
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb    # EDA, feature selection, cleaning
│   ├── 02_preprocessing.ipynb       # Fit scaler + encoders, save artifacts
│   └── 03_model_training.ipynb      # Training loop + fairness evaluation
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Column definitions, loaders, train/val split
│   ├── preprocessor.py      # CreditPreprocessor class (fit/transform/save/load)
│   └── model_builder.py     # Dual-branch Keras model + prepare_model_inputs()
│
├── models/
│   ├── best_model.keras     # Trained model weights
│   ├── scaler.pkl           # Fitted StandardScaler
│   └── encoder_objects.pkl  # Fitted LabelEncoders + cardinalities + medians
│
├── api/
│   ├── app.py               # FastAPI application (/health, /predict)
│   └── schemas.py           # Pydantic request/response models
│
├── frontend/
│   └── app.py               # Streamlit frontend
│
├── .streamlit/
│   └── config.toml          # Theme configuration
│
├── render.yaml              # Render deployment configuration
├── requirements.txt         # Python dependencies
└── README.md
```
 
---
 
## Pipeline
 
```
Raw CSV (data/raw/)
    ↓ 01_eda_and_cleaning.ipynb
Cleaned CSV (data/processed/)
    ↓ 02_preprocessing.ipynb
Processed Parquet + scaler.pkl + encoder_objects.pkl
    ↓ 03_model_training.ipynb (Google Colab, T4 GPU)
best_model.keras
    ↓
FastAPI (Render) ← Streamlit (Streamlit Cloud)
```
 


 
---
