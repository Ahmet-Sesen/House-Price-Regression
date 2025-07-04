# House Price Regression

A complete end‑to‑end pipeline for predicting house prices, from Exploratory Data Analysis (EDA) and model training with CatBoost, through to deployment via a Streamlit front‑end and FastAPI back‑end.

---

## Table of Contents

- [Project Structure](#project-structure)  
- [Data](#data)  
- [Exploratory Data Analysis](#exploratory-data-analysis)  
- [Model Training](#model-training)  
- [CatBoost Analysis](#catboost-analysis)  
- [Deployment](#deployment)  
  - [FastAPI Back‑End](#fastapi-back-end)  
  - [Streamlit Front‑End](#streamlit-front-end)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Dependencies](#dependencies)  
- [Contributing](#contributing)  
- [License](#license)  
- [Contact](#contact)  

---

## Project Structure


---

## Data

This project uses the **[House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)** dataset from Kaggle.  

1. Sign in to Kaggle and download `train.csv` and `test.csv`.  
2. Place them in a `data/` directory at the root of this repo.

---

## Exploratory Data Analysis

Open and run the notebook in `notbooks/01_data_preprocessing.ipynb` to:

- Load and inspect the data  
- Handle missing values  
- Encode categorical variables  
- Create new features  
- Visualize key relationships  

---

## Model Training

The `notbooks/02_model_training.ipynb` notebook covers:

- Data splitting (train/validation)  
- Baseline models (e.g., Linear Regression, Random Forest)  
- Training a CatBoost regressor  
- Hyperparameter tuning  
- Evaluating with RMSE, MAE, and R²  

---

## CatBoost Analysis

In `catboost_info/` you’ll find:

- **Hyperparameter tuning** notebooks demonstrating grid‑search and CatBoost’s own tools  
- **Feature importance** analysis (including SHAP plots)  

---

## Deployment

### FastAPI Back‑End

The FastAPI server (`serviceapp/api.py`) exposes a `/predict` endpoint that accepts a CSV and returns predictions.

Run it with:

```bash
uvicorn serviceapp.api:app --reload
