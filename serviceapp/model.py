import joblib
from catboost import CatBoostRegressor
import pandas as pd

# Function to train and save the model
def train_model(X_train: pd.DataFrame, y_train: pd.Series, model_path: str):
    model = CatBoostRegressor(
        depth=5,
        iterations=1700,
        l2_leaf_reg=1,
        learning_rate=0.017235477520255067,
        verbose=0 )  
    # Instantiate CatBoost Regressor
    model.fit(X_train, y_train)           # Train the model
    joblib.dump(model, model_path)        # Save the model to file
    print(f"Model saved to {model_path}")

# Function to load the saved model
def load_model(model_path: str):
    model = joblib.load(model_path)
    return model
