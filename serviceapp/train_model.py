import pandas as pd
from processing  import preprocess_input
from model import train_model

# Load your dataset
data = pd.read_csv('data/train.csv')
# Preprocess the features
df = preprocess_input(data)
# Define your target variable and features
y_train = df['SalePrice']             # Target variable
X_train = df.drop(columns=['SalePrice'])  # Features

# Train and save the model
train_model(X_train, y_train, 'models/catboost_model.pkl')
