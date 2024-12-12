# -- coding: utf-8 --
"""
Energy Prediction Project

Predicting electrical energy consumption using multiple regression models.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR  # Ensure SVR is imported
from sklearn.metrics import mean_squared_error, r2_score  # Import the necessary metrics
import joblib
import logging
import zipfile

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configurable parameters
TRAIN_ZIP = r'C:\Users\Hp\OneDrive\Desktop\ml project\train.zip'
TEST_ZIP = r'C:\Users\Hp\OneDrive\Desktop\ml project\test.zip'
MODEL_PATH = 'final_model.pkl'
SCALER_PATH = 'scaler.pkl'  # Save scaler for consistent input processing


# Functions
def load_data_from_zip(zip_path):
    """Load and combine all CSV files from a zip file into a single DataFrame."""
    data_frames = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        for file_name in z.namelist():
            if file_name.endswith('.csv'):
                logging.info(f"Reading {file_name} from {zip_path}")
                with z.open(file_name) as f:
                    df = pd.read_csv(f)
                    data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True)


def preprocess_data(data):
    """Preprocess the dataset by handling missing values and feature engineering."""
    if data.isnull().values.any():
        logging.warning("Missing values found. Imputing with column means.")
        data.fillna(data.mean(), inplace=True)

    # Using 'time' as total hours, making sure it's numeric
    if 'time' in data.columns:
        data['time'] = pd.to_numeric(data['time'], errors='coerce')  # Convert to numeric

    return data


def split_and_scale_data(data, features, target, test_size=0.2, random_state=42):
    """Split the dataset and scale the features."""
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_models(X_train, y_train):
    """Train multiple models and return a Voting Regressor."""
    lr = LinearRegression()
    rf = RandomForestRegressor(random_state=42, n_estimators=100)
    gb = GradientBoostingRegressor(random_state=42)
    svr = SVR()

    voting = VotingRegressor([('lr', lr), ('rf', rf), ('gb', gb), ('svr', svr)])
    voting.fit(X_train, y_train)
    return voting


def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return performance metrics."""
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))  # Now mean_squared_error is imported
    r2 = r2_score(y_test, predictions)
    logging.info(f"Model Evaluation - RMSE: {rmse:.3f}, R^2: {r2:.3f}")
    return rmse, r2


# Main script
if __name__ == "__main__":
    # Load and preprocess the data
    logging.info("Loading training data...")
    train_data = load_data_from_zip(TRAIN_ZIP)
    train_data = preprocess_data(train_data)

    logging.info("Loading testing data...")
    test_data = load_data_from_zip(TEST_ZIP)
    test_data = preprocess_data(test_data)

    # Define features and target
    features = ['time', 'input_voltage']  # Using time (total hours) and input_voltage as features
    target = 'el_power'  # Target is the electrical power consumption

    # Split and scale the data
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(train_data, features, target)

    # Train the model
    logging.info("Training the model...")
    model = train_models(X_train, y_train)

    # Evaluate the model
    logging.info("Evaluating the model...")
    evaluate_model(model, X_test, y_test)

    # Save the model and scaler
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    logging.info(f"Model and scaler saved to {MODEL_PATH} and {SCALER_PATH}")