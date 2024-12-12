import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Load the pre-trained model and scaler
model = joblib.load('final_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define features and target (same as used during model training)
features = ['input_voltage', 'hour']
target = 'el_power'

# Streamlit configuration
st.set_page_config(page_title="Energy Consumption Prediction", page_icon="⚡", layout="wide")

# Sidebar instructions
st.sidebar.header("How to Use")
st.sidebar.write("1. Enter the input values for voltage and hours.\n"
                 "2. The model will predict the electrical energy consumption.\n"
                 "3. Visualize trends in the dataset or predictions separately.")

# App Title
st.title("⚡ Electrical Energy Consumption Prediction")
st.markdown("### Predict electrical energy consumption and visualize trends independently!")

# Section 1: Input Data for User Prediction
st.header("User Input for Prediction")
input_voltage = st.number_input("Enter the input voltage (V)", min_value=0.0, step=0.1)

# Allow user to input any number of hours (can be fractional, greater than 24)
hour = st.number_input("Enter the number of hours", min_value=0.0, step=0.1)

if st.button("🔮 Predict for User Input"):
    # Prepare input data for prediction
    input_data = np.array([[input_voltage, hour]])

    # Scale the input data using the same scaler used during training
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)

    # Display prediction result
    st.subheader("Predicted Electrical Energy Consumption:")
    st.write(f"{prediction[0]:.2f} kW")


# Section 3: Visualize Trends in the Dataset
st.header("Visualizing Trends in the Dataset")

# Load the training data (example CSV file for training data)
import zipfile
import pandas as pd

@st.cache
def load_training_data():
    # Path to the ZIP file containing the training data
    zip_file_path = 'micro+gas+turbine+electrical+energy+prediction/train.zip'  # Adjust this to the correct path

    # List to store data frames
    data_frames = []

    # Open and extract CSV files from the ZIP file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Iterate over all files in the ZIP
        for file_name in zip_ref.namelist():
            if file_name.endswith('.csv'):
                with zip_ref.open(file_name) as f:
                    df = pd.read_csv(f)
                    data_frames.append(df)

    # Concatenate all data frames if multiple CSV files are in the ZIP
    if data_frames:
        train_data = pd.concat(data_frames, ignore_index=True)
        return train_data
    else:
        raise ValueError("No CSV files found in the ZIP file")



train_data = load_training_data()

# Plot the energy consumption over time (if time column exists)
plt.figure(figsize=(10, 5))
if 'time' in train_data.columns:
    train_data['time'] = pd.to_datetime(train_data['time'])  # Ensure time is in datetime format
    plt.plot(train_data['time'], train_data['el_power'], marker='o', linestyle='-', alpha=0.6)
    plt.title("Electrical Power Consumption Over Time")
    plt.xlabel("Time")
    plt.ylabel("Power Consumption (kW)")
    plt.xticks(rotation=45)
    plt.grid()
    st.pyplot(plt)
else:
    st.write("Time column is missing. Showing alternative trends.")
    plt.scatter(train_data['input_voltage'], train_data['el_power'], alpha=0.6)
    plt.title("Power Consumption vs Input Voltage")
    plt.xlabel("Input Voltage (V)")
    plt.ylabel("Power Consumption (kW)")
    st.pyplot(plt)

# Display Feature Correlations
st.subheader("Feature Correlation Heatmap")
plt.figure(figsize=(12, 8))
sns.heatmap(train_data.corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title("Feature Correlation Heatmap")
st.pyplot(plt)