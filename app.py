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
st.sidebar.write("1. Enter the input values for voltage and hour.\n"
                 "2. The model will predict the electrical energy consumption.\n"
                 "3. Visualize trends in the dataset or predictions separately.")

# App Title
st.title("⚡ Electrical Energy Consumption Prediction")
st.markdown("### Predict electrical energy consumption and visualize trends independently!")

# Section 1: Input Data for User Prediction
st.header("User Input for Prediction")
input_voltage = st.number_input("Enter the input voltage (V)", min_value=0.0, step=0.1)
hour = st.slider("Select the hour of the day (0-23)", min_value=0, max_value=23, step=1)

if st.button("🔮 Predict for User Input"):
    # Prepare input data for prediction
    input_data = np.array([[input_voltage, hour]])

    # Scale the input data using the same scaler used during training
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)

    # Display prediction result
    st.subheader("Predicted Electrical Power Consumption:")
    st.write(f"{prediction[0]:.2f} kW")

# Section 2: Predictions for CSV Data
st.header("Batch Prediction from CSV")
file_upload = st.file_uploader("Upload CSV file for batch prediction", type="csv")

if file_upload is not None:
    # Load uploaded data
    csv_data = pd.read_csv(file_upload)

    # Ensure required columns are present
    if all(feature in csv_data.columns for feature in features):
        csv_data_scaled = scaler.transform(csv_data[features])
        csv_predictions = model.predict(csv_data_scaled)

        # Add predictions to the DataFrame
        csv_data['Predicted_Power'] = csv_predictions

        # Display results
        st.write("### Predicted Results:")
        st.dataframe(csv_data)

        # Allow file download
        csv_data.to_csv("predicted_results.csv", index=False)
        st.download_button(label="Download Predictions", data=csv_data.to_csv(index=False), file_name="predicted_results.csv", mime="text/csv")
    else:
        st.error("The uploaded file must contain the required features: input_voltage and hour.")

# Section 3: Visualize Trends in the Dataset
st.header("Visualizing Trends in the Dataset")

# Load the training data (assumes a CSV file exists)
@st.cache
def load_data():
    train_data = pd.read_csv(r'C:\Users\Hp\OneDrive\Desktop\SEM 5\APPLIED ML\micro+gas+turbine+electrical+energy+prediction (1)\train\train\ex_20.csv')  # Replace with your actual training data path
    return train_data

train_data = load_data()

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
