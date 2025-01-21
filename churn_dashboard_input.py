import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Title of the app
st.title("Telecom Customer Churn Prediction Dashboard")

# Sidebar for inputting data
st.sidebar.header("Input Data")

# Function to input customer details
def user_input_features():
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
    partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No"])
    internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.sidebar.selectbox("Online Security", ["Yes", "No"])
    online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No"])
    device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No"])
    tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No"])
    streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No"])
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 50.0)
    total_charges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 600.0)

    data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }
    return pd.DataFrame([data])

# User input
input_df = user_input_features()

# Display user input
st.subheader("User Input Data")
st.write(input_df)

# Pre-trained model (dummy data setup and training)
# Load and preprocess dataset (simulated for this example)
# For production, replace this with your preprocessed dataset and trained model
data = {
    "gender": ["Male", "Female", "Female", "Male"],
    "SeniorCitizen": [0, 1, 0, 1],
    "Partner": ["Yes", "No", "Yes", "No"],
    "Dependents": ["No", "Yes", "No", "Yes"],
    "tenure": [1, 45, 24, 15],
    "PhoneService": ["Yes", "No", "Yes", "Yes"],
    "MultipleLines": ["No", "No", "Yes", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "DSL", "Fiber optic"],
    "OnlineSecurity": ["No", "Yes", "No", "Yes"],
    "OnlineBackup": ["Yes", "No", "Yes", "No"],
    "DeviceProtection": ["No", "Yes", "No", "Yes"],
    "TechSupport": ["Yes", "No", "Yes", "No"],
    "StreamingTV": ["No", "Yes", "Yes", "No"],
    "StreamingMovies": ["No", "Yes", "Yes", "No"],
    "Contract": ["Month-to-month", "Two year", "One year", "Month-to-month"],
    "PaperlessBilling": ["Yes", "No", "No", "Yes"],
    "PaymentMethod": ["Electronic check", "Credit card (automatic)", "Bank transfer (automatic)", "Mailed check"],
    "MonthlyCharges": [29.85, 56.95, 42.3, 25.45],
    "TotalCharges": [29.85, 1889.5, 1840.75, 245.45],
    "Churn": ["No", "Yes", "No", "Yes"],
}

df = pd.DataFrame(data)

# Preprocessing and training
df = pd.get_dummies(df, drop_first=True)
X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Match user input format with training data
input_df_encoded = pd.get_dummies(input_df)
input_df_encoded = input_df_encoded.reindex(columns=X.columns, fill_value=0)

# Prediction
prediction = model.predict(input_df_encoded)
prediction_probability = model.predict_proba(input_df_encoded)[:, 1]

# Display prediction
st.subheader("Prediction")
st.write("Churn" if prediction[0] == 1 else "No Churn")
st.write(f"Prediction Probability: {prediction_probability[0]:.2f}")
