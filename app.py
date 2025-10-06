import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model (make sure the .pkl file is present)
model = joblib.load("model.pkl")

# App title
st.title("🌾 AgriYield – Crop Yield Prediction Dashboard")

# Sidebar navigation
menu = st.sidebar.selectbox("Navigation", ["Home", "Dataset", "Visualization", "Prediction"])

# ---------------------------
# 1️⃣ Home Page
# ---------------------------
if menu == "Home":
    st.header("Welcome to AgriYield 🌱")
    st.write("""
    This Smart Farming Dashboard predicts crop yields based on key agricultural parameters like fertilizer usage, temperature, and soil nutrients (N, P, K).  
    Use the sidebar to explore:
    - 📊 Dataset and insights  
    - 🌾 Visualizations  
    - 🤖 Predict your yield
    """)

# ---------------------------
# 2️⃣ Dataset Section
# ---------------------------
elif menu == "Dataset":
    st.header("📋 Dataset Overview")
    df = pd.read_csv("crop_yield.csv")
    st.dataframe(df.head())
    st.write("Shape of data:", df.shape)

# ---------------------------
# 3️⃣ Visualization Section
# ---------------------------
elif menu == "Visualization":
    st.header("📈 Visualizations")
    df = pd.read_csv("crop_yield.csv")
    st.bar_chart(df[['Fertilizer', 'yield']])
    st.line_chart(df[['Temp', 'yield']])

# ---------------------------
# 4️⃣ Prediction Section
# ---------------------------
elif menu == "Prediction":
    st.header("🤖 Crop Yield Prediction")

    st.write("Enter the details below to predict crop yield:")

    # --- Input fields ---
    fertilizer = st.number_input("Fertilizer used (kg/ha)", min_value=0.0, step=0.1)
    temp = st.number_input("Temperature (°C)", min_value=0.0, step=0.1)
    N = st.number_input("Nitrogen content (N)", min_value=0.0, step=0.1)
    P = st.number_input("Phosphorus content (P)", min_value=0.0, step=0.1)
    K = st.number_input("Potassium content (K)", min_value=0.0, step=0.1)

    if st.button("Predict Yield"):
        if fertilizer == 0 or temp == 0 or N == 0 or P == 0 or K == 0:
            st.warning("⚠️ Please enter valid (non-zero) input values for all fields.")
        else:
            input_data = np.array([[fertilizer, temp, N, P, K]])
            prediction = model.predict(input_data)
            st.success(f"🌾 Predicted Crop Yield: **{prediction[0]:.2f} tons/hectare**")
