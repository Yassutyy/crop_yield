import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# ---------------- Load Dataset ----------------
csv_data = """
Fertilizer,Temp,N,P,K,Yield
50,30,20,10,10,3.5
60,32,25,12,12,4.0
70,28,30,14,14,4.8
80,31,28,13,16,5.0
90,29,35,15,18,5.5
40,35,15,8,7,2.8
55,33,18,9,10,3.2
65,30,22,11,11,4.1
75,34,27,12,15,4.7
85,32,33,14,17,5.2
"""
from io import StringIO
df = pd.read_csv(StringIO(csv_data))

# ---------------- Split Data ----------------
X = df[['Fertilizer', 'Temp', 'N', 'P', 'K']]
y = df['Yield']

# ---------------- Train Models ----------------
lr_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)

lr_model.fit(X, y)
rf_model.fit(X, y)
dt_model.fit(X, y)

# ---------------- Sidebar Menu ----------------
st.sidebar.title("🌾 AgriYield Dashboard")
menu = st.sidebar.radio("Navigate", ["🏠 Home", "📂 Dataset", "📊 Visualizations", "🔮 Predictions"])

# ---------------- Pages ----------------
if menu == "🏠 Home":
    st.title("🌱 AgriYield – Crop Yield Prediction Dashboard")
    st.markdown("""
    Welcome to **AgriYield**, an interactive crop yield prediction dashboard.  
    This project helps in predicting crop yield based on:
    - 🌡️ Temperature  
    - 💧 Fertilizer usage  
    - 🌿 Soil nutrients (N, P, K)  
    
    You can explore:
    - 📂 The dataset  
    - 📊 Data visualizations  
    - 🔮 Predict yield using various ML models
    """)

elif menu == "📂 Dataset":
    st.header("📂 Dataset Preview")
    st.write(df.head())

    st.subheader("📊 Summary Statistics")
    st.write(df.describe())

elif menu == "📊 Visualizations":
    st.header("📊 Data Visualizations")

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="YlGnBu", ax=ax)
    st.pyplot(fig)

    # Fertilizer vs Yield
    st.subheader("Fertilizer vs Yield")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="Fertilizer", y="Yield", color="green", s=80, ax=ax)
    st.pyplot(fig)

    # Temperature vs Yield
    st.subheader("Temperature vs Yield")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="Temp", y="Yield", color="orange", s=80, ax=ax)
    st.pyplot(fig)

elif menu == "🔮 Predictions":
    st.header("🔮 Crop Yield Prediction")

    # Model selection
    algo = st.selectbox(
        "Select Prediction Algorithm",
        ["Linear Regression", "Random Forest", "Decision Tree"]
    )

    st.markdown("### Enter the Crop Parameters:")
    fertilizer = st.number_input("Fertilizer used (kg/ha)", min_value=0.0, step=0.1)
    temp = st.number_input("Temperature (°C)", min_value=0.0, step=0.1)
    N = st.number_input("Nitrogen content (N)", min_value=0.0, step=0.1)
    P = st.number_input("Phosphorus content (P)", min_value=0.0, step=0.1)
    K = st.number_input("Potassium content (K)", min_value=0.0, step=0.1)

    if st.button("Predict Yield"):
        if fertilizer == 0 or temp == 0 or N == 0 or P == 0 or K == 0:
            st.warning("⚠️ Please enter valid (non-zero) values for all inputs.")
        else:
            input_data = np.array([[fertilizer, temp, N, P, K]])

            # Select model based on user choice
            if algo == "Linear Regression":
                prediction = lr_model.predict(input_data)
                model_name = "Linear Regression"
            elif algo == "Random Forest":
                prediction = rf_model.predict(input_data)
                model_name = "Random Forest Regressor"
            else:
                prediction = dt_model.predict(input_data)
                model_name = "Decision Tree Regressor"

            st.success(f"🌾 Predicted Crop Yield using **{model_name}**: {prediction[0]:.2f} tons/hectare")
