import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score


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
df = pd.read_csv(StringIO(csv_data))

# ---------------- Split Data ----------------
X = df[['Fertilizer', 'Temp', 'N', 'P', 'K']]
y = df['Yield']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- Train Models ----------------
lr_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)

lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)

# ---------------- Model Accuracies ----------------
lr_acc = r2_score(y_test, lr_model.predict(X_test))
rf_acc = r2_score(y_test, rf_model.predict(X_test))
dt_acc = r2_score(y_test, dt_model.predict(X_test))

median_yield = df['Yield'].median()

# ---------------- Sidebar Menu ----------------
st.sidebar.title("🌾 AgriYield Dashboard")
menu = st.sidebar.radio("Navigate", ["🏠 Home", "📂 Dataset", "📊 Visualizations", "🔮 Predictions"])

# ---------------- Pages ----------------
if menu == "🏠 Home":
    st.title("🌱 AgriYield – Smart Crop Yield Prediction System")
    st.markdown("""
    ### 🌾 Overview
    Agriculture plays a vital role in sustaining human life. The **AgriYield Dashboard** is designed to 
    help farmers, researchers, and agricultural planners make **data-driven decisions** for improving 
    crop yield.

    ### 🎯 Objective
    This system predicts the **expected crop yield (tons/hectare)** based on critical input parameters like:
    - Fertilizer usage  
    - Temperature  
    - Soil nutrients (Nitrogen, Phosphorus, and Potassium levels)

    ### 🤖 Machine Learning Integration
    The dashboard uses **three machine learning algorithms**:
    - **Linear Regression** – for basic linear trend estimation  
    - **Random Forest Regressor** – for high accuracy and stability  
    - **Decision Tree Regressor** – for clear rule-based decision making  

    Each algorithm is trained using real-world agricultural data to forecast crop performance effectively.

    ### 🌟 Key Features
    - Simple and interactive interface  
    - Multiple model selection for prediction  
    - Real-time yield estimation based on entered parameters  
    - Supports visualization for better understanding of relationships  

    ---
    **Use the sidebar to navigate** through the dataset, visualizations, and prediction modules.
    """)


elif menu == "📂 Dataset":
    st.header("📂 Dataset Preview")
    st.write(df.head())

    st.subheader("📊 Summary Statistics")
    st.write(df.describe())


elif menu == "📊 Visualizations":
    st.header("📊 Data Visualizations")

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Fertilizer vs Yield")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="Fertilizer", y="Yield", color="green", s=80, ax=ax)
    st.pyplot(fig)

    st.subheader("Temperature vs Yield")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="Temp", y="Yield", color="orange", s=80, ax=ax)
    st.pyplot(fig)


elif menu == "🔮 Predictions":
    st.header("🔮 Crop Yield Prediction")

    algo = st.selectbox(
        "Select Algorithm for Prediction:",
        ["Linear Regression", "Random Forest Regressor", "Decision Tree Regressor"]
    )

    st.markdown("### 🌾 Enter the Crop Parameters:")
    fertilizer = st.number_input("Fertilizer used (kg/ha)", min_value=0.0, step=0.1)
    temp = st.number_input("Temperature (°C)", min_value=0.0, step=0.1)
    N = st.number_input("Nitrogen content (N)", min_value=0.0, step=0.1)
    P = st.number_input("Phosphorus content (P)", min_value=0.0, step=0.1)
    K = st.number_input("Potassium content (K)", min_value=0.0, step=0.1)

    if st.button("Predict Yield"):
        if fertilizer == 0 or temp == 0 or N == 0 or P == 0 or K == 0:
            st.warning("⚠️ Please enter valid (non-zero) values for all fields.")
        else:
            input_data = np.array([[fertilizer, temp, N, P, K]])

            if algo == "Linear Regression":
                model = lr_model
                accuracy = lr_acc
            elif algo == "Random Forest Regressor":
                model = rf_model
                accuracy = rf_acc
            else:
                model = dt_model
                accuracy = dt_acc

            prediction = model.predict(input_data)
            predicted_yield = prediction[0]
            yield_type = "🌾 High Yield" if predicted_yield > median_yield else "🌱 Low Yield"

            st.success(f"**Predicted Crop Yield ({algo})**: {predicted_yield:.2f} tons/hectare")
            st.info(f"**Yield Category:** {yield_type}")
            st.caption(f"Model Accuracy (R²): {accuracy:.3f}")
