import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from io import StringIO

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

# ---------------- Train Models ----------------
X = df[['Fertilizer', 'Temp', 'N', 'P', 'K']]
y = df['Yield']

# Regression model (for yield prediction)
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X, y)

# Classification model (for yield level: High/Low)
median_yield = df['Yield'].median()
df['Yield_Class'] = (df['Yield'] >= median_yield).astype(int)
y_class = df['Yield_Class']
dt_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_clf.fit(X, y_class)

# ---------------- Sidebar Menu ----------------
st.sidebar.title("🌱 AgriYield Dashboard")
menu = st.sidebar.radio("Navigate", ["🏠 Home", "📂 Dataset", "📊 Visualizations", "🔮 Predictions"])

# ---------------- Pages ----------------
if menu == "🏠 Home":
    st.title("🌱 AgriYield – Crop Yield Prediction")
    st.markdown("""
    Welcome to the **AgriYield Dashboard** 🌾  
    This smart farming tool predicts **crop yield** based on parameters like **Fertilizer**, **Temperature**, and **Soil Nutrients (N, P, K)**.

    **Use the menu to:**  
    - 📂 Explore Dataset  
    - 📊 View Visualizations  
    - 🔮 Make Predictions  
    """)

elif menu == "📂 Dataset":
    st.header("📂 Dataset Preview")
    st.dataframe(df)

    st.subheader("📊 Summary Statistics")
    st.write(df.describe())

elif menu == "📊 Visualizations":
    st.header("📊 Visualizations")

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="RdYlBu", ax=ax)
    st.pyplot(fig)

    # Scatter Fertilizer vs Yield
    st.subheader("Fertilizer vs Yield")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="Fertilizer", y="Yield", ax=ax)
    st.pyplot(fig)

elif menu == "🔮 Predictions":
    st.header("🔮 Make Predictions")
    st.markdown("Enter the input values below to predict the expected **Crop Yield (tons/hectare)**.")

    # --- Input fields ---
    fertilizer = st.number_input("Fertilizer used (kg/ha)", min_value=0.0, step=0.1)
    temp = st.number_input("Temperature (°C)", min_value=0.0, step=0.1)
    N = st.number_input("Nitrogen content (N)", min_value=0.0, step=0.1)
    P = st.number_input("Phosphorus content (P)", min_value=0.0, step=0.1)
    K = st.number_input("Potassium content (K)", min_value=0.0, step=0.1)

    # --- Predict Button ---
    if st.button("Predict Yield"):
        if fertilizer == 0 or temp == 0 or N == 0 or P == 0 or K == 0:
            st.warning("⚠️ Please enter valid (non-zero) input values for all fields.")
        else:
            input_data = np.array([[fertilizer, temp, N, P, K]])
            prediction = rf.predict(input_data)[0]
            yield_class = dt_clf.predict(input_data)[0]

            st.success(f"🌾 **Predicted Crop Yield:** {prediction:.2f} tons/hectare")

            if yield_class == 1:
                st.info("✅ This corresponds to a **High Yield** region based on model classification.")
            else:
                st.warning("⚠️ This corresponds to a **Low Yield** region based on model classification.")
