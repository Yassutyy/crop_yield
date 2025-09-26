import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier

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

# ---------------- Train Models ----------------
X = df[['Fertilizer', 'Temp', 'N', 'P', 'K']]
y = df['Yield']

# Regression model
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X, y)

# Classification model
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
    Welcome to the **AgriYield Dashboard**.  
    This tool helps in predicting **crop yield** based on inputs like Fertilizer, Temperature, and Soil Nutrients (N, P, K).  
    
    Use the left menu to:
    - 📂 Explore dataset  
    - 📊 View visualizations  
    - 🔮 Make predictions  
    """)

elif menu == "📂 Dataset":
    st.header("📂 Dataset Preview")
    st.write(df.head())

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

    fert = st.number_input("Fertilizer", min_value=0.0)
    temp = st.number_input("Temperature", min_value=0.0)
    n = st.number_input("Nitrogen (N)", min_value=0.0)
    p = st.number_input("Phosphorus (P)", min_value=0.0)
    k = st.number_input("Potassium (K)", min_value=0.0)

    if st.button("Predict Yield"):
        input_data = np.array([[fert, temp, n, p, k]])
        predicted_yield = rf.predict(input_data)[0]
        yield_class = "High Yield 🌾" if dt_clf.predict(input_data)[0] == 1 else "Low Yield 🌿"

        st.success(f"🌱 Predicted Yield: {predicted_yield:.2f} tons/hectare")
        st.info(f"Classification: {yield_class}")
