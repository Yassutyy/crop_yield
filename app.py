import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier

st.title("🌱 AgriYield – Crop Yield Prediction Dashboard")

# ---------------- Load Dataset & Train Models ----------------
# Load your training dataset
df = pd.read_csv("crop_yield.csv")  # make sure your CSV is in 'data/' folder

# Features and target
X = df[['Fertilizer', 'Temp', 'N', 'P', 'K']]
y = df['Yield']

# Split dataset for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Train Decision Tree for classification (High vs Low Yield)
median_yield = df['Yield'].median()
df['Yield_Class'] = (df['Yield'] >= median_yield).astype(int)
y_class = df['Yield_Class']

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X, y_class, test_size=0.2, random_state=42)
dt_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_clf.fit(X_train_cls, y_train_cls)

# ---------------- Prediction Form ----------------
st.write("## 🌾 Predict Yield for New Inputs")

fert = st.number_input("Fertilizer", min_value=0.0)
temp = st.number_input("Temperature", min_value=0.0)
n = st.number_input("Nitrogen (N)", min_value=0.0)
p = st.number_input("Phosphorus (P)", min_value=0.0)
k = st.number_input("Potassium (K)", min_value=0.0)

if st.button("Predict Yield"):
    input_data = np.array([[fert, temp, n, p, k]])

    # Regression prediction
    predicted_yield = rf.predict(input_data)[0]
    st.success(f"🌱 Predicted Yield: {predicted_yield:.2f} tons/hectare")

    # Classification
    yield_class = "High Yield" if dt_clf.predict(input_data)[0] == 1 else "Low Yield"
    st.info(f"Classification: {yield_class}")

