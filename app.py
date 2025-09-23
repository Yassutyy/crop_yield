import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix

st.title("🌱 AgriYield – Crop Yield Prediction Dashboard")

# Upload dataset
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", df.head())

    # Show basic stats
    st.write("### Data Summary")
    st.write(df.describe())

    # Correlation heatmap
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="RdYlBu", ax=ax)
    st.pyplot(fig)

    # Features & Target
    X = df[['Fertilizer', 'Temp', 'N', 'P', 'K']]
    y = df['Yield']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Regression Models
    st.write("## Regression Models")

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    r2_lr = r2_score(y_test, y_pred_lr)
    st.write("### Linear Regression")
    st.write(f"RMSE: {rmse_lr:.2f}, R²: {r2_lr:.2f}")

    # Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)
    st.write("### Random Forest Regressor")
    st.write(f"RMSE: {rmse_rf:.2f}, R²: {r2_rf:.2f}")

    # Classification (High vs Low Yield)
    st.write("## Classification Models")
    median_yield = df['Yield'].median()
    df['Yield_Class'] = (df['Yield'] >= median_yield).astype(int)
    y_class = df['Yield_Class']

    X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)

    # Logistic Regression
    log_clf = LogisticRegression(max_iter=1000)
    log_clf.fit(X_train, y_train_class)
    y_pred_log = log_clf.predict(X_test)
    acc_log = accuracy_score(y_test_class, y_pred_log)
    st.write("### Logistic Regression Classifier")
    st.write(f"Accuracy: {acc_log:.2f}")

    # Decision Tree
    dt_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_clf.fit(X_train, y_train_class)
    y_pred_dt = dt_clf.predict(X_test)
    acc_dt = accuracy_score(y_test_class, y_pred_dt)
    st.write("### Decision Tree Classifier")
    st.write(f"Accuracy: {acc_dt:.2f}")

    # Confusion Matrix Heatmap (Decision Tree)
    cm = confusion_matrix(y_test_class, y_pred_dt)
    st.write("#### Confusion Matrix (Decision Tree)")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Low Yield", "High Yield"],
                yticklabels=["Low Yield", "High Yield"], ax=ax)
    st.pyplot(fig)
