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


# ---------------- Custom Professional CSS Styling ----------------
def set_professional_style():
    st.markdown("""
        <style>
        /* =======================
           🌑 PROFESSIONAL DARK THEME
           ======================= */

        /* Main App Background */
        .stApp {
            background: linear-gradient(135deg, #0d1b2a 0%, #1b263b 50%, #0d1b2a 100%);
            color: #e0e0e0;
            font-family: 'Poppins', sans-serif;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: #1b263b;
            color: #ffffff;
            border-right: 2px solid #00c853;
        }

        /* Sidebar title */
        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3 {
            color: #00e676 !important;
        }

        /* Buttons */
        div.stButton > button:first-child {
            background: linear-gradient(90deg, #00c853, #009624);
            color: white;
            font-weight: 600;
            border: none;
            border-radius: 8px;
            padding: 0.6em 1.2em;
            transition: 0.3s ease;
            box-shadow: 0px 4px 10px rgba(0, 200, 83, 0.3);
        }
        div.stButton > button:first-child:hover {
            background: linear-gradient(90deg, #00e676, #00c853);
            transform: scale(1.05);
            box-shadow: 0px 6px 15px rgba(0, 230, 118, 0.4);
        }

        /* Titles and Headers */
        h1, h2, h3, h4 {
            color: #00e676;
            font-weight: 700;
        }

        /* DataFrame Tables */
        .stDataFrame {
            background-color: #102030;
            border: 1px solid #00e676;
            border-radius: 10px;
            padding: 10px;
        }

        /* Metric boxes / success messages */
        .stAlert {
            background-color: #102a43;
            border-left: 5px solid #00c853;
            border-radius: 8px;
            color: #ffffff;
        }

        /* Inputs and SelectBoxes */
        div[data-baseweb="input"] input {
            background-color: #1b263b !important;
            color: white !important;
            border: 1px solid #00c853 !important;
            border-radius: 8px !important;
        }

        div[data-baseweb="select"] > div {
            background-color: #1b263b !important;
            color: white !important;
            border: 1px solid #00c853 !important;
            border-radius: 8px !important;
        }

        /* Radio buttons, checkboxes */
        div[role="radiogroup"] label {
            color: #e0e0e0 !important;
        }

        /* Charts area */
        .css-1ht1j8u, .stPlotlyChart, .stVegaLiteChart, .stAltairChart {
            background-color: #102030;
            border-radius: 10px;
            padding: 10px;
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
        }
        ::-webkit-scrollbar-thumb {
            background: #00c853;
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #00e676;
        }

        /* Footer / Bottom space */
        footer {
            visibility: hidden;
        }
        </style>
    """, unsafe_allow_html=True)

# Apply the theme
set_professional_style()


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
    st.title("🌱 AgriYield – Crop Yield Prediction Dashboard")
    st.markdown("""
    Welcome to **AgriYield**, an interactive dashboard for **crop yield prediction**.

    This project uses **Machine Learning algorithms** to estimate the expected yield 
    (in tons/hectare) based on:
    - Fertilizer usage  
    - Temperature  
    - Soil nutrients (N, P, K)

    ### 🤖 Models Used:
    - Linear Regression  
    - Random Forest Regressor  
    - Decision Tree Regressor
    """)

    # Display Model Accuracies
    st.subheader("📈 Model Accuracies (R² Score)")
    acc_data = {
        "Model": ["Linear Regression", "Random Forest Regressor", "Decision Tree Regressor"],
        "R² Score": [lr_acc, rf_acc, dt_acc]
    }
    acc_df = pd.DataFrame(acc_data)
    st.dataframe(acc_df)

    # Bar chart comparison
    st.subheader("📊 Model Comparison")
    fig, ax = plt.subplots()
    sns.barplot(data=acc_df, x="Model", y="R² Score", palette="YlGnBu", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Model R² Accuracy Comparison")
    st.pyplot(fig)

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
            st.write(f"📊 **Model R² Accuracy:** {accuracy:.3f}")


