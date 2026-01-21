import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# --- SAFETY TRAINER ---
def train_backup_model():
    # If the .pkl file fails, this creates a fresh one instantly
    data = {
        'income_usd': np.random.randint(500, 5000, 100),
        'family_size': np.random.randint(1, 8, 100),
        'education_level': np.random.randint(0, 10, 100),
        'is_rural': np.random.randint(0, 2, 100),
    }
    df = pd.DataFrame(data)
    df['poverty_risk'] = ((df['income_usd'] < 1500) & (df['family_size'] > 4)).astype(int)
    X = df[['income_usd', 'family_size', 'education_level', 'is_rural']]
    y = df['poverty_risk']
    model = LogisticRegression()
    model.fit(X, y)
    return model

# --- LOAD MODEL ---
try:
    model = joblib.load('poverty_model.pkl')
except:
    model = train_backup_model()

# --- WEBSITE INTERFACE ---
st.title("SDG 1: Poverty Risk Predictor")
st.write("This AI model predicts household poverty risk.")

income = st.number_input("Monthly Income (USD)", min_value=0, value=1000)
fam_size = st.slider("Family Size", 1, 15, 4)
edu_level = st.slider("Education Level (0-10)", 0, 10, 5)
location = st.selectbox("Location", options=[0, 1], format_func=lambda x: "Rural" if x == 1 else "Urban")

if st.button("Analyze Risk"):
    features = np.array([[income, fam_size, edu_level, location]])
    prediction = model.predict(features)
    
    if prediction[0] == 1:
        st.error("High Poverty Risk Detected.")
    else:
        st.success("Low Poverty Risk Detected.")