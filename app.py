import streamlit as st
import joblib
import numpy as np

# 1. Load the "Brain" we created in Colab
model = joblib.load('poverty_model.pkl')

# 2. Setup the Website Title and Description
st.title("SDG 1: Poverty Risk Predictor")
st.write("This AI model predicts if a household is at high risk of poverty based on key indicators.")

st.divider()

# 3. Create input fields for the user
st.subheader("Enter Household Details")

income = st.number_input("Monthly Income (USD)", min_value=0, value=1000)
fam_size = st.slider("Family Size", 1, 15, 4)
edu_level = st.slider("Education Level of Head of Household (0-10)", 0, 10, 5)
location = st.selectbox("Location", options=[0, 1], format_func=lambda x: "Rural" if x == 1 else "Urban")

# 4. The Prediction Button
if st.button("Analyze Risk"):
    # Arrange inputs into the format the model expects
    features = np.array([[income, fam_size, edu_level, location]])
    
    # Make the prediction
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1] # Chance of poverty

    st.divider()

    if prediction[0] == 1:
        st.error(f"High Poverty Risk Detected. (Probability: {probability:.2%})")
        st.write("Recommendation: This household may qualify for government subsidies and educational support.")
    else:
        st.success(f"Low Poverty Risk Detected. (Probability: {probability:.2%})")
        st.write("Recommendation: Focus on long-term financial stability and vocational training.")