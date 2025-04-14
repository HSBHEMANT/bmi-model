
import streamlit as st
import joblib
import pandas as pd

# Load model and encoder
model = joblib.load("diet_model.pkl")
le = joblib.load("label_encoder.pkl")

st.title("🥗 BMI-Based Diet Recommendation App")

# User input
height = st.number_input("Enter your height (cm):", min_value=50, max_value=250, value=165)
weight = st.number_input("Enter your weight (kg):", min_value=20, max_value=200, value=68)

if st.button("Get Diet Recommendation"):
    sample = pd.DataFrame([[height, weight]], columns=["Height", "Weight"])
    prediction = model.predict(sample)
    result = le.inverse_transform(prediction)[0]
    st.success(f"✅ Recommended Diet Plan: **{result}**")
