import streamlit as st
import numpy as np
from utils import get_model, get_scaler, get_prediction

st.set_page_config(page_title="Diabetes Prediction")

# Load the model and scaler
model = get_model(r"models/Diabetes_Model.pkl")
scaler = get_scaler(r"models/Diabetes_StandardScaler.pkl")

st.title("Diabetes Prediction")
st.header("Feature Descriptions")
st.markdown(
    """
- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)²)
- **DiabetesPedigreeFunction**: Diabetes pedigree function
- **Age**: Age (years)
- **Outcome**:(Not-Diabetic, Diabetic)
"""
)
st.markdown("---")

gender = st.selectbox("Gender", ["Male", "Female"])
if gender == "Female":
    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=0,
        help="Number of times pregnant",
    )
else:
    pregnancies = 0

glucose = st.number_input(
    "Glucose",
    min_value=50,
    max_value=300,
    value=120,
    help="Plasma glucose concentration during 2-hour oral glucose tolerance test",
)
blood_pressure = st.number_input(
    "Blood Pressure",
    min_value=40,
    max_value=200,
    value=80,
    help="Diastolic blood pressure (mm Hg)",
)
skin_thickness = st.number_input(
    "Skin Thickness",
    min_value=5,
    max_value=100,
    value=20,
    help="Triceps skin fold thickness (mm)",
)
insulin = st.number_input(
    "Insulin",
    min_value=0,
    max_value=1000,
    value=85,
    help="2-Hour serum insulin (mu U/ml)",
)
bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0,
    help="Body mass index (weight in kg/(height in m)²)",
)
diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.1,
    max_value=2.5,
    value=0.5,
    help="Diabetes pedigree function",
)
age = st.number_input("Age", min_value=10, max_value=120, value=30, help="Age in years")

features = [
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree,
    age,
]

if st.button("Predict"):
    prediction = get_prediction(model, scaler, features)
    if prediction == "Diabetic":
        st.error(f"The prediction is: {prediction}")
    else:
        st.success(f"The prediction is: {prediction}")
