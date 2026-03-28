import streamlit as st
import pickle
import pandas as pd

# Load trained pipeline
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Loan Approval Predictor", page_icon="💳", layout="centered")

st.title("Loan Approval Predictor")
st.write("Enter applicant details below to predict loan approval status.")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0.0, step=100.0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, step=100.0)
loan_amount = st.number_input("Loan Amount", min_value=0.0, step=1.0)
loan_amount_term = st.number_input("Loan Amount Term", min_value=0.0, step=1.0)
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

if st.button("Predict Loan Status"):
    input_df = pd.DataFrame([{
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": credit_history,
        "Property_Area": property_area
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success("Loan Approved ✅")
    else:
        st.error("Loan Not Approved ❌")

    st.info(f"Approval Probability: {probability:.2%}")