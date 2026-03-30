import pickle
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💳",
    layout="wide"
)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "probability" not in st.session_state:
    st.session_state.probability = None

st.markdown(
    """
    <style>
        .main > div {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }

        .app-title {
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }

        .app-subtitle {
            color: #9aa4b2;
            margin-bottom: 1.2rem;
            font-size: 1rem;
        }

        .section-title {
            font-size: 1.6rem;
            font-weight: 700;
            margin-top: 0.5rem;
            margin-bottom: 0.8rem;
        }

        .result-card {
            min-height: 620px;
            border-radius: 24px;
            padding: 2rem 1.5rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            color: white;
            font-weight: 600;
            box-shadow: 0 8px 24px rgba(0,0,0,0.18);
        }

        .result-neutral {
            background: linear-gradient(135deg, #6b7280, #4b5563);
        }

        .result-approved {
            background: linear-gradient(135deg, #16a34a, #15803d);
        }

        .result-rejected {
            background: linear-gradient(135deg, #dc2626, #b91c1c);
        }

        .result-title {
            font-size: 1.2rem;
            opacity: 0.95;
            margin-bottom: 0.8rem;
        }

        .result-main {
            font-size: 2.3rem;
            font-weight: 800;
            margin-bottom: 0.8rem;
        }

        .result-prob {
            font-size: 1.1rem;
            opacity: 0.95;
            margin-bottom: 0.8rem;
        }

        .small-note {
            font-size: 1rem;
            opacity: 0.92;
            line-height: 1.6;
            max-width: 420px;
        }

        div[data-testid="stButton"] > button {
            width: 100%;
            border-radius: 12px;
            height: 3rem;
            font-weight: 700;
            margin-top: 1.6rem;
        }

        @media (max-width: 1200px) {
            .result-card {
                min-height: 540px;
            }
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="app-title">Loan Approval Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Enter applicant details on the left and view the decision instantly on the right.</div>',
    unsafe_allow_html=True
)

left_col, right_col = st.columns([1.3, 1], gap="large")

with left_col:
    st.markdown('<div class="section-title">Applicant Details</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with c2:
        married = st.selectbox("Married", ["Yes", "No"])

    c3, c4 = st.columns(2)
    with c3:
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    with c4:
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])

    c5, c6 = st.columns(2)
    with c5:
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    with c6:
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    st.markdown('<div class="section-title">Financial Details</div>', unsafe_allow_html=True)

    c7, c8 = st.columns(2)
    with c7:
        applicant_income = st.number_input(
            "Applicant Income",
            min_value=0.0,
            step=100.0,
            value=5000.0
        )
    with c8:
        coapplicant_income = st.number_input(
            "Coapplicant Income",
            min_value=0.0,
            step=100.0,
            value=0.0
        )

    c9, c10 = st.columns(2)
    with c9:
        loan_amount = st.number_input(
            "Loan Amount",
            min_value=0.0,
            step=1.0,
            value=120.0
        )
    with c10:
        loan_amount_term = st.number_input(
            "Loan Amount Term",
            min_value=0.0,
            step=1.0,
            value=360.0
        )

    c11, c12 = st.columns(2)
    with c11:
        credit_history = st.selectbox(
            "Credit History",
            [1.0, 0.0],
            format_func=lambda x: "Good (1.0)" if x == 1.0 else "Poor (0.0)"
        )
    with c12:
        predict_clicked = st.button("Predict Loan Status")

    if predict_clicked:
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

        st.session_state.prediction_made = True
        st.session_state.prediction = int(prediction)
        st.session_state.probability = float(probability)

with right_col:
    if not st.session_state.prediction_made:
        st.markdown(
            """
            <div class="result-card result-neutral">
                <div class="result-title">Prediction Panel</div>
                <div class="result-main">Waiting for Input</div>
                <div class="result-prob">Fill the form and click Predict</div>
                <div class="small-note">
                    The decision result will appear here.<br>
                    Approved applications will show in green.<br>
                    Rejected applications will show in red.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        if st.session_state.prediction == 1:
            st.markdown(
                f"""
                <div class="result-card result-approved">
                    <div class="result-title">Loan Decision</div>
                    <div class="result-main">Approved ✅</div>
                    <div class="result-prob">Approval Probability: {st.session_state.probability:.2%}</div>
                    <div class="small-note">
                        This applicant shows a stronger approval profile based on the model.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="result-card result-rejected">
                    <div class="result-title">Loan Decision</div>
                    <div class="result-main">Not Approved ❌</div>
                    <div class="result-prob">Approval Probability: {st.session_state.probability:.2%}</div>
                    <div class="small-note">
                        This applicant shows a weaker approval profile based on the model.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )