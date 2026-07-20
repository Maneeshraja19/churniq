import streamlit as st
import pandas as pd
import joblib
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="ChurnIQ", page_icon="📊")

st.title("📊 ChurnIQ — Customer Churn Predictor")
st.write("Enter a customer's details to predict their churn risk and get an AI-generated retention recommendation.")

@st.cache_resource
def load_model():
    return joblib.load("models/churn_model.pkl")

@st.cache_resource
def load_generator():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return tokenizer, model

model = load_model()
gen_tokenizer, gen_model = load_generator()

st.header("Customer Details")

col1, col2 = st.columns(2)
with col1:
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
with col2:
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])


if st.button("Predict Churn Risk"):
    # Build a single-row dataframe matching the model's expected columns
    input_dict = {
        "SeniorCitizen": 0,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": monthly_charges * tenure,
        "Contract_One year": 1 if contract == "One year" else 0,
        "Contract_Two year": 1 if contract == "Two year" else 0,
        "InternetService_Fiber optic": 1 if internet_service == "Fiber optic" else 0,
        "InternetService_No": 1 if internet_service == "No" else 0,
        "PaymentMethod_Electronic check": 1 if payment_method == "Electronic check" else 0,
        "PaymentMethod_Mailed check": 1 if payment_method == "Mailed check" else 0,
        "PaymentMethod_Credit card (automatic)": 1 if payment_method == "Credit card (automatic)" else 0,
        "PaperlessBilling_Yes": 1 if paperless_billing == "Yes" else 0,
    }

    input_df = pd.DataFrame([input_dict])

    # Fill in any columns the model expects that we didn't explicitly build, with 0
    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model.feature_names_in_]  # match exact column order

    probability = model.predict_proba(input_df)[0][1]

    st.subheader(f"Churn Probability: {probability:.0%}")
    if probability > 0.5:
        st.error("High Risk")
    else:
        st.success("Low Risk")

    with st.spinner("Generating AI recommendation..."):
        prompt = f"Question: A customer has a {probability:.0%} chance of churning, uses a {contract} contract and {internet_service} internet. What is one specific action to retain this customer? Answer:"
        input_ids = gen_tokenizer(prompt, return_tensors="pt").input_ids
        output_ids = gen_model.generate(input_ids, max_new_tokens=60)
        ai_response = gen_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    st.subheader("AI Recommendation")
    st.write(ai_response)