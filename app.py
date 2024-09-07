import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.preprocessing import MinMaxScaler

# Load the encoding map
with open('data/encoding_map.json', 'r') as f:
    encoding_map = json.load(f)

# Load the trained meta-model
with open('models/catboost_model.pkl', 'rb') as f:
    meta_model = pickle.load(f)

# Define the expected input features (excluding 'Default')
expected_features = [
    'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
    'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio',
    'Education_0', 'Education_1', 'Education_2', 'Education_3',
    'EmploymentType_0', 'EmploymentType_1', 'EmploymentType_2', 'EmploymentType_3',
    'MaritalStatus_0', 'MaritalStatus_1', 'MaritalStatus_2',
    'HasMortgage_0', 'HasMortgage_1',
    'HasDependents_0', 'HasDependents_1',
    'LoanPurpose_0', 'LoanPurpose_1', 'LoanPurpose_2', 'LoanPurpose_3', 'LoanPurpose_4',
    'HasCoSigner_0', 'HasCoSigner_1'
]

# Helper function to display a user-friendly dropdown
def create_input_options(label, key, encoding_map):
    options = {v: k for k, v in encoding_map.items()}
    user_choice = st.selectbox(label, options.keys())
    return int(options[user_choice])

# Display the title
st.title("Loan Default Prediction App")

# Create a form for user input
with st.form("user_input_form"):
    st.header("Please provide the following information:")

    # Numerical inputs
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Income (in USD)", min_value=0, value=50000)
    loan_amount = st.number_input("Loan Amount (in USD)", min_value=1000, value=10000)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    months_employed = st.number_input("Months Employed", min_value=0, value=12)
    num_credit_lines = st.number_input("Number of Credit Lines", min_value=0, value=5)
    interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=5.0)
    loan_term = st.number_input("Loan Term (in Months)", min_value=1, max_value=360, value=36)
    dti_ratio = st.number_input("Debt-to-Income Ratio (%)", min_value=0.0, max_value=100.0, value=20.0)

    # Categorical inputs
    education = create_input_options("Education", "Education", encoding_map['Education'])
    employment_type = create_input_options("Employment Type", "EmploymentType", encoding_map['EmploymentType'])
    marital_status = create_input_options("Marital Status", "MaritalStatus", encoding_map['MaritalStatus'])
    has_mortgage = create_input_options("Has Mortgage", "HasMortgage", encoding_map['HasMortgage'])
    has_dependents = create_input_options("Has Dependents", "HasDependents", encoding_map['HasDependents'])
    loan_purpose = create_input_options("Loan Purpose", "LoanPurpose", encoding_map['LoanPurpose'])
    has_cosigner = create_input_options("Has Co-Signer", "HasCoSigner", encoding_map['HasCoSigner'])

    # Submit button
    submitted = st.form_submit_button("Submit")

    # Handle form submission
    if submitted:
        # Create DataFrame for input data
        input_data = pd.DataFrame({
            'Age': [age],
            'Income': [income],
            'LoanAmount': [loan_amount],
            'CreditScore': [credit_score],
            'MonthsEmployed': [months_employed],
            'NumCreditLines': [num_credit_lines],
            'InterestRate': [interest_rate],
            'LoanTerm': [loan_term],
            'DTIRatio': [dti_ratio],
            'Education': [education],
            'EmploymentType': [employment_type],
            'MaritalStatus': [marital_status],
            'HasMortgage': [has_mortgage],
            'HasDependents': [has_dependents],
            'LoanPurpose': [loan_purpose],
            'HasCoSigner': [has_cosigner]
        })

        # Encode the input data as required by the model
        input_encoded = pd.get_dummies(input_data)

        # Add missing columns with zeros (as you already do)
        for col in expected_features:
            if col not in input_encoded.columns:
                input_encoded[col] = 0

        # Handle potential extra features that might have been added during training
        # This can occur if an extra feature was mistakenly included
        input_encoded = input_encoded[expected_features]

        # Reorder the columns to match the training data
        input_encoded = input_encoded[expected_features]

        # Scale the input
        scaler = MinMaxScaler()
        input_scaled = scaler.fit_transform(input_encoded)

        # Make prediction
        prediction = meta_model.predict(input_scaled)[0]

        # Display the result
        if prediction == 0:
            st.success("Prediction: The applicant is not likely to default on the loan.")
        else:
            st.error("Prediction: The applicant is likely to default on the loan.")
