import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Config
st.set_page_config(page_title="EMI/Loan Eligibility Predictor", layout="wide")

@st.cache_resource
def load_models():
    """Load the trained models and label encoder."""
    try:
        # Load the best models as identified
        clf_pipe = joblib.load('pipeline_classification_xgb_tuned_fast.joblib')
        reg_pipe = joblib.load('pipeline_regression_xgb_focused.joblib')
        
        # Load label classes (target names)
        # Note: If this is an array like ['Eligible', 'High_Risk', 'Not_Eligible']
        label_map = joblib.load('label_encoder_classes.joblib')
        
        return clf_pipe, reg_pipe, label_map
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None

def main():
    st.title("💳 EMI Eligibility & Prediction System")
    st.write("Predict your loan eligibility and maximum safe EMI based on financial profile.")

    # Load models
    clf_model, reg_model, label_classes = load_models()
    
    if clf_model is None:
        st.warning("Please ensure model files (.joblib) are in the same directory.")
        return

    st.sidebar.header("Applicant Profile")
    
    # --- Input Fields ---
    # Demographics
    age = st.sidebar.number_input("Age", 18, 70, 30)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
    marital = st.sidebar.selectbox("Marital Status", ["Married", "Single", "Divorced", "Widowed"])
    education = st.sidebar.selectbox("Education", ["Graduate", "Post Graduate", "Under Graduate", "High School", "Others"]) # Common standardized values
    family_size = st.sidebar.number_input("Family Size", 1, 20, 4)
    dependents = st.sidebar.number_input("Dependents", 0, 10, 2)
    
    # Employment
    st.sidebar.header("Employment")
    # Checking values from notebook context implies these might be standard
    emp_type = st.sidebar.selectbox("Employment Type", ["Salaried", "Self_Employed", "Business", "Freelancer"]) 
    company_type = st.sidebar.selectbox("Company Type", ["Private", "Public", "Government", "Other"]) 
    years_emp = st.sidebar.number_input("Years of Employment", 0.0, 50.0, 5.0)
    monthly_salary = st.sidebar.number_input("Monthly Salary (INR)", 0.0, 10000000.0, 50000.0)
    
    # Assets & Expenses
    st.sidebar.header("Financials")
    house_type = st.sidebar.selectbox("House Type", ["Owned", "Rented", "Mortgaged", "Family"])
    rent = st.sidebar.number_input("Monthly Rent", 0.0, value=10000.0 if house_type=="Rented" else 0.0)
    
    # Specific Expenses
    school_fees = st.sidebar.number_input("School Fees", 0.0, value=0.0)
    college_fees = st.sidebar.number_input("College Fees", 0.0, value=0.0)
    travel_exp = st.sidebar.number_input("Travel Expenses", 0.0, value=2000.0)
    groc_util = st.sidebar.number_input("Groceries & Utilities", 0.0, value=5000.0)
    other_exp = st.sidebar.number_input("Other Monthly Expenses", 0.0, value=3000.0)
    
    # Financial State
    exist_loans = st.sidebar.number_input("Existing Loans Count", 0, 10, 0)
    curr_emi = st.sidebar.number_input("Current EMI Amount", 0.0, value=0.0)
    credit_score = st.sidebar.number_input("Credit Score", 300, 900, 750)
    bank_bal = st.sidebar.number_input("Bank Balance", 0.0, value=20000.0)
    emerg_fund = st.sidebar.number_input("Emergency Fund", 0.0, value=10000.0)
    
    # Loan Request
    st.sidebar.header("Loan Requirements")
    emi_scenario = st.sidebar.selectbox("EMI Scenario", ["Home_Loan", "Car_Loan", "Personal_Loan", "Education_Loan", "Business_Loan"])
    req_amount = st.sidebar.number_input("Requested Loan Amount", 0.0, value=500000.0)
    req_tenure = st.sidebar.number_input("Requested Tenure (Months)", 1.0, 360.0, 24.0)

    # --- Feature Engineering (Logic inferred to match Pipeline expectations) ---
    
    # 1. Total Expenses
    total_expenses = rent + school_fees + college_fees + travel_exp + groc_util + other_exp
    
    # 2. Ratios
    # Avoid div by zero
    income = monthly_salary if monthly_salary > 0 else 1.0
    
    debt_to_inc = curr_emi / income
    exp_to_inc = total_expenses / income
    savings_rat = (income - total_expenses - curr_emi) / income
    
    # Create Dataframe with EXACT column names expected by model
    # Order doesn't strictly matter for dataframe input but names DO.
    
    input_df = pd.DataFrame([{
        'age': age,
        'gender': gender,
        'marital_status': marital,
        'education': education,
        'monthly_salary': monthly_salary,
        'employment_type': emp_type,
        'years_of_employment': years_emp,
        'company_type': company_type,
        'house_type': house_type,
        'monthly_rent': rent,
        'family_size': family_size,
        'dependents': dependents,
        'school_fees': school_fees,
        'college_fees': college_fees,
        'travel_expenses': travel_exp,
        'groceries_utilities': groc_util,
        'other_monthly_expenses': other_exp,
        'existing_loans': exist_loans,
        'current_emi_amount': curr_emi,
        'credit_score': credit_score,
        'bank_balance': bank_bal,
        'emergency_fund': emerg_fund,
        'emi_scenario': emi_scenario,
        'requested_amount': req_amount,
        'requested_tenure': req_tenure,
        # Engineered Features
        'debt_to_income_ratio': debt_to_inc,
        'expense_to_income_ratio': exp_to_inc,
        'savings_ratio': savings_rat
    }])
    
    # --- Prediction Section ---
    st.write("### Applicant Summary")
    st.write(f"**Monthly Salary:** ₹{monthly_salary:,.2f} | **Credit Score:** {credit_score}")
    st.write(f"**Calculated Ratios:** DTI: {debt_to_inc:.2f} | ETI: {exp_to_inc:.2f} | Savings: {savings_rat:.2f}")

    if st.button("Analyze Eligibility"):
        
        with st.spinner("Analyzing financial profile..."):
            try:
                # 1. Classification Prediction
                # The model pipeline handles preprocessing (encoding/scaling)
                pred_raw = clf_model.predict(input_df)[0]
                
                # Check if output needs mapping
                eligibility_status = pred_raw
                if isinstance(pred_raw, (int, np.integer)):
                    if label_classes is not None and len(label_classes) > pred_raw:
                        eligibility_status = label_classes[pred_raw]
                
                # Display Classification Result
                st.markdown("---")
                if eligibility_status == "Eligible":
                     st.success(f"## ✅ Status: {eligibility_status}")
                elif eligibility_status == "High_Risk":
                     st.warning(f"## ⚠️ Status: {eligibility_status}")
                else:
                     st.error(f"## ❌ Status: {eligibility_status}")
                
                # 2. Regression Prediction (Max EMI)
                # Regression model expects 'emi_eligibility' as a feature
                # We pass the status (string) as input
                
                reg_input_df = input_df.copy()
                reg_input_df['emi_eligibility'] = eligibility_status
                
                max_emi_pred = reg_model.predict(reg_input_df)[0]
                
                st.markdown(f"### 💰 Maximum Safe Monthly EMI: ₹{max_emi_pred:,.2f}")
                
                if max_emi_pred > 0:
                    st.info(f"Based on your profile, you can afford an EMI up to ₹{max_emi_pred:,.0f}.")
                else:
                    st.warning("Calculated safe EMI is zero or negative. Please reduce obligations.")

            except Exception as e:
                st.error(f"Prediction Error: {e}")
                st.write("Debug Info (Input Data Types):")
                st.write(input_df.dtypes)

if __name__ == "__main__":
    main()
