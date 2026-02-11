import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page Config
st.set_page_config(page_title="Prediction | FinRisk", page_icon="🏦", layout="wide")

# Load CSS
def load_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
load_css("assets/style.css")

# --- Model Loading ---
@st.cache_resource
def load_models():
    """Load the trained models and label encoder."""
    try:
        # Load the best models based on our MLFlow experiment
        clf_path = 'pipeline_classification_best.joblib' if os.path.exists('pipeline_classification_best.joblib') else 'pipeline_classification_xgb_tuned_fast.joblib'
        reg_path = 'pipeline_regression_best.joblib' if os.path.exists('pipeline_regression_best.joblib') else 'pipeline_regression_xgb_focused.joblib'
        LE_path = 'label_encoder_classes.joblib'
        
        clf_pipe = joblib.load(clf_path)
        reg_pipe = joblib.load(reg_path)
        label_map = joblib.load(LE_path)
        
        return clf_pipe, reg_pipe, label_map
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None

def main():
    st.markdown("<h1>💳 Loan Eligibility & EMI <span style='color:#3b82f6'>Prediction Engine</span></h1>", unsafe_allow_html=True)
    st.markdown("---")

    # Load models
    clf_model, reg_model, label_classes = load_models()
    
    if clf_model is None:
        st.warning("⚠️ Model files not found. Please ensure the ML pipeline has run successfully.")
        return

    # --- Sidebar Inputs ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2666/2666508.png", width=60)
        st.header("Applicant Profile")
        
        # Group: Personal
        with st.expander("👤 Personal Demographics", expanded=True):
            age = st.number_input("Age", 18, 70, 30)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            marital = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Widowed"])
            education = st.selectbox("Education", ["Graduate", "Post Graduate", "Under Graduate", "High School", "Others"])
            family_size = st.number_input("Family Size", 1, 20, 4)
            dependents = st.number_input("Dependents", 0, 10, 2)
        
        # Group: Employment
        with st.expander("💼 Employment / Income", expanded=True):
            emp_type = st.selectbox("Employment Type", ["Salaried", "Self_Employed", "Business", "Freelancer"]) 
            company_type = st.selectbox("Company Type", ["Private", "Public", "Government", "Other"]) 
            years_emp = st.number_input("Years of Employment", 0.0, 50.0, 5.0)
            monthly_salary = st.number_input("Monthly Salary (INR)", 0.0, 10000000.0, 50000.0, step=1000.0)
        
        # Group: Financials
        with st.expander("🏠 Housing & Expenses", expanded=False):
            house_type = st.selectbox("House Type", ["Owned", "Rented", "Mortgaged", "Family"])
            rent = st.number_input("Monthly Rent", 0.0, value=10000.0 if house_type=="Rented" else 0.0, step=500.0)
            school_fees = st.number_input("School Fees", 0.0, value=0.0, step=500.0)
            college_fees = st.number_input("College Fees", 0.0, value=0.0, step=500.0)
            travel_exp = st.number_input("Travel Expenses", 0.0, value=2000.0, step=100.0)
            groc_util = st.number_input("Groceries & Utilities", 0.0, value=5000.0, step=100.0)
            other_exp = st.number_input("Other Expenses", 0.0, value=3000.0, step=100.0)

        # Group: Assets/Liabilities
        with st.expander("💳 Assets & Liabilities", expanded=False):
            exist_loans = st.number_input("Existing Loans Count", 0, 10, 0)
            curr_emi = st.number_input("Current EMI Amount", 0.0, value=0.0, step=500.0)
            credit_score = st.slider("Credit Score (CIBIL)", 300, 900, 750)
            bank_bal = st.number_input("Bank Balance", 0.0, value=20000.0, step=1000.0)
            emerg_fund = st.number_input("Emergency Fund", 0.0, value=10000.0, step=1000.0)

        # Group: Limit Proposal
        with st.expander("💰 Loan Request", expanded=True):
            emi_scenario = st.selectbox("EMI Scenario", ["Home_Loan", "Car_Loan", "Personal_Loan", "Education_Loan", "Shopping_Loan", "Business_Loan"])
            req_amount = st.number_input("Requested Loan Amount", 0.0, value=500000.0, step=10000.0)
            req_tenure = st.number_input("Requested Tenure (Months)", 1.0, 360.0, 24.0, step=6.0)

    # --- Feature Engineering & Main Layout ---
    
    # 1. Logic
    total_expenses = rent + school_fees + college_fees + travel_exp + groc_util + other_exp
    income = monthly_salary if monthly_salary > 0 else 1.0
    
    debt_to_inc = curr_emi / income
    exp_to_inc = total_expenses / income
    savings_rat = (income - total_expenses - curr_emi) / income
    
    # 2. Layout: Summary Cards
    col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
    with col_sum1:
        st.metric("Net Monthly Income", f"₹ {monthly_salary:,.0f}")
    with col_sum2:
        st.metric("Total Expenses", f"₹ {total_expenses + curr_emi:,.0f}", delta=f"-{(total_expenses+curr_emi)/income:.0%}")
    with col_sum3:
        st.metric("Disposable Income", f"₹ {income - (total_expenses + curr_emi):,.0f}")
    with col_sum4:
        st.metric("Credit Score", credit_score, delta=credit_score-700)

    # 3. Input Dataframe Construction
    input_data = {
        'age': age, 'gender': gender, 'marital_status': marital, 'education': education,
        'monthly_salary': monthly_salary, 'employment_type': emp_type, 'years_of_employment': years_emp,
        'company_type': company_type, 'house_type': house_type, 'monthly_rent': rent,
        'family_size': family_size, 'dependents': dependents, 'school_fees': school_fees,
        'college_fees': college_fees, 'travel_expenses': travel_exp, 'groceries_utilities': groc_util,
        'other_monthly_expenses': other_exp, 'existing_loans': exist_loans, 'current_emi_amount': curr_emi,
        'credit_score': credit_score, 'bank_balance': bank_bal, 'emergency_fund': emerg_fund,
        'emi_scenario': emi_scenario, 'requested_amount': req_amount, 'requested_tenure': req_tenure,
        # Engineered Features
        'total_expenses': total_expenses,
        'debt_to_income_ratio': debt_to_inc, 
        'expense_to_income_ratio': exp_to_inc, 
        'savings_ratio': savings_rat
    }
    input_df = pd.DataFrame([input_data])

    st.markdown("---")
    
    # --- Prediction Button ---
    col_pred1, col_pred2 = st.columns([1, 2])
    
    with col_pred1:
        st.write("### AI Analysis")
        st.write("Click below to run the dual-model inference engine.")
        analyze_btn = st.button("🚀 Analyze Eligibility", use_container_width=True)
    
    with col_pred2:
        if analyze_btn:
            with st.spinner("Running Inference on XGBoost Pipelines..."):
                try:
                    # Classification
                    pred_raw = clf_model.predict(input_df)[0]
                    
                    # Map result
                    eligibility_status = pred_raw
                    # If model returns index, map it
                    if isinstance(pred_raw, (int, np.integer)):
                        if label_classes is not None and len(label_classes) > pred_raw:
                            eligibility_status = label_classes[pred_raw]
                    
                    # Regression Input (needs status)
                    reg_input_df = input_df.copy()
                    reg_input_df['emi_eligibility'] = eligibility_status
                    
                    max_emi_pred = reg_model.predict(reg_input_df)[0]

                    # Display Results
                    
                    # Result Card Style
                    if eligibility_status == "Eligible":
                        st.success(f"## ✅ Outcome: {eligibility_status}")
                        st.balloons()
                    elif eligibility_status == "High_Risk":
                         st.warning(f"## ⚠️ Outcome: {eligibility_status}")
                    else:
                         st.error(f"## ❌ Outcome: {eligibility_status}")
                    
                    st.markdown(f"""
                    <div style="background: rgba(30, 64, 175, 0.3); padding: 20px; border-radius: 12px; border: 1px solid #3b82f6;">
                        <h2 style="margin:0; text-align:center;">Maximum Safe EMI: ₹{max_emi_pred:,.2f}</h2>
                        <p style="text-align:center; color:#9ca3af; margin-top:5px;">Based on regression analysis of similar profiles.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if max_emi_pred > 0:
                        st.info(f"💡 Recommendation: You can safely handle a monthly EMI of ₹{max_emi_pred:,.0f}. Ensure new loans stay within this limit.")
                    else:
                        st.warning("Calculated safe EMI is negligible. Reduce debt or expenses before applying.")

                    # Add specific warning for Home Loans if amount is high
                    if emi_scenario == "Home_Loan" and req_amount > 1500000:
                         st.info("ℹ️ **Note for Large Loans:** The model is highly accurate for consumer loans up to ₹15L. For multi-crore loans, please combine this with traditional bank DSR (Debt Service Ratio) checks.")

                except Exception as e:
                    st.error(f"Prediction Error: {e}")
                    st.write(input_df.dtypes)

    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #6b7280; font-size: 0.9rem;">
            <b>Project:</b> FinRisk AI - Guvi Capstone | <b>Tech:</b> Python, XGBoost, MLflow, Streamlit
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
