import streamlit as st
import pandas as pd
import os

# Page Config
st.set_page_config(
    page_title="FinRisk | AI EMI Prediction",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

if os.path.exists("assets/style.css"):
    load_css("assets/style.css")

# --- Hero Section ---
st.markdown("""
<div style="background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%); padding: 3rem; border-radius: 1rem; margin-bottom: 2rem; text-align: center;">
    <h1 style="font-size: 3.5rem; margin-bottom: 0.5rem; color: white;">FinRisk AI 🚀</h1>
    <h3 style="color: #bfdbfe; font-weight: 300;">Next-Gen Financial Risk Assessment & EMI Prediction</h3>
    <p style="font-size: 1.2rem; color: #e2e8f0; max-width: 800px; margin: 1rem auto;">
        Leveraging advanced Machine Learning to automate loan eligibility and calculate maximum safe EMI limits with 90%+ accuracy.
    </p>
</div>
""", unsafe_allow_html=True)

# --- Features Grid ---
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="stContainer">
        <h3 style="text-align: center">🤖 Dual ML Engine</h3>
        <p style="text-align: center; color: #9ca3af;">
            Combines XGBoost Classification for eligibility checks and Regression for precise EMI limits.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stContainer">
        <h3 style="text-align: center">📊 400K+ Records</h3>
        <p style="text-align: center; color: #9ca3af;">
            Trained on a massive dataset of financial profiles across 5 distinct lending scenarios.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stContainer">
        <h3 style="text-align: center">📈 MLflow Tracking</h3>
        <p style="text-align: center; color: #9ca3af;">
            Full experiment tracking and model versioning integrated into the deployment pipeline.
        </p>
    </div>
    """, unsafe_allow_html=True)

# --- Quick Actions ---
st.markdown("### 🚀 Get Started")
c1, c2 = st.columns(2)
with c1:
    if st.button("Start Prediction Engine", use_container_width=True):
        st.switch_page("pages/1_Predict.py")
with c2:
    if st.button("View Model Performance", use_container_width=True):
        st.switch_page("pages/3_Model_Dashboard.py")

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #6b7280; font-size: 0.8rem;">
        &copy; 2024 FinRisk AI Platform. Powered by Streamlit & XGBoost.
    </div>
    """, 
    unsafe_allow_html=True
)
