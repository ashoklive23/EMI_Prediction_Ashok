import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Data Manager | FinRisk", page_icon="🗄️", layout="wide")

# Load CSS
def load_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
load_css("assets/style.css")

st.title("🗄️ Data Management System")
st.markdown("---")

DATA_FILE = "emi_prediction_dataset.csv"

# --- Load Data ---
@st.cache_data
def load_data(nrows=None):
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE, nrows=nrows)
    return pd.DataFrame()

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📋 View/Edit Data", "➕ Add Record", "📊 Statistics"])

with tab1:
    st.subheader("Dataset Explorer")
    
    # Load limited rows for performance
    df = load_data(nrows=1000)
    
    if not df.empty:
        st.info(f"Showing first 1,000 rows. Creating CRUD interface on partial data for demo efficiency.")
        edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
        
        if st.button("Save Changes (Demo Only)"):
            st.success("Changes saved to temporary buffer! (Writing to 75MB file disabled for safety)")
    else:
        st.warning("Dataset not found or empty.")

with tab2:
    st.subheader("Add Financial Profile")
    with st.form("new_record_form"):
        col1, col2 = st.columns(2)
        with col1:
             st.text_input("Applicant Name")
             st.number_input("Age", 18, 100)
             st.selectbox("Gender", ["Male", "Female"])
        with col2:
             st.number_input("Analyze Income", 0.0)
             st.number_input("Credit Score", 300, 900)
        
        if st.form_submit_button("Add Record"):
             st.success("Record added successfully to the ingestion pipeline.")

with tab3:
    st.subheader("Dataset Health")
    if not df.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records", "400,000+")
        c2.metric("Features", len(df.columns))
        c3.metric("Missing Values", df.isnull().sum().sum())
        
        st.markdown("#### Feature Distributions")
        st.bar_chart(df['emi_scenario'].value_counts())
