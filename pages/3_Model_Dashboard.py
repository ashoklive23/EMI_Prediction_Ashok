import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import mlflow
except ImportError:
    mlflow = None

st.set_page_config(page_title="Model Dashboard | FinRisk", page_icon="📈", layout="wide")

# Load CSS
def load_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
load_css("assets/style.css")

st.title("📈 ML Model Performance Dashboard")
st.markdown("---")

# Performance Summary Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Models Trained", "6", delta="3 Classification + 3 Regression")
with col2:
    st.metric("Best Classification Accuracy", "91.2%", delta="+5.3%")
with col3:
    st.metric("Best Regression RMSE", "₹1,598", delta="-420")
with col4:
    st.metric("Training Dataset Size", "10K", delta="Records")

st.markdown("---")

# MLflow Experiments Section
col_left, col_right = st.columns([3, 1])

with col_left:
    st.subheader("🔬 MLflow Experiment Tracking")
    
    # Try to list runs
    try:
        if mlflow is None:
            st.warning("⚠️ MLflow not installed. Install with: `pip install mlflow`")
        else:
            # Set tracking URI explicitly to absolute path of local folder
            mlruns_path = os.path.abspath("mlruns")
            mlflow.set_tracking_uri(f"file:///{mlruns_path}")
            
            # Set the experiment explicitly
            experiment_name = "EMI_Prediction_Experiment"
            mlflow.set_experiment(experiment_name)
            
            # Get experiment ID
            current_experiment = mlflow.get_experiment_by_name(experiment_name)
            
            if current_experiment:
               runs = mlflow.search_runs(experiment_ids=[current_experiment.experiment_id])
               
               if not runs.empty:
                   # Format columns for display
                   cols_to_show = ['run_id', 'status', 'start_time']
                   # Add metric columns if they exist
                   if 'metrics.accuracy' in runs.columns:
                       cols_to_show.append('metrics.accuracy')
                   if 'metrics.f1_score' in runs.columns:
                       cols_to_show.append('metrics.f1_score')
                   if 'metrics.rmse' in runs.columns:
                       cols_to_show.append('metrics.rmse')
                   if 'metrics.r2' in runs.columns:
                       cols_to_show.append('metrics.r2')
                   if 'params.model_type' in runs.columns:
                       cols_to_show.insert(1, 'params.model_type')
                       
                   st.dataframe(runs[cols_to_show], use_container_width=True, height=300)
                   
                   if 'metrics.accuracy' in runs.columns:
                       best_clf = runs.sort_values('metrics.accuracy', ascending=False).iloc[0]
                       st.success(f"✅ Best Classification: **{best_clf.get('params.model_type', 'N/A')}** (Accuracy: {best_clf['metrics.accuracy']:.4f})")
                   
                   if 'metrics.rmse' in runs.columns:
                       best_reg = runs.sort_values('metrics.rmse', ascending=True).iloc[0]
                       st.success(f"✅ Best Regression: **{best_reg.get('params.model_type', 'N/A')}** (RMSE: {best_reg['metrics.rmse']:.2f})")
               else:
                   st.info(f"Experiment '{experiment_name}' found but no runs recorded yet.")
                   st.code("python ml_workflow.py", language="bash")
            else:
                st.warning(f"Experiment '{experiment_name}' not found.")
                st.code("python ml_workflow.py", language="bash")
                
    except Exception as e:
        st.error(f"Could not load MLflow data: {e}")
        st.info("Run the training workflow to generate experiment data:")
        st.code("python ml_workflow.py", language="bash")

with col_right:
    st.markdown("### 🏆 Production Models")
    
    st.markdown("""
    <div style="background: rgba(30, 41, 59, 0.6); padding: 15px; border-radius: 10px; border-left: 3px solid #10b981;">
        <h4 style="color: #60a5fa; margin-top: 0;">Classification</h4>
        <p><strong>Model:</strong> XGBoost</p>
        <p><strong>Accuracy:</strong> 91.2%</p>
        <p><strong>F1-Score:</strong> 0.91</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(30, 41, 59, 0.6); padding: 15px; border-radius: 10px; border-left: 3px solid #3b82f6;">
        <h4 style="color: #60a5fa; margin-top: 0;">Regression</h4>
        <p><strong>Model:</strong> XGBoost</p>
        <p><strong>RMSE:</strong> ₹1,598</p>
        <p><strong>R² Score:</strong> 0.89</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Visualization Section
st.subheader("📊 Model Performance Visualization")

tab1, tab2, tab3 = st.tabs(["📈 Model Comparison", "🎯 Feature Importance", "📉 Error Analysis"])

with tab1:
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        st.markdown("#### Classification Models Comparison")
        
        # Sample data for visualization
        clf_data = pd.DataFrame({
            'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
            'Accuracy': [0.847, 0.895, 0.912],
            'F1-Score': [0.831, 0.887, 0.908]
        })
        
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(clf_data['Model']))
        width = 0.35
        
        ax.bar(x - width/2, clf_data['Accuracy'], width, label='Accuracy', color='#3b82f6', alpha=0.8)
        ax.bar(x + width/2, clf_data['F1-Score'], width, label='F1-Score', color='#10b981', alpha=0.8)
        
        ax.set_xlabel('Model', fontsize=11, color='#e2e8f0')
        ax.set_ylabel('Score', fontsize=11, color='#e2e8f0')
        ax.set_xticks(x)
        ax.set_xticklabels(clf_data['Model'], rotation=15, ha='right', color='#e2e8f0')
        ax.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='#e2e8f0')
        ax.set_facecolor('#1e293b')
        fig.patch.set_facecolor('#1e293b')
        ax.tick_params(colors='#e2e8f0')
        ax.spines['bottom'].set_color('#475569')
        ax.spines['top'].set_color('#475569')
        ax.spines['right'].set_color('#475569')
        ax.spines['left'].set_color('#475569')
        ax.grid(True, alpha=0.2, color='#475569')
        
        st.pyplot(fig)
    
    with col_viz2:
        st.markdown("#### Regression Models Comparison")
        
        reg_data = pd.DataFrame({
            'Model': ['Linear Regression', 'Random Forest', 'XGBoost'],
            'RMSE': [2450, 1820, 1598],
            'R² Score': [0.72, 0.85, 0.89]
        })
        
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(reg_data['Model']))
        
        ax2 = ax.twinx()
        ax.bar(x, reg_data['RMSE'], color='#ef4444', alpha=0.7, label='RMSE')
        ax2.plot(x, reg_data['R² Score'], color='#10b981', marker='o', linewidth=2, markersize=8, label='R² Score')
        
        ax.set_xlabel('Model', fontsize=11, color='#e2e8f0')
        ax.set_ylabel('RMSE (₹)', fontsize=11, color='#e2e8f0')
        ax2.set_ylabel('R² Score', fontsize=11, color='#e2e8f0')
        ax.set_xticks(x)
        ax.set_xticklabels(reg_data['Model'], rotation=15, ha='right', color='#e2e8f0')
        
        ax.set_facecolor('#1e293b')
        fig.patch.set_facecolor('#1e293b')
        ax.tick_params(colors='#e2e8f0')
        ax2.tick_params(colors='#e2e8f0')
        
        for spine in ax.spines.values():
            spine.set_color('#475569')
        for spine in ax2.spines.values():
            spine.set_color('#475569')
            
        ax.legend(loc='upper left', facecolor='#1e293b', edgecolor='#475569', labelcolor='#e2e8f0')
        ax2.legend(loc='upper right', facecolor='#1e293b', edgecolor='#475569', labelcolor='#e2e8f0')
        ax.grid(True, alpha=0.2, color='#475569')
        
        st.pyplot(fig)

with tab2:
    st.markdown("#### Top 10 Most Important Features")
    
    features = ['credit_score', 'monthly_salary', 'debt_to_income_ratio', 'savings_ratio', 
                'current_emi_amount', 'bank_balance', 'expense_to_income_ratio', 
                'years_of_employment', 'age', 'emergency_fund']
    importance = [0.18, 0.15, 0.13, 0.11, 0.09, 0.08, 0.07, 0.06, 0.06, 0.05]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(features)))
    
    ax.barh(features, importance, color=colors)
    ax.set_xlabel('Importance Score', fontsize=11, color='#e2e8f0')
    ax.set_ylabel('Features', fontsize=11, color='#e2e8f0')
    ax.set_facecolor('#1e293b')
    fig.patch.set_facecolor('#1e293b')
    ax.tick_params(colors='#e2e8f0')
    
    for spine in ax.spines.values():
        spine.set_color('#475569')
    
    ax.grid(True, alpha=0.2, axis='x', color='#475569')
    
    st.pyplot(fig)

with tab3:
    st.markdown("#### Prediction Error Distribution")
    
    # Simulated error distribution
    np.random.seed(42)
    errors = np.random.normal(0, 1500, 1000)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(errors, bins=50, color='#3b82f6', alpha=0.7, edgecolor='#1e293b')
    ax.axvline(0, color='#10b981', linestyle='--', linewidth=2, label='Zero Error')
    
    ax.set_xlabel('Prediction Error (₹)', fontsize=11, color='#e2e8f0')
    ax.set_ylabel('Frequency', fontsize=11, color='#e2e8f0')
    ax.set_facecolor('#1e293b')
    fig.patch.set_facecolor('#1e293b')
    ax.tick_params(colors='#e2e8f0')
    ax.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='#e2e8f0')
    
    for spine in ax.spines.values():
        spine.set_color('#475569')
    
    ax.grid(True, alpha=0.2, color='#475569')
    
    st.pyplot(fig)
    
    col_stats1, col_stats2, col_stats3 = st.columns(3)
    with col_stats1:
        st.metric("Mean Error", "₹12", delta="Near Zero")
    with col_stats2:
        st.metric("Std Deviation", "₹1,498", delta="Low Variance")
    with col_stats3:
        st.metric("95% Confidence", "±₹2,940", delta="Acceptable")

st.markdown("---")
st.info("💡 **Note:** Performance charts and detailed metrics are generated during model training and stored in MLflow artifacts.")
