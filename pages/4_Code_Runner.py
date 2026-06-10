import streamlit as st
import os
import sys
import io
import contextlib
import time

st.set_page_config(page_title="Code Runner | FinRisk", page_icon="💻", layout="wide")

# Load CSS
def load_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
load_css("assets/style.css")

# ── Custom Styles ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .code-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #7c3aed 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .code-header h1 { color: white; margin: 0; font-size: 2.2rem; }
    .code-header p { color: #c4b5fd; margin: 0.3rem 0 0 0; font-size: 1.1rem; }
    .section-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        transition: border-color 0.3s;
    }
    .section-card:hover { border-color: #60a5fa; }
    .section-card h4 { color: #60a5fa; margin: 0 0 0.3rem 0; }
    .section-card p { color: #94a3b8; margin: 0; font-size: 0.85rem; }
    .run-output {
        background: #0f172a;
        border: 1px solid #1e40af;
        border-radius: 10px;
        padding: 1rem;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        color: #a5f3fc;
        max-height: 500px;
        overflow-y: auto;
    }
    .mlflow-banner {
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .mlflow-banner h4 { color: #a7f3d0; margin: 0; }
    .mlflow-banner p { color: #d1fae5; margin: 0; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="code-header">
    <h1>💻 Code Runner & Viewer</h1>
    <p>Browse, select, and execute ML workflow code sections</p>
</div>
""", unsafe_allow_html=True)

# ── MLflow Banner ────────────────────────────────────────────────────────────
st.markdown("""
<div class="mlflow-banner">
    <div>
        <h4>📊 MLflow UI is Live</h4>
        <p>Access experiment tracking at <a href="http://localhost:5000" target="_blank" style="color: #6ee7b7; font-weight: bold;">http://localhost:5000</a></p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Define Code Sections ─────────────────────────────────────────────────────
CODE_SECTIONS = {
    "⚙️ Configuration & Imports": {
        "description": "Library imports, file paths, MLflow settings, and constants used across the pipeline.",
        "file_lines": (1, 24),
        "code": '''import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os

# --- Configuration ---
DATA_PATH = 'emi_prediction_dataset.csv'
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "EMI_Prediction_Experiment"
ARTIFACT_PATH = "models"
''',
        "runnable": False,
    },

    "🔧 Feature Engineering": {
        "description": "Domain-specific feature creation: total expenses, financial ratios (DTI, ETI, savings), category alignment.",
        "file_lines": (26, 77),
        "code": '''def engineer_features(df):
    """
    Applies domain-specific feature engineering and aligns categories.
    """
    # 0. Category Alignment (Matching UI labels in 1_Predict.py)
    scenario_map = {
        'Home Appliances EMI': 'Home_Loan',
        'Vehicle EMI': 'Car_Loan',
        'Personal Loan EMI': 'Personal_Loan',
        'Education EMI': 'Education_Loan',
        'E-commerce Shopping EMI': 'Shopping_Loan'
    }
    if 'emi_scenario' in df.columns:
        df['emi_scenario'] = df['emi_scenario'].replace(scenario_map)
    
    # Data Cleaning and Type Conversion
    numeric_cols_to_clean = [
        'monthly_salary', 'monthly_rent', 'school_fees', 'college_fees',
        'travel_expenses', 'groceries_utilities', 'other_monthly_expenses',
        'current_emi_amount', 'requested_amount'
    ]
    
    for col in numeric_cols_to_clean:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(
                r'(\\d+\\.\\d+)\\..*', r'\\1', regex=True
            )
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if col == 'monthly_salary':
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(0)

    df['monthly_salary'] = df['monthly_salary'].replace(0, 1)
    
    # 1. Total Monthly Expenses
    df['total_expenses'] = (
        df['monthly_rent'] + df['school_fees'] + df['college_fees'] +
        df['travel_expenses'] + df['groceries_utilities'] +
        df['other_monthly_expenses']
    )
    
    # 2. Financial Ratios
    df['debt_to_income_ratio'] = df['current_emi_amount'] / df['monthly_salary']
    df['expense_to_income_ratio'] = df['total_expenses'] / df['monthly_salary']
    df['savings_ratio'] = (
        df['monthly_salary'] - df['total_expenses'] - df['current_emi_amount']
    ) / df['monthly_salary']
    
    return df
''',
        "runnable": False,
    },

    "🏗️ Preprocessing Pipeline": {
        "description": "Builds sklearn ColumnTransformer with numeric (impute + scale) and categorical (impute + one-hot encode) branches.",
        "file_lines": (79, 97),
        "code": '''def get_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor
''',
        "runnable": False,
    },

    "🤖 Classification Training (3 Models)": {
        "description": "Trains Logistic Regression, Random Forest, and XGBoost classifiers with SMOTE oversampling. Logs to MLflow.",
        "file_lines": (147, 214),
        "code": '''# --- Classification Task ---
clf_models = {
    "Logistic_Regression": LogisticRegression(max_iter=500, random_state=42),
    "Random_Forest_Clf": RandomForestClassifier(
        n_estimators=20, max_depth=10, random_state=42, n_jobs=-1
    ),
    "XGBoost_Clf": XGBClassifier(
        n_estimators=50, max_depth=6, use_label_encoder=False,
        eval_metric='mlogloss', random_state=42, n_jobs=-1
    )
}

best_clf_name = None
best_clf_score = -1

for name, model in clf_models.items():
    with mlflow.start_run(run_name=f"CLF_{name}"):
        mlflow.set_tag("task", "classification")
        mlflow.set_tag("model_name", name)

        pipeline = ImbPipeline(steps=[
            ('preproc', get_preprocessor(numeric_features, categorical_features)),
            ('oversample', SMOTE(random_state=42)),
            ('clf', model)
        ])
        
        pipeline.fit(X_train, y_cls_train_enc)
        y_pred = pipeline.predict(X_test)
        
        acc = accuracy_score(y_cls_test_enc, y_pred)
        f1 = f1_score(y_cls_test_enc, y_pred, average='weighted')
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        
        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(pipeline, "model", signature=signature,
                                  registered_model_name=f"EMI_CLF_{name}")
        
        joblib.dump(pipeline, f'model_clf_{name.lower()}.joblib')
        
        if acc > best_clf_score:
            best_clf_score = acc
            best_clf_name = name
            best_clf_pipeline = pipeline

joblib.dump(best_clf_pipeline, 'pipeline_classification_best.joblib')
''',
        "runnable": False,
    },

    "📈 Regression Training (3 Models)": {
        "description": "Trains Linear Regression, Random Forest, and XGBoost regressors for EMI amount prediction. Logs to MLflow.",
        "file_lines": (216, 285),
        "code": '''# --- Regression Task ---
reg_models = {
    "Linear_Regression": LinearRegression(),
    "Random_Forest_Reg": RandomForestRegressor(
        n_estimators=20, max_depth=10, random_state=42, n_jobs=-1
    ),
    "XGBoost_Reg": XGBRegressor(
        n_estimators=50, max_depth=6, objective='reg:squarederror',
        random_state=42, n_jobs=-1
    )
}

best_reg_name = None
best_reg_score = float('inf')

for name, model in reg_models.items():
    with mlflow.start_run(run_name=f"REG_{name}"):
        mlflow.set_tag("task", "regression")
        mlflow.set_tag("model_name", name)

        pipeline = Pipeline(steps=[
            ('pre', get_preprocessor(numeric_features, cat_features_reg)),
            ('model', model)
        ])
        
        pipeline.fit(X_train_reg, y_reg_train)
        preds = pipeline.predict(X_test_reg)
        
        rmse = np.sqrt(mean_squared_error(y_reg_test, preds))
        r2 = r2_score(y_reg_test, preds)
        mae = mean_absolute_error(y_reg_test, preds)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        
        signature = infer_signature(X_test_reg, preds)
        mlflow.sklearn.log_model(pipeline, "model", signature=signature,
                                  registered_model_name=f"EMI_REG_{name}")
        
        joblib.dump(pipeline, f'model_reg_{name.lower()}.joblib')
        
        if rmse < best_reg_score:
            best_reg_score = rmse
            best_reg_name = name
            best_reg_pipeline = pipeline

joblib.dump(best_reg_pipeline, 'pipeline_regression_best.joblib')
''',
        "runnable": False,
    },

    "🚀 Full Training Pipeline": {
        "description": "Execute the complete end-to-end ML workflow: load data → engineer features → train 6 models → log to MLflow → save best models.",
        "file_lines": (100, 297),
        "code": '''# Runs the full ml_workflow.py pipeline
# This calls: run_training_pipeline()
# Which performs:
#   1. Load 100K rows from emi_prediction_dataset.csv
#   2. Apply feature engineering
#   3. Train 3 Classification models (Logistic Reg, RF, XGBoost)
#   4. Train 3 Regression models (Linear Reg, RF, XGBoost)
#   5. Log all experiments to MLflow (sqlite:///mlflow.db)
#   6. Save best pipelines as .joblib files
''',
        "runnable": True,
    },
}


# ── Sidebar: Section Selector ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Code Sections")
    st.markdown("---")
    
    selected_section = st.radio(
        "Select a section to view:",
        list(CODE_SECTIONS.keys()),
        index=0,
        label_visibility="collapsed",
    )
    
    st.markdown("---")
    
    # Source file info
    st.markdown("""
    <div style="background: rgba(30, 41, 59, 0.7); padding: 12px; border-radius: 8px; border-left: 3px solid #7c3aed;">
        <p style="color: #a78bfa; font-weight: 600; margin: 0 0 4px 0;">📄 Source File</p>
        <p style="color: #94a3b8; margin: 0; font-size: 0.85rem;">ml_workflow.py</p>
        <p style="color: #64748b; margin: 4px 0 0 0; font-size: 0.75rem;">297 lines · 11.7 KB</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    
    # Quick links
    st.markdown("### 🔗 Quick Links")
    st.markdown("""
    - [🌐 MLflow UI](http://localhost:5000)
    - [📊 Model Dashboard](#)
    - [🏦 Prediction Engine](#)
    """)


# ── Main Content ─────────────────────────────────────────────────────────────
section_data = CODE_SECTIONS[selected_section]

# Section header
col_info, col_action = st.columns([3, 1])

with col_info:
    st.markdown(f"### {selected_section}")
    st.markdown(f"*{section_data['description']}*")
    
    line_start, line_end = section_data['file_lines']
    st.caption(f"📍 `ml_workflow.py` — Lines {line_start}–{line_end}")

with col_action:
    if section_data['runnable']:
        run_clicked = st.button("▶️ Run Full Pipeline", use_container_width=True, type="primary")
    else:
        run_clicked = False
        st.info("👁️ View Only")

st.markdown("---")

# ── Code Display ─────────────────────────────────────────────────────────────
st.markdown("#### 📝 Source Code")
st.code(section_data['code'], language="python", line_numbers=True)

# ── Run Output ───────────────────────────────────────────────────────────────
if section_data['runnable'] and run_clicked:
    st.markdown("---")
    st.markdown("#### ⚡ Execution Output")
    
    output_area = st.empty()
    progress_bar = st.progress(0, text="Initializing pipeline...")
    
    # Capture stdout during execution
    output_buffer = io.StringIO()
    
    try:
        progress_bar.progress(5, text="Loading ml_workflow module...")
        
        # Import and run the workflow
        import importlib
        
        # Add project root to path if needed
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Change to project directory for correct file paths
        original_cwd = os.getcwd()
        os.chdir(project_root)
        
        progress_bar.progress(10, text="Starting training pipeline...")
        
        # Redirect stdout to capture print output
        with contextlib.redirect_stdout(output_buffer):
            # Import fresh copy of the module
            if 'ml_workflow' in sys.modules:
                importlib.reload(sys.modules['ml_workflow'])
            import ml_workflow
            
            progress_bar.progress(15, text="Module loaded. Training models...")
            
            # Run the pipeline
            ml_workflow.run_training_pipeline()
        
        os.chdir(original_cwd)
        
        progress_bar.progress(100, text="✅ Pipeline completed!")
        
        # Display captured output
        captured = output_buffer.getvalue()
        if captured:
            st.markdown(f'<div class="run-output"><pre>{captured}</pre></div>', unsafe_allow_html=True)
        
        st.success("🎉 Training pipeline finished! Models saved and logged to MLflow.")
        
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("""
            <div style="background: rgba(16, 185, 129, 0.15); padding: 15px; border-radius: 10px; border: 1px solid #10b981;">
                <h4 style="color: #6ee7b7; margin-top: 0;">✅ Next Steps</h4>
                <ul style="color: #94a3b8;">
                    <li>View results in <a href="http://localhost:5000" target="_blank" style="color: #60a5fa;">MLflow UI</a></li>
                    <li>Go to <strong>Model Dashboard</strong> to compare metrics</li>
                    <li>Test predictions in <strong>Prediction Engine</strong></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with col_r2:
            st.markdown("""
            <div style="background: rgba(59, 130, 246, 0.15); padding: 15px; border-radius: 10px; border: 1px solid #3b82f6;">
                <h4 style="color: #93c5fd; margin-top: 0;">📦 Saved Artifacts</h4>
                <ul style="color: #94a3b8; font-size: 0.9rem;">
                    <li>pipeline_classification_best.joblib</li>
                    <li>pipeline_regression_best.joblib</li>
                    <li>label_encoder_classes.joblib</li>
                    <li>MLflow runs in sqlite:///mlflow.db</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
    except Exception as e:
        progress_bar.progress(100, text="❌ Error occurred")
        
        # Show any partial output
        captured = output_buffer.getvalue()
        if captured:
            st.markdown(f'<div class="run-output"><pre>{captured}</pre></div>', unsafe_allow_html=True)
        
        st.error(f"Pipeline execution error: {e}")
        st.code(str(e), language="text")
        
        # Restore cwd
        try:
            os.chdir(original_cwd)
        except:
            pass

# ── Full Source File Viewer ──────────────────────────────────────────────────
st.markdown("---")

with st.expander("📄 View Complete ml_workflow.py Source", expanded=False):
    try:
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "ml_workflow.py"
        )
        with open(source_path, 'r', encoding='utf-8') as f:
            full_source = f.read()
        st.code(full_source, language="python", line_numbers=True)
    except Exception as e:
        st.error(f"Could not load source file: {e}")

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.8rem;">
    💻 Code Runner — Part of the FinRisk AI Platform &nbsp;|&nbsp; 
    📊 <a href="http://localhost:5000" target="_blank" style="color: #60a5fa;">MLflow Dashboard</a>
</div>
""", unsafe_allow_html=True)
