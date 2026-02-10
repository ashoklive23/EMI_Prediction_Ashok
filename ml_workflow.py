import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
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
MLFLOW_EXPERIMENT_NAME = "EMI_Prediction_Experiment"
ARTIFACT_PATH = "models"

# --- Feature Engineering ---
def engineer_features(df):
    """
    Applies domain-specific feature engineering.
    """
    # Data Cleaning and Type Conversion
    numeric_cols_to_clean = [
        'monthly_salary', 'monthly_rent', 'school_fees', 'college_fees',
        'travel_expenses', 'groceries_utilities', 'other_monthly_expenses', 'current_emi_amount'
    ]
    
    for col in numeric_cols_to_clean:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Avoid division by zero
    df['monthly_salary'] = df['monthly_salary'].replace(0, 1)
    
    # 1. Total Monthly Expenses
    df['total_expenses'] = (
        df['monthly_rent'] + 
        df['school_fees'] + 
        df['college_fees'] + 
        df['travel_expenses'] + 
        df['groceries_utilities'] + 
        df['other_monthly_expenses']
    )
    
    # 2. Financial Ratios
    df['debt_to_income_ratio'] = df['current_emi_amount'] / df['monthly_salary']
    df['expense_to_income_ratio'] = df['total_expenses'] / df['monthly_salary']
    df['savings_ratio'] = (df['monthly_salary'] - df['total_expenses'] - df['current_emi_amount']) / df['monthly_salary']
    
    return df

# --- Preprocessing Pipeline ---
def get_preprocessor(numeric_features, categorical_features):
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

# --- Training Workflow ---
def run_training_pipeline():
    print("Starting ML Workflow...")
    
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    print(f"Loading data from {DATA_PATH}...")
    try:
        # Load a smaller subset for fast dev/demo cycle
        # Increase this to None for full production training
        print("Loading subset of data for fast training...")
        df = pd.read_csv(DATA_PATH, low_memory=False, nrows=10000) 
        print(f"Data Loaded: {df.shape}")
    except Exception as e:
        print(f"Failed to read data: {e}")
        return

    # 2. Feature Engineering
    print("Engineering features...")
    df = engineer_features(df)

    # 3. Define Features
    target_cls = 'emi_eligibility'
    target_reg = 'max_monthly_emi'
    
    # Drop targets and non-informative columns
    drop_cols = [target_cls, target_reg]
    X = df.drop(columns=drop_cols, errors='ignore')
    y_cls = df[target_cls]
    y_reg = df[target_reg]

    # Identify column types
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # 4. Train-Test Split
    X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
        X, y_cls, y_reg, test_size=0.2, random_state=42
    )

    # 5. MLflow Setup
    try:
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    except:
        print("MLflow experiment setup failed, continuing locally...")

    # --- Classification Task (3 Models) ---
    print("\n--- Training Classification Models ---")
    
    # Encode Target
    le = LabelEncoder()
    y_cls_train_enc = le.fit_transform(y_cls_train)
    y_cls_test_enc = le.transform(y_cls_test)
    joblib.dump(le.classes_, 'label_encoder_classes.joblib')

    clf_models = {
        "Logistic_Regression": LogisticRegression(max_iter=500, random_state=42),
        "Random_Forest_Clf": RandomForestClassifier(n_estimators=20, max_depth=10, random_state=42, n_jobs=-1),
        "XGBoost_Clf": XGBClassifier(n_estimators=50, max_depth=6, use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_jobs=-1)
    }

    best_clf_name = None
    best_clf_score = -1
    best_clf_pipeline = None

    for name, model in clf_models.items():
        print(f"Training {name}...")
        with mlflow.start_run(run_name=f"CLF_{name}"):
            pipeline = ImbPipeline(steps=[
                ('preproc', get_preprocessor(numeric_features, categorical_features)),
                ('oversample', SMOTE(random_state=42)),
                ('clf', model)
            ])
            
            pipeline.fit(X_train, y_cls_train_enc)
            y_pred = pipeline.predict(X_test)
            
            # Metrics
            acc = accuracy_score(y_cls_test_enc, y_pred)
            f1 = f1_score(y_cls_test_enc, y_pred, average='weighted')
            
            print(f"  {name} Accuracy: {acc:.4f}")
            
            mlflow.log_param("model_type", name)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            mlflow.sklearn.log_model(pipeline, "model")
            
            # Save local copy for reference
            joblib.dump(pipeline, f'model_clf_{name.lower()}.joblib')
            
            if acc > best_clf_score:
                best_clf_score = acc
                best_clf_name = name
                best_clf_pipeline = pipeline

    print(f"✅ Best Classification Model: {best_clf_name} (Acc: {best_clf_score:.4f})")
    joblib.dump(best_clf_pipeline, 'pipeline_classification_best.joblib')

    # --- Regression Task (3 Models) ---
    print("\n--- Training Regression Models ---")
    
    # Add eligibility to features for regression
    X_train_reg = X_train.copy()
    X_train_reg['emi_eligibility'] = y_cls_train
    X_test_reg = X_test.copy()
    X_test_reg['emi_eligibility'] = y_cls_test
    cat_features_reg = categorical_features + ['emi_eligibility']

    reg_models = {
        "Linear_Regression": LinearRegression(),
        "Random_Forest_Reg": RandomForestRegressor(n_estimators=20, max_depth=10, random_state=42, n_jobs=-1),
        "XGBoost_Reg": XGBRegressor(n_estimators=50, max_depth=6, objective='reg:squarederror', random_state=42, n_jobs=-1)
    }

    best_reg_name = None
    best_reg_score = float('inf') # Minimizing RMSE
    best_reg_pipeline = None

    for name, model in reg_models.items():
        print(f"Training {name}...")
        with mlflow.start_run(run_name=f"REG_{name}"):
            pipeline = Pipeline(steps=[
                ('pre', get_preprocessor(numeric_features, cat_features_reg)),
                ('model', model)
            ])
            
            pipeline.fit(X_train_reg, y_reg_train)
            preds = pipeline.predict(X_test_reg)
            
            # Metrics
            rmse = np.sqrt(mean_squared_error(y_reg_test, preds))
            r2 = r2_score(y_reg_test, preds)
            mae = mean_absolute_error(y_reg_test, preds)

            print(f"  {name} RMSE: {rmse:.2f}")
            
            mlflow.log_param("model_type", name)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)
            mlflow.sklearn.log_model(pipeline, "model")
            
            # Save local copy for reference
            joblib.dump(pipeline, f'model_reg_{name.lower()}.joblib')
            
            if rmse < best_reg_score:
                best_reg_score = rmse
                best_reg_name = name
                best_reg_pipeline = pipeline

    print(f"✅ Best Regression Model: {best_reg_name} (RMSE: {best_reg_score:.2f})")
    joblib.dump(best_reg_pipeline, 'pipeline_regression_best.joblib')

    print("\nWorkflow Completed successfully.")

if __name__ == "__main__":
    run_training_pipeline()
