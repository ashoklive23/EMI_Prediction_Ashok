import joblib
import sys
import numpy as np
import pandas as pd

def inspect(path, f):
    f.write(f"\n--- Inspecting {path} ---\n")
    try:
        model = joblib.load(path)
        f.write(f"Type: {type(model)}\n")
        
        # Check pipeline feature names
        if hasattr(model, 'feature_names_in_'):
            f.write("Pipeline/Model feature_names_in_ found:\n")
            for name in model.feature_names_in_:
                f.write(f"{name}\n")
        
        # If pipeline, verify steps and estimator features too
        if hasattr(model, 'steps'):
            f.write(f"Pipeline steps: {list(model.named_steps.keys())}\n")
            estimator = model.steps[-1][1]
            if hasattr(estimator, 'feature_names_in_'):
                f.write("Estimator feature_names_in_ details (first 5):\n")
                for name in list(estimator.feature_names_in_)[:5]:
                    f.write(f"{name}\n")
            
        # Label encoder check
        if isinstance(model, np.ndarray):
            f.write(f"Numpy Array Content: {model}\n")
        elif isinstance(model, list):
             f.write(f"List Content: {model}\n")
        elif hasattr(model, 'classes_'):
             f.write(f"Classes: {model.classes_}\n")
             
    except Exception as e:
        f.write(f"Error loading {path}: {e}\n")

if __name__ == "__main__":
    with open('model_features.txt', 'w', encoding='utf-8') as f:
        inspect('pipeline_classification_xgb_tuned_fast.joblib', f)
        inspect('pipeline_regression_xgb_focused.joblib', f)
        inspect('label_encoder_classes.joblib', f)
