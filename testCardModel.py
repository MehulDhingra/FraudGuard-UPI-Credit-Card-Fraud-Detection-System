import joblib
import numpy as np
import pandas as pd

# Step 1: Load the saved model
model = joblib.load("xgb_fraud_model.pkl")
print("✅ Model loaded successfully")

# Step 2: Define example input
# Replace these with actual values from your dataset structure
example = {
    'Time': 100000,
    'V1': -1.359807,
    'V2': -0.072781,
    'V3': 2.536346,
    'V4': 1.378155,
    'V5': -0.338321,
    'V6': 0.462388,
    'V7': 0.239599,
    'V8': 0.098698,
    'V9': 0.363787,
    'V10': 0.090794,
    'V11': -0.551600,
    'V12': -0.617801,
    'V13': -0.991390,
    'V14': -0.311169,
    'V15': 1.468177,
    'V16': -0.470401,
    'V17': 0.207971,
    'V18': 0.025791,
    'V19': 0.403993,
    'V20': 0.251412,
    'V21': -0.018307,
    'V22': 0.277838,
    'V23': -0.110474,
    'V24': 0.066928,
    'V25': 0.128539,
    'V26': -0.189115,
    'V27': 0.133558,
    'V28': -0.021053,
    'Amount': 149.62
}

# Step 3: Convert to DataFrame and reshape
input_df = pd.DataFrame([example])

# Step 4: Predict
probability = model.predict_proba(input_df)[0][1]
prediction = model.predict(input_df)[0]

# Step 5: Print results
print(f"\n🔍 Fraud Probability: {probability:.4f}")
print(f"🔔 Prediction: {'FRAUD' if prediction == 1 else 'LEGITIMATE'}")
