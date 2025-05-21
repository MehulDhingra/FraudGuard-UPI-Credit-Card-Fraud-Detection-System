import pandas as pd
import pickle
import joblib
from datetime import datetime

# === Import your custom class BEFORE loading pickle model for UPI ===
from Upi_fraud_model_class import FraudDetectionEnsemble

def predict_upi_fraud(model, input_tx):
    """
    Given a loaded FraudDetectionEnsemble model and a single UPI transaction dict,
    return fraud prediction and reconstruction error.
    """
    df = pd.DataFrame([input_tx])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Minimal feature engineering (adapt based on your actual model's needs)
    df['TimeSinceLastTX'] = 0
    df['Hour'] = df['Timestamp'].dt.hour
    df['AvgAmountSender'] = df['Amount (INR)']
    df['AvgAmountDevice'] = df['Amount (INR)']
    df['Note'] = df['Note'].fillna('')
    df['NoteFreq'] = 1
    
    # One-hot encode Transaction Type using model's encoder
    trans_type_ohe = model.ohe.transform(df[['Transaction Type']])
    trans_type_df = pd.DataFrame(trans_type_ohe, columns=model.ohe.get_feature_names_out(['Transaction Type']))
    df = pd.concat([df.reset_index(drop=True), trans_type_df.reset_index(drop=True)], axis=1)
    
    # Drop unused columns
    df.drop(columns=[
        'Transaction ID', 'Timestamp', 'Sender Name', 'Receiver Name',
        'Sender UPI ID', 'Receiver UPI ID', 'Note',
        'Device Type', 'Device ID', 'Transaction Type'
    ], inplace=True, errors='ignore')
    
    # Predict
    pred, error = model.predict(df)
    return pred[0], error[0]

def predict_card_fraud(model, input_features):
    """
    Given a loaded XGBoost model and a dict of features for a credit card transaction,
    return fraud probability and prediction.
    """
    df = pd.DataFrame([input_features])
    probability = model.predict_proba(df)[0][1]
    prediction = model.predict(df)[0]
    return prediction, probability

def main():
    print("=== Loading Models ===")
    # Load UPI fraud detection ensemble (pickle)
    
    with open('final_fraud_model.pkl', 'rb') as f:
        upi_model = pickle.load(f)
    print("✅ Loaded UPI fraud detection model.")
    
    # Load credit card fraud XGBoost model (joblib)
    
    card_model = joblib.load("xgb_fraud_model.pkl")
    print("✅ Loaded Credit Card fraud detection model.\n")
    
    # Example UPI transaction input
    upi_tx = {
        'Transaction ID': 'TXN6008975948',
        'Timestamp': '2025-02-11 21:07',
        'Sender Name': 'Jeevika Mukhopadhyay',
        'Receiver Name': 'Ekbal Chokshi',
        'Sender UPI ID': 'jeevikamukhopadhyay66@oksbi',
        'Receiver UPI ID': 'ekbalchokshi52@upi',
        'Amount (INR)': 1500,
        'Note': 'Rent',
        'Device Type': 'Android',
        'Device ID': 'DEVXMZSWU6U9I',
        'Transaction Type': 'P2P',
    }
    
    # Example Credit Card transaction input (features based on dataset)
    card_tx = {
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
    
    print("=== Running Fraud Predictions ===\n")
    
    # UPI prediction
    upi_pred, upi_err = predict_upi_fraud(upi_model, upi_tx)
    print("💳 UPI Transaction Fraud Detection")
    print(f"Transaction ID: {upi_tx['Transaction ID']}")
    print(f"Reconstruction Error: {upi_err:.6f}")
    print(f"Prediction: {'🚨 FRAUD' if upi_pred == 1 else '✅ GENUINE'}\n")
    
    # Credit Card prediction
    card_pred, card_prob = predict_card_fraud(card_model, card_tx)
    print("💳 Credit Card Transaction Fraud Detection")
    print(f"Fraud Probability: {card_prob:.4f}")
    print(f"Prediction: {'🚨 FRAUD' if card_pred == 1 else '✅ LEGITIMATE'}\n")

if __name__ == "__main__":
    main()
