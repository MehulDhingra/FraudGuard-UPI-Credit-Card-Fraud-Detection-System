# predictUPI.py
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from Upi_fraud_model_class import FraudDetectionEnsemble  # Important for unpickling

# === Load saved model ===
with open('final_fraud_model.pkl', 'rb') as f:
    model = pickle.load(f)

# === Example Single Input Transaction ===
input_tx = {
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

# === Convert to DataFrame ===
df = pd.DataFrame([input_tx])
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# === Minimal feature engineering ===
df['TimeSinceLastTX'] = 0  # No history
df['Hour'] = df['Timestamp'].dt.hour
df['AvgAmountSender'] = df['Amount (INR)']
df['AvgAmountDevice'] = df['Amount (INR)']
df['Note'] = df['Note'].fillna('')
df['NoteFreq'] = 1  # Assume note appears once

# === One-hot encode Transaction Type ===
trans_type_ohe = model.ohe.transform(df[['Transaction Type']])
trans_type_df = pd.DataFrame(trans_type_ohe, columns=model.ohe.get_feature_names_out(['Transaction Type']))
df = pd.concat([df.reset_index(drop=True), trans_type_df.reset_index(drop=True)], axis=1)

# === Drop unused columns ===
df.drop(columns=[
    'Transaction ID', 'Timestamp', 'Sender Name', 'Receiver Name',
    'Sender UPI ID', 'Receiver UPI ID', 'Note',
    'Device Type', 'Device ID', 'Transaction Type'
], inplace=True, errors='ignore')

# === Predict using model ===
pred, error = model.predict(df)

# === Output ===
print(f"Reconstruction Error: {error[0]:.6f}")
print(f"Prediction: {'🚨 FRAUD' if pred[0] == 1 else '✅ GENUINE'}")

