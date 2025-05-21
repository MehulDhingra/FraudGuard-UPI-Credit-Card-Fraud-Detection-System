import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

# Initialize
fake = Faker('en_IN')

# Constants
bank_suffixes = ['@okaxis', '@oksbi', '@okhdfcbank', '@okicici', '@paytm', '@upi']
device_types = ['Android', 'iOS', 'Windows', 'Mac', 'Linux']
transaction_types = ['P2P', 'Merchant', 'Bill', 'Recharge', 'QR Scan', 'Bank Transfer']
notes_pool = [
    "Rent for May", "Dinner payment", "Electricity Bill", "Phone Recharge", "For books",
    "Gift", "Birthday", "EMI", "Subscription", "Loan repayment", "", "", "", "", "", "Test","Loan",""
]

# Helper functions
def generate_upi_id(name):
    username = name.lower().replace(' ', '') + str(random.randint(10, 99))
    return username + random.choice(bank_suffixes)

def generate_transaction_id():
    return "TXN" + str(random.randint(1000000000, 9999999999))

def generate_device_id():
    return "DEV" + ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=10))

def generate_timestamp():
    start = datetime.now() - timedelta(days=180)
    random_days = random.randint(0, 180)
    random_time = timedelta(seconds=random.randint(0, 86400))
    return start + timedelta(days=random_days) + random_time

def is_fraudulent(transaction_type, amount, time_hour):
    if transaction_type == 'QR Scan' and amount > 20000:
        return 1
    if transaction_type == 'P2P' and amount > 50000:
        return 1
    if 1 <= time_hour <= 4 and amount > 10000:
        return 1
    if random.random() < 0.01:  # 1% random frauds
        return 1
    return 0

# Dataset generation
num_records = 100000
data = []

for _ in range(num_records):
    sender_name = fake.name()
    receiver_name = fake.name()
    txn_time = generate_timestamp()
    txn_hour = txn_time.hour
    amount = round(random.uniform(1, 100000), 2)
    txn_type = random.choice(transaction_types)

    entry = {
        'Transaction ID': generate_transaction_id(),
        'Timestamp': txn_time.strftime('%Y-%m-%d %H:%M:%S'),
        'Sender Name': sender_name,
        'Sender UPI ID': generate_upi_id(sender_name),
        'Receiver Name': receiver_name,
        'Receiver UPI ID': generate_upi_id(receiver_name),
        'Amount (INR)': amount,
        'Transaction Type': txn_type,
        'Note': random.choice(notes_pool),
        'Device Type': random.choice(device_types),
        'Device ID': generate_device_id(),
        'Fraud': is_fraudulent(txn_type, amount, txn_hour)
    }

    data.append(entry)

# Create DataFrame
df_upi = pd.DataFrame(data)

# Save to CSV
df_upi.to_csv('synthetic_upi_transactions.csv', index=False)
print(" Synthetic UPI dataset generated and saved to 'synthetic_upi_transactions.csv'")

# Preview
df_upi.head()
