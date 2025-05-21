<div align="center">FraudGuard 🔐</div>
<div align="center">A Dual Fraud Detection System for UPI & Credit Card Transactions</div><br>
FraudGuard is a robust and intelligent system for detecting fraudulent transactions across Unified Payments Interface (UPI) and Credit Card platforms. It utilizes an unsupervised autoencoder ensemble for UPI transactions and a supervised XGBoost classifier for credit card fraud detection. The models are trained and fine-tuned to maximize precision, recall, and overall fraud capture accuracy.

Features 📃
<ul> <li>Detects UPI fraud using Autoencoder-based anomaly detection</li> <li>Detects Credit Card fraud using supervised XGBoost classifier</li> <li>Smart feature engineering including: <ul> <li>User transaction velocity</li> <li>Suspicious note pattern recognition</li> <li>Device and user behavior profiling</li> <li>Time-based and contextual features</li> </ul> </li> <li>Precision-Recall AUC based threshold optimization (Youden's J)</li> <li>Bootstrapped ensemble for generalization and stability</li> <li>Comprehensive metric tracking: <ul> <li>F1-Score</li> <li>ROC AUC</li> <li>Precision & Recall</li> <li>PR AUC</li> </ul> </li> </ul>
Technology Used 💻
<ul> <li><strong>IDE:</strong> Visual Studio Code</li> <li><strong>UPI Model:</strong> Autoencoder (TensorFlow, Keras)</li> <li><strong>Credit Card Model:</strong> XGBoost</li> <li><strong>Thresholding:</strong> PR AUC Optimization</li> <li><strong>Feature Engineering:</strong> Pandas, NumPy</li> <li><strong>Evaluation:</strong> Scikit-learn, Matplotlib</li> <li><strong>Model Serialization:</strong> Joblib, Pickle</li> <li><strong>Version Control:</strong> Git, GitHub</li> </ul>
Results 📊
<ul> <li><strong>UPI Autoencoder Ensemble:</strong> <ul> <li>F1-Score: <strong>0.8065</strong></li> <li>ROC AUC: <strong>0.92</strong></li> <li>PR AUC: <strong>0.9076</strong></li> </ul> </li> <li><strong>Credit Card XGBoost:</strong> <ul> <li>F1-Score: ~<strong>0.91</strong></li> <li>ROC AUC: > <strong>0.95</strong></li> <li>PR AUC: ~<strong>0.95</strong></li> </ul> </li> </ul>
To Run the Project 🚀
shell
Copy
Edit
pip install -r requirements.txt
python finalTesting.py
You’ll see output like:

yaml
Copy
Edit
✅ Loaded UPI fraud detection model.
✅ Loaded Credit Card fraud detection model.

🧪 Testing sample UPI transaction...
Prediction: FRAUD 🚨 | Score: 0.000315

🧪 Testing sample Credit Card transaction...
Prediction: NOT FRAUD ✅ | Probability: 0.0217
Project Structure 🗂
<ul> <li><strong>finalTesting.py</strong> - Main script for both models</li> <li><strong>upi_autoencoder_model.h5</strong> - Trained UPI model</li> <li><strong>xgb_fraud_model.pkl</strong> - Trained Credit Card model</li> <li><strong>utils_preprocessing.py</strong> - Feature transformers</li> <li><strong>threshold_optimizer.py</strong> - Threshold tuner using PR AUC</li> <li><strong>requirements.txt</strong> - All required Python packages</li> </ul>
Why FraudGuard? 🤔
<ul> <li><strong>Dual-Mode Detection:</strong> Supports UPI and Credit Card transactions</li> <li><strong>Unsupervised + Supervised Models:</strong> Best of both ML worlds</li> <li><strong>Better Accuracy:</strong> Optimized for fraud recall & precision</li> <li><strong>Behavioral Profiling:</strong> Goes beyond just transaction values</li> <li><strong>Customizable & Scalable:</strong> Easily extend to wallets, NEFT, IMPS</li> </ul>
Contribution 🤝
Fork the repo, improve the logic or add new features, and create a pull request!

<div align="center">🛡️ Built to protect your digital payments. Safe. Accurate. Scalable. 🛡️</div>
