# fraud_model.py

import numpy as np
import pandas as pd

class FraudDetectionEnsemble:
    def __init__(self, models, scaler, vt, ohe, threshold):
        self.models = models
        self.scaler = scaler
        self.vt = vt
        self.ohe = ohe
        self.threshold = threshold

    def predict(self, X_raw):
        # Apply VarianceThreshold
        X_clean = pd.DataFrame(self.vt.transform(X_raw))
        
        # Standard scaling
        X_scaled = self.scaler.transform(X_clean)

        # Ensemble reconstruction errors
        errors = []
        for model in self.models:
            pred = model.predict(X_scaled)
            err = np.mean(np.square(X_scaled - pred), axis=1)
            errors.append(err)

        avg_error = np.mean(errors, axis=0)
        predictions = (avg_error > self.threshold).astype(int)
        return predictions, avg_error
