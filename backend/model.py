import joblib
import numpy as np

# Load your pre-trained model
model = joblib.load("models/your_model.pkl")

def predict(data: pd.DataFrame) -> np.ndarray:
    # Make predictions
    predictions = model.predict(data)
    return predictions