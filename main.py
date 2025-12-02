import pandas as pd
import joblib
import os
from src.pipeline import DepressionAnalysisSystem

# Constants
DATA_PATH = 'dataset/clean_depression_dataset.csv'
MODEL_PATH = 'model/full_depression_model.joblib'

def main():
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    print(f"Loading data from {DATA_PATH}...")
    try:
        data = pd.read_csv(DATA_PATH)
        print(f"Data loaded. Shape: {data.shape}")
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # 2. Load or Train Model
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}...")
        try:
            system = joblib.load(MODEL_PATH)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {e}")
            return
    else:
        print("Model not found. Training new model...")
        system = DepressionAnalysisSystem()
        try:
            system.fit(data)
            print("Model training completed.")
            
            # Save Model
            print(f"Saving model to {MODEL_PATH}...")
            joblib.dump(system, MODEL_PATH)
            print("Model saved.")
        except Exception as e:
            print(f"Model training failed: {e}")
            return

    # 3. Prediction
    print("Generating predictions...")
    try:
        predictions = system.predict(data)
        print("Predictions generated.")
        print(predictions.head())
        
        print("Generating probabilities...")
        probs = system.predict_proba(data)
        print("Probabilities generated.")
        print(probs.head())
        
    except Exception as e:
        print(f"Prediction failed: {e}")

if __name__ == "__main__":
    main()
