import joblib
import pandas as pd

# Load preprocessing pipeline
preprocessor = joblib.load("preprocessor_Car.pkl")

def preprocess_input(data):
    df = pd.DataFrame(data)
    return preprocessor.transform(df)
