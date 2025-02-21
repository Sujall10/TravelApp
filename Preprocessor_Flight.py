import joblib
import pandas as pd

# Load preprocessing pipeline
preprocessor_Flight = joblib.load("preprocessor_Flight.pkl")

def preprocess_input(data):
    if isinstance(data, dict):  
        df = pd.DataFrame([data])  
    else:
        df = pd.DataFrame(data)  
    
    return preprocessor_Flight.transform(df)