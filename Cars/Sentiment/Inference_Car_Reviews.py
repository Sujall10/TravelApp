import joblib
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import pandas as pd
from CarReviews_model_pipeline import TextCleaner

# Load the trained model
sentiment = joblib.load("sentiment.pkl")
# print(sentiment)

# Load environment variables
load_dotenv()
DB_URL = os.getenv("DB_URL")

# Establish database connection
engine = create_engine(DB_URL)

with engine.connect() as conn:
    inference_Review_car = pd.read_sql('SELECT * FROM Car_review', conn)


def predict_sentiment(review_text):
    review_df = pd.DataFrame({"review": [review_text]})  # Convert to DataFrame
    prediction = sentiment.predict(review_df["review"])[0]  # Predict
    sentiment_labels = {1: "Positive", 0: "Negative", 2: "Neutral"}
    return sentiment_labels[prediction]

inference_Review_car["Predicted_Sentiment"] = inference_Review_car["Review"].apply(lambda x: predict_sentiment(x))

print(inference_Review_car)

