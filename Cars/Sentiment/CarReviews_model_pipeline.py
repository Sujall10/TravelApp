import re
import numpy as np
import pandas as pd
import joblib
import pickle
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv


class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(self.clean_text)

    @staticmethod
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'\W', ' ', text)  
        text = re.sub(r'\s+', ' ', text).strip()  
        return text

vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))


best_model = joblib.load("random_forest.pkl")  


sentiment_pipeline = Pipeline([
    ('text_cleaning', TextCleaner()),  
    ('vectorizer', vectorizer),  
    ('classifier', best_model)  
])

joblib.dump(sentiment_pipeline, "sentiment.pkl")
