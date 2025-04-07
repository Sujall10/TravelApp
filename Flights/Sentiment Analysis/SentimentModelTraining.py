import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
import re
import string
import joblib  # For saving the model
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine
import nltk


# Load data
positive_reviews = pd.read_csv('notebooks\\flight_positive_reviews.csv')
neutral_reviews = pd.read_csv('notebooks\\flight_neutral_reviews.csv')
negative_reviews = pd.read_csv('notebooks\\flight_negative_reviews.csv')

# Rename columns
positive_reviews.rename(columns={'positive reviews': 'Reviews'}, inplace=True)
neutral_reviews.rename(columns={'neutral reviews': 'Reviews'}, inplace=True)
negative_reviews.rename(columns={'negative reviews': 'Reviews'}, inplace=True)



# Merge data
merged_df = pd.concat([positive_reviews, neutral_reviews, negative_reviews], ignore_index=True)
merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Define a preprocessing function
def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = word_tokenize(text)  # Tokenize
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(words)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    merged_df['Reviews'],
    merged_df['sentiment'],
    test_size=0.2,
    random_state=42
)
def preprocess_reviews(x):
    return [preprocess_text(str(doc)) for doc in x]  # Convert to string before processing


# Create a pipeline with preprocessing, vectorization, and your model
pipeline = Pipeline([
    ('preprocessing', FunctionTransformer(preprocess_reviews)),
    ('tfidf', TfidfVectorizer()),
    ('classifier', LogisticRegression())
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = pipeline.predict(X_test)
y = pipeline.predict(X_train)
accuracy = accuracy_score(y_test, y_pred)
acc = accuracy_score(y_train, y)
print(f"Testing TF-IDF + Logistic Regression Accuracy: {accuracy:.4f}")
print(f"Training TF-IDF + Logistic Regression Accuracy: {acc:.4f}")

# Save the pipeline for future use
# joblib.dump(pipeline, 'Sentiment_Analysis_Pipeline.pkl')