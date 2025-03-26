import numpy as np
import pandas as pd
import random
import re
import joblib
import pickle
import xgboost as xgb
import tensorflow as tf
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.stats import uniform, randint

#loading dataset
positive_reviews = pd.read_csv('data//car_positive_reviews.csv')
neutral_reviews = pd.read_csv('data//car_neutral_reviews.csv')
negative_reviews = pd.read_csv('data//car_negative_reviews.csv')
positive_reviews.rename(columns={'positive reviews':'review'},inplace=True)
neutral_reviews.rename(columns={'neutral reviews':'review'},inplace=True)
negative_reviews.rename(columns={'negative reviews':'review'},inplace=True)
merged_df = pd.concat([positive_reviews,neutral_reviews,negative_reviews],ignore_index=True)

merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
print(merged_df)

# Text Cleaning Function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Sklearn Pipeline
text_pipeline = Pipeline([
    ('text_cleaning', FunctionTransformer(lambda x: x.apply(clean_text))),  # Apply cleaning
    ('vectorizer', TfidfVectorizer())  # Convert text to vectors
])

# Transforming the reviews using the pipeline
X_transformed = text_pipeline.fit_transform(merged_df["review"])



# Encoding Sentiment Labels
sentiment_mapping = {"positive": 1, "negative": 0, "neutral": 2}
merged_df["sentiment"] = merged_df["sentiment"].map(sentiment_mapping)

# Splitting Data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(merged_df["cleaned_review"], merged_df["sentiment"], test_size=0.2, random_state=42, stratify=merged_df["sentiment"])

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=50000)  # Limits vocab size for efficiency
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Saving vectorizer
pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))

#  Function to Evaluate Models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return {"Accuracy": acc, "MAE": mae, "RMSE": rmse, "R2 Score": r2}

#  Model 1: Random Forest with Hyperparameter Tuning
param_dist_rf = {
    'n_estimators': randint(100, 300),
    'max_depth': [20, 30, None],
    'min_samples_split': [2, 5, 10]
}

rf = RandomizedSearchCV(RandomForestClassifier(), param_dist_rf, cv=3, scoring='accuracy', n_jobs=-1, n_iter=10, random_state=42)
rf.fit(X_train_tfidf, y_train)

# Saving Random Forest model
joblib.dump(rf.best_estimator_, "random_forest.pkl")

# Evaluating Random Forest
rf_results = evaluate_model(rf.best_estimator_, X_test_tfidf, y_test)
print("Random Forest Results:", rf_results)

# Model 2: XGBoost with Hyperparameter Tuning
param_dist_xgb = {
    'n_estimators': randint(100, 500),
    'learning_rate': uniform(0.01, 0.3),
    'max_depth': randint(3, 10)
}

xgb_model = RandomizedSearchCV(xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss"), param_dist_xgb, cv=3, scoring='accuracy', n_iter=10, n_jobs=-1, random_state=42)
xgb_model.fit(X_train_tfidf, y_train)

# Saving XGBoost model
joblib.dump(xgb_model.best_estimator_, "xgboost.pkl")

# Evaluating XGBoost
xgb_results = evaluate_model(xgb_model.best_estimator_, X_test_tfidf, y_test)
print("XGBoost Results:", xgb_results)

# Model 3: Deep Learning Model (TensorFlow)
# tokenizer = Tokenizer(num_words=50000, oov_token="<OOV>")
# tokenizer.fit_on_texts(X_train)

# X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=100, padding="post")
# X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=100, padding="post")

# # Building Deep Learning Model
# tf_model = Sequential([
#     Embedding(input_dim=50000, output_dim=32, input_length=100),
#     GlobalAveragePooling1D(),
#     Dense(64, activation="relu"),
#     Dense(3, activation="softmax")  # 3 output classes (Positive, Negative, Neutral)
# ])

# tf_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# tf_model.fit(X_train_seq, y_train, epochs=10, batch_size=256, validation_data=(X_test_seq, y_test))

# # Saving TensorFlow Model
# tf_model.save("tf_sentiment_model.h5")
# pickle.dump(tokenizer, open("tokenizer.pkl", "wb"))

# # Evaluating Deep Learning Model
# y_pred_dl = np.argmax(tf_model.predict(X_test_seq), axis=1)
# dl_acc = accuracy_score(y_test, y_pred_dl)
# dl_mae = mean_absolute_error(y_test, y_pred_dl)
# dl_rmse = np.sqrt(mean_squared_error(y_test, y_pred_dl))
# dl_r2 = r2_score(y_test, y_pred_dl)

# dl_results = {"Accuracy": dl_acc, "MAE": dl_mae, "RMSE": dl_rmse, "R2 Score": dl_r2}
# print("Deep Learning Model Results:", dl_results)

#  Comparing Model Performance
model_comparisons = {
    "Random Forest": rf_results,
    "XGBoost": xgb_results
    # "Deep Learning (TF)": dl_results
}

best_model = max(model_comparisons, key=lambda x: model_comparisons[x]["Accuracy"])
print("\n🔹 Best Model:", best_model, "with accuracy:", model_comparisons[best_model]["Accuracy"])

