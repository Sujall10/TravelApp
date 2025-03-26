import os
import joblib
import numpy as np
import pandas as pd
import optuna
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

# Load environment variables
load_dotenv()
db_url = os.getenv("a")
engine = create_engine(db_url)

# Load dataset from database
with engine.connect() as conn:
    df_flight = pd.read_sql("SELECT * FROM flight_RoundTrip", conn)

# Convert date columns to datetime format
date_cols = ['departure_date', 'Arrival_Date', 'ReturnDeparture_Date', 'ReturnArrival_Date']
for col in date_cols:
    df_flight[col] = pd.to_datetime(df_flight[col], errors='coerce')

# Feature engineering
df_flight['flight_duration'] = (df_flight['Arrival_Date'] - df_flight['departure_date']).dt.total_seconds() / 3600  # Duration in hours

for col in date_cols:
    df_flight[f"{col}_hour"] = df_flight[col].dt.hour
    df_flight[f"{col}_day_of_week"] = df_flight[col].dt.dayofweek
    df_flight[f"{col}_month"] = df_flight[col].dt.month

# Drop unnecessary columns
drop_columns = ['travelcode', 'user_id'] + date_cols
df_flight.drop(columns=drop_columns, inplace=True)

# Define features and target
X = df_flight.drop(columns=['flight_price'])
y = df_flight['flight_price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numerical and categorical features
num_features = X.select_dtypes(include=['int64', 'int32', 'float64']).columns.tolist()
cat_features = X.select_dtypes(include=['object']).columns.tolist()

# Data transformation pipelines
num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
]) if cat_features else 'passthrough'

preprocessor = ColumnTransformer([
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

# Hyperparameter tuning function
def tune_model(trial, model_type):
    param_grid = {
        "random_forest": {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5)
        },
        "xgboost": {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        },
        "lightgbm": {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        }
    }
    
    models = {
        "random_forest": RandomForestClassifier(**param_grid[model_type], random_state=42),
        "xgboost": XGBClassifier(**param_grid[model_type], random_state=42, eval_metric='mlogloss'),
        "lightgbm": LGBMClassifier(**param_grid[model_type], random_state=42)
    }
    
    model = models[model_type]
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean()
    return score

# Function to optimize models
def optimize_model(model_type, trials=30):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: tune_model(trial, model_type), n_trials=trials)
    return study.best_params

# Tune all models
best_params = {model: optimize_model(model) for model in ["random_forest", "xgboost", "lightgbm"]}

# Define best models
rf_best = RandomForestClassifier(**best_params['random_forest'], random_state=42)
xgb_best = XGBClassifier(**best_params['xgboost'], random_state=42, eval_metric='mlogloss')
lgb_best = LGBMClassifier(**best_params['lightgbm'], random_state=42)

# Stacking classifier
stacking_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", StackingClassifier(
        estimators=[("rf", rf_best), ("xgb", xgb_best), ("lgb", lgb_best)],
        final_estimator=LogisticRegression()
    ))
])

# Train and evaluate
stacking_model.fit(X_train, y_train)
y_pred = stacking_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Stacking Model Accuracy: {accuracy:.4f}")
