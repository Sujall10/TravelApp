import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
import os
import numpy as np
import optuna
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


load_dotenv()
DB_URL = os.getenv("a")
engine = create_engine(DB_URL)

with engine.connect() as conn:
    df_flight = pd.read_sql("SELECT * FROM flight_oneway", conn)

# if not np.issubdtype(df_flight["flight_duration"].dtype, np.number):
#     df_flight["flight_duration"] = pd.to_timedelta(df_flight["flight_duration"]).dt.total_seconds() / 60

# df_flight["departure_date"] = pd.to_datetime(df_flight["departure_date"])
# df_flight["arrival_date"] = df_flight["departure_date"] + pd.to_timedelta(df_flight["flight_duration"], unit='m')

df_flight["departure_hour"] = df_flight["Departure_date"].dt.hour
df_flight["departure_day_of_week"] = df_flight["Departure_date"].dt.dayofweek
df_flight["departure_month"] = df_flight["Departure_date"].dt.month
df_flight["arrival_hour"] = df_flight["Arrival_Date"].dt.hour
df_flight["arrival_day_of_week"] = df_flight["Arrival_Date"].dt.dayofweek
df_flight["arrival_month"] = df_flight["Arrival_Date"].dt.month

df = df_flight.drop(columns=["travelCode", "User_ID", "flight_number", "Departure_date", "Arrival_Date"])

X = df.drop(columns=["flight_price"])
y = df["flight_price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = X.select_dtypes(include=['object']).columns.tolist()

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

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0),
    }

    model = XGBRegressor(**params, random_state=42, eval_metric="rmse", n_jobs=-1)
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_score = cross_val_score(pipeline, X_train, y_train, cv=kfold, scoring="neg_root_mean_squared_error").mean()
    
    return -cv_score  

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

best_params = study.best_params
print("Best Parameters:", best_params)

final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBRegressor(**best_params, random_state=42, eval_metric="rmse", n_jobs=-1))
])

final_pipeline.fit(X_train, y_train)

y_pred_test = final_pipeline.predict(X_test)
y_pred_train = final_pipeline.predict(X_train)

def evaluate_model(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{name} Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

evaluate_model("Training Set", y_train, y_pred_train)
evaluate_model("Test Set", y_test, y_pred_test)

joblib.dump(final_pipeline, "Price_prediction_pipeline.pkl")
print("\nPipeline saved successfully!")

