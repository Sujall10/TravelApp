import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

load_dotenv()
DB_URL = os.getenv("a")
engine = create_engine(DB_URL)

with engine.connect() as conn:
    df_flight = pd.read_sql("SELECT * FROM flight_RoundTrip", conn)

df_flight['departure_date'] = pd.to_datetime(df_flight['departure_date'])
df_flight['Arrival_Date'] = pd.to_datetime(df_flight['Arrival_Date'])

df_flight['flight_duration_seconds'] = (df_flight['Arrival_Date'] - df_flight['departure_date']).dt.total_seconds()
df_flight.drop(columns=['departure_date', 'Arrival_Date'], inplace=True)

X = df_flight.drop(columns=['flight_price'])  
y = df_flight['flight_price']  

num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = X.select_dtypes(include=['object']).columns.tolist()

num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

def tune_model(trial, model_type):
    if model_type == "random_forest":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        }
        model = RandomForestRegressor(**params, random_state=42)
    elif model_type == "xgboost":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        }
        model = XGBRegressor(**params, random_state=42)
    
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    return scores.mean()

best_params = {}
for model_name in ["random_forest", "xgboost"]:
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: tune_model(trial, model_name), n_trials=30)
    best_params[model_name] = study.best_params
    print(f"Best {model_name} parameters: {best_params[model_name]}")

rf_best = RandomForestRegressor(**best_params["random_forest"], random_state=42)
xgb_best = XGBRegressor(**best_params["xgboost"], random_state=42)

def tune_final_estimator(trial):
    params = {
        'alpha': trial.suggest_float('alpha', 0.1, 10)
    }
    final_estimator = Ridge(**params)
    
    stacking_model = StackingRegressor(
        estimators=[("rf", rf_best), ("xgb", xgb_best)],
        final_estimator=final_estimator
    )
    
    score = cross_val_score(stacking_model, X_train, y_train, cv=5, scoring='r2').mean()
    return score

study = optuna.create_study(direction="maximize")
study.optimize(tune_final_estimator, n_trials=30)

best_params_final_estimator = study.best_params
print(f"Best final estimator parameters: {best_params_final_estimator}")

final_estimator_optimized = Ridge(**best_params_final_estimator)

stacking_model = StackingRegressor(
    estimators=[("rf", rf_best), ("xgb", xgb_best)],
    final_estimator=final_estimator_optimized
)

stacking_model.fit(X_train, y_train)

y_pred_test = stacking_model.predict(X_test)
y_pred_train = stacking_model.predict(X_train)

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
