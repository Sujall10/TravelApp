import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


load_dotenv()
DB_URL = os.getenv("a")
engine = create_engine(DB_URL)

# Load Data
with engine.connect() as conn:
    df_flight = pd.read_sql("SELECT * FROM flight_oneway", conn)

# df_flight['flight_duration'] = pd.to_timedelta(df_flight['flight_duration']).dt.total_seconds() / 60  
df_flight['price_per_km'] = df_flight['flight_price'] / df_flight['flight_distance']

# df_flight['departure_date'] = pd.to_datetime(df_flight['departure_date'])
df_flight['flight_duration'] = pd.to_timedelta(df_flight['flight_duration'])

df_flight['arrival_date'] = df_flight['departure_date'] + df_flight['flight_duration']

df = df_flight.drop(columns=['travelcode', 'user_id','flight_number'])
X = df.drop(columns=['flight_price'])
y = df['flight_price']


#Additional Features
def feature_engineering(df):
    df['departure_hour'] = df['departure_date'].dt.hour
    df['departure_day_of_week'] = df['departure_date'].dt.dayofweek  
    df['departure_month'] = df['departure_date'].dt.month
    df['arrival_hour'] = df['arrival_date'].dt.hour
    df['arrival_day_of_week'] = df['arrival_date'].dt.dayofweek
    df['arrival_month'] = df['arrival_date'].dt.month

    return df

# Preprocessing Pipeline
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

full_pipeline = Pipeline([
    ('feature_engineering', FunctionTransformer(feature_engineering, validate=False)),  # Apply feature engineering
    ('preprocessor', preprocessor)  # Apply preprocessing
])

# Train-Test Split
X_transformed = full_pipeline.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

#Hyper parameter Tuning
rf = RandomForestRegressor(random_state=42)
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}
random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=10, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
random_search.fit(X_train, y_train)
best_rf_params = random_search.best_params_

# Grid Search
rf.set_params(**best_rf_params)
grid_params = {
    'n_estimators': [max(50, best_rf_params['n_estimators'] - 50), best_rf_params['n_estimators'], best_rf_params['n_estimators'] + 50],
    'max_depth': [best_rf_params['max_depth']],
    'min_samples_split': [max(2, best_rf_params['min_samples_split'] - 1), best_rf_params['min_samples_split'], best_rf_params['min_samples_split'] + 1]
}
grid_search = GridSearchCV(rf, param_grid=grid_params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf_params = grid_search.best_params_


# Optuna Optimization
def objective(trial):
    model = XGBRegressor(
        n_estimators=trial.suggest_int('n_estimators', 100, 500),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2),
        max_depth=trial.suggest_int('max_depth', 3, 10),
        subsample=trial.suggest_float('subsample', 0.5, 1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
        random_state=42,
        seed=42,
        n_jobs=1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred)

study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=20)
best_xgb_params = study.best_params

# Final Model
best_xgb = XGBRegressor(**best_xgb_params, random_state=42, seed=42, n_jobs=1)
best_xgb.fit(X_train, y_train)
y_pred_xgb = best_xgb.predict(X_test)
y_xgb = best_xgb.predict(X_train)

def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Performance:")
    print(f"R² Score: {r2_score(y_true, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")


evaluate_model("Optimized XGBoost", y_test, y_pred_xgb)