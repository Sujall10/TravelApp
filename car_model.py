from dotenv import load_dotenv
import os
import optuna
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load environment variables
load_dotenv()
DB_url = os.getenv('DB_url')
engine = create_engine(DB_url)

df = pd.read_sql("SELECT * FROM car_rent", engine)

df['month'] = pd.to_datetime(df['Rent_Date']).dt.month
df.drop(['Rent_ID', 'User_ID', 'TravelCode'], axis=1, inplace=True)

X = df.drop(columns=['Total_Rent_Price'])
y = df['Total_Rent_Price']

num_features = X.select_dtypes(include=['int64', 'float64']).columns
cat_features = X.select_dtypes(include=['object']).columns

num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Hyperparameter tuning with Optuna
def objective(trial):
    model_type = trial.suggest_categorical("model", ["RandomForest", "XGBoost", "LightGBM", "GradientBoosting", "ExtraTrees"])
    
    if model_type == "RandomForest":
        model = RandomForestRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 200),
            max_depth=trial.suggest_int("max_depth", 5, 20)
        )
    elif model_type == "XGBoost":
        model = XGBRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 200),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3)
        )
    elif model_type == "LightGBM":
        model = LGBMRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 200),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3)
        )
    elif model_type == "GradientBoosting":
        model = GradientBoostingRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 200),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3)
        )
    else:
        model = ExtraTreesRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 200),
            max_depth=trial.suggest_int("max_depth", 5, 20)
        )
    
    score = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=5).mean()
    return -score

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

best_params = study.best_params
print("Best Parameters:", best_params)

# Train best model
if best_params['model'] == "RandomForest":
    best_model = RandomForestRegressor(**{k: v for k, v in best_params.items() if k != "model"})
elif best_params['model'] == "XGBoost":
    best_model = XGBRegressor(**{k: v for k, v in best_params.items() if k != "model"})
elif best_params['model'] == "LightGBM":
    best_model = LGBMRegressor(**{k: v for k, v in best_params.items() if k != "model"})
elif best_params['model'] == "GradientBoosting":
    best_model = GradientBoostingRegressor(**{k: v for k, v in best_params.items() if k != "model"})
else:
    best_model = ExtraTreesRegressor(**{k: v for k, v in best_params.items() if k != "model"})

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
print("Best Model MSE:", mean_squared_error(y_test, y_pred))
print("Best Model R²:", r2_score(y_test, y_pred))

# Neural Network Model
def build_nn():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

nn_model = build_nn()
nn_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)
y_nn_pred = nn_model.predict(X_test)
print("Neural Network MSE:", mean_squared_error(y_test, y_nn_pred))
print("Neural Network R²:", r2_score(y_test, y_nn_pred))

# Final Model Selection
models = {"Best ML Model": best_model, "Neural Network": nn_model}
final_model = max(models, key=lambda k: r2_score(y_test, models[k].predict(X_test) if k == "Best ML Model" else models[k](X_test).flatten()))
print("Final Selected Model:", final_model)












