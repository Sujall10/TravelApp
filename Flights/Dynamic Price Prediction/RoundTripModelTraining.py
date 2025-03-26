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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline


# Load Environment Variables
load_dotenv()
DB_URL = os.getenv("a")
engine = create_engine(DB_URL)

with engine.connect() as conn:
    df_flight = pd.read_sql("SELECT * FROM flight_RoundTrip", conn)

df_flight.drop(columns={'flight_duration'}, inplace=True)

df_flight['flight_duration'] = df_flight['Arrival_Date'] - df_flight['departure_date']

df_flight["departure_hour"] = df_flight["departure_date"].dt.hour
df_flight["departure_day_of_week"] = df_flight["departure_date"].dt.dayofweek
df_flight["departure_month"] = df_flight["departure_date"].dt.month
df_flight["arrival_hour"] = df_flight["Arrival_Date"].dt.hour
df_flight["arrival_day_of_week"] = df_flight["Arrival_Date"].dt.dayofweek
df_flight["arrival_month"] = df_flight["Arrival_Date"].dt.month

df_flight["return_departure_hour"] = df_flight["ReturnDeparture_Date"].dt.hour
df_flight["return_departure_day_of_week"] = df_flight["ReturnDeparture_Date"].dt.dayofweek
df_flight["return_departure_month"] = df_flight["ReturnDeparture_Date"].dt.month
df_flight["return_arrival_hour"] = df_flight["ReturnArrival_Date"].dt.hour
df_flight["return_arrival_day_of_week"] = df_flight["ReturnArrival_Date"].dt.dayofweek
df_flight["return_arrival_month"] = df_flight["ReturnArrival_Date"].dt.month

columns_to_drop = ['travelcode','user_id','departure_date','Arrival_Date','ReturnDeparture_Date','ReturnArrival_Date']
df_flight.drop(columns_to_drop, axis=1, inplace=True)

X = df_flight.drop(columns=["flight_price"])
y = df_flight["flight_price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_features = X.select_dtypes(include=['int64', 'int32', 'float64']).columns.tolist()
cat_features = X.select_dtypes(include=['object']).columns.tolist()

X = pd.get_dummies(X, columns=cat_features, drop_first=True)


# df_flight['flight_duration_seconds'] = df_flight['flight_duration'].dt.total_seconds()
# num_features.append('flight_duration_seconds')

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

# def objective(trial):
#     # params = {
#     #     'n_estimators': trial.suggest_int('n_estimators', 50, 300),
#     #     'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
#     #     'max_depth': trial.suggest_int('max_depth', 3, 10),
#     #     'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#     #     'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#     #     'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0),
#     #     'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0),
#     # }
#     # params = {
#     #     'n_estimators': trial.suggest_int('n_estimators', 100, 500),
#     #     'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
#     #     'max_depth': trial.suggest_int('max_depth', 3, 7),
#     #     'subsample': trial.suggest_float('subsample', 0.6, 0.9),
#     #     'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
#     #     'reg_lambda': trial.suggest_float('reg_lambda', 5.0, 15.0),
#     #     'reg_alpha': trial.suggest_float('reg_alpha', 5.0, 15.0),
#     # }
#     # params = {
#     #     'n_estimators': trial.suggest_int('n_estimators', 100, 300),
#     #     'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
#     #     'max_depth': trial.suggest_int('max_depth', 3, 5),
#     #     'subsample': trial.suggest_float('subsample', 0.6, 0.8),
#     #     'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.8),
#     #     'reg_lambda': trial.suggest_float('reg_lambda', 10.0, 20.0),
#     #     'reg_alpha': trial.suggest_float('reg_alpha', 10.0, 20.0),
#     # }
#     # params = {
#     #     'n_estimators': trial.suggest_int('n_estimators', 50, 200),
#     #     'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
#     #     'max_depth': trial.suggest_int('max_depth', 2, 4),
#     #     'subsample': trial.suggest_float('subsample', 0.5, 0.7),
#     #     'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.7),
#     #     'reg_lambda': trial.suggest_float('reg_lambda', 15.0, 30.0),
#     #     'reg_alpha': trial.suggest_float('reg_alpha', 15.0, 30.0),
#     # }
#     # params = {
#     #     'n_estimators': trial.suggest_int('n_estimators', 50, 150),
#     #     'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
#     #     'max_depth': trial.suggest_int('max_depth', 2, 3),
#     #     'subsample': trial.suggest_float('subsample', 0.6, 0.8),
#     #     'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.6),
#     #     'reg_lambda': trial.suggest_float('reg_lambda', 20.0, 40.0),
#     #     'reg_alpha': trial.suggest_float('reg_alpha', 20.0, 40.0),
#     # }
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
#         'max_depth': trial.suggest_int('max_depth', 3, 10),
#         'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
#         'reg_lambda': trial.suggest_float('reg_lambda', 1, 50, log=True),  # L2 regularization
#         'reg_alpha': trial.suggest_float('reg_alpha', 1, 50, log=True),  # L1 regularization
#         'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
#         'gamma': trial.suggest_float('gamma', 0, 0.5),
#         'random_state': 42,  # Set seed for reproducibility
#         'n_jobs': -1  # Use all CPU cores
#     }

#     model = XGBRegressor(**params)
    
#     pipeline = Pipeline([
#         ("preprocessor", preprocessor),
#         ("model", model)
#     ])
    
#     kfold = KFold(n_splits=5, shuffle=True, random_state=42)
#     cv_score = cross_val_score(pipeline, X_train, y_train, cv=kfold, scoring="neg_root_mean_squared_error").mean()
    
#     return -cv_score  # Minimize RMSE

# # Run Optuna Optimization
# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=50)

# # Get Best Hyperparameters
# best_params = study.best_params
# print("Best Parameters:", best_params)

# # Final Pipeline with Best Parameters
# final_pipeline = Pipeline([
#     ("preprocessor", preprocessor),
#     ("model", XGBRegressor(**best_params, random_state=42, n_jobs=-1))  
# ])

# # Train Final Model
# final_pipeline.fit(X_train, y_train)

# # Predictions
# y_pred_test = final_pipeline.predict(X_test)
# y_pred_train = final_pipeline.predict(X_train)

# # Evaluation Function
# def evaluate_model(name, y_true, y_pred):
#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#     mae = mean_absolute_error(y_true, y_pred)
#     r2 = r2_score(y_true, y_pred)
    
#     print(f"\n{name} Performance:")
#     print(f"R² Score: {r2:.4f}")
#     print(f"MAE: {mae:.4f}")
#     print(f"RMSE: {rmse:.4f}")

# # Evaluate Model on Train & Test Data
# evaluate_model("Training Set", y_train, y_pred_train)
# evaluate_model("Test Set", y_test, y_pred_test)



def tune_model(trial, model_type):
    if model_type == "random_forest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5)
        }
        model = RandomForestClassifier(**params, random_state=42)
    elif model_type == "xgboost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        }
        model = XGBClassifier(**params, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    elif model_type == "lightgbm":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        }
        model = LGBMClassifier(**params, random_state=42)
    
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean()
    return score

# # Tune each base model
# for model_name in ["random_forest", "xgboost", "lightgbm"]:
#     study = optuna.create_study(direction="maximize")
#     study.optimize(lambda trial: tune_model(trial, model_name), n_trials=30)
#     print(f"Best {model_name} parameters: {study.best_params}")


best_params = {}
for model_name in ["random_forest", "xgboost", "lightgbm"]:
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: tune_model(trial, model_name), n_trials=30)
    best_params[model_name] = study.best_params  # Store best parameters separately
    print(f"Best {model_name} parameters: {best_params[model_name]}")


# Define tuned models
rf_best = RandomForestClassifier(**study.best_params, random_state=42)
xgb_best = XGBClassifier(**study.best_params, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
lgb_best = LGBMClassifier(**study.best_params, random_state=42)

# Define stacking classifier
# final_estimator = make_pipeline(preprocessor, LogisticRegression())

final_estimator = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression())  
])

stacking_model = StackingClassifier(
    estimators=[("rf", rf_best), ("xgb", xgb_best), ("lgb", lgb_best)],
    final_estimator=final_estimator
)

# Train and evaluate
stacking_model.fit(X_train, y_train)
y_pred = stacking_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Stacking Model Accuracy: {accuracy:.4f}")

# print(cat_features)
# print(df_flight.info())
# print(df_flight)