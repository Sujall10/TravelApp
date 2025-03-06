import optuna
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()
DB_url = os.getenv('DB_url')
engine = create_engine(DB_url)

# Load dataset
df = pd.read_sql("SELECT * FROM car_rent LIMIT 20000", engine)

# Feature Engineering
df['month'] = pd.to_datetime(df['Rent_Date']).dt.month
df['day_of_week'] = pd.to_datetime(df['Rent_Date']).dt.dayofweek
df.drop(['Rent_Date', 'User_ID', 'TravelCode', 'Pickup_Location', 'Dropoff_Location', 'Rent_ID'], axis=1, inplace=True)

# Define features and target
X = df.drop(columns=['Total_Rent_Price'])
y = df['Total_Rent_Price']

# Split into Train & Test to Avoid Overfitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify Numerical & Categorical Features
num_features = X_train.select_dtypes(include=['int64', 'float64']).columns
cat_features = X_train.select_dtypes(include=['object']).columns

# Pipelines
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

# Transform Train & Test Data
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Save Preprocessor
joblib.dump(preprocessor, "preprocessor_Car.pkl")
print("Preprocessor saved successfully!")

# Get Correct Feature Names
cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_features)
all_feature_names = np.concatenate([num_features, cat_feature_names])

# Hyperparameter Tuning
def objective(trial, model_name, X_train, y_train):
    params = {}
    if model_name == 'Random Forest':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 5, 30, step=5),
            'min_samples_split': trial.suggest_int('min_samples_split', 5, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 10),
            'max_features': trial.suggest_float('max_features', 0.6, 1.0),
            'random_state': 42,
            'n_jobs': -1
        }
        model = RandomForestRegressor(**params)

    elif model_name == 'Gradient Boosting':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400, step=50),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'min_samples_split': trial.suggest_int('min_samples_split', 5, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 10),
            'random_state': 42
        }
        model = GradientBoostingRegressor(**params)

    elif model_name == 'XGBoost':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400, step=50),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 0.01, 1.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 0.1, 1.0),
            'random_state': 42,
            'n_jobs': -1
        }
        model = xgb.XGBRegressor(**params)

    # Perform Cross-Validation Only on Training Data
    scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
    return -scores.mean()

# Optimize Models
best_params = {}
models = {}
for model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, model_name, X_train_transformed, y_train), n_trials=20)
    best_params[model_name] = study.best_params

    # Initialize and Train Best Model
    models[model_name] = (
        RandomForestRegressor if model_name == 'Random Forest' else
        GradientBoostingRegressor if model_name == 'Gradient Boosting' else
        xgb.XGBRegressor
    )(**best_params[model_name])

    # Train Model on Training Data
    models[model_name].fit(X_train_transformed, y_train)

# Hyperparameter Tuning for Stacking
def meta_objective(trial):
    alpha = trial.suggest_loguniform('alpha', 0.01, 10.0)
    meta_model = Ridge(alpha=alpha)
    stacking_model = StackingRegressor(
        estimators=[('rf', models['Random Forest']), ('gb', models['Gradient Boosting']), ('xgb', models['XGBoost'])],
        final_estimator=meta_model
    )
    scores = cross_val_score(stacking_model, X_train_transformed, y_train, scoring='neg_mean_squared_error', cv=5)
    return -scores.mean()

meta_study = optuna.create_study(direction='minimize')
meta_study.optimize(meta_objective, n_trials=10)

# Train Final Stacking Model
best_alpha = meta_study.best_params['alpha']
stacking_model = StackingRegressor(
    estimators=[('rf', models['Random Forest']), ('gb', models['Gradient Boosting']), ('xgb', models['XGBoost'])],
    final_estimator=Ridge(alpha=best_alpha)
)

stacking_model.fit(X_train_transformed, y_train)

# Evaluate Model on Both Training & Testing Data
def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Performance:")
    print(f"R² Score: {r2_score(y_true, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")

y_train_pred = stacking_model.predict(X_train_transformed)
y_test_pred = stacking_model.predict(X_test_transformed)

evaluate_model("Training Set", y_train, y_train_pred)
evaluate_model("Test Set", y_test, y_test_pred)

# Save model
joblib.dump(stacking_model, "best_model_Car.pkl")
print("Model saved as best_model_Car.pkl")


# joblib.dump(stacking_model, "best_model_Car.pkl")
# joblib.dump(preprocessor, "preprocessor_Car.pkl")
# print("Model saved as best_model_Car.pkl")