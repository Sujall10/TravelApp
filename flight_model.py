import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from xgboost import XGBRegressor
import optuna
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


# Load environment variables
load_dotenv()
DB_URL = os.getenv("DB_URL")
engine = create_engine(DB_URL)

# Load Data
with engine.connect() as conn:
    df_flight = pd.read_sql("SELECT * FROM flight", conn)
    df_user = pd.read_sql("SELECT * FROM passenger", conn)

df_user.rename(columns={'usercode': 'user_id'}, inplace=True)
df = pd.merge(df_user, df_flight, on='user_id', how='inner').drop(columns=['name'])

# Feature Engineering
df['flight_date'] = pd.to_datetime(df['departure_date'])
df['is_weekend_flight'] = (df['flight_date'].dt.weekday >= 5).astype(int)
df['flight_month'] = df['flight_date'].dt.month
df = df.drop(columns=['travelcode', 'user_id'])
X = df.drop(columns=['flight_price'])
y = df['flight_price']

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

# Train-Test Split
X_transformed = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Hyperparameter Tuning
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

# # Feature Importance (XGBoost)
# feature_importance = best_xgb.feature_importances_
# feature_names = X.columns if isinstance(X, pd.DataFrame) else range(len(feature_importance))

# sorted_idx = np.argsort(feature_importance)[::-1]

# plt.figure(figsize=(10, 6))
# sns.barplot(x=feature_importance[sorted_idx][:10], y=np.array(feature_names)[sorted_idx][:10], palette="viridis")
# plt.xlabel("Feature Importance")
# plt.ylabel("Top 10 Features")
# plt.title("XGBoost Feature Importance")
# plt.show()


# Stacking Model
stacked_model = StackingRegressor(
    estimators=[
        ('rf', RandomForestRegressor(**best_rf_params, random_state=42)),
        ('xgb', best_xgb)
    ],
    final_estimator=XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
)
stacked_model.fit(X_train, y_train)
y_pred_stack = stacked_model.predict(X_test)

# Neural Network Model
def build_nn(units=[128, 64], dropout_rate=0.3, lr=0.01):
    model = Sequential()
    model.add(Dense(units[0], activation='relu', kernel_regularizer=l2(0.001), input_shape=(X_train.shape[1],)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    if len(units) > 1:
        model.add(Dense(units[1], activation='relu', kernel_regularizer=l2(0.001)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])
    return model

nn_model = build_nn()
nn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, verbose=1, callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5), EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])


# Second Neural Network Model
nn_model1 = build_nn(units=[64, 32], dropout_rate=0.2, lr=0.005)
nn_model1.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, verbose=1, 
              callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5), 
                         EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])

# Third Neural Network Model
nn_model2 = build_nn(units=[256, 128], dropout_rate=0.4, lr=0.01)
nn_model2.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, verbose=1, 
              callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5), 
                         EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])

# Predictions
y_pred_nn = nn_model.predict(X_test).flatten()
y_pred_nn1 = nn_model1.predict(X_test).flatten()
y_pred_nn2 = nn_model2.predict(X_test).flatten()

# Model Evaluation
def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Performance:")
    print(f"R² Score: {r2_score(y_true, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")

evaluate_model("Optimized XGBoost", y_test, y_pred_xgb)
evaluate_model("Stacking Model", y_test, y_pred_stack)
evaluate_model("Neural Network", y_test, y_pred_nn)
evaluate_model("Neural Network 1", y_test, y_pred_nn1)
evaluate_model("Neural Network 2", y_test, y_pred_nn2)
