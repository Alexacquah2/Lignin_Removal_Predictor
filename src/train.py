import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import xgboost as xgb
from skopt import BayesSearchCV

# Load processed data
df = pd.read_csv('../data/processed_biomass_lignin_removal.csv')

# Split features and target
X = df.drop('lignin_removal', axis=1)
y = df['lignin_removal']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Model
rf_model = RandomForestRegressor(
    n_estimators=310,
    max_leaf_nodes=235,
    min_samples_split=3,
    random_state=30
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)
print(f"Random Forest - RMSE: {rmse_rf:.4f}, R²: {r2_rf:.4f}")

# XGBoost Model
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    learning_rate=0.11,
    n_estimators=430,
    max_depth=6,
    random_state=30
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)
print(f"XGBoost - RMSE: {rmse_xgb:.4f}, R²: {r2_xgb:.4f}")

# Hyperparameter Optimization (Bayesian)
param_space = {
    'n_estimators': (200, 500),
    'max_depth': (10, 30),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 4)
}

bayes_search = BayesSearchCV(
    RandomForestRegressor(),
    param_space,
    n_iter=50,
    cv=5,
    scoring='r2'
)
bayes_search.fit(X_train, y_train)
best_model = bayes_search.best_estimator_

# Save best model
joblib.dump(best_model, '../models/optimized_lignin_model.pkl')
print("Model training complete. Best model saved to ../models/optimized_lignin_model.pkl")
