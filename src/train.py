#reference
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# Load cleaned data
df = pd.read_csv('../data/cleaned_biomass_lignin_removal.csv')

# Separate features and target
X = df.drop('lignin_removal', axis=1)
y = df['lignin_removal']

# Handle missing values (if any)
X = X.fillna(X.mean(numeric_only=True))

# Encode categorical variables (if any)
X = pd.get_dummies(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'R² score: {r2:.4f}')
print(f'RMSE: {rmse:.4f}')

# Save model
joblib.dump(model, '../models/ridge_model.pkl')
print('Model saved to ../models/ridge_model.pkl')

