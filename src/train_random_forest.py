import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

# Load dataset
df = pd.read_csv("data/insurance.csv")

# Separate features and target
X = df.drop("charges", axis=1)   # replace charges with your target column
y = df["charges"]

# One-hot encoding for categorical columns
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Random Forest
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf_model.fit(X_train, y_train)

# Predictions
preds = rf_model.predict(X_test)

# Metrics
r2 = r2_score(y_test, preds)
mae = mean_absolute_error(y_test, preds)
rmse = root_mean_squared_error(y_test, preds)

print("R2:", r2)
print("MAE:", mae)
print("RMSE:", rmse)

# Save model
joblib.dump(rf_model, "models/randomforest.pkl")

print("Model saved successfully!")