import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv("data/insurance.csv")

df = pd.get_dummies(df, drop_first=True)

df["risk_score"] = df["age"] * df["bmi"]

df["family_risk"] = df["children"] * df["bmi"]

X = df.drop("charges", axis=1)
print(X.columns.tolist())

import joblib
joblib.dump(X.columns.tolist(), "models/feature_order.pkl")

y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

estimators = [
    ("rf", RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ))
]

stack_model = StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression()
)

stack_model.fit(X_train, y_train)

predictions = stack_model.predict(X_test)

print("R2 Score:", r2_score(y_test, predictions))
print("MAE:", mean_absolute_error(y_test, predictions))

joblib.dump(
    stack_model,
    "models/premium_model.pkl"
)