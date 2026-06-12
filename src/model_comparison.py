import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# Load Data
df = pd.read_csv("data/insurance.csv")

# Feature Engineering
df["risk_score"] = df["age"] * df["bmi"]
df["family_risk"] = df["children"] * df["bmi"]

# Encoding
df = pd.get_dummies(df, drop_first=True)

# Features & Target
X = df.drop("charges", axis=1)
y = df["charges"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Linear Regression
# -----------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

# -----------------------------
# Random Forest
# -----------------------------
rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

# -----------------------------
# Stacking Model
# -----------------------------
estimators = [
    ("rf", RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ))
]

stack = StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression()
)

stack.fit(X_train, y_train)

stack_pred = stack.predict(X_test)

# -----------------------------
# Results
# -----------------------------
print("\nLINEAR REGRESSION")
print("R2 :", r2_score(y_test, lr_pred))
print("MAE:", mean_absolute_error(y_test, lr_pred))
print("RMSE:", mean_squared_error(y_test, lr_pred)**0.5)

print("\nRANDOM FOREST")
print("R2 :", r2_score(y_test, rf_pred))
print("MAE:", mean_absolute_error(y_test, rf_pred))
print("RMSE:", mean_squared_error(y_test, rf_pred)**0.5)

print("\nSTACKED MODEL")
print("R2 :", r2_score(y_test, stack_pred))
print("MAE:", mean_absolute_error(y_test, stack_pred))
print("RMSE:", mean_squared_error(y_test, stack_pred)**0.5)

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "Stacked RF-LR"
    ],
    "R2": [
        r2_score(y_test, lr_pred),
        r2_score(y_test, rf_pred),
        r2_score(y_test, stack_pred)
    ]
})

results.to_csv(
    "reports/model_results.csv",
    index=False
)

print("\nResults saved!")