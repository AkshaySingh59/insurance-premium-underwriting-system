from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import pandas as pd

df = pd.read_csv("data/insurance.csv")

df = pd.get_dummies(df, drop_first=True)

X = df.drop("charges", axis=1)

y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)

print("Training Complete")