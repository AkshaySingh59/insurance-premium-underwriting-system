import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

df = pd.read_csv("data/insurance.csv")

df = pd.get_dummies(df, drop_first=True)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(df)

dbscan = DBSCAN(
    eps=1.5,
    min_samples=5
)

clusters = dbscan.fit_predict(X_scaled)

df["cluster"] = clusters

clean_df = df[df["cluster"] != -1]

print("Original:", len(df))
print("Cleaned:", len(clean_df))