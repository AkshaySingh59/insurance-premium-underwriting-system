import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/insurance.csv")

encoder = LabelEncoder()

df["sex"] = encoder.fit_transform(df["sex"])
df["smoker"] = encoder.fit_transform(df["smoker"])
df["region"] = encoder.fit_transform(df["region"])

print(df.head())