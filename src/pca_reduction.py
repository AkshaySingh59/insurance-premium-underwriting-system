from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import pandas as pd

df = pd.read_csv("data/insurance.csv")

df = pd.get_dummies(df, drop_first=True)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(df)

pca = PCA(n_components=0.95)

X_pca = pca.fit_transform(X_scaled)

print("Original Features:", X_scaled.shape[1])

print("Reduced Features:", X_pca.shape[1])