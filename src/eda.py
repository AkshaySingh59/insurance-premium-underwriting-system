import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/insurance.csv")

print(df.describe())

sns.histplot(df["charges"])

plt.show()

df_encoded = pd.get_dummies(df, drop_first=True)

sns.heatmap(df_encoded.corr(), annot=True)

plt.show()