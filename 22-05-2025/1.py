import pandas as pd

df = pd.read_csv('Titanic-Dataset.csv')

print("\nFirst 10 rows of the Titanic dataset:\n")
print(df.head(10))
print("\nSummary statistics using .describe():\n")
print(df.describe())
print("\nInformation about the dataset using .info():\n")
print(df.info())
print("\nShape of the dataset:",df.shape)