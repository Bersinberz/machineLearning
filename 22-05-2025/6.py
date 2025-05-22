import pandas as pd

df = pd.read_csv('Titanic-Dataset.csv')

df = df.drop(['PassengerId','Ticket','Name'], axis=1)

print(df.columns)