import pandas as pd

df = pd.read_csv('Titanic-Dataset.csv')

df['AgeGroup'] = pd.cut(df['Age'], bins=5)

print(df[['Age' , 'AgeGroup']].head(50))