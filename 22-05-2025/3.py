import pandas as pd

df = pd.read_csv('Titanic-Dataset.csv')

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1


df['Title'] = df['Name'].str.extract(r',\s*([^\.]*)\.', expand=False)
print(df[['Name', 'FamilySize', 'Title']].head())