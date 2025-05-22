import pandas as pd

df = pd.read_csv('Titanic-Dataset.csv')

encode = pd.get_dummies(df, columns=['Sex','Embarked'],drop_first=True)

print(encode.head())