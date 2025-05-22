import pandas as pd


df = pd.read_csv('Titanic-Dataset.csv')




df = df.drop(['PassengerId', 'Ticket', 'Name'], axis=1)

df = df.dropna()


df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)


X = df.drop('Survived', axis=1) 
y = df['Survived']              


df_cleaned = pd.concat([X, y], axis=1)


df_cleaned.to_csv('titanic_cleaned.csv', index=False)

print("✅ Dataset cleaned, split, and saved as 'titanic_cleaned.csv'")
