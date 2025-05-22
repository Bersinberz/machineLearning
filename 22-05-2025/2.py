import pandas as pd

df = pd.read_csv('Titanic-Dataset.csv')

print("\nMissing values:\n")
print(df.isnull().sum())

# Filling Missing Age values with the median
df['Age'] = df['Age'].fillna(df['Age'].median())

# Filling Missing embarked values with frequent category
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

#Drop cabin due to lot of error
df = df.drop(columns='Cabin')

print("\nMissing values after handling:\n")
print(df.isnull().sum())