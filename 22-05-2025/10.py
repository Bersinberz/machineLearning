import pandas as pd

# Load the dataset
df = pd.read_csv('Titanic-Dataset.csv')

# ----- CLEANING & PREPROCESSING -----

# Drop irrelevant columns
df = df.drop(['PassengerId', 'Ticket', 'Name'], axis=1)

# Drop rows with missing values
df = df.dropna()


df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)


X = df.drop('Survived', axis=1)  # Input features
y = df['Survived']               # Target label

# Combine X and y into a final DataFrame (if needed)
df_cleaned = pd.concat([X, y], axis=1)

# Save to CSV
df_cleaned.to_csv('titanic_cleaned.csv', index=False)

print("✅ Dataset cleaned, split, and saved as 'titanic_cleaned.csv'")
