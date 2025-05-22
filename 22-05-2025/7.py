import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Titanic-Dataset.csv')

plt.figure(figsize=(6, 4))
plt.hist(df['Age'].dropna(), bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Number of Passengers')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
survival_by_sex = df.groupby('Sex')['Survived'].sum()
survival_by_sex.plot(kind='bar', color=['orange', 'lightgreen'])
plt.title('Survival Count by Gender')
plt.xlabel('Gender')
plt.ylabel('Number of Survivors')
plt.xticks(rotation=0)
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
df.boxplot(column='Fare', by='Pclass', grid=True)
plt.title('Fare Distribution by Passenger Class')
plt.suptitle('')
plt.xlabel('Passenger Class (Pclass)')
plt.ylabel('Fare')
plt.tight_layout()
plt.show()