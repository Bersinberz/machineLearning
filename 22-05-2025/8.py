import pandas as pd
import numpy as np

df = pd.read_csv('Titanic-Dataset.csv')

df = df[['Age', 'Fare']].dropna()

age_array = df['Age'].values
fare_array = df['Fare'].values

age_mean = np.mean(age_array)
age_median = np.median(age_array)
age_std = np.std(age_array)

fare_mean = np.mean(fare_array)
fare_median = np.median(fare_array)
fare_std = np.std(fare_array)

print("Age - Mean :",age_mean)
print("Age - Median :",age_median)
print("Age - std Dev :",age_std)

print("Fare - Mean :",fare_mean)
print("Fare - Median :",fare_median)
print("Fare - std Dev :",fare_std)

age_min = np.min(age_array)
age_max = np.max(age_array)
age_norm = (age_array - age_min) / (age_max - age_min)

fare_min = np.min(fare_array)
fare_max = np.max(fare_array)
fare_norm = (fare_array - fare_min) / (fare_max - fare_min)

print("\nNormalized Age (first 5):", age_norm[:5])
print("Normalized Fare (first 5):", fare_norm[:5])