import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Titanic-Dataset.csv')

numeric_df = df.select_dtypes(include=[np.number])

numeric_df = numeric_df.dropna()

corr_matrix = numeric_df.corr()

plt.figure(figsize=(8, 6))
plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar(label='Correlation Coefficient')
plt.title('Correlation Matrix Heatmap')

tick_marks = range(len(corr_matrix.columns))
plt.xticks(tick_marks, corr_matrix.columns, rotation=45, ha='right')
plt.yticks(tick_marks, corr_matrix.columns)

plt.tight_layout()
plt.show()