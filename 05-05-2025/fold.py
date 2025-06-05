from sklearn.datasets import load_wine
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load and shuffle
X, y = load_wine(return_X_y=True)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X, y = shuffle(X, y, random_state=42)

# Manual K-Fold Cross Validation
k = 5
fold_size = len(X) // k
accuracies = []

for i in range(k):
    start = i * fold_size
    end = (i + 1) * fold_size

    X_test = X[start:end]
    y_test = y[start:end]

    X_train = np.concatenate((X[:start], X[end:]), axis=0)
    y_train = np.concatenate((y[:start], y[end:]), axis=0)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

# Print fold results
for i, acc in enumerate(accuracies):
    print(f"Fold {i+1} Accuracy: {acc:.4f}")
print(f"Average Accuracy: {np.mean(accuracies):.4f}")
