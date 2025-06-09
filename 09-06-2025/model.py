from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from modelfn import rank_models
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(max_iter=2000)
dt = DecisionTreeClassifier()
lr.fit(X_train_scaled, y_train)
dt.fit(X_train, y_train)

models = {
    "Logistic Regression": lr,
    "Decision Tree": dt
}

rank =  rank_models(models, X_test, y_test, metric='f1')

print(rank)