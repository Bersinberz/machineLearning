import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

def load_and_prepare_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    
    print("Dataset Information:")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Target distribution:\n{y.value_counts()}")
    print("\nFirst 5 samples:")
    print(X.head())
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, data.feature_names

def build_and_evaluate_models(X_train, X_test, y_train, y_test):
    
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=10000, random_state=42),
            'params': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
        },
        'SVM': {
            'model': SVC(probability=True, random_state=42),
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=42, eval_metric='logloss'),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            }
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance']
            }
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        },
        'Naive Bayes': {
            'model': GaussianNB(),
            'params': {}
        }
    }

    results = []
    best_models = {}

    for name, config in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}...")
        
        grid = GridSearchCV(
            estimator=config['model'],
            param_grid=config['params'],
            scoring='f1',
            cv=5,
            n_jobs=-1,
            verbose=1
        )
        grid.fit(X_train, y_train)
        
        best_model = grid.best_estimator_
        best_models[name] = best_model
        
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        results.append({
            'Model': name,
            'Best Parameters': grid.best_params_,
            'F1 Score': f1,
            'AUC-ROC': auc
        })
        
        print(f"\n{name} Classification Report:")
        print(classification_report(y_test, y_pred))
        
        print(f"\n{name} Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    return results, best_models
2z
def visualize_results(results, feature_names, best_models):
    """Create visualizations and leaderboard of model results"""
    
    leaderboard = pd.DataFrame(results).sort_values(by=['F1 Score', 'AUC-ROC'], ascending=False)
    leaderboard.reset_index(drop=True, inplace=True)
    
    plt.figure(figsize=(12, 6))
    leaderboard.set_index('Model')[['F1 Score', 'AUC-ROC']].plot(kind='bar')
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    for name, model in best_models.items():
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(12, 6))
            importances = pd.Series(model.feature_importances_, index=feature_names)
            importances.sort_values().plot(kind='barh')
            plt.title(f'{name} Feature Importances')
            plt.tight_layout()
            plt.show()
    
    print("\nFINAL LEADERBOARD:")
    print(leaderboard.to_string(index=False))
    
    return leaderboard

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data()
    
    results, best_models = build_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    leaderboard = visualize_results(results, feature_names, best_models)
    
    best_model_name = leaderboard.iloc[0]['Model']
    best_model = best_models[best_model_name]
    print(f"\nBest model: {best_model_name}")