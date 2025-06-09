import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def rank_models(models, X_test, y_test, metric="accuracy"):
    results = []

    for name, model in models.items():
        y_pred = model.predict(X_test)

        if metric == "roc_auc":
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)[:, 1]
                score = roc_auc_score(y_test, y_score)
            else:
                score = None
        elif metric == "accuracy":
            score = accuracy_score(y_test, y_pred)
        elif metric == "f1":
            score = f1_score(y_test, y_pred)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        results.append({
            "Model": name,
            metric.capitalize(): score
        })
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=metric.capitalize(), ascending=False).reset_index(drop=True)

    return results_df
