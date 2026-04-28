from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, recall_score
recall = make_scorer(recall_score)


def tune_model(pipeline, param_dist, x_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=20,
        scoring=recall,
        cv=cv,
        n_jobs=-1,
        random_state=42,
        verbose=2
    )

    search.fit(x_train, y_train)
    return search
