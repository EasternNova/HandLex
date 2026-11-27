from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier


class ModelA:
    """Random Forest implementation for landmark features."""

    def __init__(self, n_estimators: int = 300):
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1,
            max_features="sqrt",
        )

    def fit(self, X, y):
        self.classifier.fit(X, y)
        return self

    def predict(self, X):
        return self.classifier.predict(X)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)
