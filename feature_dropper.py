from sklearn.base import BaseEstimator, TransformerMixin


class FeatureDropper(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_drop: list[str]):
        self.features_to_drop = features_to_drop

    def fit(self, y=None):
        return self

    def transform(self, X):
        updated_X = X.drop(self.features_to_drop, axis=1)
        return updated_X
