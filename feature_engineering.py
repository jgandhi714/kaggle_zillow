from sklearn.base import BaseEstimator, TransformerMixin
from datetime import date
import pandas as pd


class CreateYearFeatures(BaseEstimator, TransformerMixin):
    """
    Creates new features converting dates into years from present.
    Eg: 1989 is converted into present_year(2021) - 1989 which is 32.
    """

    def __init__(self, date_features):
        self.date_features = date_features
        self.current_year = date.today().year

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for var in self.date_features.keys():
            new_var_name = self.date_features[var]
            X[new_var_name] = self.current_year - X[var]
            X = X.drop(var, axis=1)
        return X


class CreateDateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        dt = pd.to_datetime(X['transactiondate']).dt
        X['transaction_month'] = ((dt.year - 2016) * 12 + dt.month)
        X = X.drop(['transactiondate'], axis=1)
        return X


class CreateDerivedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['halfbathcnt'] = X.bathroomcnt - X.fullbathcnt
        X['unfinished_sqft'] = X.lotsizesquarefeet - X.calculatedfinishedsquarefeet
        X['unfinished_sqft_pct'] = X.unfinished_sqft / X.lotsizesquarefeet
        X['finished_area_pct'] = X.calculatedfinishedsquarefeet / X.lotsizesquarefeet
        X['property_tax_per_sqft'] = X.taxamount / X.calculatedfinishedsquarefeet
        X['avg_finished_area_per_bedroom'] = X.calculatedfinishedsquarefeet / X.bedroomcnt
