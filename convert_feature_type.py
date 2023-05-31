from sklearn.base import BaseEstimator, TransformerMixin


class ConvertFeatureType(BaseEstimator, TransformerMixin):
    def __init__(self, convert_to_int=[], convert_to_bool=[], convert_to_string=[], convert_to_float=[]):
        self.convert_to_int = convert_to_int
        self.convert_to_bool = convert_to_bool
        self.convert_to_string = convert_to_string
        self.convert_to_float = convert_to_float
        self.features = {"int": convert_to_int, "float": convert_to_float, "boolean": convert_to_bool, "str": convert_to_string}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.map_bool_features(X)
        for data_type in self.features.keys():
            X = self.convert_feature_types(X, data_type)
        return X

    def map_bool_features(self, X):
        for var in self.convert_to_bool:
            X[var][X[var].notnull()] = True
            X[var] = X[var].fillna(False)

    def convert_feature_types(self, X, data_type):
        for var in self.features[data_type]:
            X[var] = X[var].astype(data_type)
        return X



