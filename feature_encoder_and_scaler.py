from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
import pandas as pd


class FeatureEncoderAndScaler(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_encode=None, features_to_scale=None, numeric_types=["float"]):
        self.features_to_encode = features_to_encode
        self.features_to_scale = features_to_scale
        self.numeric_types = numeric_types
        self.feature_encoder_and_scaler = None

    def fit(self, X, y=None):
        if not self.features_to_encode:
            self.features_to_encode = X.select_dtypes(include=["object"]).columns
        if not self.features_to_scale:
            self.features_to_scale = X.select_dtypes(include=self.numeric_types).columns

        feature_encoder_scaler = ColumnTransformer(
            [
                ("ohe_cats", OneHotEncoder(handle_unknown='ignore', sparse=False), self.features_to_encode),
                ("num_scaler", RobustScaler(), self.features_to_scale)
            ],
            remainder='passthrough'
        )

        self.feature_encoder_scaler = feature_encoder_scaler.fit(X)
        return self

    def transform(self, X):
        X_np = self.feature_encoder_scaler.transform(X)
        X = pd.DataFrame(
            X_np,
            columns=self.feature_encoder_scaler.get_feature_names_out()
        )
        X = self.convert_feature_types(X)
        return X

    def convert_feature_types(self, X):
        """Convert feature types to object, float, bool based on the column name.
        Columns with `ohe_cats` are object, `num_scaler` are float, `remainder` are bool"""
        for column in X:
            if 'ohe_cats' in column:
                X[column] = X[column].astype("object")
            elif 'num_scaler' in column:
                X[column] = X[column].astype("float")
            elif 'remainder' in column:
                X[column] = X[column].astype("boolean")
        return X
