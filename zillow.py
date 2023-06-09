import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

from feature_dropper import FeatureDropper
from feature_engineering import CreateYearFeatures, CreateDateFeatures

warnings.filterwarnings('ignore')

color = sns.color_palette()
sns.set_style('darkgrid')

RANDOM_SEED = 10
PROPERTIES_DF_REDUNDANT_COLUMNS_TO_DROP = ['calculatedbathnbr', 'poolcnt', 'hashottuborspa', 'taxdelinquencyyear',
                                           'rawcensustractandblock', 'threequarterbathnbr', 'taxvaluedollarcnt']
# properties with mostly null values that can't be reliably imputed
PROPERTIES_DF_NULL_COLUMNS_TO_DROP = ['architecturalstyletypeid', 'assessmentyear', 'basementsqft', 'buildingclasstypeid',
                                      'finishedfloor1squarefeet', 'finishedsquarefeet6', 'finishedsquarefeet13',
                                      'finishedsquarefeet15', 'finishedsquarefeet50', 'storytypeid',
                                      'typeconstructiontypeid', 'yardbuildingsqft26', 'numberofstories', 'censustractandblock']
ALL_COLUMNS_TO_DROP = PROPERTIES_DF_REDUNDANT_COLUMNS_TO_DROP + PROPERTIES_DF_NULL_COLUMNS_TO_DROP
PROPERTIES_DF_CAT_COLUMNS = [
    'airconditioningtypeid', 'architecturalstyletypeid', 'buildingclasstypeid', 'buildingqualitytypeid',
    'decktypeid', 'fips', 'hashottuborspa', 'heatingorsystemtypeid', 'pooltypeid10', 'pooltypeid2',
    'pooltypeid7', 'propertycountylandusecode', 'propertylandusetypeid', 'propertyzoningdesc',
    'rawcensustractandblock', 'regionidcity', 'regionidcounty', 'regionidneighborhood', 'regionidzip',
    'storytypeid', 'typeconstructiontypeid', 'fireplaceflag', 'assessmentyear',
    'taxdelinquencyflag', 'taxdelinquencyyear', 'censustractandblock',
]
ZERO_IMPUTATION_COLUMNS = ['fireplacecnt', 'pooltypeid2', 'pooltypeid7', 'pooltypeid10', 'decktypeid', 'poolsizesum', 'yardbuildingsqft17']
ONE_IMPUTATION_COLUMNS = ['fullbathcnt', 'unitcnt', 'bedroomcnt', 'bathroomcnt']
MODE_IMPUTATION_COLUMNS = PROPERTIES_DF_CAT_COLUMNS
UNIVARIATE_IMPUTATION_COLUMNS = ZERO_IMPUTATION_COLUMNS + ONE_IMPUTATION_COLUMNS + MODE_IMPUTATION_COLUMNS
BEST_XGB_PARAMS = {
    'colsample_bynode': 0.75,
    'lambda': 2.,
    'alpha': 2.,
    'gamma': 0.201,
    'max_depth': 5,
    'subsample': 0.95,
    'tree_method': 'exact',
    'learning_rate': 0.01,
    'n_estimators': 10000,
    'random_state': RANDOM_SEED,
}
DATE_FEATURES = {"yearbuilt": "house_age"}
TEST_SIZE = 0.1


def load_properties_df(properties_csv_fp: str) -> pd.DataFrame:
    print("loading properties dataframe from csv " + properties_csv_fp)
    properties_df = pd.read_csv(properties_csv_fp)
    numeric_columns = properties_df.select_dtypes(include=['float']).columns.tolist()
    properties_df[numeric_columns] = properties_df[numeric_columns].astype('float32')
    print("done loading properties dataframe " + properties_csv_fp)
    return properties_df


def target_variable_eda(logerror_data: np.ndarray[np.float32]) -> None:
    print("beginning target variable eda")

    logerror_data.hist(bins=100, figsize=(8, 5))
    plt.show()

    logerror_data.describe()

    sns.distplot(logerror_data, fit=stats.norm)

    (mu, sigma) = stats.norm.fit(logerror_data)
    print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
               loc='best')
    plt.ylabel('Frequency')
    plt.title('LogError distribution')
    plt.figure()
    stats.probplot(logerror_data, plot=plt)
    plt.show()

    print("done with target variable eda")


def drop_outliers(train_df: pd.DataFrame, num_stdevs: int) -> pd.DataFrame:
    mean = train_df['logerror'].mean()
    std = train_df['logerror'].std()
    high_thresh = mean + num_stdevs * std
    low_thresh = mean - num_stdevs * std
    return train_df[train_df['logerror'].between(low_thresh, high_thresh)]


# def impute_properties_df(df: pd.DataFrame) -> pd.DataFrame:
#     df['calculatedfinishedsquarefeet'] = df['calculatedfinishedsquarefeet'].combine_first(
#         df['finishedsquarefeet12'])
#     df.loc[df['fireplaceflag'], 'fireplacecnt'] = df.loc[
#         df['fireplaceflag'], 'fireplacecnt'].fillna(1)
#     df = df.drop(['fireplaceflag', 'finishedsquarefeet12'], axis=1)
#
#     df[ZERO_IMPUTATION_COLUMNS] = df[ZERO_IMPUTATION_COLUMNS].fillna(0)
#     df[ONE_IMPUTATION_COLUMNS] = df[ONE_IMPUTATION_COLUMNS].fillna(1)
#     df[ONE_IMPUTATION_COLUMNS] = df[ONE_IMPUTATION_COLUMNS].replace(0, 1)
#     df[MODE_IMPUTATION_COLUMNS] = df[MODE_IMPUTATION_COLUMNS].fillna(
#         df[MODE_IMPUTATION_COLUMNS].mode().iloc[0])
#     for category_col in PROPERTIES_DF_CATEGORY_COLUMNS:
#         df[category_col] = pd.Categorical(df[category_col],
#                                           categories=df[category_col].unique())
#
#     # use the already imputed columns to impute the remaining columns
#     impute_x_df = df[UNIVARIATE_IMPUTATION_COLUMNS]
#     # use xgboost to impute the rest of the missing values in each column
#     columns_to_impute = (set(df.columns) - set(UNIVARIATE_IMPUTATION_COLUMNS))
#     columns_to_impute.remove('parcelid')
#     columns_to_impute.remove('taxdelinquencyflag')
#     for col in columns_to_impute:
#         print("imputing column: " + col)
#         impute_train_df = impute_x_df.copy(deep=True)
#         impute_train_df[col] = df[col]
#         impute_train_df = impute_train_df.dropna()
#         impute_train_data_dmatrix = xgb.DMatrix(data=impute_train_df[UNIVARIATE_IMPUTATION_COLUMNS],
#                                                 label=impute_train_df[col].values, enable_categorical=True)
#         impute_test_data_dmatrix = xgb.DMatrix(data=impute_x_df, enable_categorical=True)
#         xgb_model = xgb.train(dtrain=impute_train_data_dmatrix, params=IMPUTE_XGB_PARAMS)
#         predictions = pd.Series(xgb_model.predict(impute_test_data_dmatrix))
#         df[col] = df[col].fillna(predictions)
#         print("done imputing column: " + col)
#     return df


def merge_df_with_csvs(properties_df: pd.DataFrame, csvs: list[str]) -> pd.DataFrame:
    print("joining properties dataframe and training data from csvs " + str(csvs))
    merged_df = pd.concat([pd.read_csv(train_csv) for train_csv in csvs])
    merged_df = merged_df.merge(properties_df, how='left', on='parcelid')
    print("done joining properties dataframe and training data from csvs " + str(csvs))
    return merged_df


# load/merge data
properties_2017 = load_properties_df('properties_2017.csv')
properties_df = merge_df_with_csvs(properties_2017, ["train_2016_v2.csv", "train_2017.csv"])

# remove outliers and check for normality
# target_variable_eda(properties_df.logerror)
properties_df = drop_outliers(properties_df, 2.2)
print("Finished dropping outliers")
# target_variable_eda(properties_df.logerror)

# imputation pipelines
# univariate_impute_pipe = ColumnTransformer(
#     [
#         ("impute_0", SimpleImputer(strategy="constant", fill_value=0), ZERO_IMPUTATION_COLUMNS),
#         ("impute_1", SimpleImputer(strategy="constant", fill_value=1), ONE_IMPUTATION_COLUMNS),
#         ("impute_mode", SimpleImputer(strategy="most_frequent"), MODE_IMPUTATION_COLUMNS)
#     ],
#     remainder='passthrough'
# )
#
# columns_to_impute = (set(properties_df.columns) - set(UNIVARIATE_IMPUTATION_COLUMNS))
# columns_to_impute.remove('parcelid')
# cat_columns_to_impute = [col for col in columns_to_impute if col in PROPERTIES_DF_CAT_COLUMNS]
# numeric_columns_to_impute = [col for col in columns_to_impute if col not in cat_columns_to_impute]
# # TODO: experiment with xgboost for multivariate imputation
# multivariate_impute_pipe = ColumnTransformer(
#     [
#         ("impute_cats", SimpleImputer(strategy='constant', fill_value='missing'), cat_columns_to_impute),
#         ("impute_num", IterativeImputer(estimator=RandomForestRegressor(n_estimators=1, max_depth=30, min_samples_leaf=32), random_state=0, max_iter=1), numeric_columns_to_impute)
#     ],
#     remainder='passthrough'
# )

# split data into train, validation, test
X_train, X_test, y_train, y_test = train_test_split(properties_df.drop('logerror', axis=1), properties_df['logerror'], test_size=TEST_SIZE, random_state=RANDOM_SEED)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=TEST_SIZE, random_state=RANDOM_SEED)

# drop features
X_train_preprocessed = X_train.copy()
X_val_preprocessed = X_val.copy()
feat_dropper = FeatureDropper(features_to_drop=['parcelid'])
year_feat_creator = CreateYearFeatures(date_features=DATE_FEATURES)
date_feat_creator = CreateDateFeatures()
feature_encoder = ColumnTransformer([
    ("ohe_cats", OneHotEncoder(handle_unknown='ignore'), PROPERTIES_DF_CAT_COLUMNS)
],
    remainder='passthrough'
)
boosting_preprocessor = Pipeline(
    [
        ('feature_dropper', feat_dropper),
        ('date_feat_creator', date_feat_creator),
        ('year_feat_creator', year_feat_creator),
        ('feature_encoder', feature_encoder)
    ]
)
print("preprocessing data")
boosting_preprocessor.fit(X_train_preprocessed)
X_train_preprocessed = boosting_preprocessor.transform(X_train_preprocessed)
X_val_preprocessed = boosting_preprocessor.transform(X_val_preprocessed)
print("done preprocessing data")

X_train_xgb = X_train_preprocessed.copy()
X_val_xgb = X_val_preprocessed.copy()

xgb_base = xgb.XGBRegressor(**BEST_XGB_PARAMS)

xgb_fit_params = {
    'early_stopping_rounds': 15,
    'eval_metric': 'mae',
    'verbose': False,
    'eval_set': [[X_val_xgb, y_val]]
}
print("fitting xgboost")
xgb_base.fit(X_train_xgb, y_train, **xgb_fit_params)
print("done fitting xgboost")

# print("scoring xgboost")
# scores = -cross_val_score(xgb_base, X_train_xgb, y_train, scoring='neg_mean_absolute_error', cv=3, fit_params=xgb_fit_params)
# print("xgb scores ", scores)
# print("\nMean: ", scores.mean())
# print("\nStandard deviation: ", scores.std())

# #hyperparameter tuning
# xgb_param_grid = {
#     'learning_rate': [0.25],
#     'n_estimators': [100],
#     'max_depth': [4],
#     'colsample_bynode': [0.75],
#     'lambda': [2., 2.5],
#     'alpha': [2., 2.5],
#     'gamma': [0.201],
#     'subsample': [0.95],
#     'tree_method': ['exact'],
#     'random_state': [RANDOM_SEED]
# }
# clf = GridSearchCV(xgb.XGBRegressor(), param_grid=xgb_param_grid, scoring='neg_mean_absolute_error', cv=3)
# clf.fit(X_train_xgb, y_train)
# results = pd.DataFrame(clf.cv_results_)

print("generating predictions")
predictions_df = properties_2017[['parcelid']]
properties_2017['transactiondate'] = pd.Timestamp('2016-12-01')
properties_2017 = boosting_preprocessor.transform(properties_2017)
predictions = xgb_base.predict(properties_2017)
print("done generating predictions")
for prediction_date_string in ["20161001", "20161101", "20161201", "20171001", "20171101", "20171201"]:
    predictions_df[prediction_date_string[:6]] = predictions

print("outputting predictions to gzip")
predictions_df.to_csv('submission.gz', index=False, compression='gzip')
print("done outputting predictions to gzip")
