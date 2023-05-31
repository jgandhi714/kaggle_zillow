import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from feature_dropper import FeatureDropper
from feature_engineering import CreateYearFeatures, CreateDateFeatures, CreateDerivedFeatures

warnings.filterwarnings('ignore')

color = sns.color_palette()
sns.set_style('darkgrid')

RANDOM_SEED = 10
PROPERTIES_DF_CATEGORY_COLUMNS = ['airconditioningtypeid', 'buildingqualitytypeid', 'fips', 'heatingorsystemtypeid',
                                  'propertycountylandusecode', 'propertylandusetypeid', 'propertyzoningdesc',
                                  'regionidcity', 'regionidneighborhood', 'regionidzip']
PROPERTIES_DF_REDUNDANT_COLUMNS_TO_DROP = ['calculatedbathnbr', 'poolcnt', 'hashottuborspa', 'taxdelinquencyyear',
                                           'rawcensustractandblock', 'threequarterbathnbr', 'taxvaluedollarcnt']
# properties with mostly null values that can't be reliably imputed
PROPERTIES_DF_NULL_COLUMNS_TO_DROP = ['architecturalstyletypeid', 'assessmentyear', 'basementsqft', 'buildingclasstypeid',
                                      'finishedfloor1squarefeet', 'finishedsquarefeet6', 'finishedsquarefeet13',
                                      'finishedsquarefeet15', 'finishedsquarefeet50', 'storytypeid',
                                      'typeconstructiontypeid', 'yardbuildingsqft26', 'numberofstories', 'censustractandblock']
ALL_COLUMNS_TO_DROP = PROPERTIES_DF_REDUNDANT_COLUMNS_TO_DROP + PROPERTIES_DF_NULL_COLUMNS_TO_DROP
ZERO_IMPUTATION_COLUMNS = ['fireplacecnt', 'pooltypeid2', 'pooltypeid7', 'pooltypeid10', 'decktypeid', 'poolsizesum', 'yardbuildingsqft17']
ONE_IMPUTATION_COLUMNS = ['fullbathcnt', 'unitcnt', 'bedroomcnt', 'bathroomcnt']
MODE_IMPUTATION_COLUMNS = PROPERTIES_DF_CATEGORY_COLUMNS
CONVERT_TO_BOOL_COLUMNS = ['fireplaceflag', 'taxdelinquencyflag']
UNIVARIATE_IMPUTATION_COLUMNS = ZERO_IMPUTATION_COLUMNS + ONE_IMPUTATION_COLUMNS + MODE_IMPUTATION_COLUMNS
IMPUTE_XGB_PARAMS = {'max_depth': 3, 'alpha': 2.}  # arbitrary params
PREDICT_XGB_PARAMS = {'eta': 0.2, 'lambda': 2.5, 'alpha': 2.5, 'max_depth': 2}
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
    # properties_df = properties_df.drop(PROPERTIES_DF_REDUNDANT_COLUMNS_TO_DROP, axis=1)
    # drop columns with mostly null values that can't be reliably imputed
    # properties_df = properties_df.drop(PROPERTIES_DF_NULL_COLUMNS_TO_DROP, axis=1)
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


# TODO: change to transformer?
def drop_outliers(train_df: pd.DataFrame, num_stdevs: int) -> pd.DataFrame:
    mean = train_df['logerror'].mean()
    std = train_df['logerror'].std()
    high_thresh = mean + num_stdevs * std
    low_thresh = mean - num_stdevs * std
    return train_df[train_df['logerror'].between(low_thresh, high_thresh)]


def convert_columns_to_bool(properties_df: pd.DataFrame, columns_to_convert: list[str]) -> pd.DataFrame:
    for column_to_convert in columns_to_convert:
        properties_df[column_to_convert] = properties_df[column_to_convert].map({'Y': True}).fillna(False).astype(bool)
    return properties_df


def impute_properties_df(df: pd.DataFrame) -> pd.DataFrame:
    df = convert_columns_to_bool(df, CONVERT_TO_BOOL_COLUMNS)
    df['calculatedfinishedsquarefeet'] = df['calculatedfinishedsquarefeet'].combine_first(
        df['finishedsquarefeet12'])
    df.loc[df['fireplaceflag'], 'fireplacecnt'] = df.loc[
        df['fireplaceflag'], 'fireplacecnt'].fillna(1)
    df = df.drop(['fireplaceflag', 'finishedsquarefeet12'], axis=1)

    df[ZERO_IMPUTATION_COLUMNS] = df[ZERO_IMPUTATION_COLUMNS].fillna(0)
    df[ONE_IMPUTATION_COLUMNS] = df[ONE_IMPUTATION_COLUMNS].fillna(1)
    df[ONE_IMPUTATION_COLUMNS] = df[ONE_IMPUTATION_COLUMNS].replace(0, 1)
    df[MODE_IMPUTATION_COLUMNS] = df[MODE_IMPUTATION_COLUMNS].fillna(
        df[MODE_IMPUTATION_COLUMNS].mode().iloc[0])
    for category_col in PROPERTIES_DF_CATEGORY_COLUMNS:
        df[category_col] = pd.Categorical(df[category_col],
                                          categories=df[category_col].unique())

    # use the already imputed columns to impute the remaining columns
    impute_x_df = df[UNIVARIATE_IMPUTATION_COLUMNS]
    # use xgboost to impute the rest of the missing values in each column
    columns_to_impute = (set(df.columns) - set(UNIVARIATE_IMPUTATION_COLUMNS))
    columns_to_impute.remove('parcelid')
    columns_to_impute.remove('taxdelinquencyflag')
    for col in columns_to_impute:
        print("imputing column: " + col)
        impute_train_df = impute_x_df.copy(deep=True)
        impute_train_df[col] = df[col]
        impute_train_df = impute_train_df.dropna()
        impute_train_data_dmatrix = xgb.DMatrix(data=impute_train_df[UNIVARIATE_IMPUTATION_COLUMNS],
                                                label=impute_train_df[col].values, enable_categorical=True)
        impute_test_data_dmatrix = xgb.DMatrix(data=impute_x_df, enable_categorical=True)
        xgb_model = xgb.train(dtrain=impute_train_data_dmatrix, params=IMPUTE_XGB_PARAMS)
        predictions = pd.Series(xgb_model.predict(impute_test_data_dmatrix))
        df[col] = df[col].fillna(predictions)
        print("done imputing column: " + col)
    return df


def engineer_features_properties_df(properties_df: pd.DataFrame) -> pd.DataFrame:
    properties_df['halfbathcnt'] = properties_df.bathroomcnt - properties_df.fullbathcnt
    properties_df['unfinished_sqft'] = properties_df.lotsizesquarefeet - properties_df.calculatedfinishedsquarefeet
    properties_df['unfinished_sqft_pct'] = properties_df.unfinished_sqft / properties_df.lotsizesquarefeet
    properties_df['finishedareapct'] = properties_df.calculatedfinishedsquarefeet / properties_df.lotsizesquarefeet
    properties_df['property_tax_per_sqft'] = properties_df.taxamount / properties_df.calculatedfinishedsquarefeet
    properties_df['avg_finished_area_per_room'] = properties_df.calculatedfinishedsquarefeet / properties_df.bedroomcnt
    return properties_df


def merge_df_with_csvs(properties_df: pd.DataFrame, csvs: list[str]) -> pd.DataFrame:
    print("joining properties dataframe and training data from csvs " + str(csvs))
    merged_df = pd.concat([pd.read_csv(train_csv) for train_csv in csvs])
    # merged_df['transactiondate'] = pd.to_datetime(merged_df['transactiondate'])
    # merged_df['transaction_month'] = ((merged_df.transactiondate - merged_df.transactiondate.min()) // np.timedelta64(1, 'M')) + 1
    # merged_df = merged_df.drop("transactiondate", axis=1)
    merged_df = merged_df.merge(properties_df, how='left', on='parcelid')
    print("done joining properties dataframe and training data from csvs " + str(csvs))
    return merged_df


# load data
properties_df = load_properties_df('properties_2017.csv')
properties_df = merge_df_with_csvs(properties_df, ["train_2016_v2.csv", "train_2017.csv"])

# remove outliers and check for normality
# target_variable_eda(properties_df.logerror)
properties_df = drop_outliers(properties_df, 2.2)
print("Finished dropping outliers")
# target_variable_eda(properties_df.logerror)

# # imputation pipelines
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
# cat_columns_to_impute = [col for col in columns_to_impute if col in PROPERTIES_DF_CATEGORY_COLUMNS]
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
cat_vars = ['transaction_month',
            'airconditioningtypeid', 'architecturalstyletypeid', 'buildingclasstypeid', 'buildingqualitytypeid',
            'decktypeid', 'fips',  'hashottuborspa', 'heatingorsystemtypeid',  'pooltypeid10', 'pooltypeid2',
            'pooltypeid7', 'propertycountylandusecode', 'propertylandusetypeid', 'propertyzoningdesc',
            'rawcensustractandblock', 'regionidcity', 'regionidcounty', 'regionidneighborhood', 'regionidzip',
            'storytypeid',  'typeconstructiontypeid', 'fireplaceflag',  'assessmentyear',
            'taxdelinquencyflag', 'taxdelinquencyyear', 'censustractandblock',
           ]
feature_encoder = ColumnTransformer([
    ("ohe_cats", OneHotEncoder(handle_unknown='ignore'), cat_vars)
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
# xgb_preprocessor.fit(X_val_preprocessed)
X_val_preprocessed = boosting_preprocessor.transform(X_val_preprocessed)
print("done preprocessing data")

X_train_xgb = X_train_preprocessed.copy()
X_val_xgb = X_val_preprocessed.copy()

xgb_params = {
    'learning_rate': 0.3,
    'n_estimators': 10000,
    'random_state': RANDOM_SEED
}

xgb_base = xgb.XGBRegressor(**BEST_XGB_PARAMS)

xgb_fit_params = {
    'early_stopping_rounds': 15,
    'eval_metric': 'mae',
    'verbose': False,
    'eval_set': [[X_val_xgb, y_val]]
}
print("fitting xgboost")
xgb_base.fit(X_train_xgb, y_train, **xgb_fit_params)
print("scoring xgboost")
scores = -cross_val_score(xgb_base, X_train_xgb, y_train, scoring='neg_mean_absolute_error', cv=3, fit_params=xgb_fit_params)
print("xgb scores ", scores)
print("\nMean: ", scores.mean())
print("\nStandard deviation: ", scores.std())

# xgb_param_grid = {
#     'learning_rate': [0.3],
#     'n_estimators': [10000],
#     'max_depth': list(range(1,7)),
#     'random_state': [RANDOM_SEED]
# }
# clf = GridSearchCV(xgb.XGBRegressor(), param_grid=xgb_param_grid, scoring='neg_mean_absolute_error', cv=3)
# clf.fit(X_train_xgb, y_train)

print("generating predictions")
predictions_df = properties_df[['parcelid']]
x_train = properties_df.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
# y_train = properties_df['logerror'].values
# data_dmatrix = xgb.DMatrix(data=x_train, label=y_train, enable_categorical=True)
# xgb_model = xgb.train(PREDICT_XGB_PARAMS, dtrain=data_dmatrix, num_boost_round=35)

for prediction_date_string in ["20161001", "20161101", "20161201", "20171001", "20171101", "20171201"]:
    # x_train = properties_df.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
    # transaction_month = ((pd.to_datetime(prediction_date_string) - properties_df.transactiondate.min()) // np.timedelta64(1, 'M')) + 1
    # x_train['transaction_month'] = transaction_month
    predictions_df[prediction_date_string[:6]] = xgb_base.predict(
        xgb.DMatrix(x_train.drop('parcelid', axis=1), enable_categorical=True))
print("done generating predictions")
print("outputting predictions to gzip")
predictions_df.to_csv('submission.gz', index=False, compression='gzip')






properties_df = impute_properties_df(properties_df)
properties_df = engineer_features_properties_df(properties_df)
properties_df = merge_df_with_csvs(properties_df, ["train_2016_v2.csv", "train_2017.csv"])
target_variable_eda(properties_df.logerror)
properties_df = drop_outliers(properties_df, 2.2)

X_train, X_test, y_train, y_test = train_test_split(properties_df.drop(['parcelid', 'logerror', 'transactiondate'], axis=1), )

# predictions_df = properties_df[['parcelid']]
x_train = properties_df.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
# y_train = properties_df['logerror'].values
# data_dmatrix = xgb.DMatrix(data=x_train, label=y_train, enable_categorical=True)
# xgb_model = xgb.train(PREDICT_XGB_PARAMS, dtrain=data_dmatrix, num_boost_round=35)

for prediction_date_string in ["20161001", "20161101", "20161201", "20171001", "20171101", "20171201"]:
    # x_train = properties_df.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
    # transaction_month = ((pd.to_datetime(prediction_date_string) - properties_df.transactiondate.min()) // np.timedelta64(1, 'M')) + 1
    # x_train['transaction_month'] = transaction_month
    predictions_df[prediction_date_string[:6]] = xgb_model.predict(
        xgb.DMatrix(x_train.drop('parcelid', axis=1), enable_categorical=True))

predictions_df.to_csv('submission.gz', index=False, compression='gzip')

# # data imputation
# # TODO: try to impute these rather than dropping
# properties_df_all.loc[properties_df_all['bathroomcnt'] == 0 | properties_df_all['bathroomcnt'].isna(), 'bathroomcnt'] = \
#     properties_df_all.loc[properties_df_all['bathroomcnt'] != 0, 'bathroomcnt'].mode()[0]
# # data imputation
# properties_df_all.calculatedfinishedsquarefeet = properties_df_all.calculatedfinishedsquarefeet.combine_first(
#     properties_df_all.finishedsquarefeet12)
# properties_df_all.calculatedfinishedsquarefeet = properties_df_all.calculatedfinishedsquarefeet.fillna(
#     properties_df_all.calculatedfinishedsquarefeet.median())
#
# properties_df_all.loc[properties_df_all.fireplaceflag == True, 'fireplacecnt'] = properties_df_all.loc[
#     properties_df_all.fireplaceflag == True, 'fireplacecnt'].fillna(1)
# # drop newly redundant columns
# properties_df_all = properties_df_all.drop(['finishedsquarefeet12', 'fireplaceflag'], axis=1)
#
# # TODO: fillna() with less calls
# properties_df_all.fireplacecnt = properties_df_all.fireplacecnt.fillna(0)
# properties_df_all.airconditioningtypeid = properties_df_all.airconditioningtypeid.fillna(
#     properties_df_all.airconditioningtypeid.mode()[0])
# properties_df_all.buildingqualitytypeid = properties_df_all.buildingqualitytypeid.fillna(0)
# properties_df_all.fullbathcnt = properties_df_all.fullbathcnt.fillna(1)
# properties_df_all.garagecarcnt = properties_df_all.garagecarcnt.fillna(0)
# properties_df_all.heatingorsystemtypeid = properties_df_all.heatingorsystemtypeid.fillna(
#     properties_df_all.heatingorsystemtypeid.mode()[0])
# properties_df_all.lotsizesquarefeet = properties_df_all.lotsizesquarefeet.combine_first(
#     properties_df_all.calculatedfinishedsquarefeet)
# properties_df_all.pooltypeid10 = properties_df_all.pooltypeid10.fillna(0)
# properties_df_all.pooltypeid2 = properties_df_all.pooltypeid2.fillna(0)
# properties_df_all.pooltypeid7 = properties_df_all.pooltypeid7.fillna(0)
# properties_df_all.regionidneighborhood = properties_df_all.regionidneighborhood.fillna(0)
# properties_df_all.unitcnt = properties_df_all.unitcnt.fillna(1)
# properties_df_all.yardbuildingsqft17 = properties_df_all.yardbuildingsqft17.fillna(0)
# properties_df_all.numberofstories = properties_df_all.numberofstories.fillna(1)

# garage_size_df = properties.loc[(properties.garagetotalsqft != 0) & (properties.garagecarcnt != 0)]
# # properties = properties[properties.bathroomcnt != 0 & ~properties.bathroomcnt.isna()]
# garage_size_train_df = garage_size_df[
#     ['garagecarcnt', 'calculatedfinishedsquarefeet', 'garagetotalsqft']].dropna()
# garage_size_train_x_df = garage_size_train_df[['garagecarcnt', 'calculatedfinishedsquarefeet']]
# garage_size_train_y_df = garage_size_train_df['garagetotalsqft']
# imputer = LinearRegression()  # TODO: this can be changed to something like miss forest for better results
# imputer.fit(garage_size_train_x_df, garage_size_train_y_df)
# properties['garagetotalsqft'] = properties['garagetotalsqft'].fillna(
#     pd.Series(imputer.predict(
#         properties[properties['garagetotalsqft'].isna()][['garagecarcnt', 'calculatedfinishedsquarefeet']])))
# properties.loc[
#     (properties.garagecarcnt > 0) & (properties.garagetotalsqft == 0), 'garagetotalsqft'] = pd.Series(
#     imputer.predict(
#         properties.loc[(properties.garagecarcnt > 0) & (properties.garagetotalsqft == 0),
#         ['garagecarcnt', 'calculatedfinishedsquarefeet']]))

# for col in properties_df_all.columns:
#     properties_df_all[col] = properties_df_all[col].fillna(properties_df_all[col].mode()[0])
#
# properties_df_all[
#     ['airconditioningtypeid', 'buildingqualitytypeid', 'fips', 'heatingorsystemtypeid', 'propertycountylandusecode',
#      'propertylandusetypeid', 'propertyzoningdesc', 'regionidcity', 'regionidneighborhood', 'regionidzip',
#      'censustractandblock']] = properties_df_all[
#     ['airconditioningtypeid', 'buildingqualitytypeid', 'fips', 'heatingorsystemtypeid', 'propertycountylandusecode',
#      'propertylandusetypeid', 'propertyzoningdesc', 'regionidcity', 'regionidneighborhood', 'regionidzip',
#      'censustractandblock']].astype('category')
# properties_df_all['halfbathcnt'] = properties_df_all.bathroomcnt - properties_df_all.fullbathcnt
# properties_df_all[
#     'unfinishedsquarefeet'] = properties_df_all.lotsizesquarefeet - properties_df_all.calculatedfinishedsquarefeet
# properties_df_all[
#     'finishedareapct'] = properties_df_all.calculatedfinishedsquarefeet / properties_df_all.lotsizesquarefeet
# properties_df_all = properties_df_all.drop(['lotsizesquarefeet', 'bathroomcnt'], axis=1)
#
# # load training data and merge with properties
# properties_df_2016_train, properties_df_2016_test = properties_df_2016_train[
#     properties_df_2016_train['transactiondate'] < pd.to_datetime('2016-10-15')], properties_df_2016_train[
#     properties_df_2016_train['transactiondate'] >= pd.to_datetime('2016-10-15')]
# train_2016 = pd.read_csv('train_2016_v2.csv')
# df_train = train_2016.merge(properties_df_all, how='left', on='parcelid')
#
# # change transaction date to months since dec 31 2015
# df_train['transactiondate'] = pd.to_datetime(df_train['transactiondate'])  # TODO: do this when loading in properties
# min_date = df_train.transactiondate.min()
# df_train['transaction_month'] = ((df_train.transactiondate - min_date) // np.timedelta64(1, 'M')) + 1
# df_train = df_train.drop(['transactiondate'], axis=1)

# x_train = df_train.drop(['parcelid', 'logerror'], axis=1)
# y_train = df_train['logerror'].values
# data_dmatrix = xgb.DMatrix(data=x_train, label=y_train, enable_categorical=True)
#
# params = {'eta': 0.2, 'lambda': 2.5, 'alpha': 2.5, 'max_depth': 2}
# xgb_model = xgb.train(params, dtrain=data_dmatrix, num_boost_round=35)
#
# predictions_df = properties_df_all[['parcelid']]
# for prediction_date_string in ["20161001", "20161101", "20161201", "20171001", "20171101", "20171201"]:
#     x_train = properties_df_all
#     transaction_month = ((pd.to_datetime(prediction_date_string) - min_date) // np.timedelta64(1, 'M')) + 1
#     x_train['transaction_month'] = transaction_month
#     predictions_df[prediction_date_string[:6]] = xgb_model.predict(
#         xgb.DMatrix(x_train.drop('parcelid', axis=1), enable_categorical=True))
#
# predictions_df.to_csv('first_submission.gz', index=False, compression='gzip')

# param_grid = {
#     'eta': [0.2],
#     'lambda': [2.5],
#     'alpha': [2.5],
#     'max_depth': [2]
# }
# param_grid_list = list(ParameterGrid(param_grid))
# results_dict = {}
# for i, params in enumerate(param_grid_list):
#     print("testing " + str(params))
#     cv_results = xgboost.cv(dtrain=data_dmatrix, params=params, nfold=5, seed=1, metrics=['mae'],
#                             num_boost_round=35)
#     results_dict[i] = {'params': params, 'cv_results': cv_results}
#
# df = pd.DataFrame()
# for round_num, round_results in results_dict.items():
#
#     # extract the parameter dictionary for this boosting round
#     params_dict = round_results['params']
#
#     # extract the cv_results for this boosting round
#     cv_results = round_results['cv_results']
#
#     # iterate over the rows in cv_results
#     for i in range(len(cv_results)):
#
#         # create a dictionary to store the row data
#         row_dict = {}
#         row_dict['boosting_round'] = [i]
#
#         # add the parameter values to the row dictionary
#         for param_name, param_value in params_dict.items():
#             row_dict[param_name] = [param_value]
#
#         # add the cv_results to the row dictionary
#         for result_name, result_values in cv_results.items():
#             row_dict[result_name] = [result_values[i]]
#
#         # add the row dictionary to the DataFrame
#         df = pd.concat([df, pd.DataFrame.from_dict(row_dict)], ignore_index=True)
