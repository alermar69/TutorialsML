import pandas as pd
import numpy as np
from handson.GetData import load_housing_data
from handson.TransformerData import housingPrepared, splitData
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

df = load_housing_data()

df.info()
df['ocean_proximity'].value_counts()
col =list(df)
desc = df.describe()
df['households'].hist(bins=50, figsize=(15,7))
plt.hist(np.log(df['households']),bins=50)

df.hist(bins=50, figsize=(15,7))
# df['median_income'].hist(bins=50, figsize=(15,7))
plt.show()

start_train, start_test = splitData(df)

df_train = start_train.copy()
df_train = pd.DataFrame(df_train)
df_train.plot(kind='scatter', x='longitude', y='latitude')
plt.show()

df_train.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, figsize=(10,7),
       s=df_train['population']/100, label='population',
       c=df_train['median_house_value'], cmap='jet', colorbar=True)
plt.show()

corr_matrix = df_train.corr()


scatter_matrix(df_train[['median_house_value', 'median_income',
                   'total_rooms', 'housing_median_age']])

df_train.plot(kind='scatter', x='median_income' , y='median_house_value', alpha=0.1);

df_train["rooms_per_household"] = df_train["total_rooms"]/df_train["households"]
df_train["bedrooms_per_room"] = df_train["total_bedrooms"]/df_train["total_rooms"]
df_train["population_per_household"]=df_train["population"]/df_train["households"]

corr_matrix = df_train.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

df_train.plot(kind="scatter", x="rooms_per_household", y="median_house_value",
             alpha=0.2)
plt.axis([0, 5, 0, 520000])




ix = dict(zip(list(df),  list(range(0, len(df.columns)))))


def log_features(X, columns):
    # columns = ['median_income', 'households', 'population', 'total_bedrooms', 'total_rooms']
    X[columns] = np.log(X[columns])
    return X

df1 = df.copy()
log_features(df1)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer


columns = ['median_income', 'households', 'population', 'total_bedrooms', 'total_rooms']

pipeline1 = Pipeline([('log', FunctionTransformer(np.log1p, validate=False)), ])
pipeline1.fit_transform(df)

num_attribs = list(df.drop('ocean_proximity', axis=1))
cat_attribs = ["ocean_proximity"]



full_pipeline = ColumnTransformer([
        ("num", pipeline1, columns),
    ])

df1 = full_pipeline.fit_transform(df)
df2 = pd.DataFrame(df1, columns=columns)
df3 = df.copy()

df3[columns] = df2

df.dtypes
df2[columns] = df2[columns].astype(float)
df2.dtypes

df3.hist(bins=50, figsize=(15,7))
plt.show()




from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer


def log_features(X):
    return np.log1p(X)

def log_features(X):
    return np.log1p(X)

num_pipe = Pipeline([
    ("num", SimpleImputer()),
    ('log', FunctionTransformer(log_features))
])

ct = ColumnTransformer([
    ("num",  SimpleImputer(), num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])
ct1 = ColumnTransformer([
    ('log', FunctionTransformer(log_features), columns)
])

pipe = Pipeline([
    ('ct', ct),
    ('ct1', ct1),
])

df1 = df.copy()
pipe.fit_transform(df1)

df1.hist(bins=50, figsize=(15,7))
plt.show()