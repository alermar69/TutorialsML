import pandas as pd
import numpy as np
from handson.GetData import *
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

#fetch_housing_data()
df = load_housing_data()

df.info()
df['ocean_proximity'].value_counts()
df.describe()

df.hist(bins=50, figsize=(15,7))
plt.show()

df['income_cat'] = np.ceil(df['median_income'] / 1.5)
df['income_cat'].where(df['income_cat']<5, 5, inplace=True)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df['income_cat']):
    start_train = df.loc[train_index]
    start_test = df.loc[test_index]

#ls = list(split.split(df, df['income_cat']))

df['income_cat'].value_counts() / len(df)

del start_train['income_cat']
del start_test['income_cat']

df = start_train.copy()

df.plot(kind='scatter', x='longitude', y='latitude')
plt.show()

df.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, figsize=(10,7),
       s=df['population']/100, label='population',
       c=df['median_house_value'], cmap='jet', colorbar=True)
plt.show()

corr_matrix = df.corr()

from pandas.plotting import scatter_matrix
scatter_matrix(df[['median_house_value', 'median_income',
                   'total_rooms', 'housing_median_age']])

df.plot(kind='scatter', x='median_income' , y='median_house_value', alpha=0.1);

df["rooms_per_household"] = df["total_rooms"]/df["households"]
df["bedrooms_per_room"] = df["total_bedrooms"]/df["total_rooms"]
df["population_per_household"]=df["population"]/df["households"]

corr_matrix = df.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

df.plot(kind="scatter", x="rooms_per_household", y="median_house_value",
             alpha=0.2)
plt.axis([0, 5, 0, 520000])

#-------------Подготовка данных-------------------

X_train = start_train.drop('median_house_value', axis=1)
y_train = start_train['median_house_value'].copy()

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")

X_train_num = X_train.drop('ocean_proximity', axis=1)
imputer.fit(X_train_num)
imputer.transform(X_train_num)

X_train_num = pd.DataFrame(imputer.fit_transform(X_train_num), columns=X_train_num.columns)

X_train_num.info()

X_train_cat = X_train['ocean_proximity']
X_train_cat_encode, X_train_cat_categories = X_train_cat.factorize()


#pd.get_dummies(X_train_cat)
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
X_train_cat_1hot = encoder.fit_transform(X_train_cat[:,np.newaxis]).toarray()


from sklearn.base import BaseEstimator, TransformerMixin

# column index
col_names = "total_rooms", "total_bedrooms", "population", "households"
rooms_ix, bedrooms_ix, population_ix, households_ix = [
    X_train.columns.get_loc(c) for c in col_names] # get the column indices

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(X_train.values)

housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(X_train.columns)+["rooms_per_household", "population_per_household"],
    index=X_train.index)



from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(X_train_num)


from sklearn.compose import ColumnTransformer

num_attribs = list(X_train_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(X_train)