import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import OneHotEncoder, StandardScaler, KBinsDiscretizer, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import Ridge

from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

from sklearn.base import BaseEstimator


DATA_PATH = os.path.join("datasets", "HousingPrices")

train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
y = train.pop('SalePrice').values

test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))

#------------------------------------------------------------
vc =train['HouseStyle'].value_counts()
hs_train = train[['HouseStyle']]

ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
hs_transf = ohe.fit_transform(hs_train)
feature_names = ohe.get_feature_names()

# Проверка корректности преобразования
hs_inv = ohe.inverse_transform(hs_transf)
np.array_equal(hs_train, hs_inv)


hs_test = test[['HouseStyle']]
hs_test_tf = ohe.transform(hs_test)

#-------------------------------------------------------------------------
hs_train = train[['HouseStyle']].copy()
hs_train.iloc[0,0] = np.nan

si = SimpleImputer(strategy='constant', fill_value='MISSING')
hs_imp = si.fit_transform(hs_train)
hs_transf = ohe.fit_transform(hs_imp)
ohe.get_feature_names()

hs_test = test[['HouseStyle']].copy()
hs_test.iloc[0,0] = 'unique value to test set'
hs_test.iloc[1,0] = np.nan
hs_test_imp = si.transform(hs_test)
hs_test_tf = ohe.transform(hs_test_imp)


#--------------------------------------------------------------------------
pipe = Pipeline([
    ('si', SimpleImputer(strategy='constant', fill_value='MISSING')),
    ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore')),
])

hs_train = train[['HouseStyle']].copy()
hs_train.iloc[0,0] = np.nan
hs_transf = pipe.fit_transform(hs_train)

hs_test = test[['HouseStyle']].copy()
hs_test_tf = pipe.transform(hs_test)

pipe.named_steps['ohe'].get_feature_names()

#--------------------------------------------------------------------------------
cat_pipe = Pipeline([
    ('si', SimpleImputer(strategy='constant', fill_value='MISSING')),
    ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))
])

ct = ColumnTransformer([
            ("cat", cat_pipe, ['RoofMatl', 'HouseStyle']),
        ],
    remainder='passthrough')

X_cat_tf = ct.fit_transform(train)
X_cat_tf_test = ct.transform(test)

pl = ct.named_transformers_['cat']
ohe = pl.named_steps['ohe']
ohe.get_feature_names()

feature_names = ct.named_transformers_['cat'].named_steps['ohe'].get_feature_names()

#------------------------------------------------------------------------------
# выводим тип каждой переменной в виде одной буквы
kinds = np.array([dt.kind for dt in train.dtypes])

all_columns = train.columns.values
is_num = (kinds == 'i') | (kinds == 'f')
num_cols = all_columns[is_num]

# cat_cols = all_columns[~is_num]
cat_cols = all_columns[kinds == 'O']

num_pipe = Pipeline([
    ('si', SimpleImputer(strategy='mean')),
    ('ss', StandardScaler())
])
ct = ColumnTransformer([
    ('num', num_pipe, num_cols)
])

X_num_tf = ct.fit_transform(train)

#------------------------------------------------------------------------------
ct = ColumnTransformer([
    ('cat', cat_pipe, cat_cols),
    ('num', num_pipe, num_cols)
])

X = ct.fit_transform(train)

#------------------------------------------------------------------------------
ml_pipe = Pipeline([
    ('transform', ct),
    ('ridge', Ridge())
])

ml_pipe.fit(train, y)
ml_pipe.score(train, y)

#-------------------------------------------------------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=123)
cross_val_score(ml_pipe, train, y, cv=kf).mean()

#-------------------------------------------------------------------------------
param_grid = {
'transform__num__si__strategy': ['mean', 'median'],
'ridge__alpha': [.001, 0.1, 1.0, 5, 10, 50, 100, 1000]
}

gs = GridSearchCV(ml_pipe, param_grid, cv=kf, return_train_score=True)
gs.fit(train, y)
gs.best_params_
gs.best_score_

cv_results = pd.DataFrame(gs.cv_results_)
#------------------------------------------------------------------------------
class BasicTransformer(BaseEstimator):

    def __init__(self, cat_threshold=None, num_strategy='median', return_df=False):
        # храним параметры как публичные аттрибуты
        self.cat_threshold = cat_threshold

        if num_strategy not in ['median', 'mean']:
            raise ValueError('num_strategy must be either "mean" or "median"')

        self.num_strategy = num_strategy
        self.return_df = return_df

    def fit(self, X, y=None):
        # подразумевает, что X - это объект DataFrame
        self._columns = X.columns.values

        # разбиваем данные на категориальные и количественные признаки
        self._dtypes = X.dtypes.values
        self._kinds = np.array([dt.kind for dt in X.dtypes])
        self._column_dtypes = {}
        is_cat = self._kinds == 'O'
        self._column_dtypes['cat'] = self._columns[is_cat]
        self._column_dtypes['num'] = self._columns[~is_cat]
        self._feature_names = self._column_dtypes['num']

        # создаем словарь на основе категориального признака,
        # где ключом будет уникальное значение выше порога
        self._cat_cols = {}
        for col in self._column_dtypes['cat']:
            vc = X[col].value_counts()
        if self.cat_threshold is not None:
            vc = vc[vc > self.cat_threshold]
        vals = vc.index.values
        self._cat_cols[col] = vals
        self._feature_names = np.append(self._feature_names, col + '_' + vals)

        # вычисляем общее количество новых категориальных признаков
        self._total_cat_cols = sum([len(v) for col, v in self._cat_cols.items()])

        # вычисляем среднее или медиану
        self._num_fill = X[self._column_dtypes['num']].agg(self.num_strategy)
        return self

    def transform(self, X):

        # проверяем, есть ли у нас объект DataFrame с теми же именами столбцов,
        # что и в том объекте DataFrame, который использовался для обучения
        if set(self._columns) != set(X.columns):
            raise ValueError('Passed DataFrame has diff cols than fit DataFrame')
        elif len(self._columns) != len(X.columns):
            raise ValueError('Passed DataFrame has diff number of cols than fit DataFrame')

        # заполняем пропуски
        X_num = X[self._column_dtypes['num']].fillna(self._num_fill)

        # стандартизируем количественные признаки
        std = X_num.std()
        X_num = (X_num - X_num.mean()) / std
        zero_std = np.where(std == 0)[0]

        # Если стандартное отклонение 0, то все значения идентичны. Задаем их равными 0.
        if len(zero_std) > 0:
            X_num.iloc[:, zero_std] = 0
        X_num = X_num.values

        # создаем отдельный массив для преобразованных категориальных признаков
        X_cat = np.empty((len(X), self._total_cat_cols), dtype='int')
        i = 0
        for col in self._column_dtypes['cat']:
            vals = self._cat_cols[col]
        for val in vals:
            X_cat[:, i] = X[col] == val
        i += 1

        # конкатенируем преобразованные количественные и категориальные признаки
        data = np.column_stack((X_num, X_cat))

        # возвращаем либо DataFrame, либо массив
        if self.return_df:
            return pd.DataFrame(data=data, columns=self._feature_names)
        else:
            return data

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def get_feature_names(self):
        return self._feature_names

bt = BasicTransformer(cat_threshold=3, return_df=True)
# train_tf = bt.fit_transform(train)

basic_pipe = Pipeline([('bt', bt), ('ridge', Ridge())])
# basic_pipe.fit(train, y)
# basic_pipe.score(train, y)
#
# cross_val_score(basic_pipe, train, y, cv=kf).mean()

param_grid = {
    'bt__cat_threshold': [0, 1, 2, 3, 5],
    'ridge__alpha': [.1, 1, 10, 100]
}
gs = GridSearchCV(basic_pipe, param_grid, cv=kf)
gs.fit(train, y)
gs.best_params_
gs.best_score_

#-------------------------------------------------------------

kbd = KBinsDiscretizer(encode='onehot-dense')
# выполняем биннинг
year_built_transformed = kbd.fit_transform(train[['YearBuilt']])

train['YearBuilt'].value_counts()
year_built_transformed
kbd._encoder.get_feature_names()

num_highposskew_pipe = Pipeline([
    ('sqrt', FunctionTransformer(np.sqrt, validate=False)),
    ('kbd', KBinsDiscretizer(n_bins=5, encode='onehot-dense'))
])
year_built_transformed1 = num_highposskew_pipe.fit_transform(train[['YearBuilt']])
num_highposskew_pipe.named_steps['kbd']._encoder.get_feature_names()


# убедимся, что бины содержать примерно
# одинаковое количество наблюдений
year_built_transformed.sum(axis=0)

# посмотрим на границы бинов
kbd.bin_edges_

# выделяем переменные с годами в отдельный список
year_cols = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']
# создаем булев массив
not_year = ~np.isin(num_cols, year_cols + ['Id'])
# выделяем количественные переменные, исключив переменные с годами и Id
num_cols2 = num_cols[not_year]

year_pipe = Pipeline([
    ('si', SimpleImputer(strategy='median')),
    ('kbd', KBinsDiscretizer(n_bins=5, encode='onehot-dense')),
])
cat_pipe = Pipeline([
    ('si', SimpleImputer(strategy='constant', fill_value='MISSING')),
    ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore')),
])
num_pipe = Pipeline([
    ('si', SimpleImputer(strategy='mean')),
    ('ss', StandardScaler())
])

ct = ColumnTransformer([
    ('cat', cat_pipe, cat_cols),
    ('num', num_pipe, num_cols2),
    ('year', year_pipe, year_cols)
])

X = ct.fit_transform(train)

ml_pipe = Pipeline([('transform', ct), ('ridge', Ridge())])
# cross_val_score(ml_pipe, train, y, cv=kf).mean()
param_grid = {
'transform__year__kbd__n_bins': [4, 6, 8, 10],
'ridge__alpha': [.1, .5, 1, 5, 10, 100]
}
gs = GridSearchCV(ml_pipe, param_grid, cv=kf)
gs.fit(train, y)
gs.best_params_
gs.best_score_