import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestRegressor

from HousingPrices.GetData import load_data

from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, FunctionTransformer, KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import Ridge

from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

from sklearn.base import BaseEstimator

kf = KFold(n_splits=5, shuffle=True, random_state=123)

train = load_data()

# создаем массив меток
y = train.pop('SalePrice').values

# удаляем переменную Id
train.drop('Id', axis=1, inplace=True)

# выделим категориальные и количественные признаки
cat_columns = train.dtypes[train.dtypes == 'object'].index
num_columns = train.dtypes[train.dtypes != 'object'].index

# вычислим коэффициент асимметрии для количественных признаков
for i in num_columns:
    print(i, train[i].skew())

# выделяем список признаков с небольшой отрицательной асимметрией
neg_skew_num_columns = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'GarageCars']

# выделяем список признаков с высокой положительной асимметрией
high_pos_skew_num_columns = ['MiscVal', 'PoolArea', '3SsnPorch', 'LotArea', 'LowQualFinSF']

# создадим булев массив
not_neg_high_pos_skew_num_columns = ~np.isin(
    num_columns, high_pos_skew_num_columns + neg_skew_num_columns)

# из списка количественных признаков удалим количественные признаки
# с небольшой отрицательной и высокой положительной асимметрией
num_columns = num_columns[not_neg_high_pos_skew_num_columns]

# создаем конвейер преобразований для количественных признаков
# с небольшой отрицательной асимметрией
num_negskew_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('square', FunctionTransformer(np.square, validate=False)),
    ('scaler', StandardScaler())
])

# создаем конвейер преобразований для количественных признаков
# с небольшой и средней положительной асимметрией
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('log', FunctionTransformer(np.log1p, validate=False)),
    ('scaler', RobustScaler())
])

# создаем конвейер преобразований для количественных
# признаков с высокой асимметрией
num_highposskew_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('sqrt', FunctionTransformer(np.sqrt, validate=False)),
    ('kbd', KBinsDiscretizer(n_bins=5, encode='onehot-dense'))
])

# создаем конвейер преобразований для категориальных признаков
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
    ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))
])
transformers = [('num_negskew', num_negskew_pipe, neg_skew_num_columns),
                ('num', num_pipe, num_columns),
                ('num_highposskew', num_highposskew_pipe, high_pos_skew_num_columns),
                ('cat', cat_pipe, cat_columns)]
transformer = ColumnTransformer(transformers=transformers)

ml_pipe = Pipeline([('transform', transformer), ('ridge', Ridge())])
cross_val_score(ml_pipe, train, y, cv=kf).mean()

param_grid = {
    'ridge__alpha': [.1, .5, 1, 5, 10, 100],
    'transform__num_highposskew__kbd__n_bins': [3, 4, 5, 6, 7]
}
gs = GridSearchCV(ml_pipe, param_grid, cv=kf)
gs.estimator.get_params().keys()
gs.fit(train, y)
gs.best_params_
gs.best_score_

# --------------------------------------------------------------------------------------------------
ml1_pipe = Pipeline([
    ('transform', transformer),
    ('rf', RandomForestRegressor())
])
param_grid = [
    {'rf__n_estimators': [300, 500],
     'rf__max_features': [100, 200],
     'transform__num_highposskew__kbd__n_bins': [3, 7]},
    {'rf__bootstrap': [False],
     'rf__n_estimators': [300, 500],
     'rf__max_features': [100, 200],
     'transform__num_highposskew__kbd__n_bins': [3, 7]},

]

gs = GridSearchCV(ml1_pipe, param_grid, cv=kf)
gs.fit(train, y)
gs.best_params_
gs.best_score_
