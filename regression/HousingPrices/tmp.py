import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer, RobustScaler, KBinsDiscretizer

DATA_PATH = os.path.join("datasets", "HousingPrices")
# train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))

pp = r'e:\Projects\ML\datasets\HousingPrices\train.csv '
train = pd.read_csv(pp)

y = train.pop('SalePrice').values


class BasicTransformer(BaseEstimator):

    def __init__(self, cat_threshold=None, num_strategy='median', return_df=False, asymmetry=True):
        # храним параметры как публичные аттрибуты
        self.cat_threshold = cat_threshold

        if num_strategy not in ['median', 'mean']:
            raise ValueError('num_strategy must be either "mean" or "median"')

        self.num_strategy = num_strategy
        self.return_df = return_df
        self.asymmetry = asymmetry

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

        if self.asymmetry:
            self._skew = X.skew()

            # выделяем список признаков с небольшой отрицательной асимметрией
            self._neg_skew_num_columns = self._skew[self._skew < 0].index.values

            # выделяем список признаков с высокой положительной асимметрией
            self._high_pos_skew_num_columns = self._skew[self._skew > 7].index.values

            # создадим булев массив
            not_neg_high_pos_skew_num_columns = ~np.isin(
                self._column_dtypes['num'], np.r_[self._high_pos_skew_num_columns, self._neg_skew_num_columns])

            # из списка количественных признаков удалим количественные признаки
            # с небольшой отрицательной и высокой положительной асимметрией
            self._mean_skew_num_columns =  self._column_dtypes['num'][not_neg_high_pos_skew_num_columns]

            # создаем конвейер преобразований для количественных признаков
            # с небольшой отрицательной асимметрией
            self._num_negskew_pipe = Pipeline([
                ('square', FunctionTransformer(np.square, validate=False)),
                ('scaler', StandardScaler())
            ])

            # создаем конвейер преобразований для количественных признаков
            # с небольшой и средней положительной асимметрией
            self._num_meanskewpipe = Pipeline([
                ('log', FunctionTransformer(np.log1p, validate=False)),
                ('scaler', RobustScaler())
            ])

            # создаем конвейер преобразований для количественных
            # признаков с высокой асимметрией
            self._num_highposskew_pipe = Pipeline([
                ('sqrt', FunctionTransformer(np.sqrt, validate=False)),
                ('kbd', KBinsDiscretizer(n_bins=5, encode='onehot-dense'))
            ])

            X_num = X[self._column_dtypes['num']]
            self._num_negskew_pipe.fit(X_num[self._neg_skew_num_columns])
            self._num_meanskewpipe.fit(X_num[self._mean_skew_num_columns])
            self._num_highposskew_pipe.fit(X_num[self._high_pos_skew_num_columns])

            self._cat_names = self._feature_names[~np.isin(self._feature_names, self._column_dtypes['num'])]
            # self._feature_names = self._feature_names[~np.isin(self._feature_names, self._high_pos_skew_num_columns)]

            self._bins_names = self._num_highposskew_pipe.named_steps['kbd']._encoder.get_feature_names()
            self._num_names = np.r_[self._neg_skew_num_columns,
                              self._mean_skew_num_columns,
                              self._bins_names,
            ]
            self._feature_names = np.append(self._num_names, self._cat_names)

        return self

    def transform(self, X):

        # проверяем, есть ли у нас объект DataFrame с теми же именами столбцов,
        # что и в том объекте DataFrame, который использовался для обучения
        if set(self._columns) != set(X.columns):
            raise ValueError('Passed DataFrame has diff cols than fit DataFrame')
        elif len(self._columns) != len(X.columns):
            raise ValueError('Passed DataFrame has diff number of cols than fit DataFrame')

        #Числовые признаки------------------------------------------------------------------
        # заполняем пропуски
        X_num = X[self._column_dtypes['num']].fillna(self._num_fill)
        # X_num = X[self._column_dtypes['num']]
        # X_num = pd.DataFrame(SimpleImputer(strategy=self.num_strategy).fit_transform(X_num),
        #                      columns=X_num.columns)

        if not self.asymmetry:
            # стандартизируем количественные признаки
            std = X_num.std()
            X_num = (X_num - X_num.mean()) / std
            zero_std = np.where(std == 0)[0]

            # Если стандартное отклонение 0, то все значения идентичны. Задаем их равными 0.
            if len(zero_std) > 0:
                X_num.iloc[:, zero_std] = 0


        if self.asymmetry:
            X_num_neg_skew = X_num[self._neg_skew_num_columns]
            X_num_high_pos_skew = X_num[self._high_pos_skew_num_columns]
            X_num_mean_skew = X_num[self._mean_skew_num_columns]

            X_num_neg_skew = pd.DataFrame(self._num_negskew_pipe.transform(X_num_neg_skew),
                                 columns=X_num_neg_skew.columns)
            X_num_mean_skew = pd.DataFrame(self._num_meanskewpipe.transform(X_num_mean_skew),
                                          columns=X_num_mean_skew.columns)
            X_num_high_pos_skew = pd.DataFrame(self._num_highposskew_pipe.transform(X_num_high_pos_skew),
                                          columns=self._bins_names)

            X_num = pd.concat([X_num_neg_skew, X_num_mean_skew, X_num_high_pos_skew], axis=1)

        X_num = X_num.values



        #Категориальные признаки-------------------------------------------------------------
        # создаем отдельный массив для преобразованных категориальных признаков
        X_cat = np.empty((len(X), self._total_cat_cols), dtype='int')
        # X_cat = np.zeros((len(X), self._total_cat_cols), dtype='int')
        i = 0
        for col in self._column_dtypes['cat']:
            vals = self._cat_cols[col]
            for val in vals:
                X_cat[:, i] = X[col] == val
                i += 1

        #------------------------------------------------------------------------------------
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




# создаем класс BasicTransformer
class BasicTransformer1(BaseEstimator):
    def __init__(self, cat_threshold=None, num_strategy='median', return_df=False):
        # храним параметры как публичные аттрибуты
        self.cat_threshold = cat_threshold
        if num_strategy not in ['mean', 'median']:
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
train_tf = bt.fit_transform(train)

x1 = train.iloc[5:10]
x2 = bt.transform(x1)

train_tf