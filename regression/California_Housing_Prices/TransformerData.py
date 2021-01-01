import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler, KBinsDiscretizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer


def splitData(df):
    df['income_cat'] = np.ceil(df['median_income'] / 1.5)
    df['income_cat'].where(df['income_cat'] < 5, 5, inplace=True)

    # df["income_cat"] = pd.cut(df["median_income"],
    #                                bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    #                                labels=[1, 2, 3, 4, 5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(df, df['income_cat']):
        start_train = df.loc[train_index]
        start_test = df.loc[test_index]

    del start_train['income_cat']
    del start_test['income_cat']

    return start_train, start_test

def housingPrepared(X_train):
    X_train_num = X_train.drop('ocean_proximity', axis=1)

    # column index
    col_names = "total_rooms", "total_bedrooms", "population", "households"
    rooms_ix, bedrooms_ix, population_ix, household_ix = [
        X_train.columns.get_loc(c) for c in col_names] # get the column indices

    ix = dict(zip(list(X_train),  list(range(0, len(X_train.columns)))))

    class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
        def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
            self.add_bedrooms_per_room = add_bedrooms_per_room
        def fit(self, X, y=None):
            return self  # nothing else to do
        def transform(self, X):
            rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
            population_per_household = X[:, population_ix] / X[:, household_ix]
            if self.add_bedrooms_per_room:
                bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
                return np.c_[X, rooms_per_household, population_per_household,
                             bedrooms_per_room]
            else:
                return np.c_[X, rooms_per_household, population_per_household]

    def add_extra_features(X, add_bedrooms_per_room=True):
        rooms_per_household = X[:, ix['total_rooms']] / X[:, ix['households']]
        population_per_household = X[:, ix['population']] / X[:, ix['households']]
        if add_bedrooms_per_room:
            bedrooms_per_room = X[:, ix['total_bedrooms']] / X[:, ix['total_rooms']]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

    def log_features(X):

        return np.log1p(X)

    num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            #('attribs_adder', CombinedAttributesAdder()),
            ('attribs_adder', FunctionTransformer(add_extra_features, validate=False)),
            ('std_scaler', StandardScaler()),
        ])


    num_attribs = list(X_train_num)
    cat_attribs = ["ocean_proximity"]

    full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        # ('log', FunctionTransformer(log_features), [''])
        ])

    full_pipeline.fit(X_train)

    # if isFit:
    #     housing_prepared = full_pipeline.fit_transform(X_train)
    # else:
    #     housing_prepared = full_pipeline.transform(X_train)

    return full_pipeline


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

            if len(self._neg_skew_num_columns) > 0:
                self._num_negskew_pipe.fit(X_num[self._neg_skew_num_columns])
            if len(self._mean_skew_num_columns) > 0:
                self._num_meanskewpipe.fit(X_num[self._mean_skew_num_columns])

            self._bins_names = np.empty(0)
            if len(self._high_pos_skew_num_columns) > 0:
                self._num_highposskew_pipe.fit(X_num[self._high_pos_skew_num_columns])
                self._bins_names = self._num_highposskew_pipe.named_steps['kbd']._encoder.get_feature_names()

            self._cat_names = self._feature_names[~np.isin(self._feature_names, self._column_dtypes['num'])]
            # self._feature_names = self._feature_names[~np.isin(self._feature_names, self._high_pos_skew_num_columns)]


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

            X_num = pd.DataFrame()

            if len(self._neg_skew_num_columns) > 0:
                X_num_neg_skew = pd.DataFrame(self._num_negskew_pipe.transform(X_num_neg_skew),
                                    columns=X_num_neg_skew.columns)
                X_num = pd.concat([X_num, X_num_neg_skew], axis=1)

            if len(self._mean_skew_num_columns) > 0:
                X_num_mean_skew = pd.DataFrame(self._num_meanskewpipe.transform(X_num_mean_skew),
                                              columns=X_num_mean_skew.columns)
                X_num = pd.concat([X_num, X_num_mean_skew], axis=1)

            if len(self._high_pos_skew_num_columns) > 0:
                X_num_high_pos_skew = pd.DataFrame(self._num_highposskew_pipe.transform(X_num_high_pos_skew),
                                              columns=self._bins_names)
                X_num = pd.concat([X_num, X_num_high_pos_skew], axis=1)

            # X_num = pd.concat([X_num_neg_skew, X_num_mean_skew, X_num_high_pos_skew], axis=1)

        X_num = X_num.values



        #Категориальные признаки-------------------------------------------------------------
        # создаем отдельный массив для преобразованных категориальных признаков
        # X_cat = np.empty((len(X), self._total_cat_cols), dtype='int')
        X_cat = np.zeros((len(X), self._total_cat_cols), dtype='int')
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