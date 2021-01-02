import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


class DataAlg:
    def __init__(self, X, y=None, column_target=None):

        self.X = None
        self.y = None
        self.y_pred = None

        if isinstance(X, pd.core.frame.DataFrame):
            self.data = X.copy()
            self.X = X.copy()
        if isinstance(y, pd.core.frame.DataFrame):
            self.y = y.copy()

        # Если X или y явлется list, Series или ndarray, делаем их DataFrame
        if isinstance(X, (list, pd.core.series.Series, np.ndarray)):
            self.data = pd.DataFrame(X)
            self.X = self.data.copy()
        if isinstance(y, (list, pd.core.series.Series, np.ndarray)):
            self.y = pd.DataFrame(y)

        # Если указан имя столбца по которому надо сделать целевое значение
        if column_target:
            if column_target in self.X.columns.values:
                self.y = self.X.pop(column_target)

        self.X_y = (self.X, self.y)
        self.split_num_cat()

    def split_train_test(self, count=0):
        if count != 0:
            X_train, X_test, y_train, y_test = self.X[:count], self.X[count:], self.y[:count], self.y[count:]

    def split_num_cat(self):
        # подразумевает, что X - это объект DataFrame
        self._columns = self.X.columns.values

        # разбиваем данные на категориальные и количественные признаки
        self._dtypes = self.X.dtypes.values
        self._kinds = np.array([dt.kind for dt in self.X.dtypes])
        self._column_dtypes = {}
        is_cat = self._kinds == 'O'

        self._column_dtypes['cat'] = self._columns[is_cat]
        self._column_dtypes['num'] = self._columns[~is_cat]

        self._feature_names = np.empty(0)
        self._feature_names = np.append(self._feature_names, self._column_dtypes['num'])
        self._feature_names = np.append(self._feature_names, self._column_dtypes['cat'])

        # ассиметрия----------------------------------------------------------------------------------
        self._skew = self.X.skew()

        # выделяем список признаков с небольшой отрицательной асимметрией
        self._neg_skew_num_columns = self._skew[self._skew < 0].index.values

        # выделяем список признаков с высокой положительной асимметрией
        self._high_pos_skew_num_columns = self._skew[self._skew > 7].index.values

        # создадим булев массив
        not_neg_high_pos_skew_num_columns = ~np.isin(
            self._column_dtypes['num'], np.r_[self._high_pos_skew_num_columns, self._neg_skew_num_columns])

        # из списка количественных признаков удалим количественные признаки
        # с небольшой отрицательной и высокой положительной асимметрией
        self._mean_skew_num_columns = self._column_dtypes['num'][not_neg_high_pos_skew_num_columns]

    def desc(self):
        print('КОЛИЧЕСТВЕННЫЕ ПРИЗНАКИ:')
        print(' ' * 5 + 'Имя:    ', self._column_dtypes['num'])
        print(' ' * 5 + 'Количество:    ', len(self._column_dtypes['num']))
        print()
        print('-' * 50)
        print('КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ:')
        print(' ' * 5 + 'Имя:    ', self._column_dtypes['cat'])
        print(' ' * 5 + 'Количество:    ', len(self._column_dtypes['cat']))


def display_scores(scores, name_model=''):
    print("Model:", name_model, '------------------------------------------------------------------')
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    print()


def choise_model(data):
    # lin_reg, lin_scores = model_fit_cross(LinearRegression(), *data.X_y)
    # tree_reg, tree_scores = model_fit_cross(DecisionTreeRegressor(), *data.X_y)
    # forest_reg, forest_scores = model_fit_cross(RandomForestRegressor(), *data.X_y)
    # svm_reg, svm_scores = model_fit_cross(SVR(kernel="linear"), *data.X_y)

    lin_reg = model_fit_cross(LinearRegression(), *data.X_y)
    tree_reg = model_fit_cross(DecisionTreeRegressor(), *data.X_y)
    forest_reg = model_fit_cross(RandomForestRegressor(), *data.X_y)
    svm_reg = model_fit_cross(SVR(kernel="linear"), *data.X_y)

    model_reg = min([lin_reg, tree_reg, forest_reg, svm_reg], key=lambda x: x[1].mean())

    print()
    print()
    print('Наилучшаяя модель: ', model_reg[0])
    print('Ошибка: ', model_reg[1].mean())
    print('Отклонение: ', model_reg[1].std())

    return model_reg


def model_fit_cross(model, X, y, scoring="neg_mean_squared_error", cv=5):
    scores = cross_val_score(model, X, y, scoring=scoring, cv=cv)
    rmse_scores = np.sqrt(-scores)
    display_scores(rmse_scores, type(model))
    return model, rmse_scores
