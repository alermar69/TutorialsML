import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

from handson.GetData import load_housing_data
from handson.TransformerData import housingPrepared, splitData, BasicTransformer
from handson.Model import DataAlg, choise_model, display_scores, model_fit_cross
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from scipy.stats import randint

df = load_housing_data()

start_train, start_test = splitData(df)

X_train = start_train.copy()
# y_train = X_train.pop('median_house_value')


train = DataAlg(X_train)
train1 = DataAlg(X_train, column_target='median_house_value')
train1.desc()


bt = BasicTransformer(cat_threshold=3, return_df=True)
X_train_prepared1 = bt.fit_transform(X_train)

# X_train = start_train.drop('median_house_value', axis=1)
# y_train = start_train['median_house_value'].copy()

full_pipeline = housingPrepared(X_train)
X_train_prepared = full_pipeline.transform(X_train)

train = DataAlg(X_train_prepared, y_train)
train1 = DataAlg(X_train_prepared1, y_train)

# X_train.hist(bins=50, figsize=(15,7))
# plt.show()

# X_train_prepared1.hist(bins=50, figsize=(15,7))
# plt.show()

model = choise_model(train)
model1 = choise_model(train1)

# --------------------------------------------------------------------------------------------------
param_grid = [
    {'n_estimators': [3, 10, 30, 100], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10, 50], 'max_features': [2, 3, 4]},
]

param_grid = [
    {'n_estimators': [100, 200, 300], 'max_features': [8, 10, 12]},
    {'bootstrap': [False], 'n_estimators': [50, 100, 200, 300], 'max_features': [4, 8, 10]},
]

forest_reg = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(*train.X_y)
grid_search.best_params_
np.sqrt(-grid_search.best_score_)

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

#----------------------------------------------------------------------
basic_pipe = Pipeline([
    ('bt', BasicTransformer(cat_threshold=3, return_df=True)),
    ('rf', RandomForestRegressor(random_state=42))
])

param_grid = [
    # {'rf__n_estimators': [100, 200, 300], 'max_features': [8, 10, 12]},
    {'rf__bootstrap': [False], 'rf__n_estimators': [300], 'rf__max_features': [6, 10],
     'bt__asymmetry': [False, True], 'bt__num_strategy': ['median', 'mean']},
]

gs = GridSearchCV(basic_pipe, param_grid, cv=3)
gs.fit(*train1.X_y)
gs.best_params_
gs.best_score_

# --------------------------------------------------------------------------------------------------
param_distribs = {
    'n_estimators': randint(low=1, high=200),
    'max_features': randint(low=1, high=8),
}

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(*train)

cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

# ------------------------------------------------------------------------------------------------
num_attribs = list(X_train.drop('ocean_proximity', axis=1))

feature_importances = grid_search.best_estimator_.feature_importances_
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
# cat_encoder = cat_pipeline.named_steps["cat_encoder"] # old solution
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)

# -----------------------------------------------------------------------------------------------
final_model = grid_search.best_estimator_

X_test = start_test.drop("median_house_value", axis=1)
y_test = start_test["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)













# lin_reg = LinearRegression()
# lin_reg.fit(X_train_prepared, y_train)
#
# some_X = X_train.iloc[:5]
# some_y = y_train.iloc[:5]
# some_X_prepared = full_pipeline.transform(some_X)
# lin_reg.predict(some_X_prepared)
#
# y_pred = lin_reg.predict(X_train_prepared)
# lin_mse = mean_squared_error(y_train, y_pred)
# lin_rmse = np.sqrt(lin_mse)
# lin_mae = mean_absolute_error(y_train, y_pred)
#
# tree_reg = DecisionTreeRegressor()
# tree_reg.fit(*train.X_y)
# y_pred = tree_reg.predict(X_train_prepared)
# tree_mse = mean_squared_error(y_train, y_pred)
# tree_rmse = np.sqrt(tree_mse)
# tree_mae = mean_absolute_error(y_train, y_pred)
#
# scores = cross_val_score(tree_reg, X_train_prepared, y_train, scoring='neg_mean_squared_error', cv=10)
# tree_rmse_scores = np.sqrt(-scores)
# display_scores(tree_rmse_scores)
#
# lin_scores = cross_val_score(lin_reg, X_train_prepared, y_train, scoring="neg_mean_squared_error", cv=10)
# lin_rmse_scores = np.sqrt(-lin_scores)
# display_scores(lin_rmse_scores)
#
# forest_reg = RandomForestRegressor()
# forest_reg.fit(*train)
# y_pred = forest_reg.predict(train[0])
# forest_mse = mean_squared_error(y_train, y_pred)
# forest_rmse = np.sqrt(forest_mse)
# forest_scores = cross_val_score(forest_reg, *train, scoring="neg_mean_squared_error", cv=10)
# forest_rmse_scores = np.sqrt(-forest_scores)
# display_scores(forest_rmse_scores)
#
# svm_reg = SVR(kernel="linear")
# svm_reg.fit(*train)
# y_pred = svm_reg.predict(train[0])
# svm_mse = mean_squared_error(y_train, y_pred)
# svm_rmse = np.sqrt(svm_mse)
# svm_rmse
