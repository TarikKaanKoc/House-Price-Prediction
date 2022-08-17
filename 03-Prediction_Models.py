################################################
##                                            ##
##           03-Prediction_Models             ##
##                                            ##
################################################


'''
# Used Models:

# Linear Models:

    - Multiple Linear Regression
    - Lasso Regression
    - Ridge Regression
    - ElasticNet Regression

# Non Linear Models:

    - K-Nearest Neighbors Regression
    - Support Vector Machines
    - Classification and Regression Trees
    - DecisionTreeRegressor
    - RandomForestRegressor
    - Gradient Boosting Regressor
    - AdaBoostRegressor
    - XGBoost - XGBRegressor
    - LightGBM - LGBMRegressor
    - CatBoost - CatBoostRegressor
'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('display.max_rows', None)
pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


train = pd.read_pickle("train_final_v.pkl")
test = pd.read_pickle("test_final_v.pkl")


'''
# Linear Models:
   
    - Multiple Linear Regression
    - Lasso Regression
    - Ridge Regression
    - ElasticNet Regression
'''

# split data - dependent variable and Independent variables
X = train.drop('SalePrice', axis=1)
y = np.ravel(train[["SalePrice"]])

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20,
                                                    random_state=123,
                                                    shuffle=True
                                                    )
# Dimension adjustment (for dependent variable! (y))
y_train = np.ravel(y_train)


# Ridge - Model and prediction + Model Tuning

ridge_model = Ridge().fit(X_train, y_train)
ridge_model.coef_[0:10]
'''
array([ 2.69660025e+03,  1.13385719e+03, -5.80409834e+03, -8.37413372e+00,
       -1.55927425e+04, -5.05150393e+04,  1.97114222e+03,  1.59774632e+04,
        1.05892976e+02, -5.86922785e+04])
'''

ridge_model.intercept_ # Out: 111742.86058632667

ridge_model.alpha # Out: 1.0

# Train error
np.sqrt(mean_squared_error(y_train, ridge_model.predict(X_train)))
'''
Train Error Out: 21024.577005001644

'''
# Test error
np.sqrt(mean_squared_error(y_test, ridge_model.predict(X_test)))
'''
Test Error Out: 25626.953350852058
'''

ridge_params = {"alpha": 10 ** np.linspace(10, -2, 100) * 0.5}
ridge_model = Ridge()
ridge_cv_model = GridSearchCV(ridge_model, ridge_params, cv=10).fit(X_train, y_train)
ridge_cv_model.best_params_

'''
OUT: {'alpha': 0.03527401155359316}
'''
# Result :
ridge_tuned = Ridge(**ridge_cv_model.best_params_).fit(X_train, y_train)
y_pred = ridge_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
'''
OUT: 24151.723487933647
'''

# Lasso Regression - Model and prediction + Model Tuning
lasso_model = Lasso(normalize=True).fit(X_train, y_train)
lasso_model.intercept_ # Out: -351542.3093497799
lasso_model.coef_[0:10]
'''
array([ 3.28579890e+03,  2.12416874e+03,  0.00000000e+00, -3.34335228e+00,
       -1.31797801e+04, -8.66442349e+03, -8.56381795e+02,  4.51718493e+03,
        0.00000000e+00, -0.00000000e+00])
'''

# Train Error
lasso_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, lasso_model.predict(X_train)))
'''
Train Error Out: 19850.87171630218
'''

# Test Error
lasso_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, lasso_model.predict(X_test)))
'''
Train Error Out: 23725.701331664066
'''

lasso_params = {"alpha": [1.0, 10 ** np.linspace(10, -2, 100) * 0.5]}
lasso_model = Lasso()
lasso_cv_model = GridSearchCV(lasso_model, lasso_params, cv=10).fit(X_train, y_train)
lasso_cv_model.best_params_  # OUT:  {'alpha': 1.0}

# Result Model
lasso_tuned = Lasso(**lasso_cv_model.best_params_).fit(X_train, y_train)
np.sqrt(mean_squared_error(y_test, lasso_tuned.predict(X_test)))
'''
Out:  24146.952375270077
'''

# ElasticNet Regression - Model and prediction + Model Tuning - Result

elastic_net = ElasticNet().fit(X_train, y_train)
elastic_net.intercept_ # Out : 172627.67627511887
elastic_net.coef_[0:10]
'''
array([ 3.82564666e+03,  5.47293041e+03,  5.32379263e+03, -1.32158277e+00,
       -8.53580374e+01,  7.48115264e+03,  6.56626984e+03,  4.53141414e+03,
       -2.33664941e+01,  1.02578929e+04])
'''

# Train Error
elastic_net.predict(X_train)
np.sqrt(mean_squared_error(y_train, elastic_net.predict(X_train)))
'''
Train Error Out: 34767.25809455722
'''

# Test Error
elastic_net.predict(X_test)
np.sqrt(mean_squared_error(y_test, elastic_net.predict(X_test)))
'''
Test Error Out: 33579.660933337254
'''

elastic_net_params = {"l1_ratio": [0.1, 0.4, 0.5, 0.6, 0.8, 1],
               "alpha": [0.1, 0.01, 0.001, 0.2, 0.3, 0.5, 0.8, 0.9, 1]}
enet_model = ElasticNet()

elastic_net_cv_model = GridSearchCV(enet_model, elastic_net_params, cv=10).fit(X_train, y_train)
elastic_net_cv_model.best_params_ # Out: {'alpha': 0.1, 'l1_ratio': 0.8}

elastic_net_tuned = ElasticNet(**elastic_net_cv_model.best_params_).fit(X_train, y_train)
np.sqrt(mean_squared_error(y_test, elastic_net_tuned.predict(X_test)))
'''
Out: 26538.43872718721
'''

# MODELLING - RECAP


# Evaluate each model in turn by looking at train and test errors and scores
def summary_models(models):
    # Define lists to track names and results for models
    model_name = []
    train_rmse = []
    test_rmse = []
    train_r2_scores = []
    test_r2_scores = []

    print('################ RMSE and R2_score values for test set for the models: ################\n')
    for name, model in models:
        model.fit(X_train, y_train)

        model_train_rmse_result = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
        model_test_rmse_result = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        train_rmse.append(model_train_rmse_result)
        test_rmse.append(model_test_rmse_result)

        train_r2_score = model.score(X_train, y_train)
        test_r2_score = model.score(X_test, y_test)
        train_r2_scores.append(train_r2_score)
        test_r2_scores.append(test_r2_score)

        model_name.append(name)
        msg = "%s: %f --> %f" % (name, model_test_rmse_result, test_r2_score)
        print(msg)

    print('\n################ Train and test results for the model: ################\n')
    data_result = pd.DataFrame({'models': model_name,
                                'rmse_train': train_rmse,
                                'rmse_test': test_rmse,
                                'r2_score_train': train_r2_scores,
                                'r2_score_test': test_r2_scores
                                })
    print(data_result)

    # Plot the results
    plt.figure(figsize = (15, 12))
    sns.barplot(x='rmse_test', y='models', data=data_result, color="r")
    plt.xlabel('RMSE values')
    plt.ylabel('Models')
    plt.title('RMSE For Test Set')
    plt.show()


# See the results for base models
base_models = [#('LinearRegression', LinearRegression()),
               ('Ridge', Ridge()),
               ('Lasso', Lasso()),
               ('ElasticNet', ElasticNet())]

summary_models(base_models)


'''
models          rmse_train  rmse_test  r2_score_train  r2_score_test
0       Ridge   21024.577  25626.953           0.930          0.894
1       Lasso   19684.874  24146.952           0.939          0.906
2  ElasticNet   34767.258  33579.661           0.809          0.818

'''

# See the results for tuned models
tuned_models = [('LinearRegression', LinearRegression()),
                ('Ridge', ridge_tuned),
                ('Lasso', lasso_tuned),
                ('ElasticNet', elastic_net_tuned)]

summary_models(tuned_models)

'''
             models  rmse_train  rmse_test  r2_score_train  r2_score_test
0  LinearRegression   18128.054  24252.594           0.948          0.905
1             Ridge   18892.720  24151.723           0.944          0.906
2             Lasso   19684.874  24146.952           0.939          0.906
3        ElasticNet   25735.288  26538.439           0.896          0.886

'''
################################################################################################################
##################                          NonLinear Models                                  ##################
################################################################################################################

'''
# Used Models:
    - K-Nearest Neighbors Regression
    - Support Vector Machines - NonLinear
    - Classification and Regression Trees 
    - DecisionTreeRegressor
    - RandomForestRegressor
    - Gradient Boosting Regressor
    - AdaBoostRegressor
    - XGBoost - XGBRegressor
    - LightGBM - LGBMRegressor
    - CatBoost - CatBoostRegressor
'''

train = pd.read_pickle("train_final_v.pkl")
test = pd.read_pickle("test_final_v.pkl")


# split data - dependent variable and Independent variables
X = train.drop('SalePrice', axis=1)
y = np.ravel(train[["SalePrice"]])

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20,
                                                    random_state=123,
                                                    shuffle=True
                                                    )
# Dimension adjustment (for dependent variable! (y))
y_train = np.ravel(y_train)



# K NEAREST NEIGHBORS - Model - Prediction - Model Tun - Result
knn_model = KNeighborsRegressor().fit(X_train, y_train)
np.sqrt(mean_squared_error(y_test, knn_model.predict(X_test)))
'''
OUT: 49731.61622365995
'''

# Model Tun
knn_params = {"n_neighbors": np.arange(0, 40, 1)}
knn_model = KNeighborsRegressor()
knn_cv_model = GridSearchCV(knn_model, knn_params, cv=10).fit(X_train, y_train)
knn_cv_model.best_params_
'''
Out: BEST PARAMS {'n_neighbors': 19}
'''

# Result
knn_tuned = KNeighborsRegressor(**knn_cv_model.best_params_).fit(X_train, y_train)
np.sqrt(mean_squared_error(y_test, knn_tuned.predict(X_test)))
'''
Out: 48433.48383975602
'''

# NON-Linear SVR - Model - Model Tun- Result

svr_model = SVR()
svr_params = {"C": [0.01, 0.001, 0.2, 0.1, 0.5, 0.8, 0.9, 1, 10, 100, 500, 1000]}
svr_cv_model = GridSearchCV(svr_model, svr_params, cv=5, n_jobs=-1, verbose=2).fit(X_train, y_train)
svr_cv_model.best_params_
'''
Out: Best Params {'C': 1000}
'''

# Result Model
svr_tuned = SVR(**svr_cv_model.best_params_).fit(X_train, y_train)
np.sqrt(mean_squared_error(y_test, svr_tuned.predict(X_test)))
'''
Out: 79420.23619719454
'''


# CART - Model - Prd - Model Tun - Result

cart_model = DecisionTreeRegressor(random_state=52)
cart_model.fit(X_train, y_train)
np.sqrt(mean_squared_error(y_test, cart_model.predict(X_test)))
'''
Out: 36026.4209478019
'''

# Model Tun
cart_model = DecisionTreeRegressor()
cart_params = {"max_depth": [2, 3, 4, 5, 10, 20, 100, 1000],
              "min_samples_split": [2, 10, 5, 30, 50, 10],
              "criterion" : ["mse", "friedman_mse", "mae"]}
cart_cv_model = GridSearchCV(cart_model, cart_params, cv=10).fit(X_train, y_train)
cart_cv_model.best_params_
''' 
Out: {'criterion': 'mae', 'max_depth': 1000, 'min_samples_split': 50}
'''
# Result
cart_tuned = DecisionTreeRegressor(**cart_cv_model.best_params_).fit(X_train, y_train)
mean_squared_error(y_test, cart_tuned.predict(X_test))
'''
Out: 34020.2145957801
'''
# RF -  Model - Prd - Model Tun - Result

rf_model = RandomForestRegressor(random_state=12345).fit(X_train, y_train)
np.sqrt(mean_squared_error(y_test, rf_model.predict(X_test)))

'''
Out: 27285.072754100634
'''

# Model Tun

rf_params = {"max_depth": [3, 5, 8, 10, 15, None],
           "max_features": [5, 10, 15, 20, 50, 100],
           "n_estimators": [200, 500, 1000],
           "min_samples_split": [2, 5, 10, 20, 30, 50]}

rf_cv_model = GridSearchCV(rf_model, rf_params, cv=10, n_jobs=-1, verbose=1).fit(X_train, y_train)
rf_cv_model.best_params_

'''
Out: Best Params 
{
 'max_depth': 15,
 'max_features': 50,
 'min_samples_split': 2,
 'n_estimators': 1000
 }
'''

# Result:

rf_tuned = RandomForestRegressor(**rf_cv_model.best_params_).fit(X_train, y_train)
np.sqrt(mean_squared_error(y_test, rf_tuned.predict(X_test)))
'''
Out: 25966.276588360863
'''


# Gradient Boosting Regressor -  Model - Prd - Model Tun - Result

gbm_model = GradientBoostingRegressor().fit(X_train, y_train)
np.sqrt(mean_squared_error(y_test, gbm_model.predict(X_test)))
'''
Out: 26013.718696569016
'''

# Model Tun

gbm_model = GradientBoostingRegressor()

gbm_params = {"learning_rate": [0.001, 0.1, 0.01, 0.05],
              "max_depth": [3, 5, 8, 10, 20, 30],
              "n_estimators": [200, 500, 1000, 1500, 5000],
              "subsample": [1, 0.4, 0.5, 0.7],
              "loss": ["ls", "lad", "quantile"]}

gbm_cv_model = GridSearchCV(gbm_model, gbm_params, cv=10, n_jobs=-1, verbose=2).fit(X_train, y_train)
gbm_cv_model.best_params_
'''
Out Best Params: learning_rate=0.001, loss=ls, max_depth=5, n_estimators=1000, subsample=0.5
'''

# Result:
gbm_tuned = GradientBoostingRegressor(**gbm_cv_model.best_params_).fit(X_train, y_train)
print(np.sqrt(mean_squared_error(y_test, gbm_tuned.predict(X_test)))) # 21693.081005924025


# XGBoost - Model - Prd - Model Tun - Result


xgb_model = XGBRegressor().fit(X_train, y_train)
np.sqrt(mean_squared_error(y_test, xgb_model.predict(X_test))) # 26777.163722217178

# Model Tuning

xgb_params = {"learning_rate": [0.1, 0.01, 0.5],
             "max_depth": [5, 8, 15, 20],
             "n_estimators": [100, 200, 500, 1000],
             "colsample_bytree": [0.4, 0.7, 1]}

xgb_cv_model = GridSearchCV(xgb_model, xgb_params, cv=10, n_jobs=-1, verbose=2).fit(X_train, y_train)
xgb_cv_model.best_params_

''''
{'colsample_bytree': 0.4, 'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 1000}
'''

# Result:
xgb_tuned = XGBRegressor(**xgb_cv_model.best_params_).fit(X_train, y_train)
np.sqrt(mean_squared_error(y_test,  xgb_tuned.predict(X_test))) # 22613.514160521838


# LightGBM

lgbm_model = LGBMRegressor().fit(X_train, y_train)
np.sqrt(mean_squared_error(y_test, lgbm_model.predict(X_test))) # 24466.61545191471

# Model Tun

lgbm_model = LGBMRegressor()

lgbm_params = {"learning_rate": [0.01, 0.001, 0.1, 0.5, 1],
              "n_estimators": [200, 500, 1000, 5000],
              "max_depth": [6, 8, 10, 15, 20],
              "colsample_bytree": [1, 0.8, 0.5, 0.4]}


lgbm_cv_model = GridSearchCV(lgbm_model, lgbm_params, cv=10, n_jobs=-1, verbose=2).fit(X_train, y_train)
lgbm_cv_model.best_params_ # {'colsample_bytree': 0.4, 'learning_rate': 0.01, 'max_depth': 6, 'n_estimators': 5000}

# Result:

lgbm_tuned = LGBMRegressor(**lgbm_cv_model.best_params_).fit(X_train, y_train)
y_pred = lgbm_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred)) # 21418.842759852396

# CatBoost -  Model - Prd - Model Tun - Result

catb_model = CatBoostRegressor(verbose=False).fit(X_train, y_train)
np.sqrt(mean_squared_error(y_test, catb_model.predict(X_test))) # 21122.3517877923

# Modle Tun
catb_model = CatBoostRegressor()
catb_params = {"iterations": ['None', 200, 500],
               "learning_rate": ['None', 0.01, 0.1],
               "depth": ['None', 3, 6]}

catb_cv_model = GridSearchCV(catb_model, catb_params, cv=5, n_jobs=-1, verbose=2).fit(X_train, y_train)
catb_cv_model.best_params_ # {'depth': 6, 'iterations': 500, 'learning_rate': 0.1}

#Result
catb_tuned = CatBoostRegressor(**catb_cv_model.best_params_).fit(X_train, y_train)
np.sqrt(mean_squared_error(y_test, y_pred)) # 21122.3517877923


# Evaluate each model in turn by looking at train and test errors and scores
def summary_models(models):
    # Define lists to track names and results for models
    model_name = []
    train_rmse = []
    test_rmse = []
    train_r2_scores = []
    test_r2_scores = []

    print('################ RMSE and R2_score values for test set for the models: ################\n')
    for name, model in models:
        model.fit(X_train, y_train)

        model_train_rmse_result = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
        model_test_rmse_result = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        train_rmse.append(model_train_rmse_result)
        test_rmse.append(model_test_rmse_result)

        train_r2_score = model.score(X_train, y_train)
        test_r2_score = model.score(X_test, y_test)
        train_r2_scores.append(train_r2_score)
        test_r2_scores.append(test_r2_score)

        model_name.append(name)
        msg = "%s: %f --> %f" % (name, model_test_rmse_result, test_r2_score)
        print(msg)

    print('\n################ Train and test results for the model: ################\n')
    data_result = pd.DataFrame({'models': model_name,
                                'rmse_train': train_rmse,
                                'rmse_test': test_rmse,
                                'r2_score_train': train_r2_scores,
                                'r2_score_test': test_r2_scores
                                })
    print(data_result)

    # Plot the results
    plt.figure(figsize = (15, 12))
    sns.barplot(x='rmse_test', y='models', data=data_result, color="r")
    plt.xlabel('RMSE values')
    plt.ylabel('Models')
    plt.title('RMSE For Test Set')
    plt.show()


# See the results for base models
base_models = [#linear algorithms
               ('LinearRegression', LinearRegression()),
               ('Ridge', Ridge()),
               ('Lasso', Lasso()),
               ('ElasticNet', ElasticNet()),
               # nonlinear algorithms
               ('KNN', KNeighborsRegressor()),
               ('SVR', SVR()),
               ('CART', DecisionTreeRegressor()),
               ('RF', RandomForestRegressor()),
               ('GBM', GradientBoostingRegressor()),
               ("XGBoost", XGBRegressor()),
               ("LightGBM", LGBMRegressor()),
               ("CatBoost", CatBoostRegressor(verbose=False))
             ]

summary_models(base_models)

# See the results for tuned models
tuned_models = [('KNN', knn_tuned),
                ('SVR', svr_tuned),
                ('CART', cart_tuned),
                ('RF', rf_tuned),
                ('GBM', gbm_tuned),
                ("XGBoost", xgb_tuned),
                ("LightGBM", lgbm_tuned),
                ("CatBoost", catb_tuned),
                ]

summary_models(tuned_models)