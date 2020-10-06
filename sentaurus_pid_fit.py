"""
https://www.pluralsight.com/guides/non-linear-regression-trees-scikit-learn
"""
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import  DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from joblib import dump, load
import platform
import os

base_path = r'G:\My Drive\Research\PVRD1\FENICS\SUPG_TRBDF2\simulations\sentaurus_fitting'
sentarus_dataset = r'sentaurus_ml_db.csv'

if __name__ == "__main__":
    if platform.system() == 'Windows':
        base_path = r'\\?\\' + base_path

    df = pd.read_csv(os.path.join(base_path, sentarus_dataset))
    # If fitting pmpp uncomment the next line
    column_list = list(set(list(df.columns)) - set(['Rsh (Ohms cm2)', 'time (s)']))
    # If fitting rsh uncomment the next line
    # column_list = list(set(list(df.columns)) - set(['pd_mpp (mW/cm2)', 'time (s)']))
    column_list.sort()
    df = df[column_list]
    print(df)
    print(df.describe())

    # If fitting pmpp uncomment the next line
    target_column = ['pd_mpp (mW/cm2)']
    # If fitting rsh uncomment the next line
    # target_column = ['Rsh (Ohms cm2)']
    predictors = list(set(list(df.columns)) - set(target_column))
    df[predictors] = df[predictors] / df[predictors].max()
    print(df.describe())

    X = df[predictors].values
    y = df[target_column].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=100)
    print(X_train.shape)
    print(X_test.shape)

    # RF model
    model_rf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=100)
    model_rf.fit(X_train, y_train.ravel())

    # Multi layer perceptron regressor
    model_mlp = MLPRegressor(
        random_state=100, hidden_layer_sizes=(500,), max_iter=100000, solver='adam', activation='relu', alpha=1E-4,
    )
    model_mlp.fit(X_train, y_train.ravel())

    pred_train_rf = model_rf.predict(X_train)
    print('******* Random Forest Regressor train *********')
    print('Sqrt(mean variance): {0}'.format(np.sqrt(mean_squared_error(y_train, pred_train_rf))))
    print('Coefficient of determination: {0}'.format(r2_score(y_train, pred_train_rf)))

    pred_test_rf = model_rf.predict(X_test)
    print('******* Random Forest Regressor test *********')
    print('Sqrt(mean variance): {0}'.format(np.sqrt(mean_squared_error(y_test, pred_test_rf))))
    print('Coefficient of determination: {0}'.format(r2_score(y_test, pred_test_rf)))

    pred_train_mlp = model_mlp.predict(X_train)
    print('******* Multilevel Perceptron Train *********')
    print('Sqrt(mean variance): {0}'.format(np.sqrt(mean_squared_error(y_train, pred_train_mlp))))
    print('Coefficient of determination: {0}'.format(r2_score(y_train, pred_train_mlp)))

    pred_test_mlp = model_mlp.predict(X_test)
    print('******* Multilevel Perceptron Test *********')
    print('Sqrt(mean variance): {0}'.format(np.sqrt(mean_squared_error(y_test, pred_test_mlp))))
    print('Coefficient of determination: {0}'.format(r2_score(y_test, pred_test_mlp)))

    # If fitting mpp append mpp to the pickle, if fitting rsh append rsh
    dump(model_rf, os.path.join(base_path, 'random_forest_mpp.joblib'))
    dump(model_rf, os.path.join(base_path, 'multilevel_perceptron_mpp.joblib'))

