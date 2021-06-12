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
import logging
from datetime import datetime

base_path = r'G:\My Drive\Research\PVRD1\FENICS\SUPG_TRBDF2\simulations\sentaurus_fitting'
sentarus_dataset = r'sentaurus_ml_db.csv'

if __name__ == "__main__":
    if platform.system() == 'Windows':
        base_path = r'\\?\\' + base_path

    # Create a logger
    logFile = 'fit_{0}.log'.format(datetime.today().strftime('%Y-%m-%d'))
    logFile = os.path.join(base_path, logFile)
    my_logger = logging.getLogger('fitlog')
    my_logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(logFile)
    fh.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # add the handlers to the logger
    my_logger.addHandler(fh)
    my_logger.addHandler(ch)

    df = pd.read_csv(os.path.join(base_path, sentarus_dataset))
    # If fitting pmpp uncomment the next line
    column_list_pmpp = list(set(list(df.columns)) - set(['Rsh (Ohms cm2)', 'time (s)']))
    # If fitting rsh uncomment the next line
    column_list_rsh = list(set(list(df.columns)) - set(['pd_mpp (mW/cm2)', 'time (s)']))
    column_list_pmpp.sort()
    column_list_rsh.sort()
    df_mpp = df[column_list_pmpp]
    df_rsh = df[column_list_rsh]
    my_logger.info(df_mpp.tail())
    my_logger.info(df_mpp.describe())

    my_logger.info(df_rsh.tail())
    my_logger.info(df_rsh.describe())

    # If fitting pmpp uncomment the next line
    target_column_mpp = ['pd_mpp (mW/cm2)']
    # If fitting rsh uncomment the next line
    target_column_rsh = ['Rsh (Ohms cm2)']
    predictors = list(set(list(df_mpp.columns)) - set(target_column_mpp))
    # predictors_rsh = list(set(list(df_rsh.columns)) - set(target_column_rsh))
    # df[predictors] = df[predictors]
    print(df.describe())

    X_mpp = df_mpp[predictors].values
    y_mpp = df_mpp[target_column_mpp].values
    # If Rsh take the log10
    X_rsh = df_rsh[predictors].values
    y_rsh = df_rsh[target_column_rsh].values
    y_rsh = np.log10(y_rsh)

    X_train_mpp, X_test_mpp, y_train_mpp, y_test_mpp = train_test_split(
        X_mpp, y_mpp, test_size=0.30, random_state=100
    )

    X_train_rsh, X_test_rsh, y_train_rsh, y_test_rsh = train_test_split(
        X_rsh, y_rsh, test_size=0.30, random_state=100
    )

    my_logger.info('Shape of X_train_mpp')
    my_logger.info(X_train_mpp.shape)
    my_logger.info('Shape of X_test_mpp')
    my_logger.info(X_test_mpp.shape)

    # RF model
    # model_rf = RandomForestRegressor(n_estimators=1000, oob_score=True, random_state=100)
    # model_rf.fit(X_train, y_train.ravel())

    # Multi layer perceptron regressor (mpp settings)
    hls_mpp = [X_mpp.shape[1]*2 for i in range(40)]
    hls_mpp = tuple(hls_mpp)
    model_mlp_mpp = MLPRegressor(
        random_state=100, hidden_layer_sizes=hls_mpp, max_iter=1000000, tol=1E-6, max_fun=1000000,
        solver='adam', activation='relu', alpha=1E-3, verbose=True, learning_rate_init=1E-4,
        learning_rate='adaptive', n_iter_no_change=20
    )

    # Rsh settings
    hls_rsh = [X_rsh.shape[1]*2 for i in range(40)]
    hls_rsh = tuple(hls_rsh)
    model_mlp_rsh = MLPRegressor(
        random_state=100, hidden_layer_sizes=hls_rsh, max_iter=1000000, tol=1E-6, max_fun=1000000,
        solver='adam', activation='relu', alpha=1E-3, verbose=True, learning_rate_init=1E-4,
        learning_rate='adaptive', n_iter_no_change=20
    )

    model_mlp_mpp.fit(X_train_mpp, y_train_mpp.ravel())
    model_mlp_rsh.fit(X_train_rsh, y_train_rsh.ravel())

    # pred_train_rf = model_rf.predict(X_train)
    # print('******* Random Forest Regressor train *********')
    # print('Sqrt(mean variance): {0}'.format(np.sqrt(mean_squared_error(y_train, pred_train_rf))))
    # print('Coefficient of determination: {0}'.format(r2_score(y_train, pred_train_rf)))
    #
    # pred_test_rf = model_rf.predict(X_test)
    # print('******* Random Forest Regressor test *********')
    # print('Sqrt(mean variance): {0}'.format(np.sqrt(mean_squared_error(y_test, pred_test_rf))))
    # print('Coefficient of determination: {0}'.format(r2_score(y_test, pred_test_rf)))

    pred_train_mlp_mpp = model_mlp_mpp.predict(X_train_mpp)
    my_logger.info('******* Multilevel Perceptron Train MPP *********')
    my_logger.info('Sqrt(mean variance): {0}'.format(
        np.sqrt(mean_squared_error(y_train_mpp, pred_train_mlp_mpp))
    ))
    my_logger.info('Coefficient of determination: {0}'.format(
        r2_score(y_train_mpp, pred_train_mlp_mpp)
    ))

    pred_test_mlp_mpp = model_mlp_mpp.predict(X_test_mpp)
    my_logger.info('******* Multilevel Perceptron Test MPP *********')
    my_logger.info('Sqrt(mean variance): {0}'.format(
        np.sqrt(mean_squared_error(y_test_mpp, pred_test_mlp_mpp))
    ))
    my_logger.info('Coefficient of determination: {0}'.format(
        r2_score(y_test_mpp, pred_test_mlp_mpp)
    ))

    # Rsh

    pred_train_mlp_rsh = model_mlp_rsh.predict(X_train_rsh)
    my_logger.info('******* Multilevel Perceptron Train RSH *********')
    my_logger.info('Sqrt(mean variance): {0}'.format(
        np.sqrt(mean_squared_error(y_train_rsh, pred_train_mlp_rsh))
    ))
    my_logger.info('Coefficient of determination: {0}'.format(
        r2_score(y_train_rsh, pred_train_mlp_rsh)
    ))

    pred_test_mlp_rsh = model_mlp_rsh.predict(X_test_rsh)
    my_logger.info('******* Multilevel Perceptron Test Rsh *********')
    my_logger.info('Sqrt(mean variance): {0}'.format(
        np.sqrt(mean_squared_error(y_test_rsh, pred_test_mlp_rsh))
    ))
    my_logger.info('Coefficient of determination: {0}'.format(
        r2_score(y_test_rsh, pred_test_mlp_rsh)
    ))

    # print("number of untis in layer 1: {0}".format(X.shape[1]))
    """
    -------------------- MPP ---------------------
    ******* Random Forest Regressor train *********
    Sqrt(mean variance): 0.2834322409863376
    Coefficient of determination: 0.9988144922159324
    ******* Random Forest Regressor test *********
    Sqrt(mean variance): 0.5020983002872765
    Coefficient of determination: 0.9954081556964257
    ******* Multilevel Perceptron Train *********
    Sqrt(mean variance): 1.314241657720306
    Coefficient of determination: 0.9745107905104172
    ******* Multilevel Perceptron Test *********
    Sqrt(mean variance): 1.7160158298772783
    Coefficient of determination: 0.9463645126447843
    -------------------- RSH ---------------------
    Training loss did not improve more than tol=0.000001 for 10 consecutive epochs. Stopping.
    ******* Random Forest Regressor train *********
    Sqrt(mean variance): 0.04549899799729554
    Coefficient of determination: 0.9993709137710272
    ******* Random Forest Regressor test *********
    Sqrt(mean variance): 0.2105159670086715
    Coefficient of determination: 0.9860048262502275
    ******* Multilevel Perceptron Train *********
    Sqrt(mean variance): 0.44047169932064073
    Coefficient of determination: 0.9410420284684089
    ******* Multilevel Perceptron Test *********
    Sqrt(mean variance): 0.5423786549344012
    Coefficient of determination: 0.9071004968394786
    number of untis in layer 1: 200
    """

    # If fitting mpp append mpp to the pickle, if fitting rsh append rsh
    # dump(model_rf, os.path.join(base_path, 'random_forest_mpp.joblib'))
    dump(model_mlp_mpp, os.path.join(base_path, 'multilevel_perceptron_mpp.joblib'))
    dump(model_mlp_rsh, os.path.join(base_path, 'multilevel_perceptron_rsh.joblib'))

    print('is rsh in predictors?: {0}'.format(target_column_rsh in predictors))
    print('is mpp in predictors?: {0}'.format(target_column_mpp in predictors))
    print('length of predictors: {0}'.format(len(predictors)))