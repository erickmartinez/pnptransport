import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import validation_curve
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from joblib import dump, load
import platform
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt

base_path = r'./'
sentarus_dataset = r'sentaurus_ml_db.csv'
results_path = r'./results'


def plot_learning_curve(
        estimator, ax, title, X, y, ylim=None, cv=None,
        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 10)
):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
    ax:
        Figure axes

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    train_sizes:
        x1 = np.linspace(0, 10, 8, endpoint=True) produces
        8 evenly spaced points in the range 0 to 10


    """

    # plt.figure()
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='neg_mean_squared_error')
    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = -np.std(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    test_scores_std = -np.std(test_scores, axis=1)
    plt.grid()

    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1,
                    color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
            label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
            label="Cross-validation score")

    ax.legend(loc="best")
    return plt


def plot_cv_curve(ax, title, grid_search_df, alphas):
    ax.fill_between(
        alphas,
        -grid_search_df['mean_train_score'] - grid_search_df['std_train_score'],
        - grid_search_df['mean_train_score'] + grid_search_df['std_train_score'],
        color='C0', alpha=0.25
    )

    ax.fill_between(
        alphas,
        -grid_search_df['mean_test_score'] - grid_search_df['std_test_score'],
        - grid_search_df['mean_test_score'] + grid_search_df['std_test_score'],
        color='C1', alpha=0.25
    )

    ax.plot(alphas, -grid_search_df['mean_train_score'], color='C0', label='Training')
    ax.plot(alphas, -grid_search_df['mean_test_score'], color='C1', label='Test')

    ax.set_xscale('log')
    ax.legend(loc='best', frameon=True)
    ax.set_xlabel(r'n estimators')
    ax.set_ylabel(r'MSE')

    ax.set_title(title)
    plt.grid()

    return plt


if __name__ == "__main__":
    if platform.system() == 'Windows':
        base_path = r'\\?\\' + base_path
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Create a logger
    # logFile = 'fit_{0}.log'.format(datetime.today().strftime('%Y-%m-%d'))
    # logFile = os.path.join(base_path, logFile)
    # my_logger = logging.getLogger('fitlog')
    # my_logger.setLevel(logging.DEBUG)
    # # create file handler which logs even debug messages
    # fh = logging.FileHandler(logFile)
    # fh.setLevel(logging.DEBUG)
    # # create console handler and set level to debug
    # ch = logging.StreamHandler()
    # ch.setLevel(logging.DEBUG)
    # # add the handlers to the logger
    # my_logger.addHandler(fh)
    # my_logger.addHandler(ch)

    df = pd.read_csv(os.path.join(base_path, sentarus_dataset))
    # If fitting pmpp uncomment the next line
    column_list_pmpp = list(set(list(df.columns)) - set(['Rsh (Ohms cm2)', 'time (s)']))
    # If fitting rsh uncomment the next line
    column_list_rsh = list(set(list(df.columns)) - set(['pd_mpp (mW/cm2)', 'time (s)']))
    column_list_pmpp.sort()
    column_list_rsh.sort()
    df_mpp = df[column_list_pmpp]
    df_rsh = df[column_list_rsh]
    # my_logger.info(df_mpp.tail())
    # my_logger.info(df_mpp.describe())
    #
    # my_logger.info(df_rsh.tail())
    # my_logger.info(df_rsh.describe())

    # If fitting pmpp uncomment the next line
    target_column_mpp = ['pd_mpp (mW/cm2)']
    # If fitting rsh uncomment the next line
    target_column_rsh = ['Rsh (Ohms cm2)']
    # predictors = list(set(list(df_mpp.columns)) - set(target_column_mpp))
    # predictors_rsh = list(set(list(df_rsh.columns)) - set(target_column_rsh))
    # df[predictors] = df[predictors]
    n_c_points = 201
    # The maximum depth in um to take for the concentration profile
    x_max = 1.
    x_inter = np.linspace(start=0., stop=x_max, num=n_c_points)
    # x_inter = utils.geometric_series_spaced(max_val=x_max, steps=(n_c_points-1), min_delta=1E-5)
    predictors = ['sigma at {0:.3E} um'.format(x) for x in x_inter]
    print(df.describe())

    X_mpp = df_mpp[predictors].values
    y_mpp = df_mpp[target_column_mpp].values
    # If Rsh take the log10
    X_rsh = df_rsh[predictors].values
    y_rsh = df_rsh[target_column_rsh].values
    y_rsh = np.log10(y_rsh)

    n_estimators = [30, 60, 100, 300, 600, 1000, 3000]  # list(np.logspace(start=-3, stop=1, num=9))

    model_gbr_mpp = GradientBoostingRegressor()
    model_gbr_rsh = GradientBoostingRegressor()

    pipe_mpp = Pipeline(steps=[('normalize', StandardScaler()), ('gbr', model_gbr_mpp)])
    pipe_rsh = Pipeline(steps=[('normalize', StandardScaler()), ('gbr', model_gbr_rsh)])
    param_grid = {
        'gbr__n_estimators': n_estimators,
        'gbr__subsample': [1.0],
        'gbr__max_depth': [10],
        'gbr__max_features': ['auto'],
        'gbr__learning_rate': [1E-1],
        'gbr__criterion': ['mse'],
        'gbr__n_iter_no_change': [20],
        'gbr__tol': [1E-6],
        'gbr__random_state': [10]
    }

    kf = KFold(n_splits=10, shuffle=True, random_state=10)

    gs_mpp = GridSearchCV(
        pipe_mpp,
        param_grid=param_grid,
        cv=kf,
        verbose=1,
        n_jobs=-1,
        return_train_score=True,
        scoring='neg_mean_squared_error'
    )

    gs_rsh = GridSearchCV(
        pipe_rsh,
        param_grid=param_grid,
        cv=kf,
        verbose=1,
        n_jobs=-1,
        return_train_score=True,
        scoring='neg_mean_squared_error'
    )

    regr_mpp = gs_mpp.fit(X_mpp, y_mpp.ravel())
    regr_rsh = gs_rsh.fit(X_rsh, y_rsh.ravel())

    grid_search_mpp_df = pd.DataFrame(regr_mpp.cv_results_)
    print(grid_search_mpp_df.describe())

    grid_search_rsh_df = pd.DataFrame(regr_rsh.cv_results_)
    print(grid_search_rsh_df.describe())

    fig_cv_mpp = plt.figure(1)
    fig_cv_mpp.set_size_inches(4.75, 3.0, forward=True)
    ax1_cv_mpp = fig_cv_mpp.add_subplot(1, 1, 1)
    title = r"Cross Validation Curve (GBR) MPP"

    plot_cv_curve(ax=ax1_cv_mpp, title=title, grid_search_df=grid_search_mpp_df, alphas=n_estimators)

    fig_cv_rsh = plt.figure(2)
    fig_cv_rsh.set_size_inches(4.75, 3.0, forward=True)
    ax1_cv_rsh = fig_cv_rsh.add_subplot(1, 1, 1)
    title = r"Cross Validation Curve (GBR) RSH"

    plot_cv_curve(ax=ax1_cv_rsh, title=title, grid_search_df=grid_search_rsh_df, alphas=n_estimators)

    fig_lc_mpp = plt.figure(3)
    fig_lc_mpp.set_size_inches(4.75, 3.0, forward=True)
    ax_lc_mpp = fig_lc_mpp.add_subplot(1, 1, 1)

    title = r"Learning Curve (GBR) estimators={0:g} MPP".format(regr_mpp.best_params_['gbr__n_estimators'])
    kf = KFold(n_splits=10, shuffle=True, random_state=10)
    plot_learning_curve(
        estimator=regr_mpp.best_estimator_, ax=ax_lc_mpp, title=title, X=X_mpp, y=y_mpp.ravel(),
        ylim=None, cv=kf, n_jobs=-1
    )

    fig_lc_rsh = plt.figure(4)
    fig_lc_rsh.set_size_inches(4.75, 3.0, forward=True)
    ax_lc_rsh = fig_lc_rsh.add_subplot(1, 1, 1)

    title = r"Learning Curve (GBR) estimators={0:g} RSH".format(regr_rsh.best_params_['gbr__n_estimators'])

    plot_learning_curve(
        estimator=regr_rsh.best_estimator_, ax=ax_lc_rsh, title=title, X=X_rsh, y=y_rsh.ravel(),
        ylim=None, cv=kf, n_jobs=-1
    )

    fig_cv_mpp.tight_layout()
    fig_lc_mpp.tight_layout()
    fig_cv_rsh.tight_layout()
    fig_lc_rsh.tight_layout()
    plt.show()

    dump(
        regr_mpp.best_estimator_,
        os.path.join(results_path, 'gbr_optimized_mpp.joblib'),
        compress=1
    )

    dump(
        regr_rsh.best_estimator_,
        os.path.join(results_path, 'gbr_optimized_rsh.joblib'),
        compress=1
    )

    fig_cv_mpp.savefig(
        os.path.join(results_path, 'cross_validation_curve_mpp_gbr.png'),
        dpi=300
    )
    fig_lc_mpp.savefig(
        os.path.join(results_path, 'learning_curve_mpp_gbr.png'), dpi=300
    )
    fig_cv_rsh.savefig(
        os.path.join(results_path, 'cross_validation_rsh_gbr.png'),
        dpi=300
    )
    fig_lc_rsh.savefig(
        os.path.join(results_path, 'learning_curve_rsh_gbr.png'), dpi=300
    )

    # Save SVGs
    fig_cv_mpp.savefig(
        os.path.join(results_path, 'cross_validation_curve_mpp_gbr.svg'),
        dpi=300
    )
    fig_lc_mpp.savefig(
        os.path.join(results_path, 'learning_curve_mpp_gbr.svg'), dpi=600
    )
    fig_cv_rsh.savefig(
        os.path.join(results_path, 'cross_validation_rsh_gbr.svg'),
        dpi=600
    )
    fig_lc_rsh.savefig(
        os.path.join(results_path, 'learning_curve_rsh_gbr.svg'), dpi=600
    )
